import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from apps_core import ASSETS, DATA, canonical, digest, load_cases, read, score, write
from apps_model_runner import PROJECT, run_folder
from apps_sft import build_examples


def run(command, root, label):
    folder = None
    print("STAGE:", label, flush=True)
    with (root / (label + ".txt")).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace",
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
            if line.startswith("EVIDENCE: "):
                folder = Path(line.removeprefix("EVIDENCE: ").strip())
        if process.wait() != 0:
            raise RuntimeError("Stage failed: " + label)
    stages_path = root / "stages.json"
    stages = read(stages_path) if stages_path.exists() else {}
    stages[label] = str(folder) if folder else "completed"
    write(stages_path, stages)
    return folder


def diagnose(root, previous):
    from restricted_python import RestrictedFunction
    cases, manifest = load_cases("validation")
    by_id = {case["case_id"]: case for case in cases}
    rows = [json.loads(line) for line in (Path(previous["baseline_validation"]["path"]) / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    counts = Counter()
    for row in rows:
        case = by_id[row["case_id"]]
        for sample in row["samples"]:
            recomputed = score(sample["raw_output"], sample["terminated"], case)
            for key in ("valid_format", "correct_expected", "bug_detected", "reward"):
                if recomputed[key] != sample[key]:
                    raise RuntimeError("Baseline verifier mismatch")
            counts["bug_detected" if sample["bug_detected"] else "correct_expected_no_bug" if sample["correct_expected"] else "wrong_expected" if sample["valid_format"] else sample.get("error", "invalid")] += 1
    witnesses = {row["case_id"]: row["witness"] for row in read(DATA / "audit.json")["cases"]}
    accepted = 0
    for case in cases:
        witness = witnesses[case["case_id"]]
        raw = canonical({"input": witness["input"], "expected": witness["expected"]})
        checked = score(raw, True, case)
        if not checked["bug_detected"] or checked["reward"] != 1:
            raise RuntimeError("Verifier rejected a known valid bug-detecting test")
        bad = canonical({"input": witness["input"], "expected": {"wrong_expected_control": True}})
        if score(bad, True, case)["bug_detected"]:
            raise RuntimeError("Verifier counted a wrong expected answer as a bug")
        accepted += 1
    report = {"status": "passed", "original_baseline_samples_rescored": sum(counts.values()),
              "failure_counts": dict(counts), "validation_witnesses_accepted": accepted,
              "wrong_expected_controls_rejected": accepted, "dataset_id": manifest["dataset_id"],
              "note": "No blanket false-negative checker failure found in these checks. This does not prove all implementation details are bug-free.",
              "model_test_evaluation": False}
    write(root / "diagnosis.json", report)
    print("DIAGNOSIS:", canonical(report), flush=True)


def main(args):
    root = run_folder("round-v2")
    print("EVIDENCE:", root, flush=True)
    os.environ.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                      HF_HOME=str(ASSETS / "cache/huggingface"), TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"))
    previous = read(PROJECT / "APPS_STATE.json")
    target_order = read(PROJECT / "configs/active-dataset.json").get("sft_target_order", "sorted-v1")
    write(root / "previous-state.json", previous)
    write(root / "previous-active-config.json", read(PROJECT / "configs/active-dataset.json"))
    source = root / "source"
    source.mkdir()
    hashes = {}
    for name in ("apps_core.py", "apps_model_runner.py", "apps_sft.py", "apps_round.py", "audit_apps_run.py", "apps_profiles.py", "restricted_python.py", "local_model_loader.py", "train_static_grpo.py", "check_lora_update.py"):
        shutil.copy2(Path(__file__).with_name(name), source / name)
        hashes[name] = digest(source / name)
    plan = {"status": "planned", "dataset_id": previous["dataset_id"], "prompt_version": "apps-call-v2",
            "source_hashes": hashes, "sft_steps": 64, "grpo_steps": args.grpo_steps, "seed": 42, "sft_target_order": target_order,
            "validation_tests_per_case": 4, "max_completion_tokens": 96,
            "selection": "Final adapter at the declared budget; no best-checkpoint selection or Test evaluation",
            "lineage": "fresh base -> SFT on Train only -> static GRPO on Train only", "adversary_curriculum": False}
    write(root / "plan.json", plan)
    result = {"status": "running", "plan": plan, "model_test_evaluation": False}
    write(root / "result.json", result)
    python = str(ASSETS / "env/Scripts/python.exe")
    runner = [python, "-B", str(PROJECT / "scripts/apps_model_runner.py")]
    try:
        run([python, "-B", str(PROJECT / "scripts/verify_apps.py")], root, "verify-data")
        diagnose(root, previous)
        targets, train_cases, manifest = build_examples(target_order=target_order)
        from transformers import AutoTokenizer
        from apps_model_runner import render_prompt
        tokenizer = AutoTokenizer.from_pretrained(ASSETS / "models/Qwen3.5-2B", local_files_only=True)
        validation, _ = load_cases("validation")
        lengths = [len(tokenizer.encode(render_prompt(tokenizer, case, "apps-call-v2"), add_special_tokens=False)) for case in train_cases + validation]
        if max(lengths) > 1536:
            raise RuntimeError("v2 prompt exceeds fixed 1536-token budget")
        write(root / "preflight.json", {"train_examples": len(targets), "train_cases": len(train_cases), "validation_cases": len(validation), "prompt_tokens_min": min(lengths), "prompt_tokens_max": max(lengths), "test_not_used": True})
        base = run(runner + ["evaluate", "--prompt-version", "apps-call-v2", "--no-publish"], root, "base-validation")
        run([python, "-B", str(PROJECT / "scripts/audit_apps_run.py"), str(base)], root, "audit-base")
        sft = run([python, "-B", str(PROJECT / "scripts/apps_sft.py"), "--steps", "64", "--target-order", target_order], root, "sft")
        run([python, "-B", str(PROJECT / "scripts/audit_apps_run.py"), str(sft)], root, "audit-sft")
        warm = run(runner + ["evaluate", "--adapter", str(sft / "adapter"), "--prompt-version", "apps-call-v2", "--no-publish"], root, "sft-validation")
        run([python, "-B", str(PROJECT / "scripts/audit_apps_run.py"), str(warm)], root, "audit-sft-validation")
        grpo = run(runner + ["train", "--steps", str(args.grpo_steps), "--init-adapter", str(sft / "adapter"), "--baseline-run", str(warm), "--prompt-version", "apps-call-v2", "--no-publish"], root, "grpo")
        run([python, "-B", str(PROJECT / "scripts/audit_apps_run.py"), str(grpo)], root, "audit-grpo")
        final = run(runner + ["evaluate", "--adapter", str(grpo / "adapter"), "--prompt-version", "apps-call-v2", "--no-publish"], root, "grpo-validation")
        run([python, "-B", str(PROJECT / "scripts/audit_apps_run.py"), str(final)], root, "audit-grpo-validation")
        result["runs"] = {"baseline_validation": str(base), "sft_training": str(sft), "sft_validation": str(warm), "pilot_training": str(grpo), "pilot_validation": str(final)}
        comparisons = []
        for folder in (base, warm, final):
            record = read(folder / "result.json")
            if not record["complete_split"] or record["metrics"]["generated_tests"] != 60:
                raise RuntimeError("Incomplete validation comparison")
            rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            comparisons.append({row["case_id"]: (row["prompt_sha256"], row["case_seed"]) for row in rows})
        if not comparisons[0] == comparisons[1] == comparisons[2]:
            raise RuntimeError("Comparison prompt or seed mismatch")
        for name, expected in hashes.items():
            if digest(Path(__file__).with_name(name)) != expected:
                raise RuntimeError("Experiment source changed during execution: " + name)
        new_state = {"dataset_id": manifest["dataset_id"], "dataset_path": str(DATA), "prompt_version": "apps-call-v2", "round_path": str(root), "sft_target_order": target_order}
        for key, folder in result["runs"].items():
            record = {"path": folder, "result_sha256": digest(Path(folder) / "result.json")}
            if key.endswith("training"):
                record.update(adapter_path=str(Path(folder) / "adapter"), adapter_sha256=digest(Path(folder) / "adapter/adapter_model.safetensors"))
            new_state[key] = record
        write(PROJECT / "APPS_STATE.json", new_state)
        config = read(PROJECT / "configs/active-dataset.json")
        config.update(prompt_version="apps-call-v2", stage="sft-and-static-grpo", grpo_start="sft")
        write(PROJECT / "configs/active-dataset.json", config)
        result.update(status="completed", comparison_prompt_and_seed_match=True,
                      metrics={key: read(Path(value) / "result.json")["metrics"] for key, value in result["runs"].items() if key.endswith("validation")})
        write(root / "result.json", result)
        print("ROUND COMPLETED:", canonical(result["metrics"]), flush=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        write(root / "result.json", result)
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--grpo-steps", type=int, default=96)
    args = parser.parse_args()
    if not 1 <= args.grpo_steps <= 96:
        parser.error("GRPO steps must be 1..96")
    main(args)
