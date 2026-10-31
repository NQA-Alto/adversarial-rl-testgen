import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from apps_core import ASSETS, DATA, canonical, digest, load_cases, metrics, read, score, write
from apps_model_runner import PROJECT, load_model, render_prompt, run_folder
from apps_ui import gpu_busy
from restricted_python import RestrictedFunction


PACKAGES = ("torch", "transformers", "peft")
LOCK = ASSETS / "evidence/APPS_FINAL_TEST_LOCK.json"


def prepare(selection_path):
    selection = read(selection_path)
    if selection["selected_variant"] not in selection["variants"] or not selection.get("selection_rule") or not selection.get("validation_evidence"):
        raise ValueError("A recorded Validation selection and its evidence are required")
    if "base" not in selection["variants"] or selection["variants"]["base"] is not None:
        raise ValueError("The frozen comparison must include the original unadapted baseline")
    variants = {}
    for name, adapter in selection["variants"].items():
        if re.fullmatch(r"[a-zA-Z0-9_-]+", name) is None:
            raise ValueError("Unsafe variant name")
        variants[name] = {"adapter": adapter, "files": {}}
        if adapter:
            for filename in ("adapter_model.safetensors", "adapter_config.json"):
                path = Path(adapter) / filename
                variants[name]["files"][str(path)] = digest(path)
    for entry in selection["validation_evidence"]:
        if digest(entry["path"]) != entry["sha256"]:
            raise ValueError("Validation evidence changed")
        if read(entry["path"]).get("model_test_evaluation") is not False:
            raise ValueError("Model selection evidence must precede Test evaluation")
    root = run_folder("final-protocol")
    source = root / "source"
    source.mkdir()
    hashes = {}
    for name in ("apps_final_evaluate.py", "apps_model_runner.py", "apps_core.py", "apps_lora.py", "apps_profiles.py", "restricted_python.py", "local_model_loader.py", "model_profiles.py"):
        path = Path(__file__).with_name(name)
        hashes[name] = digest(path)
        shutil.copy2(path, source / name)
    model = ASSETS / "models/Qwen3.5-2B"
    model_hashes = {str(path): digest(path) for path in sorted(model.iterdir()) if path.is_file()}
    manifest = read(DATA / "manifest.json")
    plan = {"frozen_at": datetime.now(timezone.utc).isoformat(), "dataset_id": manifest["dataset_id"],
            "split": "test", "split_role": "Problem-disjoint development holdout inside official APPS Train; not official APPS Test",
            "split_sha256": manifest["splits"]["test"]["sha256"], "expected_cases": manifest["splits"]["test"]["cases"],
            "selected_variant_before_test": selection["selected_variant"], "selection_rule": selection["selection_rule"],
            "validation_evidence": selection["validation_evidence"], "variants": variants, "generation_seeds": [42, 43, 44],
            "selection_sha256": digest(selection_path), "implementation_commit": selection.get("implementation_commit"),
            "tests_per_case": 4, "prompt_version": "apps-call-v2", "max_prompt_tokens": 1536,
            "generation": {"max_new_tokens": 96, "temperature": 1.0, "top_p": .95, "top_k": 20, "thinking": False},
            "overflow_policy": "Prompt longer than 1536 tokens yields four failed outputs included in the denominator; do not tune the budget on Test.",
            "policy": "One locked evaluation event for all declared variants/seeds; no training, best-checkpoint selection or design tuning from Test. Preserve failures and never overwrite the global event lock.",
            "source_hashes": hashes, "base_model_files": model_hashes,
            "versions": {name: importlib.metadata.version(name) for name in PACKAGES},
            "state_sha256": digest(PROJECT / "APPS_STATE.json"), "active_config_sha256": digest(PROJECT / "configs/active-dataset.json"),
            "test_content_read_during_preparation": False}
    write(root / "plan.json", plan)
    shutil.copy2(selection_path, root / "selection.json")
    write(root / "result.json", {"status": "prepared", "plan_sha256": digest(root / "plan.json"), "test_evaluation_started": False})
    print("FINAL PROTOCOL PREPARED; TEST NOT OPENED:", root, flush=True)


def verify(plan):
    for name, expected in plan["source_hashes"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("Frozen evaluation source changed: " + name)
    for path, expected in plan["base_model_files"].items():
        if digest(path) != expected:
            raise ValueError("Frozen base model changed")
    for item in plan["variants"].values():
        for path, expected in item["files"].items():
            if digest(path) != expected:
                raise ValueError("Frozen adapter changed")
    if {name: importlib.metadata.version(name) for name in PACKAGES} != plan["versions"]:
        raise ValueError("Frozen library versions changed")
    if digest(PROJECT / "APPS_STATE.json") != plan["state_sha256"] or digest(PROJECT / "configs/active-dataset.json") != plan["active_config_sha256"]:
        raise ValueError("Frozen project selection changed")
    for entry in plan["validation_evidence"]:
        if digest(entry["path"]) != entry["sha256"]:
            raise ValueError("Frozen Validation selection evidence changed")


def audit_saved(folder, cases):
    by_id = {case["case_id"]: case for case in cases}
    rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != len(cases) or {row["case_id"] for row in rows} != set(by_id):
        raise ValueError("Incomplete final holdout")
    count = 0
    for row in rows:
        if len(row["samples"]) != 4:
            raise ValueError("Final per-case budget changed")
        for sample in row["samples"]:
            checked = score(sample["raw_output"], sample["terminated"], by_id[row["case_id"]])
            if any(checked[key] != sample[key] for key in ("valid_format", "correct_expected", "bug_detected", "reward")):
                raise ValueError("Final raw output rescoring mismatch")
            count += 1
    result = read(folder / "result.json")
    if metrics(rows) != result["metrics"]:
        raise ValueError("Final metrics mismatch")
    write(folder / "review.json", {"status": "passed", "samples_rescored": count, "predictions_sha256": digest(folder / "predictions.jsonl"), "model_test_evaluation": True})


def execute(root, resume=False):
    if gpu_busy(ignore_pid=os.getpid()):
        raise RuntimeError("Another project GPU job is active")
    available = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0])
    if available < 6000:
        raise RuntimeError("At least 6000 MiB GPU memory free is required before the final event")
    plan, state = read(root / "plan.json"), read(root / "result.json")
    if digest(root / "plan.json") != state["plan_sha256"]:
        raise ValueError("Frozen protocol changed")
    if resume:
        lock = read(LOCK)
        if state["status"] not in ("running", "failed") or lock["plan_sha256"] != state["plan_sha256"] or Path(lock["root"]).resolve() != root.resolve():
            raise ValueError("Only the same interrupted final event may be resumed")
    elif state["status"] != "prepared":
        raise ValueError("Fresh prepared protocol required")
    verify(plan)
    os.environ.update(HF_HUB_OFFLINE="1", HF_HOME=str(ASSETS / "cache/huggingface"), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
                      TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"))
    import torch
    if not resume:
        with LOCK.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"root": str(root), "plan_sha256": state["plan_sha256"], "started_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    state.update(status="running", test_evaluation_started=True, model_test_evaluation=True,
                 runs=state.get("runs", []) if resume else [], selected_variant_before_test=plan["selected_variant_before_test"])
    if resume:
        state["technical_resume_count"] = state.get("technical_resume_count", 0) + 1
    write(root / "result.json", state)
    try:
        cases, manifest = load_cases("test", allow_test=True)
        if manifest["dataset_id"] != plan["dataset_id"] or digest(DATA / "test.jsonl") != plan["split_sha256"] or len(cases) != plan["expected_cases"]:
            raise ValueError("Frozen holdout changed")
        signatures = {}
        completed = {(row["variant"], row["generation_seed"]): row for row in state["runs"]}
        for row in completed.values():
            folder = Path(row["path"])
            if digest(folder / "result.json") != row["result_sha256"]:
                raise ValueError("Completed final evidence changed")
            audit_saved(folder, cases)
            saved = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            signature = {r["case_id"]: (r["prompt_sha256"], r["case_seed"]) for r in saved}
            seed = row["generation_seed"]
            if seed in signatures and signatures[seed] != signature:
                raise ValueError("Completed final signatures differ")
            signatures[seed] = signature
        for name, item in plan["variants"].items():
            pending = [seed for seed in plan["generation_seeds"] if (name, seed) not in completed]
            if not pending:
                continue
            print("FINAL VARIANT:", name, flush=True)
            model, tokenizer, eos = load_model(Path(item["adapter"]) if item["adapter"] else None)
            functions = {c["case_id"]: (RestrictedFunction(c["reference"], c["arguments"]), RestrictedFunction(c["buggy_code"], c["arguments"])) for c in cases}
            for seed in pending:
                folder = root / (name + "-seed-" + str(seed))
                folder.mkdir(exist_ok=resume)
                old_rows = []
                if (folder / "predictions.jsonl").exists():
                    previous = read(folder / "result.json")
                    if not resume or previous.get("protocol_sha256") != state["plan_sha256"]:
                        raise ValueError("Existing final output is not part of this frozen event")
                    old_rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                    if [r["case_id"] for r in old_rows] != [c["case_id"] for c in cases[:len(old_rows)]]:
                        raise ValueError("Interrupted predictions are not an exact completed prefix")
                    if previous["status"] == "evaluation_completed" and len(old_rows) == len(cases):
                        audit_saved(folder, cases)
                        signature = {r["case_id"]: (r["prompt_sha256"], r["case_seed"]) for r in old_rows}
                        if seed in signatures and signatures[seed] != signature:
                            raise ValueError("Recovered final signatures differ")
                        signatures[seed] = signature
                        state["runs"].append({"variant":name,"generation_seed":seed,"path":str(folder),"result_sha256":digest(folder / "result.json"),"metrics":previous["metrics"]})
                        write(root / "result.json", state)
                        continue
                    shutil.copy2(folder / "result.json", folder / f'result-before-resume-{state["technical_resume_count"]}.json')
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                result = {"status": "running", "action": "evaluate", "variant": name, "dataset_id": plan["dataset_id"], "split": "test",
                          "split_sha256": plan["split_sha256"], "seed": seed, "tests_per_case": 4, "generation": plan["generation"],
                          "prompt_version": plan["prompt_version"], "adapter": item["adapter"],
                          "adapter_sha256": item["files"].get(str(Path(item["adapter"]) / "adapter_model.safetensors")) if item["adapter"] else None,
                          "protocol_sha256": state["plan_sha256"], "versions": plan["versions"],
                          "resumed_cases_preserved": len(old_rows),
                          "elapsed_seconds_scope": "remaining cases in resumed session" if old_rows else "all cases in this seed",
                          "training": False, "model_test_evaluation": True, "complete_split": True}
                write(folder / "result.json", result)
                predictions = old_rows[:]
                for case in cases[len(old_rows):]:
                    prompt = render_prompt(tokenizer, case, plan["prompt_version"])
                    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
                    case_seed = (seed + int(hashlib.sha256(case["case_id"].encode()).hexdigest()[:8], 16)) % 2**32
                    torch.manual_seed(case_seed)
                    samples = []
                    if inputs.input_ids.shape[1] > plan["max_prompt_tokens"]:
                        for _ in range(4):
                            sample = score("", False, case)
                            sample.update(generated_tokens=0, reason="prompt_budget_exceeded")
                            samples.append(sample)
                    else:
                        inputs = inputs.to("cuda:0")
                        with torch.inference_mode():
                            output = model.generate(**inputs, num_return_sequences=4, max_new_tokens=96, max_time=120, do_sample=True,
                                                    temperature=1.0, top_p=.95, top_k=20, eos_token_id=eos,
                                                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, use_cache=True)
                        for tokens in output[:, inputs.input_ids.shape[1]:].tolist():
                            end = next((i for i, token in enumerate(tokens) if token in eos), None)
                            actual = tokens[:end + 1] if end is not None else tokens
                            sample = score(tokenizer.decode(actual, skip_special_tokens=True).strip(), end is not None, case, *functions[case["case_id"]])
                            if sample.get("error") == "verifier_limit":
                                raise RuntimeError("Verifier limit during final evaluation")
                            sample["generated_tokens"] = len(actual)
                            samples.append(sample)
                        del output
                    row = {"case_id": case["case_id"], "source_id": case["source_id"], "case_seed": case_seed,
                           "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "samples": samples}
                    predictions.append(row)
                    with (folder / "predictions.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(canonical(row) + "\n")
                    del inputs
                result.update(status="evaluation_completed", metrics=metrics(predictions), elapsed_seconds=round(time.perf_counter() - started, 3),
                              peak_vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3), gpu=torch.cuda.get_device_name(0))
                write(folder / "result.json", result)
                audit_saved(folder, cases)
                signature = {row["case_id"]: (row["prompt_sha256"], row["case_seed"]) for row in predictions}
                if seed in signatures and signatures[seed] != signature:
                    raise ValueError("Final prompts or per-case seeds differ across variants")
                signatures[seed] = signature
                state["runs"].append({"variant": name, "generation_seed": seed, "path": str(folder), "result_sha256": digest(folder / "result.json"), "metrics": result["metrics"]})
                write(root / "result.json", state)
                print("FINAL SEED COMPLETED:", name, seed, canonical(result["metrics"]), flush=True)
            del model, tokenizer, functions
            gc.collect()
            torch.cuda.empty_cache()
        verify(plan)
        state.update(status="completed", selected_variant_unchanged=True, comparison_prompt_and_seed_match=True,
                     no_training=True, official_apps_test_evaluated=False)
        write(root / "result.json", state)
        print("FINAL HOLDOUT COMPLETED:", root, flush=True)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        write(root / "result.json", state)
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path)
    action.add_argument("--run", type=Path)
    action.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    else:
        execute(args.resume or args.run, resume=bool(args.resume))
