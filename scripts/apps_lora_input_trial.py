import argparse
import ast
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from apps_core import ASSETS, digest, read, write
from apps_lora import coverage
from apps_model_runner import PROJECT, run_folder
from apps_round import run
from apps_sft import build_examples
from apps_ui import gpu_busy
from audit_apps_run import audit


FIELDS = ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")
OUTPUT = PROJECT / "results/apps-lora-input-first"


def prepare():
    previous = read(PROJECT / "results/apps-shared-sft-budget/metrics.json")
    study = Path(previous["root"])
    if digest(study / "result.json") != previous["trial_result_sha256"]:
        raise ValueError("Saved SFT comparison changed")
    control = read(study / "result.json")
    parent = Path(control["training"])
    audit(parent)
    audit(Path(control["probes"]["256"]["path"]))
    trained = read(parent / "result.json")
    if trained["configuration"].get("target_order") != "input-first-v1" or trained["optimizer_steps"] != 256:
        raise ValueError("Input-first SFT256 control required")
    targets, _, manifest = build_examples(42, "input-first-v1")
    if len(targets) != trained["training_examples"]:
        raise ValueError("Supervised pool size changed")
    root = run_folder("lora-input-first")
    (root / "source").mkdir()
    names = ("apps_lora_input_trial.py", "apps_sft.py", "apps_model_runner.py", "apps_core.py", "apps_lora.py",
             "apps_profiles.py", "apps_output_contracts.py", "restricted_python.py", "local_model_loader.py",
             "check_lora_update.py", "train_static_grpo.py", "apps_round.py", "audit_apps_run.py")
    hashes = {}
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), root / "source" / name)
        hashes[name] = digest(root / "source" / name)
    comparison = {}
    for name in ("apps_sft.py", "apps_model_runner.py", "apps_core.py", "apps_lora.py", "apps_profiles.py",
                 "restricted_python.py", "local_model_loader.py", "check_lora_update.py", "train_static_grpo.py"):
        if digest(parent / "source" / name) != hashes[name]:
            old = ast.parse((parent / "source" / name).read_text(encoding="utf-8"))
            new = ast.parse((root / "source" / name).read_text(encoding="utf-8"))
            if name != "apps_sft.py" or ast.dump(ast.Module(body=old.body[:-1], type_ignores=[])) != ast.dump(ast.Module(body=new.body[:-1], type_ignores=[])):
                raise ValueError("Control training source differs: " + name)
            comparison[name] = "All definitions/imports unchanged; CLI target-order default now reads active config. Both runs explicitly pass input-first-v1."
        else:
            comparison[name] = "Identical bytes"
    plan = {"recorded_at": datetime.now(timezone.utc).isoformat(),
            "hypothesis": "Adding linear-attention projections to LoRA may improve expected prediction at the same SFT256 budget.",
            "control_training": str(parent), "control_training_sha256": digest(parent / "result.json"),
            "control_adapter_sha256": trained["adapter"]["adapter_sha256"],
            "control_probe": control["probes"]["256"],
            "control_validation": [row for row in control["evaluations"] if row["sft_steps"] == 256],
            "dataset_id": manifest["dataset_id"], "train_sha256": manifest["splits"]["train"]["sha256"],
            "target_sha256": trained["target_sha256"], "order_sha256": digest(parent / "training-order.json"),
            "steps": 256, "seed": 42, "target_order": "input-first-v1", "prompt_version": "apps-call-v2",
            "candidate_profile": "hybrid-qkv-v1", "coverage": {name: coverage(name) for name in ("attention-qv-v1", "hybrid-qkv-v1")},
            "fixed": "Original BF16 model, rank4 alpha8, learning rate5e-5, four examples/update, same Train labels/order, final256 checkpoint only.",
            "train_gate": "Expected at least control+4/164; valid, bug-test and detected-code counts must not decrease. Otherwise stop before Validation.",
            "validation_gate": "Sum across seeds42/43/44: expected at least control+9/180; valid, bug-test and detected-code totals must not decrease.",
            "evaluation": "Four tests/code,96 tokens/test, v2 prompt, temperature1, top-p0.95, top-k20; all three seeds if Train gate passes.",
            "limits": "One training seed; reused audited narrow SFT256 control, fresh wider adapter. Parameter count and adapted layers change together. Repeated Validation exposure; Test remains untouched. SFT only, not an RL gain.",
            "minimum_free_mib": 5700,
            "memory_basis": "Earlier hybrid SFT64 used4.374 GiB allocated;5700 MiB leaves about1.2 GiB for runtime/headroom. Abort on allocation failure.",
            "source_hashes": hashes, "control_source_comparison": comparison, "state_sha256": digest(PROJECT / "APPS_STATE.json"),
            "config_sha256": digest(PROJECT / "configs/active-dataset.json"), "model_test_evaluation": False}
    write(root / "plan.json", plan)
    write(root / "result.json", {"status": "prepared", "plan_sha256": digest(root / "plan.json"), "evaluations": [], "model_test_evaluation": False})
    print("EVIDENCE:", root, flush=True)


def execute(root):
    plan, result = read(root / "plan.json"), read(root / "result.json")
    if result["status"] != "prepared" or digest(root / "plan.json") != result["plan_sha256"]:
        raise ValueError("Fresh prepared experiment required")
    if gpu_busy(ignore_pid=os.getpid()):
        raise RuntimeError("Another project GPU task is active")
    available = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
    if available < plan["minimum_free_mib"]:
        raise RuntimeError(f'GPU requires {plan["minimum_free_mib"]} MiB free; available {available}')
    for name, expected in plan["source_hashes"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("Prepared source changed: " + name)
    for filename, key in (("APPS_STATE.json", "state_sha256"), ("configs/active-dataset.json", "config_sha256")):
        if digest(PROJECT / filename) != plan[key]:
            raise ValueError("Active state changed")
    os.environ.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                      HF_HOME=str(ASSETS / "cache/huggingface"), TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"))
    python = str(ASSETS / "env/Scripts/python.exe")
    runner = [python, "-B", str(PROJECT / "scripts/apps_model_runner.py")]
    result["status"] = "running"
    write(root / "result.json", result)
    try:
        training = run([python, "-B", str(PROJECT / "scripts/apps_sft.py"), "--steps", "256", "--target-order", "input-first-v1",
                        "--lora-profile", plan["candidate_profile"]], root, "hybrid-sft256")
        audit(training)
        trained = read(training / "result.json")
        if digest(training / "targets.jsonl") != plan["target_sha256"] or digest(training / "training-order.json") != plan["order_sha256"]:
            raise ValueError("Supervised labels or training order differ")
        if trained["lora_coverage"] != plan["coverage"][plan["candidate_profile"]] or not trained["update_verified"]:
            raise ValueError("Coverage or saved update mismatch")
        for target in trained["lora_coverage"]["targets"]:
            if not any(target["module"] + ".lora_B.weight" in name for name in trained["changed_adapter_tensors"]):
                raise ValueError("A declared target has no saved update")
        result["training"] = str(training)
        write(root / "result.json", result)
        probe = run(runner + ["evaluate", "--split", "train", "--adapter", str(training / "adapter"),
                              "--prompt-version", plan["prompt_version"], "--no-publish"], root, "train-probe")
        audit(probe)
        candidate = read(probe / "result.json")["metrics"]
        control = plan["control_probe"]["metrics"]
        passed = candidate["correct_expected"] >= control["correct_expected"] + 4 and all(candidate[f] >= control[f] for f in ("valid_tests", "bug_detecting_tests", "cases_detected"))
        result.update(probe={"path": str(probe), "result_sha256": digest(probe / "result.json"), "metrics": candidate}, train_gate_passed=passed)
        write(root / "result.json", result)
        if passed:
            for seed in (42, 43, 44):
                folder = run(runner + ["evaluate", "--adapter", str(training / "adapter"), "--seed", str(seed),
                                      "--prompt-version", plan["prompt_version"], "--no-publish"], root, f'validation-{seed}')
                audit(folder)
                result["evaluations"].append({"generation_seed": seed, "path": str(folder),
                                               "result_sha256": digest(folder / "result.json"), "metrics": read(folder / "result.json")["metrics"]})
                write(root / "result.json", result)
        for name, expected in plan["source_hashes"].items():
            if digest(Path(__file__).with_name(name)) != expected:
                raise ValueError("Source changed during experiment")
        for filename, key in (("APPS_STATE.json", "state_sha256"), ("configs/active-dataset.json", "config_sha256")):
            if digest(PROJECT / filename) != plan[key]:
                raise ValueError("Active state changed during experiment")
        result.update(status="completed" if passed else "stopped_at_train_gate", active_checkpoint_unchanged=True)
        write(root / "result.json", result)
        print("LORA INPUT-FIRST COMPLETE:", root, result["status"], flush=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        write(root / "result.json", result)
        raise


def summarize(root):
    plan, result = read(root / "plan.json"), read(root / "result.json")
    if digest(root / "plan.json") != result["plan_sha256"] or result["status"] not in ("completed", "stopped_at_train_gate"):
        raise ValueError("Completed experiment with original plan required")
    parent = Path(plan["control_training"])
    if digest(parent / "result.json") != plan["control_training_sha256"] or digest(parent / "adapter/adapter_model.safetensors") != plan["control_adapter_sha256"]:
        raise ValueError("Control changed")
    audit(parent)
    audit(Path(result["training"]))
    train_rows = []
    for variant, probe in (("attention-qv-v1", plan["control_probe"]), ("hybrid-qkv-v1", result["probe"])):
        path = Path(probe["path"])
        audit(path)
        info = read(path / "result.json")
        if digest(path / "result.json") != probe["result_sha256"] or info["split"] != "train" or info["metrics"] != probe["metrics"]:
            raise ValueError("Train probe changed")
        train_rows.append({"variant": variant, **{f: info["metrics"][f] for f in FIELDS}})
    table, signatures, decoding = [], {}, None
    if result["train_gate_passed"]:
        for variant, runs in (("attention-qv-v1", plan["control_validation"]), ("hybrid-qkv-v1", result["evaluations"])):
            if len(runs) != 3 or {r["generation_seed"] for r in runs} != {42, 43, 44}:
                raise ValueError("Three planned generation seeds required")
            for row in runs:
                path = Path(row["path"])
                audit(path)
                info = read(path / "result.json")
                if digest(path / "result.json") != row["result_sha256"] or info["split"] != "validation" or not info["complete_split"] or info["metrics"] != row["metrics"]:
                    raise ValueError("Validation evidence changed")
                predictions = [json.loads(line) for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                sig = {r["case_id"]: (r["prompt_sha256"], r["case_seed"]) for r in predictions}
                seed = row["generation_seed"]
                if seed in signatures and signatures[seed] != sig or decoding is not None and decoding != info["generation"]:
                    raise ValueError("Prompt, seeds or generation settings differ")
                signatures[seed], decoding = sig, info["generation"]
                table.append({"variant": variant, "generation_seed": seed, **{f: info["metrics"][f] for f in FIELDS}})
    elif result["evaluations"]:
        raise ValueError("Validation evaluated after failed Train gate")
    summary = {v: {f: {"mean": statistics.mean(r[f] for r in table if r["variant"] == v),
                       "sample_sd": statistics.stdev(r[f] for r in table if r["variant"] == v)} for f in FIELDS}
               for v in ("attention-qv-v1", "hybrid-qkv-v1")} if table else {}
    eligible = False
    if summary:
        a, b = summary["attention-qv-v1"], summary["hybrid-qkv-v1"]
        eligible = b["correct_expected"]["mean"] >= a["correct_expected"]["mean"] + 3 and all(b[f]["mean"] >= a[f]["mean"] for f in ("valid_tests", "bug_detecting_tests", "cases_detected"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write(OUTPUT / "metrics.json", {"root": str(root), "result_sha256": digest(root / "result.json"), "plan": plan,
                                    "status": result["status"], "train_rows": train_rows, "summary": summary,
                                    "validation_rows": table, "eligible_for_followup": eligible, "model_test_evaluation": False})
    for name, rows in (("train.csv", train_rows), ("metrics.csv", table)):
        if rows:
            with (OUTPUT / name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print("SUMMARY:", result["status"], "eligible:", eligible, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", type=Path)
    action.add_argument("--summarize", type=Path)
    args = parser.parse_args()
    prepare() if args.prepare else execute(args.run) if args.run else summarize(args.summarize)
