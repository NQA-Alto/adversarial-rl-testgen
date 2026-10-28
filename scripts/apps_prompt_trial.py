import argparse
import ast
import os
import shutil
import sys
from pathlib import Path

from apps_core import ASSETS, digest, load_cases, read, write
from apps_model_runner import PROJECT, run_folder
from apps_round import run
from apps_ui import gpu_busy
from audit_apps_run import audit


def scoring_signature(path):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef) and node.name in ("score", "valid_input", "metrics")}


def main(candidate):
    if gpu_busy(ignore_pid=os.getpid()):
        raise ValueError("Another project GPU job is active")
    os.environ.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1",
                      HF_HOME=str(ASSETS / "cache/huggingface"), TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"))
    root = run_folder(candidate.replace("apps-call-", "prompt-") + "-trial")
    print("EVIDENCE:", root, flush=True)
    state = read(PROJECT / "APPS_STATE.json")
    parent = Path(state["sft_training"]["adapter_path"])
    train_baseline = ASSETS / "evidence/apps-adapter-validation-20261028-155601-910485"
    val_baseline = Path(state["sft_validation"]["path"])
    old_train, old_val = read(train_baseline / "result.json"), read(val_baseline / "result.json")
    adapter_hash = digest(parent / "adapter_model.safetensors")
    for folder, baseline, split in ((train_baseline, old_train, "train"), (val_baseline, old_val, "validation")):
        audit(folder)
        if baseline["split"] != split or not baseline["complete_split"] or baseline["adapter_sha256"] != adapter_hash or baseline["prompt_version"] != "apps-call-v2" or baseline["seed"] != 42:
            raise ValueError("Baseline not compatible")
        if scoring_signature(folder / "source/apps_core.py") != scoring_signature(Path(__file__).with_name("apps_core.py")):
            raise ValueError("Scoring changed")
    _, manifest = load_cases("train")
    source = root / "source"
    source.mkdir()
    names = ("apps_prompt_trial.py", "apps_output_contracts.py", "apps_core.py", "apps_model_runner.py", "restricted_python.py", "audit_apps_run.py")
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), source / name)
    plan = {"method": "PROMPT_ONLY_ABLATION_NOT_RL", "dataset_id": manifest["dataset_id"], "adapter_sha256": adapter_hash,
            "baseline_train": str(train_baseline), "baseline_validation": str(val_baseline), "candidate": candidate,
            "change": "Preserve the complete v2 user prompt and format example; add only expected JSON type to the system instruction." if candidate == "apps-call-v4" else "Remove unrelated integer-output example; explicitly state task JSON output type and finite input contract. No reference code or concrete expected answers in prompts.",
            "output_type_provenance": "Static interface annotations for the known public APPS Train/Validation tasks, checked against finite-domain references; this is added interface metadata, not an ATGen-original feature.",
            "generation": old_train["generation"], "seed": 42, "tests_per_case": 4,
            "train_gate": "IO accuracy at least +5 percentage points; valid-test and bug-test counts must not decrease versus fixed SFT v2 Train baseline.",
            "validation_gate": "Only if Train gate passes, evaluate Validation once. Candidate remains separate unless correct-expected count strictly improves and valid/bug-test counts do not decrease versus SFT v2.",
            "source_hashes": {name: digest(source / name) for name in names}, "training": False, "model_test_evaluation": False}
    write(root / "plan.json", plan)
    result = {"status": "running", "plan": plan, "training": False, "model_test_evaluation": False}
    write(root / "result.json", result)
    python = str(ASSETS / "env/Scripts/python.exe")
    runner = [python, "-B", str(PROJECT / "scripts/apps_model_runner.py"), "evaluate", "--adapter", str(parent), "--prompt-version", candidate, "--no-publish"]
    try:
        train = run(runner + ["--split", "train"], root, "candidate-train")
        audit(train)
        candidate_train = read(train / "result.json")
        a, b = old_train["metrics"], candidate_train["metrics"]
        passed = b["io_accuracy"] >= a["io_accuracy"] + .05 and b["valid_tests"] >= a["valid_tests"] and b["bug_detecting_tests"] >= a["bug_detecting_tests"]
        result.update(train_gate_passed=passed, train_run=str(train), train_metrics=b, baseline_train_metrics=a,
                      validation_evaluated=False, eligible_prompt_candidate=False)
        write(root / "result.json", result)
        if passed:
            validation = run(runner + ["--split", "validation"], root, "candidate-validation")
            audit(validation)
            candidate_validation = read(validation / "result.json")
            a, b = old_val["metrics"], candidate_validation["metrics"]
            eligible = b["correct_expected"] > a["correct_expected"] and b["valid_tests"] >= a["valid_tests"] and b["bug_detecting_tests"] >= a["bug_detecting_tests"]
            result.update(validation_evaluated=True, validation_run=str(validation), validation_metrics=b, baseline_validation_metrics=a,
                          eligible_prompt_candidate=eligible)
        if digest(parent / "adapter_model.safetensors") != adapter_hash:
            raise ValueError("Frozen SFT adapter changed")
        for name, expected in plan["source_hashes"].items():
            if digest(Path(__file__).with_name(name)) != expected:
                raise ValueError("Trial source changed: " + name)
        result.update(status="completed", adapter_unchanged=True, active_prompt_unchanged=True,
                      interpretation="Any gain belongs to the prompt/interface change, not RL. No Test evaluation or model selection by Test.")
        write(root / "result.json", result)
        print("PROMPT TRIAL COMPLETED:", result, flush=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        write(root / "result.json", result)
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("apps-call-v3", "apps-call-v4"), default="apps-call-v4")
    main(parser.parse_args().candidate)
