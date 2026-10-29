import ast
import json
import os
import shutil
import sys
from pathlib import Path

from apps_core import ASSETS, canonical, digest, load_cases, read, write
from apps_model_runner import PROJECT, run_folder
from apps_round import run
from apps_sft import build_examples
from apps_ui import gpu_busy
from audit_apps_run import audit


def main():
    if gpu_busy(ignore_pid=os.getpid()):
        raise RuntimeError("Another project GPU job is active")
    os.environ.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                      HF_HOME=str(ASSETS / "cache/huggingface"), TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"))
    state = read(PROJECT / "APPS_STATE.json")
    before_state = digest(PROJECT / "APPS_STATE.json")
    baseline = Path(state["sft_training"]["path"])
    old = read(baseline / "result.json")
    old_val = Path(state["sft_validation"]["path"])
    audit(baseline)
    audit(old_val)
    for key in ("sft_training", "sft_validation"):
        if digest(Path(state[key]["path"]) / "result.json") != state[key]["result_sha256"]:
            raise ValueError("Baseline evidence changed")
    if old["optimizer_steps"] != 64 or old["configuration"]["prompt_version"] != "apps-call-v2":
        raise ValueError("Expected the original 64-step SFT baseline")
    root = run_folder("sft-budget-trial")
    print("EVIDENCE:", root, flush=True)
    write(root / "previous-state.json", state)
    targets, cases, manifest = build_examples(42)
    target_file = root / "targets-for-plan.jsonl"
    target_file.write_text("".join(canonical(row) + "\n" for row in targets), encoding="utf-8", newline="\n")
    if digest(target_file) != old["target_sha256"] or len(targets) != 319:
        raise ValueError("Supervised target pool differs from baseline")
    source = root / "source"
    source.mkdir()
    names = ("apps_sft_budget_trial.py", "apps_sft.py", "apps_core.py", "apps_model_runner.py", "apps_round.py",
             "local_model_loader.py", "restricted_python.py", "check_lora_update.py", "train_static_grpo.py", "audit_apps_run.py")
    hashes = {}
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), source / name)
        hashes[name] = digest(source / name)
    plan = {"method": "SUPERVISED_WARMUP_BUDGET_ABLATION_NOT_RL", "dataset_id": manifest["dataset_id"],
            "baseline_training": str(baseline), "baseline_validation": str(old_val),
            "baseline_training_result_sha256": digest(baseline / "result.json"), "target_sha256": digest(target_file),
            "training_examples": len(targets), "baseline_steps": 64, "candidate_steps": 256, "examples_per_step": 4,
            "seed": 42, "prompt_version": "apps-call-v2", "initialization": "Fresh original model and new rank-4 LoRA, same target order and optimizer settings as baseline",
            "checkpoints": [64, 128], "checkpoint_policy": "Checkpoint 64 checks replay; checkpoint 128 is retained for recovery only. Evaluate only final checkpoint 256 on Validation, no intermediate Validation selection.",
            "candidate_gate": "Validation correct-expected count at least +3/60, valid-test count >=41, bug-test count >=4 and cases detected >=3; replay relative L2 <=0.001. Passing makes a candidate for later RL comparison, not an automatic replacement of the current system.",
            "validation_tests_per_case": 4, "max_generation_tokens": 96,
            "source_hashes": hashes, "model_test_evaluation": False}
    write(root / "plan.json", plan)
    result = {"status": "running", "plan": plan, "model_test_evaluation": False, "training_method": "SFT_NOT_RL"}
    write(root / "result.json", result)
    python = str(ASSETS / "env/Scripts/python.exe")
    runner = [python, "-B", str(PROJECT / "scripts/apps_model_runner.py")]
    try:
        training = run([python, "-B", str(PROJECT / "scripts/apps_sft.py"), "--steps", "256", "--checkpoint-steps", "64", "128",
                        "--reference-adapter", str(baseline / "adapter"), "--target-order", "sorted-v1"], root, "sft-256")
        audit(training)
        trained = read(training / "result.json")
        if trained["target_sha256"] != old["target_sha256"] or digest(training / "training-order.json") != digest(baseline / "training-order.json"):
            raise ValueError("Targets or supervised order changed")
        expected = {**old["configuration"], "steps": 256, "checkpoint_steps": [64, 128]}
        if trained["configuration"] != expected:
            raise ValueError("Unexpected SFT configuration change")
        train_eval = run(runner + ["evaluate", "--split", "train", "--adapter", str(training / "adapter"), "--no-publish", "--prompt-version", "apps-call-v2"], root, "sft-256-train-evaluation")
        audit(train_eval)
        validation = run(runner + ["evaluate", "--adapter", str(training / "adapter"), "--no-publish", "--prompt-version", "apps-call-v2"], root, "sft-256-validation")
        audit(validation)
        a = read(old_val / "result.json")
        b = read(validation / "result.json")
        def signatures(folder):
            return {row["case_id"]: (row["prompt_sha256"], row["case_seed"]) for row in
                    [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]}
        if signatures(old_val) != signatures(validation) or a["generation"] != b["generation"]:
            raise ValueError("Validation generation protocol changed")
        replay = trained["step64_replay"]
        m = b["metrics"]
        eligible = m["correct_expected"] >= a["metrics"]["correct_expected"] + 3 and m["valid_tests"] >= 41 and m["bug_detecting_tests"] >= 4 and m["cases_detected"] >= 3 and replay["relative_l2_difference"] <= .001
        for name, expected_hash in hashes.items():
            if digest(Path(__file__).with_name(name)) != expected_hash:
                raise ValueError("Source changed during budget experiment: " + name)
        if digest(PROJECT / "APPS_STATE.json") != before_state:
            raise ValueError("Current experiment state changed during trial")
        result.update(status="completed", training_run=str(training), train_evaluation=str(train_eval), validation_run=str(validation),
                      train_metrics=read(train_eval / "result.json")["metrics"], validation_metrics=m,
                      baseline_validation_metrics=a["metrics"], step64_replay=replay, eligible_for_later_rl_comparison=eligible,
                      active_checkpoint_unchanged=True, comparison_prompt_and_seed_match=True)
        write(root / "result.json", result)
        print("SFT BUDGET TRIAL COMPLETED:", canonical(result), flush=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        write(root / "result.json", result)
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
