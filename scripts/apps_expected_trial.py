import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from apps_core import ASSETS, digest, read, write
from apps_model_runner import PROJECT, run_folder
from apps_round import run
from apps_ui import gpu_busy
from audit_apps_run import audit


BASELINE = ASSETS / "evidence/apps-sft-v2-20261029-143238-169569"
PROBE = ASSETS / "evidence/apps-adapter-validation-20261029-143558-352883"


def prepare():
    for path in (BASELINE, PROBE):
        audit(path)
    old, probe = read(BASELINE / "result.json"), read(PROBE / "result.json")
    if old["configuration"]["target_order"] != "input-first-v1" or old["optimizer_steps"] != 64:
        raise ValueError("Expected input-first SFT64")
    if probe["split"] != "train" or probe["adapter_sha256"] != old["adapter"]["adapter_sha256"]:
        raise ValueError("Probe does not match baseline")
    root = run_folder("expected-budget-trial")
    source = root / "source"
    source.mkdir()
    names = ("apps_expected_trial.py", "apps_sft.py", "apps_model_runner.py", "apps_core.py", "apps_lora.py",
             "apps_round.py", "apps_rewards.py", "apps_curriculum.py", "apps_profiles.py", "restricted_python.py",
             "local_model_loader.py", "train_static_grpo.py", "check_lora_update.py", "audit_apps_run.py")
    hashes = {}
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), source / name)
        hashes[name] = digest(source / name)
    plan = {"recorded_at": datetime.now(timezone.utc).isoformat(), "hypothesis": "Input-first SFT64 sees only 256 of 319 target examples. A fixed 256-step supervised warmup may improve expected accuracy; it can also overfit.",
            "only_change": "SFT budget 64 to 256 steps. Same fresh base, rank-4 attention-qv-v1 LoRA, 319 Train targets in input-first order, seed42, LR5e-5, four examples per update. Retain step64 for replay, never select intermediate checkpoints on Validation.",
            "baseline_sft": str(BASELINE), "baseline_sft_result_sha256": digest(BASELINE / "result.json"),
            "baseline_probe": str(PROBE), "baseline_probe_result_sha256": digest(PROBE / "result.json"),
            "train_gate": {"expected_gain_min": 8, "valid_drop_max": 0, "bug_tests_drop_max": 0,
                           "cases_drop_max": 0, "replay_relative_l2_max": .001},
            "if_train_gate_passes": "Fresh 96-step GRPO from final SFT256, same legacy-v1 reward, seed42 and prompt-v2. Then and only then evaluate both final checkpoints on all three Validation seeds42/43/44, reusing all SFT64 and SFT64+GRPO96 controls from the completed order trial.",
            "if_gate_fails": "Stop without new Validation or Test evaluation. Preserve failed candidate and raw outputs. No automatic retry, checkpoint search or larger budget.",
            "validation_eligibility": "At final GRPO, sum expected counts across three seeds >= SFT64+GRPO96 total +9; valid, bug-test and detected-case totals must not decrease. Every seed and SFT result remains reported even when gate fails.",
            "budget": {"sft_steps": 256, "grpo_steps": 96, "samples_per_group": 4, "max_completion_tokens": 96,
                       "validation_seeds": [42, 43, 44]},
            "limits": "Exploratory follow-up after earlier Validation results; one training seed, repeated use of only five Validation problems risks selection overfitting. This is a bounded attempt, not a guarantee of improvement. Test remains sealed.",
            "source_hashes": hashes, "state_sha256": digest(PROJECT / "APPS_STATE.json"),
            "active_config_sha256": digest(PROJECT / "configs/active-dataset.json"), "model_test_evaluation": False}
    write(root / "plan.json", plan)
    write(root / "result.json", {"status": "prepared", "plan_sha256": digest(root / "plan.json"), "model_test_evaluation": False})
    print("EVIDENCE:", root, flush=True)


def execute(root):
    if gpu_busy(ignore_pid=os.getpid()):
        raise RuntimeError("Another project GPU task is active")
    free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0])
    if free < 6000:
        raise RuntimeError("6000 MiB free required; prepared plan preserved")
    plan, result = read(root / "plan.json"), read(root / "result.json")
    if result["status"] != "prepared" or result["plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("Fresh prepared trial required")
    for name, value in plan["source_hashes"].items():
        if digest(Path(__file__).with_name(name)) != value:
            raise ValueError("Prepared source changed")
    os.environ.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                      HF_HOME=str(ASSETS / "cache/huggingface"), TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"))
    python = str(ASSETS / "env/Scripts/python.exe")
    runner = [python, "-B", str(PROJECT / "scripts/apps_model_runner.py")]
    result["status"] = "running"
    write(root / "result.json", result)
    print("EVIDENCE:", root, flush=True)
    try:
        training = run([python, "-B", str(PROJECT / "scripts/apps_sft.py"), "--steps", "256", "--target-order", "input-first-v1",
                        "--checkpoint-steps", "64", "--reference-adapter", str(BASELINE / "adapter")], root, "sft256")
        audit(training)
        trained, old = read(training / "result.json"), read(BASELINE / "result.json")
        if (digest(BASELINE / "result.json") != plan["baseline_sft_result_sha256"]
                or digest(PROBE / "result.json") != plan["baseline_probe_result_sha256"]
                or trained["target_sha256"] != old["target_sha256"]
                or digest(training / "training-order.json") != digest(BASELINE / "training-order.json")
                or trained["configuration"] != {**old["configuration"], "steps": 256, "checkpoint_steps": [64]}):
            raise ValueError("More than supervised budget changed")
        probe = run(runner + ["evaluate", "--split", "train", "--adapter", str(training / "adapter"), "--prompt-version", "apps-call-v2", "--no-publish"], root, "train-probe")
        audit(probe)
        a, b = read(PROBE / "result.json"), read(probe / "result.json")
        def signatures(folder):
            return {r["case_id"]: (r["prompt_sha256"], r["case_seed"]) for r in
                    [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]}
        if signatures(PROBE) != signatures(probe) or a["generation"] != b["generation"]:
            raise ValueError("Train probe protocol differs")
        gate = plan["train_gate"]
        passed = (b["metrics"]["correct_expected"] >= a["metrics"]["correct_expected"] + gate["expected_gain_min"]
                  and all(b["metrics"][f] >= a["metrics"][f] for f in ("valid_tests", "bug_detecting_tests", "cases_detected"))
                  and trained["step64_replay"]["relative_l2_difference"] <= gate["replay_relative_l2_max"])
        result.update(training=str(training), train_probe=str(probe), train_gate_passed=passed,
                      baseline_train_metrics=a["metrics"], candidate_train_metrics=b["metrics"], step64_replay=trained["step64_replay"])
        write(root / "result.json", result)
        if passed:
            grpo = run(runner + ["train", "--steps", "96", "--seed", "42", "--init-adapter", str(training / "adapter"),
                                "--baseline-run", str(probe), "--prompt-version", "apps-call-v2", "--reward-profile", "legacy-v1", "--no-publish"], root, "grpo96")
            audit(grpo)
            result.update(grpo=str(grpo), evaluations=[])
            write(root / "result.json", result)
            for seed in plan["budget"]["validation_seeds"]:
                for stage, parent in (("sft", training), ("grpo", grpo)):
                    evaluated = run(runner + ["evaluate", "--adapter", str(parent / "adapter"), "--seed", str(seed), "--prompt-version", "apps-call-v2", "--no-publish"], root, f'{stage}-validation-{seed}')
                    audit(evaluated)
                    info = read(evaluated / "result.json")
                    if not info["complete_split"] or info["metrics"]["generated_tests"] != 60:
                        raise ValueError("Incomplete Validation")
                    result["evaluations"].append({"stage": stage, "generation_seed": seed, "path": str(evaluated),
                                                  "result_sha256": digest(evaluated / "result.json"), "metrics": info["metrics"]})
                    write(root / "result.json", result)
        for name, value in plan["source_hashes"].items():
            if digest(Path(__file__).with_name(name)) != value:
                raise ValueError("Frozen source changed")
        if digest(PROJECT / "APPS_STATE.json") != plan["state_sha256"] or digest(PROJECT / "configs/active-dataset.json") != plan["active_config_sha256"]:
            raise ValueError("Active checkpoint changed")
        result.update(status="completed" if passed else "stopped_at_train_gate", active_checkpoint_unchanged=True)
        write(root / "result.json", result)
        print("EXPECTED BUDGET TRIAL:", result["status"], "Train:", b["metrics"], flush=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        write(root / "result.json", result)
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    execute(args.run) if args.run else prepare()
