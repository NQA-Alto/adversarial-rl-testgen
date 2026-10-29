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


def main():
    if gpu_busy(ignore_pid=os.getpid()):
        raise RuntimeError("Another GPU task is active")
    if int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()[0]) < 6000:
        raise RuntimeError("6000 MiB free GPU memory required")
    os.environ.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                      HF_HOME=str(ASSETS / "cache/huggingface"), TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"))
    root = run_folder("shared-sft-budget")
    print("EVIDENCE:", root, flush=True)
    source = root / "source"
    source.mkdir()
    names = ("apps_shared_budget_trial.py", "apps_sft.py", "apps_model_runner.py", "apps_core.py", "apps_lora.py", "apps_round.py",
             "apps_profiles.py", "restricted_python.py", "local_model_loader.py", "check_lora_update.py", "train_static_grpo.py", "audit_apps_run.py")
    hashes = {}
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), source / name)
        hashes[name] = digest(source / name)
    plan = {"recorded_at": datetime.now(timezone.utc).isoformat(), "purpose": "Compare SFT64 and SFT256 from the SAME fresh training trajectory, avoiding reliance on exact historical CUDA replay.",
            "predecessor": str(ASSETS / "evidence/apps-expected-budget-trial-20261030-000324-270571"),
            "reason_for_new_protocol": "Previous attempt stopped at its original replay threshold, with no new Validation. Targets/order/shared source hashes matched, but floating-point trajectories differed from step2. Cause not established. Do not relax the old threshold or mark that attempt successful.",
            "method": "SFT_NOT_RL", "seed": 42, "target_order": "input-first-v1", "sft_steps": 256, "control_checkpoint": 64,
            "control": "Checkpoint64 and final256 saved during one uninterrupted optimizer trajectory. Same initial LoRA, targets, optimizer and first64 updates by construction. Do not use step116 from the interrupted run.",
            "train_gate": "At256 relative to same-run64, expected +8/164 minimum; valid, bug-test and detected-case counts must not decrease. Otherwise stop without new Validation. No larger-budget retry.",
            "validation": "If Train gate passes, evaluate BOTH checkpoints on seeds42/43/44, all15 Validation codes and4 tests/code, same v2 prompt and96-token budget. Report all6 evaluations. No checkpoint selection mid-training.",
            "validation_gate": "Total expected across3 seeds +9/180 minimum; valid, bug-test and detected-case totals must not decrease. SFT result only; no claim of RL improvement or automatic replacement of a GRPO checkpoint.",
            "limits": "Exploratory budget study after earlier Validation exposure. One training seed, three sampling seeds. Nested checkpoints are paired observations, not independent training replicates. Test remains sealed.",
            "source_hashes": hashes, "state_sha256": digest(PROJECT / "APPS_STATE.json"), "active_config_sha256": digest(PROJECT / "configs/active-dataset.json"), "model_test_evaluation": False}
    write(root / "plan.json", plan)
    result = {"status": "running", "plan_sha256": digest(root / "plan.json"), "model_test_evaluation": False}
    write(root / "result.json", result)
    python = str(ASSETS / "env/Scripts/python.exe")
    runner = [python, "-B", str(PROJECT / "scripts/apps_model_runner.py")]
    try:
        training = run([python, "-B", str(PROJECT / "scripts/apps_sft.py"), "--steps", "256", "--target-order", "input-first-v1", "--checkpoint-steps", "64"], root, "sft256-shared")
        audit(training)
        result.update(training=str(training), probes={}, evaluations=[])
        write(root / "result.json", result)
        for steps, adapter in ((64, training / "checkpoint-64"), (256, training / "adapter")):
            probe = run(runner + ["evaluate", "--split", "train", "--adapter", str(adapter), "--prompt-version", "apps-call-v2", "--no-publish"], root, f'train-{steps}')
            audit(probe)
            result["probes"][str(steps)] = {"path": str(probe), "result_sha256": digest(probe / "result.json"), "metrics": read(probe / "result.json")["metrics"]}
            write(root / "result.json", result)
        a, b = result["probes"]["64"]["metrics"], result["probes"]["256"]["metrics"]
        passed = b["correct_expected"] >= a["correct_expected"] + 8 and all(b[f] >= a[f] for f in ("valid_tests", "bug_detecting_tests", "cases_detected"))
        result["train_gate_passed"] = passed
        write(root / "result.json", result)
        if passed:
            for seed in (42, 43, 44):
                for steps, adapter in ((64, training / "checkpoint-64"), (256, training / "adapter")):
                    folder = run(runner + ["evaluate", "--adapter", str(adapter), "--seed", str(seed), "--prompt-version", "apps-call-v2", "--no-publish"], root, f'validation-{steps}-{seed}')
                    audit(folder)
                    result["evaluations"].append({"sft_steps": steps, "generation_seed": seed, "path": str(folder), "result_sha256": digest(folder / "result.json"), "metrics": read(folder / "result.json")["metrics"]})
                    write(root / "result.json", result)
        for name, value in hashes.items():
            if digest(Path(__file__).with_name(name)) != value:
                raise ValueError("Frozen source changed")
        if digest(PROJECT / "APPS_STATE.json") != plan["state_sha256"] or digest(PROJECT / "configs/active-dataset.json") != plan["active_config_sha256"]:
            raise ValueError("Active checkpoint changed")
        result.update(status="completed" if passed else "stopped_at_train_gate", active_checkpoint_unchanged=True)
        write(root / "result.json", result)
        print("SHARED BUDGET COMPLETED:", root, result["status"], flush=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        write(root / "result.json", result)
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
