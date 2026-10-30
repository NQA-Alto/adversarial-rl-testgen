import argparse
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from apps_core import ASSETS, digest, read, write
from apps_model_runner import PROJECT, run_folder
from apps_round import run
from apps_rewards import PROFILES
from apps_ui import gpu_busy
from audit_apps_run import audit


def main(shared, reward_profile="legacy-v1"):
    if reward_profile not in PROFILES:
        raise ValueError("Unknown reward profile")
    if gpu_busy(ignore_pid=os.getpid()):
        raise RuntimeError("Another GPU task is active")
    summary = read(PROJECT / "results/apps-shared-sft-budget/metrics.json")
    if Path(summary["root"]).resolve() != shared.resolve() or digest(shared / "result.json") != summary["trial_result_sha256"] or not summary["eligible_for_rl_followup"]:
        raise ValueError("Completed SFT budget study must pass its original Validation gate")
    previous = read(shared / "result.json")
    parent, baseline = Path(previous["training"]), Path(previous["probes"]["256"]["path"])
    audit(parent)
    audit(baseline)
    root = run_folder("sft256-grpo-followup" if reward_profile == "legacy-v1" else "sft256-equal-reward")
    print("EVIDENCE:", root, flush=True)
    source = root / "source"
    source.mkdir()
    hashes = {}
    for name in ("apps_sft256_rl_followup.py", "apps_model_runner.py", "apps_core.py", "apps_profiles.py", "apps_round.py", "apps_lora.py",
                 "apps_rewards.py", "apps_curriculum.py", "restricted_python.py", "local_model_loader.py", "train_static_grpo.py", "audit_apps_run.py"):
        shutil.copy2(Path(__file__).with_name(name), source / name)
        hashes[name] = digest(source / name)
    reuse = [r for r in previous["evaluations"] if r["sft_steps"] == 256]
    if len(reuse) != 3:
        raise ValueError("All three parent SFT Validation seeds required")
    plan = {"recorded_at": datetime.now(timezone.utc).isoformat(), "purpose": "Measure GRPO96 from SFT256 with a declared reward profile, keeping the parent, curriculum, optimizer and evaluation settings fixed.",
            "shared_sft_trial": str(shared), "shared_trial_result_sha256": digest(shared / "result.json"),
            "parent_training": str(parent), "parent_training_result_sha256": digest(parent / "result.json"),
            "parent_adapter_sha256": digest(parent / "adapter/adapter_model.safetensors"), "parent_train_probe": str(baseline),
            "parent_validation": reuse, "steps": 96, "reward_profile": reward_profile, "reward_weights": PROFILES[reward_profile], "training_seed": 42,
            "generation_seeds": [42, 43, 44], "prompt_version": "apps-call-v2", "tests_per_case": 4, "max_new_tokens": 96,
            "selection": "Final checkpoint at96 updates; report every predeclared Validation seed. No intermediate checkpoint search or larger-budget retry. If RL fails to improve, retain and report the stronger SFT checkpoint separately.",
            "scope": "Static Train code only. This is not a new paired adversarial experiment; the existing two-round adaptive comparison retains its SFT64 parent.",
            "limits": "One training seed, three generation seeds, five Validation problems; exploratory follow-up with repeated Validation exposure. Test remains sealed.",
            "source_hashes": hashes, "state_sha256": digest(PROJECT / "APPS_STATE.json"), "active_config_sha256": digest(PROJECT / "configs/active-dataset.json"), "model_test_evaluation": False}
    write(root / "plan.json", plan)
    result = {"status": "running", "plan_sha256": digest(root / "plan.json"), "evaluations": [], "model_test_evaluation": False}
    write(root / "result.json", result)
    os.environ.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                      HF_HOME=str(ASSETS / "cache/huggingface"), TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"))
    runner = [str(ASSETS / "env/Scripts/python.exe"), "-B", str(PROJECT / "scripts/apps_model_runner.py")]
    try:
        training = run(runner + ["train", "--steps", "96", "--seed", "42", "--init-adapter", str(parent / "adapter"),
                                 "--baseline-run", str(baseline), "--prompt-version", "apps-call-v2", "--reward-profile", reward_profile, "--no-publish"], root, "grpo96")
        audit(training)
        result["training"] = str(training)
        write(root / "result.json", result)
        for seed in (42, 43, 44):
            folder = run(runner + ["evaluate", "--adapter", str(training / "adapter"), "--seed", str(seed), "--prompt-version", "apps-call-v2", "--no-publish"], root, f'validation-{seed}')
            audit(folder)
            evaluated = read(folder / "result.json")
            result["evaluations"].append({"generation_seed": seed, "path": str(folder), "result_sha256": digest(folder / "result.json"), "metrics": evaluated["metrics"]})
            write(root / "result.json", result)
        for name, value in hashes.items():
            if digest(Path(__file__).with_name(name)) != value:
                raise ValueError("Frozen source changed")
        if digest(PROJECT / "APPS_STATE.json") != plan["state_sha256"] or digest(PROJECT / "configs/active-dataset.json") != plan["active_config_sha256"]:
            raise ValueError("Active checkpoint changed")
        result.update(status="completed", active_checkpoint_unchanged=True)
        write(root / "result.json", result)
        print("SFT256 RL FOLLOWUP COMPLETED:", root, flush=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        write(root / "result.json", result)
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("shared", type=Path)
    parser.add_argument("--reward-profile", choices=tuple(PROFILES), default="legacy-v1")
    args = parser.parse_args()
    main(args.shared, args.reward_profile)
