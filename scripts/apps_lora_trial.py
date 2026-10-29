import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from apps_core import ASSETS, digest, read, write
from apps_lora import DEFAULT_PROFILE
from apps_model_runner import PROJECT, run_folder
from apps_round import run
from apps_sft import build_examples
from apps_ui import gpu_busy
from audit_apps_run import audit
from check_apps_lora import check


def prepare():
    root = run_folder("lora-coverage-trial")
    print("EVIDENCE:", root, flush=True)
    source = root / "source"
    source.mkdir()
    names = ("apps_lora_trial.py", "apps_lora.py", "check_apps_lora.py", "apps_sft.py", "apps_model_runner.py", "apps_core.py",
             "apps_round.py", "apps_rewards.py", "apps_curriculum.py", "apps_profiles.py", "restricted_python.py", "local_model_loader.py",
             "train_static_grpo.py", "check_lora_update.py", "audit_apps_run.py")
    hashes = {}
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), source / name)
        hashes[name] = digest(source / name)
    targets, cases, manifest = build_examples(42)
    state = read(PROJECT / "APPS_STATE.json")
    write(root / "previous-state.json", state)
    write(root / "coverage.json", check())
    plan = {"recorded_at": datetime.now(timezone.utc).isoformat(), "purpose": "Controlled LoRA target-coverage experiment, not a full ATGen replication",
            "hypothesis": "Historical q/v LoRA adapts only six full-attention blocks in this hybrid model. Adding rank4 in_proj_qkv adapters to 18 linear-attention blocks may improve expected prediction and usable bug-detecting tests.",
            "arms": [DEFAULT_PROFILE, "hybrid-qkv-v1"], "dataset_id": manifest["dataset_id"], "train_sha256": manifest["splits"]["train"]["sha256"],
            "training_targets": len(targets), "train_cases": len(cases), "sft_steps": 64, "grpo_steps": 96, "training_seed": 42,
            "generation_seeds": [42, 43, 44], "prompt_version": "apps-call-v2", "reward_profile": "legacy-v1",
            "fixed": "Same original BF16 model, rank4 alpha8 dropout0, Train target pool and sample order, SFT lr5e-5 and 4 examples/step, GRPO lr1e-5 and 4 samples/group, 96 tokens/completion. Fresh optimizer per stage.",
            "evaluation_order": "Train both SFT arms; evaluate each on Train to provide the same-parent baseline required by GRPO; train both GRPO arms. Then evaluate all four final checkpoints on Validation at all three sampling seeds. No intermediate selection or Test.",
            "primary": "Correct-expected count and bug detection of hybrid SFT+GRPO versus fresh narrow SFT+GRPO",
            "eligibility": "Across three Validation sampling seeds: mean correct expected improves by >=3/60, mean valid tests, bug-detecting tests and detected cases do not decrease. Also report each SFT separately. Passing only permits a later adaptive-curriculum comparison, not automatic deployment or an RL-effect claim.",
            "limits": "Single training seed; five Validation problems; more trainable parameters in the hybrid arm. This changes adapter coverage, not only rank or training steps. Static GRPO screens the representation before spending compute on adaptive refresh.",
            "source_hashes": hashes, "state_sha256": digest(PROJECT / "APPS_STATE.json"),
            "active_config_sha256": digest(PROJECT / "configs/active-dataset.json"), "model_test_evaluation": False}
    write(root / "plan.json", plan)
    write(root / "result.json", {"status": "prepared", "plan_sha256": digest(root / "plan.json"), "arms": {}, "model_test_evaluation": False})
    return root


def execute(root):
    if gpu_busy(ignore_pid=os.getpid()):
        raise RuntimeError("Another project GPU job is active")
    available = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0])
    if available < 6000:
        raise RuntimeError(f'GPU has {available} MiB free; experiment requires >=6000 MiB free before starting. Prepared plan is preserved.')
    plan = read(root / "plan.json")
    result = read(root / "result.json")
    if result["status"] != "prepared" or digest(root / "plan.json") != result["plan_sha256"]:
        raise ValueError("Fresh prepared plan required; refusing to overwrite an earlier attempt")
    for name, expected in plan["source_hashes"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("Prepared experiment source changed: " + name)
    if digest(PROJECT / "APPS_STATE.json") != plan["state_sha256"]:
        raise ValueError("Active experiment changed since preparation")
    os.environ.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                      HF_HOME=str(ASSETS / "cache/huggingface"), TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"))
    print("EVIDENCE:", root, flush=True)
    python = str(ASSETS / "env/Scripts/python.exe")
    runner = [python, "-B", str(PROJECT / "scripts/apps_model_runner.py")]
    result["status"] = "running"
    write(root / "result.json", result)
    try:
        for profile in plan["arms"]:
            trained = run([python, "-B", str(PROJECT / "scripts/apps_sft.py"), "--steps", "64", "--lora-profile", profile], root, profile + "-sft")
            audit(trained)
            info = read(trained / "result.json")
            if info["lora_coverage"]["profile"] != profile or not info["update_verified"]:
                raise ValueError("SFT profile or update mismatch")
            for target in info["lora_coverage"]["targets"]:
                if not any(target["module"] + ".lora_B.weight" in name for name in info["changed_adapter_tensors"]):
                    raise ValueError("A selected LoRA block received no saved B update: " + target["module"])
            result["arms"][profile] = {"sft": str(trained), "sft_result_sha256": digest(trained / "result.json")}
            write(root / "result.json", result)
        folders = [Path(result["arms"][profile]["sft"]) for profile in plan["arms"]]
        if digest(folders[0] / "targets.jsonl") != digest(folders[1] / "targets.jsonl") or digest(folders[0] / "training-order.json") != digest(folders[1] / "training-order.json"):
            raise ValueError("SFT targets or data order differ")
        for profile in plan["arms"]:
            arm = result["arms"][profile]
            parent = Path(arm["sft"])
            probe = run(runner + ["evaluate", "--split", "train", "--adapter", str(parent / "adapter"), "--prompt-version", "apps-call-v2", "--no-publish"], root, profile + "-train-probe")
            audit(probe)
            arm["train_probe"] = str(probe)
            trained = run(runner + ["train", "--steps", "96", "--seed", "42", "--init-adapter", str(parent / "adapter"),
                                    "--baseline-run", str(probe), "--prompt-version", "apps-call-v2", "--reward-profile", "legacy-v1", "--no-publish"], root, profile + "-grpo")
            audit(trained)
            if not read(trained / "result.json")["update_verified"]:
                raise ValueError("GRPO update not verified")
            arm.update(grpo=str(trained), grpo_result_sha256=digest(trained / "result.json"), evaluations=[])
            write(root / "result.json", result)
        orders = [[row["case_id"] for row in [json.loads(line) for line in (Path(result["arms"][profile]["grpo"]) / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()]] for profile in plan["arms"]]
        if orders[0] != orders[1] or len(orders[0]) != 96:
            raise ValueError("GRPO case order or budget mismatch")
        signatures = {}
        for seed in plan["generation_seeds"]:
            for profile in plan["arms"]:
                arm = result["arms"][profile]
                for stage in ("sft", "grpo"):
                    adapter = Path(arm[stage]) / "adapter"
                    evaluated = run(runner + ["evaluate", "--adapter", str(adapter), "--seed", str(seed), "--prompt-version", "apps-call-v2", "--no-publish"], root, f'{profile}-{stage}-validation-{seed}')
                    audit(evaluated)
                    info = read(evaluated / "result.json")
                    signature = {row["case_id"]: (row["prompt_sha256"], row["case_seed"]) for row in [json.loads(line) for line in (evaluated / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]}
                    if seed in signatures and signatures[seed] != signature:
                        raise ValueError("Validation prompts or case seeds differ")
                    signatures[seed] = signature
                    if not info["complete_split"] or info["metrics"]["generated_tests"] != 60:
                        raise ValueError("Incomplete Validation")
                    arm["evaluations"].append({"stage": stage, "generation_seed": seed, "path": str(evaluated), "result_sha256": digest(evaluated / "result.json"), "metrics": info["metrics"]})
                    write(root / "result.json", result)
        for name, expected in plan["source_hashes"].items():
            if digest(Path(__file__).with_name(name)) != expected:
                raise ValueError("Trial source changed while running")
        if digest(PROJECT / "APPS_STATE.json") != plan["state_sha256"] or digest(PROJECT / "configs/active-dataset.json") != plan["active_config_sha256"]:
            raise ValueError("Active experiment changed during trial")
        result.update(status="completed", active_checkpoint_unchanged=True, training_case_order_match=True)
        write(root / "result.json", result)
        print("LORA COVERAGE TRIAL COMPLETED:", root, flush=True)
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
