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
from check_apps_target_order import check


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare(control, probe):
    info = read(control / "result.json")
    audit(control)
    audit(probe)
    baseline = read(probe / "result.json")
    if info["optimizer_steps"] != 64 or info["trainable_parameters"] != 208896 or info["configuration"].get("target_order", "sorted-v1") != "sorted-v1":
        raise ValueError("Expected the completed narrow-profile sorted-target SFT64 control")
    if baseline["split"] != "train" or baseline["seed"] != 42 or baseline["adapter_sha256"] != info["adapter"]["adapter_sha256"]:
        raise ValueError("Control Train probe does not evaluate this exact SFT adapter")
    root = run_folder("target-order-trial")
    print("EVIDENCE:", root, flush=True)
    source = root / "source"
    source.mkdir()
    hashes = {}
    for name in ("apps_order_trial.py", "apps_sft.py", "check_apps_target_order.py", "apps_model_runner.py", "apps_core.py", "apps_lora.py",
                 "apps_round.py", "apps_rewards.py", "apps_curriculum.py", "apps_profiles.py", "restricted_python.py", "local_model_loader.py",
                 "train_static_grpo.py", "check_lora_update.py", "audit_apps_run.py"):
        shutil.copy2(Path(__file__).with_name(name), source / name)
        hashes[name] = digest(source / name)
    write(root / "preflight.json", check())
    write(root / "previous-state.json", read(PROJECT / "APPS_STATE.json"))
    plan = {"recorded_at": datetime.now(timezone.utc).isoformat(), "purpose": "Isolate top-level supervised target field order in a local SFT+GRPO pipeline",
            "hypothesis": "Teaching input before expected may let an autoregressive model condition the answer on its selected concrete input. Observation of the historical order shift is not itself causal evidence.",
            "arms": ["sorted-v1", "input-first-v1"], "control_sft": str(control), "control_sft_result_sha256": digest(control / "result.json"),
            "control_probe": str(probe), "control_probe_result_sha256": digest(probe / "result.json"),
            "reuse": "Control SFT64 and its Train probe completed before this diagnosis; no Validation from the interrupted coverage trial was run. Discard interrupted partial GRPO; train a new full 96-step control here.",
            "dataset_id": info["dataset_id"], "train_sha256": info["train_sha256"], "target_count": 319,
            "lora_profile": "attention-qv-v1", "trainable_parameters": 208896, "sft_steps": 64, "grpo_steps": 96, "training_seed": 42,
            "generation_seeds": [42, 43, 44], "prompt_version": "apps-call-v2", "reward_profile": "legacy-v1",
            "only_intended_change": "Top-level target JSON order. Canonical inner input key order, values, sample IDs and supervised sample order unchanged. No parser, reward, prompt, model, LoRA coverage, learning rate or generation budget change.",
            "evaluation_order": "Train new input-first SFT64, probe Train, train fresh GRPO96 from each SFT. Only after both GRPO arms complete, evaluate all four SFT/GRPO final checkpoints on all three Validation generation seeds.",
            "eligibility": "At final GRPO stage, input-first mean correct expected must exceed fresh sorted control by >=3/60; mean valid tests, bug-detecting tests and detected cases must not decrease. Report SFT separately. No automatic checkpoint promotion or claim of adaptive benefit.",
            "limits": "Single training seed and five Validation problems. Control SFT is reused rather than simultaneous; CUDA nondeterminism is not ruled out. Sampling SD is not a confidence interval.",
            "source_hashes": hashes, "state_sha256": digest(PROJECT / "APPS_STATE.json"),
            "active_config_sha256": digest(PROJECT / "configs/active-dataset.json"), "model_test_evaluation": False}
    write(root / "plan.json", plan)
    write(root / "result.json", {"status": "prepared", "plan_sha256": digest(root / "plan.json"), "arms": {}, "model_test_evaluation": False})


def execute(root):
    if gpu_busy(ignore_pid=os.getpid()):
        raise RuntimeError("Another project GPU job is active")
    available = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0])
    if available < 6000:
        raise RuntimeError("At least 6000 MiB free GPU memory required; prepared plan preserved")
    plan, result = read(root / "plan.json"), read(root / "result.json")
    if result["status"] != "prepared" or digest(root / "plan.json") != result["plan_sha256"]:
        raise ValueError("Fresh prepared trial required")
    for name, expected in plan["source_hashes"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("Prepared source changed")
    control, probe = Path(plan["control_sft"]), Path(plan["control_probe"])
    if digest(control / "result.json") != plan["control_sft_result_sha256"] or digest(probe / "result.json") != plan["control_probe_result_sha256"]:
        raise ValueError("Reused control evidence changed")
    os.environ.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                      HF_HOME=str(ASSETS / "cache/huggingface"), TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"))
    print("EVIDENCE:", root, flush=True)
    python = str(ASSETS / "env/Scripts/python.exe")
    runner = [python, "-B", str(PROJECT / "scripts/apps_model_runner.py")]
    result.update(status="running", arms={"sorted-v1": {"sft": str(control), "sft_result_sha256": digest(control / "result.json"), "train_probe": str(probe)}})
    write(root / "result.json", result)
    try:
        candidate = run([python, "-B", str(PROJECT / "scripts/apps_sft.py"), "--steps", "64", "--target-order", "input-first-v1"], root, "input-first-sft")
        audit(candidate)
        trained = read(candidate / "result.json")
        if trained["configuration"]["target_order"] != "input-first-v1" or trained["trainable_parameters"] != 208896:
            raise ValueError("Candidate changes more than target order")
        old_targets, new_targets = records(control / "targets.jsonl"), records(candidate / "targets.jsonl")
        if len(old_targets) != 319 or len(new_targets) != 319 or digest(control / "training-order.json") != digest(candidate / "training-order.json"):
            raise ValueError("Supervised sample order or count changed")
        for a, b in zip(old_targets, new_targets, strict=True):
            if json.loads(a["target"]) != json.loads(b["target"]) or {k:v for k,v in a.items() if k != "target"} != {k:v for k,v in b.items() if k != "target"}:
                raise ValueError("Supervised semantic labels changed")
        candidate_probe = run(runner + ["evaluate", "--split", "train", "--adapter", str(candidate / "adapter"), "--prompt-version", "apps-call-v2", "--no-publish"], root, "input-first-train-probe")
        audit(candidate_probe)
        result["arms"]["input-first-v1"] = {"sft": str(candidate), "sft_result_sha256": digest(candidate / "result.json"), "train_probe": str(candidate_probe)}
        write(root / "result.json", result)
        for order in plan["arms"]:
            arm = result["arms"][order]
            trained = run(runner + ["train", "--steps", "96", "--seed", "42", "--init-adapter", str(Path(arm["sft"]) / "adapter"),
                                    "--baseline-run", arm["train_probe"], "--prompt-version", "apps-call-v2", "--reward-profile", "legacy-v1", "--no-publish"], root, order + "-grpo")
            audit(trained)
            if not read(trained / "result.json")["update_verified"]:
                raise ValueError("No verified GRPO update")
            arm.update(grpo=str(trained), grpo_result_sha256=digest(trained / "result.json"), evaluations=[])
            write(root / "result.json", result)
        orders = [[row["case_id"] for row in records(Path(result["arms"][order]["grpo"]) / "rollouts.jsonl")] for order in plan["arms"]]
        if orders[0] != orders[1] or len(orders[0]) != 96:
            raise ValueError("GRPO case order mismatch")
        signatures = {}
        for seed in plan["generation_seeds"]:
            for order in plan["arms"]:
                arm = result["arms"][order]
                for stage in ("sft", "grpo"):
                    path = run(runner + ["evaluate", "--adapter", str(Path(arm[stage]) / "adapter"), "--seed", str(seed), "--prompt-version", "apps-call-v2", "--no-publish"], root, f'{order}-{stage}-validation-{seed}')
                    audit(path)
                    info = read(path / "result.json")
                    signature = {row["case_id"]: (row["prompt_sha256"], row["case_seed"]) for row in records(path / "predictions.jsonl")}
                    if seed in signatures and signatures[seed] != signature:
                        raise ValueError("Validation prompt or per-case seed differs")
                    signatures[seed] = signature
                    if not info["complete_split"] or info["metrics"]["generated_tests"] != 60:
                        raise ValueError("Incomplete planned evaluation")
                    arm["evaluations"].append({"stage":stage,"generation_seed":seed,"path":str(path),"result_sha256":digest(path / "result.json"),"metrics":info["metrics"]})
                    write(root / "result.json", result)
        for name, expected in plan["source_hashes"].items():
            if digest(Path(__file__).with_name(name)) != expected:
                raise ValueError("Source changed during experiment")
        if digest(PROJECT / "APPS_STATE.json") != plan["state_sha256"] or digest(PROJECT / "configs/active-dataset.json") != plan["active_config_sha256"]:
            raise ValueError("Active experiment changed")
        result.update(status="completed", active_checkpoint_unchanged=True, target_semantics_and_order_verified=True, grpo_case_order_match=True)
        write(root / "result.json", result)
        print("TARGET ORDER TRIAL COMPLETED:", root, flush=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        write(root / "result.json", result)
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--control-sft", type=Path)
    parser.add_argument("--control-probe", type=Path)
    args = parser.parse_args()
    if args.run:
        execute(args.run)
    elif args.control_sft and args.control_probe:
        prepare(args.control_sft, args.control_probe)
    else:
        parser.error("Provide --run, or both --control-sft and --control-probe")
