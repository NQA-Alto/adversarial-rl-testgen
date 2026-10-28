import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from apps_core import ASSETS, canonical, digest, load_cases, read, write
from apps_model_runner import PROJECT, run_folder
from apps_rewards import PROFILES
from apps_round import run
from apps_ui import gpu_busy
from audit_apps_run import audit
from check_apps_rewards import check


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    if gpu_busy(ignore_pid=os.getpid()):
        raise RuntimeError("Another project GPU job is active")
    os.environ.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                      HF_HOME=str(ASSETS / "cache/huggingface"), TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"))
    state_path = PROJECT / "APPS_STATE.json"
    state, state_hash = read(state_path), digest(state_path)
    active_hash = digest(PROJECT / "configs/active-dataset.json")
    for key in ("sft_training", "sft_validation"):
        if digest(Path(state[key]["path"]) / "result.json") != state[key]["result_sha256"]:
            raise ValueError("Parent evidence changed")
    parent = Path(state["sft_training"]["adapter_path"])
    parent_hash = digest(parent / "adapter_model.safetensors")
    if parent_hash != state["sft_training"]["adapter_sha256"]:
        raise ValueError("Parent adapter changed")
    baseline = Path(state["sft_validation"]["path"])
    audit(Path(state["sft_training"]["path"]))
    audit(baseline)
    cases, manifest = load_cases("train")
    if len(cases) != 41 or state["prompt_version"] != "apps-call-v2":
        raise ValueError("Unexpected fixed training protocol")
    root = run_folder("reward-ablation")
    print("EVIDENCE:", root, flush=True)
    source = root / "source"
    source.mkdir()
    names = ("apps_reward_trial.py", "apps_rewards.py", "check_apps_rewards.py", "apps_core.py", "apps_model_runner.py",
             "apps_round.py", "apps_curriculum.py", "apps_profiles.py", "apps_output_contracts.py", "restricted_python.py",
             "local_model_loader.py", "train_static_grpo.py", "check_lora_update.py", "audit_apps_run.py")
    hashes = {}
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), source / name)
        hashes[name] = digest(source / name)
    write(root / "previous-state.json", state)
    write(root / "preflight.json", check())
    plan = {"recorded_at": datetime.now(timezone.utc).isoformat(), "method": "STATIC_GRPO_REWARD_WEIGHT_ABLATION",
            "dataset_id": manifest["dataset_id"], "train_sha256": manifest["splits"]["train"]["sha256"],
            "parent_adapter": str(parent), "parent_adapter_sha256": parent_hash,
            "parent_validation": str(baseline), "parent_validation_sha256": digest(baseline / "result.json"),
            "arms_in_order": ["legacy-v1", "equal-v1"], "reward_weights": PROFILES, "steps_per_arm": 96,
            "seed": 42, "prompt_version": "apps-call-v2", "tests_per_group": 4, "max_new_tokens": 96,
            "configuration": "Same frozen 41-case static curriculum, new identical GRPO optimizer per arm, lr=1e-5, LoRA rank4, BF16, same case order. Only reward weights differ.",
            "evaluation_policy": "Train both final adapters before evaluating either. Evaluate Train for fitting, then Validation once per final adapter with identical prompt, per-case seed, 4 samples and budget. No intermediate checkpoint selection.",
            "primary_outcomes": ["bug_detecting_tests", "cases_detected"],
            "secondary_outcomes": ["correct_expected", "valid_tests"],
            "candidate_gate": "Equal reward must add >=3 correct expected tests versus fresh legacy control and not reduce valid tests, bug-detecting tests or detected cases versus either control or parent. This is eligibility for a further adversarial comparison, not automatic promotion.",
            "limits": "One training seed and five Validation problems. Static reward ablation isolates weights, not the full adaptive ATGen method. JSON+domain format reward differs from ATGen thinking tags; equal weights are normalized to total 1. CUDA nondeterminism is not ruled out.",
            "source": "https://arxiv.org/html/2510.14635v1#S4.SS1",
            "source_hashes": hashes, "state_sha256": state_hash, "active_config_sha256": active_hash,
            "model_test_evaluation": False}
    write(root / "plan.json", plan)
    result = {"status": "running", "plan": plan, "arms": {}, "model_test_evaluation": False}
    write(root / "result.json", result)
    runner = [str(ASSETS / "env/Scripts/python.exe"), "-B", str(PROJECT / "scripts/apps_model_runner.py")]
    try:
        for profile in plan["arms_in_order"]:
            folder = run(runner + ["train", "--steps", "96", "--seed", "42", "--prompt-version", "apps-call-v2",
                                 "--init-adapter", str(parent), "--baseline-run", str(baseline),
                                 "--reward-profile", profile, "--no-publish"], root, profile + "-training")
            audit(folder)
            trained = read(folder / "result.json")
            if trained["parent_adapter_sha256"] != parent_hash or not trained["parent_adapter_exact_match"] or not trained["update_verified"]:
                raise ValueError("Parent or policy update verification failed")
            result["arms"][profile] = {"training": str(folder), "training_result_sha256": digest(folder / "result.json")}
            write(root / "result.json", result)
        paths = [Path(result["arms"][key]["training"]) for key in plan["arms_in_order"]]
        orders = [[row["case_id"] for row in records(path / "rollouts.jsonl")] for path in paths]
        if orders[0] != orders[1] or len(orders[0]) != 96:
            raise ValueError("Unequal training case order or rollout budget")
        training_results = [read(path / "result.json") for path in paths]
        if training_results[0]["configuration"] != training_results[1]["configuration"]:
            raise ValueError("Training configuration mismatch")
        result["training_case_order_match"] = True
        for profile in plan["arms_in_order"]:
            arm = result["arms"][profile]
            for split in ("train", "validation"):
                folder = run(runner + ["evaluate", "--split", split, "--adapter", str(Path(arm["training"]) / "adapter"),
                                     "--seed", "42", "--prompt-version", "apps-call-v2", "--no-publish"], root, profile + "-" + split)
                audit(folder)
                arm[split] = str(folder)
                arm[split + "_result_sha256"] = digest(folder / "result.json")
                arm[split + "_metrics"] = read(folder / "result.json")["metrics"]
                write(root / "result.json", result)
        def signatures(folder):
            return {row["case_id"]: (row["prompt_sha256"], row["case_seed"]) for row in records(folder / "predictions.jsonl")}
        for split in ("train", "validation"):
            a, b = (Path(result["arms"][profile][split]) for profile in plan["arms_in_order"])
            if signatures(a) != signatures(b) or read(a / "result.json")["generation"] != read(b / "result.json")["generation"]:
                raise ValueError("Evaluation protocol mismatch")
        if signatures(baseline) != signatures(Path(result["arms"]["equal-v1"]["validation"])):
            raise ValueError("Parent Validation prompt or seeds differ")
        control, candidate = (result["arms"][profile]["validation_metrics"] for profile in plan["arms_in_order"])
        parent_metrics = read(baseline / "result.json")["metrics"]
        eligible = candidate["correct_expected"] >= control["correct_expected"] + 3 and all(
            candidate[field] >= reference[field] for field in ("valid_tests", "bug_detecting_tests", "cases_detected")
            for reference in (control, parent_metrics))
        for name, expected_hash in hashes.items():
            if digest(Path(__file__).with_name(name)) != expected_hash:
                raise ValueError("Source changed during reward trial: " + name)
        if digest(state_path) != state_hash or digest(PROJECT / "configs/active-dataset.json") != active_hash:
            raise ValueError("Active checkpoint or configuration changed during trial")
        result.update(status="completed", parent_validation_metrics=parent_metrics, controlled_evaluation_verified=True,
                      eligible_for_adversarial_followup=eligible, active_checkpoint_unchanged=True)
        write(root / "result.json", result)
        print("REWARD TRIAL COMPLETED:", canonical(result), flush=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        write(root / "result.json", result)
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
