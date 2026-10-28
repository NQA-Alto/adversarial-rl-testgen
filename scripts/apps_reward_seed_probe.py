import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from apps_core import ASSETS, digest, read, write
from apps_model_runner import PROJECT
from apps_round import run
from apps_ui import gpu_busy
from audit_apps_run import audit


def prepare(folder):
    path = folder / "sampling-plan.json"
    if path.exists():
        raise ValueError("Sampling plan already exists; refusing to overwrite")
    trial = read(folder / "result.json")
    if trial["status"] != "running" or any("validation" in arm for arm in trial["arms"].values()):
        raise ValueError("This follow-up must be planned before either new Validation result")
    source = folder / "sampling-source"
    source.mkdir()
    hashes = {}
    for name in ("apps_reward_seed_probe.py", "apps_model_runner.py", "apps_core.py", "apps_rewards.py", "audit_apps_run.py", "apps_round.py"):
        shutil.copy2(Path(__file__).with_name(name), source / name)
        hashes[name] = digest(source / name)
    plan = {"recorded_at": datetime.now(timezone.utc).isoformat(), "original_trial_plan_sha256": digest(folder / "plan.json"),
            "purpose": "Supplementary sampling variability check, not additional training or checkpoint selection",
            "variants": ["SFT64_parent", "legacy-v1", "equal-v1"], "generation_seeds": [42, 43, 44],
            "reuse": "Seed42 comes from the original trial and historical parent. Run seeds43 and44 on all three fixed final adapters after original trial completion, regardless of its outcome.",
            "policy": "Report every seed and variant; no selecting a favorable seed. Original candidate gate stays on seed42, supplementary data do not silently replace it. This is one training seed with three sampling seeds, not three trained models per method.",
            "evaluation": "Same Validation, prompt v2, 4 samples per code, 96 tokens, same per-case seeding formula. No Test.",
            "source_hashes": hashes, "model_test_evaluation": False}
    write(path, plan)
    print("Sampling follow-up planned before Validation:", path)


def execute(folder):
    if gpu_busy(ignore_pid=os.getpid()):
        raise RuntimeError("Another project GPU job is active")
    plan = read(folder / "sampling-plan.json")
    trial = read(folder / "result.json")
    if trial["status"] != "completed" or digest(folder / "plan.json") != plan["original_trial_plan_sha256"]:
        raise ValueError("Completed original trial required")
    if (folder / "sampling-result.json").exists():
        raise ValueError("Sampling result already exists; refusing to overwrite")
    state_hash = digest(PROJECT / "APPS_STATE.json")
    active_hash = digest(PROJECT / "configs/active-dataset.json")
    parent = Path(trial["plan"]["parent_adapter"])
    variants = {"SFT64_parent": (parent, Path(trial["plan"]["parent_validation"]))}
    for profile, arm in trial["arms"].items():
        variants[profile] = (Path(arm["training"]) / "adapter", Path(arm["validation"]))
    result = {"status": "running", "sampling_plan_sha256": digest(folder / "sampling-plan.json"),
              "original_trial_result_sha256": digest(folder / "result.json"), "runs": [], "model_test_evaluation": False}
    write(folder / "sampling-result.json", result)
    os.environ.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                      HF_HOME=str(ASSETS / "cache/huggingface"), TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"))
    runner = [str(ASSETS / "env/Scripts/python.exe"), "-B", str(PROJECT / "scripts/apps_model_runner.py")]
    signatures = {}
    try:
        for seed in plan["generation_seeds"]:
            for name in plan["variants"]:
                adapter, old_evaluation = variants[name]
                adapter_hash = digest(adapter / "adapter_model.safetensors")
                expected_hash = trial["plan"]["parent_adapter_sha256"] if name == "SFT64_parent" else read(Path(trial["arms"][name]["training"]) / "result.json")["adapter"]["adapter_sha256"]
                if adapter_hash != expected_hash:
                    raise ValueError("Adapter changed")
                path = old_evaluation if seed == 42 else run(runner + ["evaluate", "--adapter", str(adapter), "--seed", str(seed),
                          "--prompt-version", "apps-call-v2", "--no-publish"], folder, "sampling-" + str(seed) + "-" + name)
                audit(path)
                evaluated = read(path / "result.json")
                if evaluated["adapter_sha256"] != adapter_hash or evaluated["seed"] != seed or not evaluated["complete_split"]:
                    raise ValueError("Unexpected sampling evaluation")
                predictions = [json.loads(line) for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                current = {row["case_id"]: (row["prompt_sha256"], row["case_seed"]) for row in predictions}
                if seed in signatures and signatures[seed] != current:
                    raise ValueError("Prompt or per-case seed differs between variants")
                signatures[seed] = current
                result["runs"].append({"variant": name, "generation_seed": seed, "path": str(path),
                                       "result_sha256": digest(path / "result.json"), "adapter_sha256": adapter_hash,
                                       "metrics": evaluated["metrics"]})
                write(folder / "sampling-result.json", result)
        for name, value in plan["source_hashes"].items():
            if digest(Path(__file__).with_name(name)) != value:
                raise ValueError("Frozen sampling source changed")
        if digest(PROJECT / "APPS_STATE.json") != state_hash or digest(PROJECT / "configs/active-dataset.json") != active_hash:
            raise ValueError("Active experiment changed")
        result.update(status="completed", runs_completed=len(result["runs"]), training_seeds=1, sampling_seeds=3,
                      active_checkpoint_unchanged=True, original_candidate_gate_unchanged=True)
        write(folder / "sampling-result.json", result)
        print("Sampling probe completed:", folder)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        write(folder / "sampling-result.json", result)
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    (prepare if args.plan_only else execute)(args.folder)
