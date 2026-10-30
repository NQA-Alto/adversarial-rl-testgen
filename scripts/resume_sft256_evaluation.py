import argparse
import os
import shutil
from pathlib import Path

from apps_core import ASSETS, digest, read, write
from apps_model_runner import PROJECT
from apps_round import run
from apps_ui import gpu_busy
from audit_apps_run import audit


def resume(root):
    plan, result = read(root / "plan.json"), read(root / "result.json")
    if result["status"] != "failed" or "training" not in result or digest(root / "plan.json") != result["plan_sha256"]:
        raise ValueError("Failed evaluation after completed training required")
    if gpu_busy(ignore_pid=os.getpid()):
        raise RuntimeError("Another GPU task is active")
    for name, expected in plan["source_hashes"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("Original experiment source changed: " + name)
    for filename, key in (("APPS_STATE.json", "state_sha256"), ("configs/active-dataset.json", "active_config_sha256")):
        if digest(PROJECT / filename) != plan[key]:
            raise ValueError("Active state changed")
    training = Path(result["training"])
    audit(training)
    trained = read(training / "result.json")
    if not trained["update_verified"] or trained["parent_adapter_sha256"] != plan["parent_adapter_sha256"]:
        raise ValueError("Verified original training required")
    complete = set()
    for row in result["evaluations"]:
        path = Path(row["path"])
        audit(path)
        if digest(path / "result.json") != row["result_sha256"]:
            raise ValueError("Completed evaluation changed")
        complete.add(row["generation_seed"])
    failures = []
    for seed in plan["generation_seeds"]:
        if seed in complete:
            continue
        log = root / f'validation-{seed}.txt'
        if log.exists():
            locations = [Path(line.removeprefix("EVIDENCE: ").strip()) for line in log.read_text(encoding="utf-8").splitlines() if line.startswith("EVIDENCE: ")]
            if len(locations) != 1 or (locations[0] / "predictions.jsonl").exists():
                raise ValueError("Partial outputs require case-level recovery")
            failures.append({"generation_seed": seed, "path": str(locations[0]), "result_sha256": digest(locations[0] / "result.json"), "generated_outputs": 0})
    recovery = root / "evaluation-recovery"
    recovery.mkdir(exist_ok=False)
    shutil.copy2(root / "result.json", recovery / "failed-result.json")
    shutil.copy2(__file__, recovery / Path(__file__).name)
    write(recovery / "plan.json", {"original_plan_sha256": result["plan_sha256"],
                                   "failed_result_sha256": digest(recovery / "failed-result.json"),
                                   "resume_source_sha256": digest(__file__), "retained_seeds": sorted(complete),
                                   "restart_empty_attempts": failures,
                                   "scope": "Evaluation only; identical frozen adapter, prompts, seeds and budget. Completed outputs are retained. No training or model selection."})
    os.environ.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                      HF_HOME=str(ASSETS / "cache/huggingface"), TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"))
    runner = [str(ASSETS / "env/Scripts/python.exe"), "-B", str(PROJECT / "scripts/apps_model_runner.py")]
    result.update(status="resuming_evaluation", evaluation_recovery=str(recovery))
    write(root / "result.json", result)
    try:
        for seed in plan["generation_seeds"]:
            if seed in complete:
                continue
            path = run(runner + ["evaluate", "--adapter", str(training / "adapter"), "--seed", str(seed),
                                 "--prompt-version", plan["prompt_version"], "--no-publish"], root, f'validation-{seed}-recovery')
            audit(path)
            info = read(path / "result.json")
            result["evaluations"].append({"generation_seed": seed, "path": str(path), "result_sha256": digest(path / "result.json"), "metrics": info["metrics"]})
            write(root / "result.json", result)
        for name, expected in plan["source_hashes"].items():
            if digest(Path(__file__).with_name(name)) != expected:
                raise ValueError("Source changed during recovery")
        for filename, key in (("APPS_STATE.json", "state_sha256"), ("configs/active-dataset.json", "active_config_sha256")):
            if digest(PROJECT / filename) != plan[key]:
                raise ValueError("Active state changed during recovery")
        result.pop("error", None)
        result.update(status="completed", active_checkpoint_unchanged=True)
        write(root / "result.json", result)
        print("EVALUATION RECOVERY COMPLETED:", root, flush=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        write(root / "result.json", result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    resume(parser.parse_args().root)
