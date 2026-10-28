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
from audit_apps_run import audit


def prepare(folder):
    original = read(folder / "plan.json")
    if (folder / "paired64-plan.json").exists():
        raise ValueError("Paired control plan already exists")
    paths = list((ASSETS / "evidence").glob("apps-sft-v2-*/step64-replay.json"))
    stages = read(folder / "stages.json") if (folder / "stages.json").exists() else {}
    training = Path(stages["sft-256"]) if "sft-256" in stages else next(path.parent for path in sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)
                                                                   if read(path.parent / "result.json")["configuration"]["steps"] == 256)
    record = read(training / "result.json")
    if record["target_sha256"] != original["target_sha256"] or not (training / "step64-replay.json").exists():
        raise ValueError("Current training run does not match the budget trial")
    if (folder / "sft-256-validation.txt").exists():
        raise ValueError("Paired control must be planned before new Validation")
    checkpoint = training / "checkpoint-64"
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "reason": "Historical step-64 replay is not exact; add an exact same-trajectory budget control before viewing new Validation.",
            "original_plan_preserved": True, "training_run": str(training), "checkpoint64": str(checkpoint),
            "checkpoint64_sha256": digest(checkpoint / "adapter_model.safetensors"), "dataset_id": original["dataset_id"],
            "checkpoint_policy": "Finish training to 256 before evaluating either checkpoint on Validation. Evaluate step64 only as a control, never to select an intermediate checkpoint. Step128 is not evaluated.",
            "candidate_gate": "SFT256 must exceed both historical and same-trajectory SFT64 by >=3 correct-expected tests, with valid tests, bug tests and detected cases no lower than either control. This follow-up does not change the historical replay failure or original gate.",
            "generation": {"seed": 42, "tests_per_case": 4, "max_new_tokens": 96, "prompt_version": "apps-call-v2"},
            "source_sha256": digest(Path(__file__)), "model_test_evaluation": False}
    shutil.copy2(__file__, folder / "source/apps_sft_paired_control.py")
    write(folder / "paired64-plan.json", plan)
    print("Paired control planned before Validation:", folder)


def execute(folder):
    plan = read(folder / "paired64-plan.json")
    trial = read(folder / "result.json")
    if trial["status"] != "completed" or digest(Path(__file__)) != plan["source_sha256"]:
        raise ValueError("Original trial must finish and follow-up source must be unchanged")
    checkpoint = Path(plan["checkpoint64"])
    if digest(checkpoint / "adapter_model.safetensors") != plan["checkpoint64_sha256"]:
        raise ValueError("Same-trajectory checkpoint changed")
    os.environ.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1",
                      HF_HOME=str(ASSETS / "cache/huggingface"), TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"))
    python = str(ASSETS / "env/Scripts/python.exe")
    control = run([python, "-B", str(PROJECT / "scripts/apps_model_runner.py"), "evaluate", "--adapter", str(checkpoint),
                   "--prompt-version", "apps-call-v2", "--no-publish"], folder, "same-trajectory-64-validation")
    audit(control)
    candidate = Path(trial["validation_run"])
    audit(candidate)
    a = read(control / "result.json")
    b = read(candidate / "result.json")
    historical = read(Path(trial["plan"]["baseline_validation"]) / "result.json")
    def signatures(path):
        return {row["case_id"]: (row["prompt_sha256"], row["case_seed"]) for row in
                [json.loads(line) for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]}
    if signatures(control) != signatures(candidate) or a["generation"] != b["generation"]:
        raise ValueError("Validation protocol mismatch")
    m = b["metrics"]
    controls = [a["metrics"], historical["metrics"]]
    eligible = all(m["correct_expected"] >= base["correct_expected"] + 3 and
                   all(m[field] >= base[field] for field in ("valid_tests", "bug_detecting_tests", "cases_detected")) for base in controls)
    result = {"status": "completed", "plan": plan, "control_validation": str(control), "candidate_validation": str(candidate),
              "control_metrics": a["metrics"], "candidate_metrics": m, "historical_metrics": historical["metrics"],
              "eligible_for_later_rl_comparison": eligible, "same_trajectory_budget_comparison": True,
              "historical_step64_replay": trial["step64_replay"], "original_trial_eligibility": trial["eligible_for_later_rl_comparison"],
              "model_test_evaluation": False, "active_checkpoint_unchanged": True}
    write(folder / "paired64-result.json", result)
    print("SAME-TRAJECTORY CONTROL COMPLETED:", result, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "execute"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else execute)(args.folder)
