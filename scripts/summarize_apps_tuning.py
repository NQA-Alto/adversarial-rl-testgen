import argparse
from pathlib import Path

from apps_core import digest, read, write
from apps_model_runner import PROJECT
from audit_apps_run import audit
from result_docs import write_result_readme


def summarize(thinking, trials):
    result = read(thinking / "result.json")
    review = read(thinking / "futility-review.json")
    if result["status"] != "stopped_early_infeasible" or review["status"] != "passed":
        raise ValueError("Thinking feasibility record is incomplete")
    from apps_reasoning_probe import audit as audit_reasoning
    audit_reasoning(thinking)
    records = []
    for path in trials:
        trial = read(path / "result.json")
        if trial["status"] != "completed":
            raise ValueError("Prompt trial is not complete")
        audit(Path(trial["train_run"]))
        if trial["validation_evaluated"]:
            audit(Path(trial["validation_run"]))
        records.append({"path": str(path), "result_sha256": digest(path / "result.json"), "result": trial})
    output = PROJECT / "results/apps-tuning"
    output.mkdir(parents=True, exist_ok=True)
    write(output / "experiments.json", {"thinking_probe": {"path": str(thinking), "result": result, "review": review},
                                       "prompt_trials": records, "training": False, "model_test_evaluation": False,
                                       "active_configuration_changed": False})
    write_result_readme((output / 'README.md').parent)
    print("Saved:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thinking", type=Path, required=True)
    parser.add_argument("--trial", type=Path, action="append", required=True)
    args = parser.parse_args()
    summarize(args.thinking, args.trial)
