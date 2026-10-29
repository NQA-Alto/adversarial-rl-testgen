import json
from collections import Counter, defaultdict
from pathlib import Path

from apps_core import digest, read, write
from apps_model_runner import PROJECT
from result_docs import write_result_readme


def analyze():
    state = read(PROJECT / "APPS_STATE.json")
    training = Path(state["sft_training"]["path"])
    targets = [json.loads(line) for line in (training / "targets.jsonl").read_text(encoding="utf-8").splitlines()]
    first = dict(Counter(next(iter(json.loads(row["target"]))) for row in targets))
    if digest(training / "targets.jsonl") != read(training / "result.json")["target_sha256"]:
        raise ValueError("Historical supervised targets changed")
    sampling = read(PROJECT / "results/apps-adversarial-review/metrics.json")
    summaries = {}
    for variant in ("base", "sft", "control", "adaptive"):
        counts = defaultdict(lambda: Counter(samples=0, valid=0, correct_expected=0, bug_detected=0))
        for row in sampling["runs"]:
            if row["variant"] != variant:
                continue
            folder = Path(row["path"])
            if digest(folder / "result.json") != row["result_sha256"]:
                raise ValueError("Saved evaluation changed")
            for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
                for sample in json.loads(line)["samples"]:
                    try:
                        value = json.loads(sample["raw_output"])
                        key = next(iter(value), "empty_object") if type(value) is dict else "not_object"
                    except (ValueError, TypeError):
                        key = "parse_error"
                    counts[key].update(samples=1, valid=int(sample["valid_format"]), correct_expected=int(sample["correct_expected"]), bug_detected=int(sample["bug_detected"]))
        summaries[variant] = {key: dict(value) for key, value in counts.items()}
    output = PROJECT / "results/apps-serialization-diagnosis"
    output.mkdir(parents=True, exist_ok=True)
    report = {"target_path": str(training / "targets.jsonl"), "target_sha256": digest(training / "targets.jsonl"),
              "supervised_targets": len(targets), "first_field_in_targets": first, "target_example": targets[0]["target"],
              "validation_first_field": summaries, "evaluations_per_variant": 3, "model_test_evaluation": False,
              "interpretation": "Observed order shift, not a causal demonstration that ordering alone explains low accuracy. Proposed experiment changes only top-level training-target field order; strict JSON scoring is order invariant."}
    write(output / "analysis.json", report)
    write_result_readme((output / 'README.md').parent)
    print({"output": str(output), "target_first_fields": first, "saved_output_first_fields": summaries})


if __name__ == "__main__":
    analyze()
