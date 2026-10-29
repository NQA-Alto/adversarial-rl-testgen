import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from apps_adversary import audit_pool, rows
from apps_core import digest, read, write
from apps_model_runner import PROJECT
from audit_apps_run import audit
from result_docs import write_result_readme


def summarize(folder, output=None):
    result = read(folder / "result.json")
    if result["status"] != "completed" or not result["comparison_prompt_and_seed_match"]:
        raise ValueError("A completed matched comparison is required")
    previous = read(folder / "previous-state.json")
    runs = [("base", "Qwen gốc", Path(previous["baseline_validation"]["path"])),
            ("sft", "SFT", Path(result["plan"].get("parent_sft_validation", previous["sft_validation"]["path"])))]
    if output is None and result["plan"].get("parent_sft_validation", previous["sft_validation"]["path"]) != previous["sft_validation"]["path"]:
        raise ValueError("Research-parent comparison requires a separate --output directory")
    final = result["rounds"][-1]["arms"]
    runs += [(arm, label, Path(final[arm]["validation"]["path"])) for arm, label in
             (("control", "SFT + GRPO cố định"), ("adaptive", "SFT + GRPO đối kháng"))]
    summary = {"round_folder": str(folder), "plan": result["plan"], "runs": {}, "rounds": [],
               "model_test_evaluation": False, "adversarial_generated_test_exposures": result["adversarial_generated_test_exposures"],
               "limits": "Five Validation source problems, 15 mutated programs, one seed; finite call-based input domains. No official APPS Test evaluation. Matched optimizer-step and generation budgets, not matched total wall-clock or code-sampling compute."}
    metric_rows, problem_rows = [], []
    for key, label, path in runs:
        audit(path)
        record = read(path / "result.json")
        breakdown, problems = Counter(), {}
        for prediction in rows(path / "predictions.jsonl"):
            problems.setdefault(prediction["source_id"], []).append(prediction)
            for sample in prediction["samples"]:
                category = "bug_detected" if sample["bug_detected"] else "correct_expected_no_bug" if sample["correct_expected"] else "wrong_expected" if sample["valid_format"] else "invalid_or_truncated"
                breakdown[category] += 1
        if sum(breakdown.values()) != 60:
            raise ValueError("Incomplete Validation")
        summary["runs"][key] = {"label": label, "result": record, "result_sha256": digest(path / "result.json"), "failure_breakdown": dict(breakdown)}
        m = record["metrics"]
        metric_rows.append([key, label, m["valid_tests"], m["correct_expected"], m["bug_detecting_tests"], m["cases_detected"], 60, 15])
        for source_id, predictions in problems.items():
            samples = [sample for prediction in predictions for sample in prediction["samples"]]
            problem_rows.append([key, source_id, len(samples), sum(sample["valid_format"] for sample in samples),
                                 sum(sample["correct_expected"] for sample in samples), sum(sample["bug_detected"] for sample in samples),
                                 sum(any(sample["bug_detected"] for sample in prediction["samples"]) for prediction in predictions)])
    seen_behaviors = set()
    for stage in result["rounds"]:
        pool = Path(stage["pool"])
        audit_pool(pool)
        reviewed = read(pool / "result.json")
        for arm in stage["arms"].values():
            audit(Path(arm["training"]["path"]))
            audit(Path(arm["validation"]["path"]))
            for key in ("training", "validation"):
                entry = arm[key]
                if digest(Path(entry["path"]) / "result.json") != entry["result_sha256"]:
                    raise ValueError("Recorded run hash changed")
        train = {key: read(Path(arm["training"]["path"]) / "result.json") for key, arm in stage["arms"].items()}
        orders = {key: [event["case_id"] for event in rows(Path(arm["training"]["path"]) / "rollouts.jsonl")]
                  for key, arm in stage["arms"].items()}
        if train["control"]["configuration"] != train["adaptive"]["configuration"] or orders["control"] != orders["adaptive"]:
            raise ValueError("Training configurations or case order differ between arms")
        for key, arm in stage["arms"].items():
            expected_parent = result["plan"]["parent_adapter_sha256"] if stage["round"] == 1 else result["rounds"][stage["round"] - 2]["arms"][key]["training"]["adapter_sha256"]
            if train[key]["parent_adapter_sha256"] != expected_parent or not train[key]["parent_adapter_exact_match"]:
                raise ValueError("Parent lineage mismatch")
        stage["matched_training_configuration_and_case_order"] = True
        accepted = rows(pool / "adversarial-cases.jsonl")
        behaviors = {(row["source_id"], row["gate"]["behavior_sha256"]) for row in accepted}
        summary["rounds"].append({"round": stage["round"], "pool": reviewed, "curricula": stage["curricula"],
                                  "arms": stage["arms"], "matched_training_configuration_and_case_order": True,
                                  "novel_behaviors_vs_prior_rounds": len(behaviors - seen_behaviors),
                                  "accepted_code_examples": accepted[:2]})
        seen_behaviors.update(behaviors)
    summary["distinct_accepted_behaviors_across_rounds"] = len(seen_behaviors)
    output = Path(output) if output else PROJECT / "results/apps-adversarial"
    output.mkdir(exist_ok=True, parents=True)
    write(output / "metrics.json", summary)
    for filename, header, data in (
        ("metrics.csv", ["variant", "label", "valid", "correct_expected", "bug_detecting_tests", "cases_detected", "generated_tests", "cases_total"], metric_rows),
        ("per-problem.csv", ["variant", "apps_source_id", "tests", "valid", "correct_expected", "bug_detecting_tests", "cases_detected"], problem_rows),
    ):
        with (output / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(data)
    maximum = {field: max(entry["result"]["metrics"][field] for entry in summary["runs"].values())
               for field in ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")}
    write_result_readme((output / 'README.md').parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor="white")
    keys = list(summary["runs"])
    labels = ["Qwen gốc", "SFT", "GRPO\ncố định", "GRPO\nđối kháng"]
    for ax, field, title in zip(axes, ("valid_rate", "io_accuracy", "case_detection_at_4"),
                               ("Test đúng định dạng và miền input", "Test có expected đúng", "Code lỗi phát hiện với 4 test"), strict=True):
        values = [summary["runs"][key]["result"]["metrics"][field] * 100 for key in keys]
        bars = ax.bar(labels, values, color=["#9ca49c", "#b5bea4", "#7d9baf", "#47735b"], width=.65)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f'{value:.1f}%', ha="center", fontsize=9,
                    fontweight="bold" if value == max(values) else "normal")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Phiên bản model")
        ax.set_ylabel("Tỷ lệ (%)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("APPS Validation · cùng prompt, seed và ngân sách sinh test", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "validation-comparison.png", dpi=180, facecolor="white")
    fig.savefig(output / "validation-comparison.svg", facecolor="white")
    svg = output / "validation-comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    print("Saved:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summarize(args.folder, args.output)
