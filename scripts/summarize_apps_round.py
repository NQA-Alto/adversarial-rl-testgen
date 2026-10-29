import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from apps_core import canonical, digest, read, write
from result_docs import write_result_readme


def scoring_ast(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in ("valid_input", "score", "metrics")}


def summarize(folder):
    result = read(folder / "result.json")
    if result["status"] != "completed" or not result["comparison_prompt_and_seed_match"]:
        raise ValueError("A completed, controlled round is required")
    project = Path(__file__).resolve().parents[1]
    output = project / "results/apps-sft-grpo"
    output.mkdir(parents=True, exist_ok=True)
    names = [("baseline_validation", "Qwen gốc"), ("sft_validation", "Sau SFT"), ("pilot_validation", "Sau SFT + GRPO")]
    runs = {key: read(Path(path) / "result.json") for key, path in result["runs"].items()}
    for key, _ in names:
        review = read(Path(result["runs"][key]) / "review.json")
        if review["status"] != "passed" or review["predictions_rescored"] != 60:
            raise ValueError("Validation audit missing")
    breakdown = {}
    per_problem = []
    for key, label in names:
        counts = Counter()
        problems = defaultdict(list)
        predictions = Path(result["runs"][key]) / "predictions.jsonl"
        for row in [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]:
            problems[row["source_id"]].append(row)
            for sample in row["samples"]:
                if sample["bug_detected"]:
                    category = "bug_detected"
                elif sample["correct_expected"]:
                    category = "correct_expected_no_bug"
                elif sample["valid_format"]:
                    category = "wrong_expected"
                elif sample.get("error") == "truncated":
                    category = "truncated"
                elif sample.get("error") == "schema_or_domain":
                    parsed = json.loads(sample["raw_output"])
                    category = "input_contract" if type(parsed) is dict and set(parsed) == {"input", "expected"} else "json_schema"
                else:
                    category = "invalid_json"
                counts[category] += 1
        if sum(counts.values()) != 60:
            raise ValueError("Incomplete failure breakdown")
        breakdown[key] = dict(counts)
        for identity, rows in problems.items():
            samples = [sample for row in rows for sample in row["samples"]]
            per_problem.append([label, identity, len(samples), sum(s["valid_format"] for s in samples),
                                sum(s["correct_expected"] for s in samples), sum(s["bug_detected"] for s in samples),
                                sum(any(s["bug_detected"] for s in row["samples"]) for row in rows), len(rows)])
    previous = read(folder / "previous-state.json")
    old = Path(previous["baseline_validation"]["path"])
    base = Path(result["runs"]["baseline_validation"])
    old_result = read(old / "result.json")
    new_result = runs["baseline_validation"]
    control = {key: old_result[key] == new_result[key] for key in ("dataset_id", "split_sha256", "seed", "generation", "tests_per_case", "model_path")}
    control["same_scoring_ast"] = scoring_ast(old / "source/apps_core.py") == scoring_ast(base / "source/apps_core.py")
    control["same_interpreter"] = digest(old / "source/restricted_python.py") == digest(base / "source/restricted_python.py")
    if not all(control.values()):
        raise ValueError("Prompt-only comparison changed other controlled conditions")
    summary = {"round_folder": str(folder), "plan": result["plan"], "runs": runs, "failure_breakdown": breakdown,
               "diagnosis": read(folder / "diagnosis.json"), "prompt_comparison_controls": control,
               "prompt_only_comparison": {"apps-call-v1": old_result["metrics"], "apps-call-v2": new_result["metrics"]},
               "model_test_evaluation": False,
               "limits": "Five validation problems / 15 mutated functions, one training seed and generation seed. Static GRPO, no APPS adversary curriculum yet."}
    write(output / "metrics.json", summary)
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        fields = ("generated_tests", "valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected", "evaluated_cases")
        writer.writerow(["variant", *fields])
        for key, _ in names:
            writer.writerow([key, *(runs[key]["metrics"][field] for field in fields)])
    with (output / "per-problem.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "apps_source_id", "tests", "valid", "correct_expected", "bug_detecting_tests", "cases_detected", "cases_total"])
        writer.writerows(per_problem)
    write_result_readme((output / 'README.md').parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="white")
    colors = ["#89928b", "#719c80", "#456e92"]
    for axis, field, title, denominator in (
        (axes[0], "valid_rate", "Test hợp lệ", 60),
        (axes[1], "case_detection_at_4", "Code lỗi được phát hiện với 4 test", 15),
    ):
        values = [runs[key]["metrics"][field] * 100 for key, _ in names]
        bars = axis.bar([label for _, label in names], values, color=colors, width=.6)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 2, f'{value:.1f}%', ha="center", fontsize=10)
        axis.set_title(title, loc="left", fontsize=12)
        axis.set_ylabel("Tỷ lệ (%)")
        axis.set_xlabel("Phiên bản model")
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("APPS Validation · cùng prompt v2, seed 42 và ngân sách sinh test", fontsize=12)
    figure.tight_layout()
    figure.savefig(output / "validation-comparison.png", dpi=180, facecolor="white")
    figure.savefig(output / "validation-comparison.svg", facecolor="white")
    svg = output / "validation-comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(figure)
    print(canonical({"output": str(output), "metrics": result["metrics"], "prompt_comparison_controls": control}))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
