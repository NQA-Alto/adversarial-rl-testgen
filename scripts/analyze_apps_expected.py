import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from apps_core import canonical, digest, read, write
from apps_model_runner import PROJECT
from audit_apps_run import audit
from result_docs import write_result_readme


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def analyze():
    order = read(PROJECT / "results/apps-target-order/metrics.json")
    compared = read(PROJECT / "results/apps-input-first-review/metrics.json")
    runs = [dict(row, variant="grpo96") for row in order["runs"]["input-first-v1"]["evaluations"] if row["stage"] == "grpo"]
    runs += compared["runs"]
    totals, by_problem, examples = defaultdict(Counter), defaultdict(Counter), []
    for row in runs:
        folder = Path(row["path"])
        if digest(folder / "result.json") != row["result_sha256"]:
            raise ValueError("Evaluation changed")
        audit(folder)
        info = read(folder / "result.json")
        if info["split"] != "validation" or info["model_test_evaluation"]:
            raise ValueError("Validation only")
        for prediction in records(folder / "predictions.jsonl"):
            for index, sample in enumerate(prediction["samples"]):
                c = Counter(samples=1)
                if not sample["valid_format"]:
                    c["invalid_or_truncated"] += 1
                elif sample["correct_expected"]:
                    c["correct_expected"] += 1
                    c["bug_detected"] += int(sample["bug_detected"])
                    c["correct_without_bug"] += int(not sample["bug_detected"])
                else:
                    c["wrong_expected"] += 1
                    gold, expected = sample["gold_output"], sample["test"]["expected"]
                    c["wrong_python_type"] += int(type(gold) is not type(expected))
                    same_buggy = "buggy_output" in sample and canonical(sample["buggy_output"]) == canonical(expected)
                    c["wrong_equals_buggy_output"] += int(same_buggy)
                    opportunity = "buggy_error" in sample or ("buggy_output" in sample and canonical(sample["buggy_output"]) != canonical(gold))
                    c["input_exposes_bug_but_expected_wrong"] += int(opportunity)
                    examples.append({"variant": row["variant"], "generation_seed": row["generation_seed"],
                                     "case_id": prediction["case_id"], "sample_index": index,
                                     "input": sample["test"]["input"], "model_expected": expected, "reference_expected": gold,
                                     "equals_buggy_output": same_buggy, "input_exposes_bug": opportunity,
                                     "run": str(folder)})
                totals[row["variant"]].update(c)
                by_problem[row["variant"], prediction["source_id"]].update(c)
    output = PROJECT / "results/apps-expected-diagnosis"
    output.mkdir(parents=True, exist_ok=True)
    fields = ("samples", "invalid_or_truncated", "correct_expected", "wrong_expected", "wrong_python_type",
              "wrong_equals_buggy_output", "input_exposes_bug_but_expected_wrong", "bug_detected", "correct_without_bug")
    table = [{"variant": v, **{f: c[f] for f in fields}} for v, c in totals.items()]
    write(output / "analysis.json", {"counts": table, "runs": runs, "model_test_evaluation": False,
                                    "note": "Wrong-type and equals-buggy counts overlap. Matching buggy output is observational, not proof the model copied code. Reference is used for offline scoring only; no model output is repaired."})
    write(output / "wrong-expected-examples.json", examples)
    problem_rows = [{"variant": v, "source_id": p, **{f: c[f] for f in fields}} for (v, p), c in by_problem.items()]
    for name, rows in (("counts.csv", table), ("per-problem.csv", problem_rows)):
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_result_readme((output / 'README.md').parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"base": "Qwen gốc", "sft": "SFT64 · input trước", "grpo96": "SFT64 + GRPO96", "control": "GRPO cố định · 2 vòng", "adaptive": "GRPO đối kháng · 2 vòng"}
    variants = ("base", "sft", "grpo96", "control", "adaptive")
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor="white")
    left = [0] * len(variants)
    for field, label, color in (("invalid_or_truncated", "JSON/input không hợp lệ", "#d3c9b9"),
                                ("wrong_expected", "Input hợp lệ, expected sai", "#b56a58"),
                                ("correct_without_bug", "Expected đúng, chưa bắt lỗi", "#8ca9af"),
                                ("bug_detected", "Expected đúng và bắt lỗi", "#437052")):
        values = [totals[v][field] for v in variants]
        ax.barh([labels[v] for v in variants], values, left=left, label=label, color=color, height=.65)
        for y, n, start in zip(range(len(variants)), values, left, strict=True):
            ax.text(start + n / 2, y, str(n), ha="center", va="center", color="white" if field in ("wrong_expected", "bug_detected") else "#26362e", fontsize=10)
        left = [a + b for a, b in zip(left, values, strict=True)]
    if left != [180] * len(variants):
        raise ValueError("Error partition must cover every generated output")
    ax.set(xlim=(0, 180), xlabel="Số test /180 · ba seed sinh, một seed train", ylabel="Checkpoint cố định", title="Expected sai còn làm mất nhiều cơ hội bắt lỗi")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "expected-errors.png", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(table, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    analyze()
