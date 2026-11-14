import csv
import json
import statistics
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from result_docs import LABELS, NOTES


ROOT = Path(__file__).resolve().parents[1] / "results"
FIELDS = ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")
TITLES = ("Valid tests", "Correct expected outputs", "Bug-detecting tests", "Cases detected @4")
COLORS = ("#9ca9b5", "#427d8c", "#b79868", "#4b7054", "#877199")
LABELS = dict(LABELS, adversarial="Adversarial GRPO", **{"64": "SFT64", "256": "SFT256",
    "grpo96": "SFT64 + GRPO96", "legacy-v1": "Original reward", "equal-v1": "Equal reward"})


def read(folder, name="metrics.json"):
    return json.loads((ROOT / folder / name).read_text(encoding="utf-8"))


def rows(folder, name="metrics.csv"):
    with (ROOT / folder / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(figure, folder, name, svg=False):
    target = ROOT / folder / name
    figure.savefig(target, dpi=160, bbox_inches="tight", facecolor="white")
    if svg:
        vector = target.with_suffix(".svg")
        figure.savefig(vector, bbox_inches="tight", facecolor="white")
        vector.write_text("\n".join(line.rstrip() for line in vector.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(figure)
    print(target.relative_to(ROOT))


def style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", alpha=.16)
    axis.set_axisbelow(True)


def bars(folder, name, summary, title, tests=60, cases=15, note="", svg=False, selected=None):
    figure, axes = plt.subplots(2, 2, figsize=(13, 7.7), layout="constrained")
    names = list(summary)
    labels = [textwrap.fill(LABELS.get(key, key), 23) for key in names]
    for axis, field, heading, maximum in zip(axes.flat, FIELDS, TITLES, (tests, tests, tests, cases), strict=True):
        values = [summary[key][field] for key in names]
        heights = [value["mean"] for value in values]
        errors = [value.get("sample_sd", 0) for value in values]
        patches = axis.barh(labels, heights, xerr=errors, capsize=3, color=COLORS[:len(names)])
        for key, patch, value, error in zip(names, patches, heights, errors, strict=True):
            axis.text(value + error + maximum * .02, patch.get_y() + patch.get_height() / 2,
                      f"{value:.2f}" if error else f"{value:g}", va="center", fontsize=9)
            if key == selected:
                patch.set_edgecolor("#172b37")
                patch.set_linewidth(2)
        axis.set(title=heading, xlabel=f"Count /{maximum}", xlim=(0, maximum * 1.15))
        axis.invert_yaxis()
        style(axis)
    figure.suptitle(title + ("\n" + note if note else ""), fontsize=12)
    save(figure, folder, name, svg)


def aggregate(records, key):
    groups = {}
    for row in records:
        groups.setdefault(key(row), []).append(row)
    result = {}
    for name, group in groups.items():
        result[name] = {}
        for field in FIELDS:
            aliases = {"valid_tests": "valid", "bug_detecting_tests": "bug_tests", "cases_detected": "cases_detected_with_up_to_4_tests"}
            values = [float(row[field] if field in row else row[aliases[field]]) for row in group]
            result[name][field] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else 0}
    return result


def sampling(folder, name="sampling.png", csv_name="metrics.csv"):
    records = rows(folder, csv_name)
    names = list(dict.fromkeys(row["variant"] for row in records))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for axis, field, heading, maximum in zip(axes, ("correct_expected", "cases_detected"),
                                           ("Correct expected outputs", "Cases detected @4"), (60, 15), strict=True):
        for index, name_key in enumerate(names):
            values = sorted((int(row["generation_seed"]), int(row[field])) for row in records if row["variant"] == name_key)
            axis.plot([x for x, y in values], [y for x, y in values], marker="o", label=LABELS.get(name_key, name_key), color=COLORS[index])
        axis.set(title=heading, xlabel="Generation seed", ylabel=f"Count /{maximum}", xticks=[42, 43, 44], ylim=(0, maximum))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.16)
    axes[1].legend(loc="upper left", fontsize=9)
    figure.suptitle(NOTES[folder][0] + "\nValidation · fixed checkpoints, three generation seeds", fontsize=12)
    save(figure, folder, name)


def data_profile():
    folder = "apps-data-profile"
    data = read(folder, "profile.json")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    splits = data["split_counts_from_manifest"]
    axes[0].bar(["Train", "Validation", "Held-out Test"], [splits[key]["problems"] for key in ("train", "validation", "test")], color=COLORS[:3])
    axes[0].set(title="Disjoint problem-level splits", ylabel="Source problems")
    types = data["train_source_input_types"]
    axes[1].barh(list(types), list(types.values()), color=COLORS[1])
    axes[1].set(title="Train input types", xlabel="Problems")
    volumes = data["train_domain_sizes"]
    axes[2].bar(list(volumes), list(volumes.values()), color=COLORS[3])
    axes[2].set(title="Finite input domains", ylabel="Valid inputs", xlabel="APPS Train problem ID")
    axes[2].tick_params(axis="x", rotation=90, labelsize=8)
    figure.suptitle("Historical 29-problem APPS subset · all splits drawn from APPS Train", fontsize=12)
    save(figure, folder, "dataset-profile.png")
    figure, axis = plt.subplots(figsize=(13, 5))
    axis.set(xlim=(0, 13), ylim=(0, 5))
    axis.axis("off")
    boxes = [(1.8, 3.8, "APPS Introductory\n29 function-call problems"), (5.1, 3.8, "Problem-level split\nTrain / Validation / Test"),
             (9, 3.8, "SFT64 warm-up\non Train"), (10.7, 2, "Fixed GRPO\n2 rounds × 48 steps"),
             (6.5, 2, "Adversarial GRPO\n2 rounds × 48 steps"), (2, 2, "Frozen Qwen generates code\nFilter against current tests"),
             (8.6, .4, "Evaluate on the same Validation split\nCorrect expected output AND bug detection")]
    for x, y, label in boxes:
        axis.text(x, y, label, ha="center", va="center", fontsize=10,
                  bbox=dict(boxstyle="round,pad=.7", facecolor="#eef4f4", edgecolor="#7c999e"))
    for start, end in (((3.25,3.8),(3.55,3.8)),((6.7,3.8),(7.7,3.8)),((9.5,3.25),(10.7,2.65)),
                       ((8.5,3.25),(6.5,2.65)),((3.6,2),(4.9,2)),((6.5,1.4),(7.5,.95)),((10.7,1.4),(9.7,.95))):
        axis.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#46616b"))
    axis.text(2,.4,"Historical snapshot:\nheld-out Test not yet evaluated", ha="center", fontsize=9)
    figure.suptitle("APPS adversarial curriculum · initial experiment", fontsize=13)
    save(figure, folder, "pipeline.png")


def error_charts():
    folder = "apps-error-analysis"
    data = read(folder, "analysis.json")["runs"]
    names = list(data)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    left = [0] * len(names)
    categories = (("invalid_input_or_json", "Invalid / truncated", COLORS[0]), ("wrong_expected", "Wrong expected output", "#bd876d"),
                  ("correct_expected_no_bug", "Correct, no bug", COLORS[1]), ("bug_detected", "Bug detected", COLORS[3]))
    for field, label, color in categories:
        values = [data[key]["counts"].get(field, 0) + (data[key]["counts"].get("truncated", 0) if field == "invalid_input_or_json" else 0) for key in names]
        axes[0].barh([LABELS.get(key, key) for key in names], values, left=left, label=label, color=color)
        left = [a + b for a, b in zip(left, values, strict=True)]
    assert left == [60] * len(names)
    axes[0].set(title="Validation output categories", xlabel="Tests /60")
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.15), fontsize=9, ncol=2)
    for index, key in enumerate(names):
        axes[1].barh(index-.18, data[key]["counts"].get("input_only_attack", 0), height=.35, color=COLORS[0], label="Bug-revealing input (oracle)" if index == 0 else None)
        axes[1].barh(index+.18, data[key]["metrics"]["bug_detecting_tests"], height=.35, color=COLORS[3], label="Valid model-generated detection" if index == 0 else None)
    axes[1].set(yticks=range(len(names)), yticklabels=[LABELS.get(key, key) for key in names], xlabel="Tests /60", title="Input potential vs. valid detection")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.15), fontsize=9)
    figure.suptitle("Initial expected-first trial · generation seed 42\nOracle diagnostics do not repair or credit model outputs", fontsize=12)
    save(figure, folder, "failure-breakdown.png")
    folder = "apps-expected-diagnosis"
    records = read(folder, "analysis.json")["counts"]
    figure, axis = plt.subplots(figsize=(12, 5.8), layout="constrained")
    left = [0] * len(records)
    for field, label, color in (("invalid_or_truncated", "Invalid / truncated", COLORS[0]), ("wrong_expected", "Wrong expected output", "#bd876d"),
                               ("correct_without_bug", "Correct, no bug", COLORS[1]), ("bug_detected", "Bug detected", COLORS[3])):
        values = [row[field] for row in records]
        axis.barh([LABELS.get(row["variant"], row["variant"]) for row in records], values, left=left, label=label, color=color)
        for y, value, start in zip(range(len(records)), values, left, strict=True):
            if value:
                axis.text(start+value/2, y, str(value), ha="center", va="center", fontsize=9)
        left = [a+b for a,b in zip(left, values, strict=True)]
    assert left == [180] * len(records)
    axis.set(xlabel="Tests /180 · three generation seeds, one training seed", title="Expected-output errors · input-first checkpoints")
    axis.legend(loc="upper center", bbox_to_anchor=(.5,-.16), ncol=2, fontsize=9)
    save(figure, folder, "expected-errors.png")


def main():
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    for folder in ("apps-sft-grpo", "apps-adversarial", "apps-input-first-adversarial", "apps-reward-ablation", "apps-sft-budget"):
        bars(folder, "validation-comparison.png", aggregate(rows(folder), lambda row: row["variant"]), NOTES[folder][0],
             note="Validation · generation seed 42", svg=folder in ("apps-sft-grpo", "apps-adversarial", "apps-input-first-adversarial"))
    for folder in ("apps-adversarial-review", "apps-input-first-review"):
        sampling(folder)
    sampling("apps-reward-ablation", "sampling-variability.png", "sampling-metrics.csv")
    for folder, key in (("apps-sft256-grpo", lambda row: row["variant"]), ("apps-shared-sft-budget", lambda row: row["sft_steps"]),
                        ("apps-target-order", lambda row: ("Input-first" if row["order"] == "input-first-v1" else "Expected-first") + " · " + row["stage"].upper())):
        bars(folder, "comparison.png", aggregate(rows(folder), key), NOTES[folder][0], note="Validation · mean ± sample SD across generation seeds 42/43/44")
    folder = "apps-expanded-adaptive"
    data = read(folder)
    LABELS["sft"] = "SFT840"
    for split in ("validation", "test"):
        count = data[split + "_cases"]
        bars(folder, split + ".png", data[split], "Expanded APPS study · " + ("held-out Test" if split == "test" else "Validation"),
             tests=count*4, cases=count, note="Mean ± sample SD across generation seeds 42/43/44 · Validation-selected SFT outlined", selected="sft")
    LABELS["sft"] = "SFT64"
    folder = "grpo-seed-check"
    bars(folder, "training-seed-stability.png", aggregate(rows(folder, "by-training-seed.csv"), lambda row: row["arm"]),
         "Internal dataset · GRPO training seed sensitivity", tests=72, cases=18,
         note="Mean ± sample SD across training seeds 42/43/44 · generation seed fixed at 42")
    data_profile()
    error_charts()


if __name__ == "__main__":
    main()
