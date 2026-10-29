import argparse
import csv
import statistics
from pathlib import Path

from apps_core import digest, read, write
from apps_model_runner import PROJECT
from audit_apps_run import audit
from result_docs import write_sampling_notes


def summarize(folder):
    result = read(folder / "sampling-result.json")
    plan = read(folder / "sampling-plan.json")
    if result["status"] != "completed" or digest(folder / "sampling-plan.json") != result["sampling_plan_sha256"]:
        raise ValueError("Completed sampling probe with original plan required")
    keys = {(row["variant"], row["generation_seed"]) for row in result["runs"]}
    if len(result["runs"]) != 9 or keys != {(variant, seed) for variant in plan["variants"] for seed in plan["generation_seeds"]}:
        raise ValueError("All nine prespecified sampling evaluations are required")
    for row in result["runs"]:
        path = Path(row["path"])
        if digest(path / "result.json") != row["result_sha256"]:
            raise ValueError("Saved sampling evidence changed")
        audit(path)
        if read(path / "result.json")["metrics"] != row["metrics"]:
            raise ValueError("Sampling metrics do not match saved evaluation")
    labels = {"SFT64_parent": "SFT64 trước RL", "legacy-v1": "GRPO reward hiện tại", "equal-v1": "GRPO reward chia đều"}
    fields = ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")
    summaries = {}
    for variant in plan["variants"]:
        values = [row for row in result["runs"] if row["variant"] == variant]
        summaries[variant] = {field: {"mean": statistics.mean(row["metrics"][field] for row in values),
                                     "sample_sd": statistics.stdev(row["metrics"][field] for row in values),
                                     "values_by_seed": {str(row["generation_seed"]): row["metrics"][field] for row in values}} for field in fields}
    paired = []
    indexed = {(row["variant"], row["generation_seed"]): row["metrics"] for row in result["runs"]}
    for seed in plan["generation_seeds"]:
        paired.append({"generation_seed": seed, **{field: indexed[("equal-v1", seed)][field] - indexed[("legacy-v1", seed)][field] for field in fields}})
    summary = {"sampling_result": result, "sampling_plan": plan, "descriptive_summary": summaries,
               "equal_minus_legacy_by_sampling_seed": paired, "training_seeds": 1, "sampling_seeds": 3,
               "model_test_evaluation": False, "statistical_significance_claim": False}
    output = PROJECT / "results/apps-reward-ablation"
    output.mkdir(parents=True, exist_ok=True)
    write(output / "sampling-metrics.json", summary)
    with (output / "sampling-metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "generation_seed", *fields, "generated_tests", "cases_total"])
        for row in result["runs"]:
            m = row["metrics"]
            writer.writerow([row["variant"], row["generation_seed"], *(m[field] for field in fields), m["generated_tests"], m["evaluated_cases"]])
    write_sampling_notes((output / 'SAMPLING.md').parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
    colors = ("#aaa896", "#8da38d", "#38654e")
    for variant, color, marker in zip(plan["variants"], colors, ("o", "s", "^"), strict=True):
        for ax, field, title, denominator in zip(axes, ("correct_expected", "cases_detected"),
                                               ("Test có expected đúng", "Code phát hiện với tối đa 4 test"), (60, 15), strict=True):
            values = [indexed[(variant, seed)][field] for seed in plan["generation_seeds"]]
            ax.plot(plan["generation_seeds"], values, color=color, marker=marker, linewidth=1.5, markersize=6, label=labels[variant])
            ax.set_xticks(plan["generation_seeds"])
            ax.set_xlabel("Seed sinh test · không phải seed huấn luyện")
            ax.set_ylabel(f'Số lượng /{denominator}')
            ax.set_title(title, loc="left", fontsize=11)
            ax.set_ylim(0, denominator)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis="y", alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Biến động lấy mẫu trên cùng checkpoint · Validation", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(output / "sampling-variability.png", dpi=180, facecolor="white")
    plt.close(fig)
    print("Saved sampling analysis:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
