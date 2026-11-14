import json
from collections import Counter

from apps_core import all_inputs, load_cases, read, write
from apps_model_runner import PROJECT
from result_docs import write_result_readme


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from matplotlib.ticker import MaxNLocator
    plt.rcParams["font.family"] = "DejaVu Sans"
    output = PROJECT / "results/apps-error-analysis"
    analysis = read(output / "analysis.json")
    variants = tuple(key for key in ("base", "sft", "sft256", "control", "adaptive") if key in analysis["runs"])
    names = {"base": "Base Qwen", "sft": "SFT 64", "sft256": "SFT 256", "control": "Fixed GRPO", "adaptive": "Adversarial GRPO"}
    labels = [names[key] for key in variants]
    categories = [("invalid_input_or_json", "Invalid or truncated input/JSON", "#c9c8c0"),
                  ("wrong_expected", "Wrong expected output", "#c48e6d"),
                  ("correct_expected_no_bug", "Correct, no bug detected", "#93adbb"),
                  ("bug_detected", "Bug-detecting tests", "#47735b")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor="white")
    left = [0] * len(variants)
    for category, label, color in categories:
        values = [analysis["runs"][key]["counts"].get(category, 0) + (analysis["runs"][key]["counts"].get("truncated", 0) if category == "invalid_input_or_json" else 0) for key in variants]
        bars = axes[0].barh(labels, values, left=left, color=color, label=label)
        for bar, value in zip(bars, values, strict=True):
            if value:
                axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(value), ha="center", va="center", fontsize=10)
        left = [a + b for a, b in zip(left, values, strict=True)]
    axes[0].set_xlim(0, 60)
    axes[0].set_xlabel("Generated Validation tests (60 per variant)")
    axes[0].set_ylabel("Model variant")
    axes[0].set_title("Wrong expected outputs are the largest error category", loc="left", fontsize=12)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, -.48), frameon=False, ncol=2, fontsize=9)
    potential = [analysis["runs"][key]["counts"].get("input_only_attack", 0) for key in variants]
    usable = [analysis["runs"][key]["metrics"]["bug_detecting_tests"] for key in variants]
    x = list(range(len(variants)))
    axes[1].bar([i - .18 for i in x], potential, width=.36, color="#a4b2b6", label="Bug-revealing inputs (oracle)*")
    axes[1].bar([i + .18 for i in x], usable, width=.36, color="#47735b", label="Model-generated bug-detecting tests")
    for i, (a, b) in enumerate(zip(potential, usable, strict=True)):
        axes[1].text(i - .18, a + .25, str(a), ha="center", fontsize=10)
        axes[1].text(i + .18, b + .25, str(b), ha="center", fontsize=10)
    axes[1].set_xticks(x, labels=[names[key].replace("GRPO ", "GRPO\n") for key in variants])
    axes[1].set_ylim(0, max(potential) + 3)
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_xlabel("Model variant")
    axes[1].set_ylabel("Tests /60 generated samples")
    axes[1].set_title("Detection opportunities lost to wrong expected outputs", loc="left", fontsize=12)
    axes[1].legend(frameon=False, loc="upper right", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=.3, wspace=.3)
    fig.text(.08, .015, "*Oracle diagnostics do not repair or credit model outputs. Same Validation split, prompt v2 and generation seed 42.", fontsize=9, color="#616a62")
    fig.savefig(output / "failure-breakdown.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    train, manifest = load_cases("train")
    sources = {case["source_id"]: case for case in train}
    types, volumes = Counter(), {}
    for case in sources.values():
        fields = list(case["domain"].values())
        if any(field["type"] == "list" and field.get("alphabet") and all(type(value) is str for value in field["alphabet"]) for field in fields):
            category = "String lists"
        elif any(field["type"] == "list" for field in fields):
            category = "Numeric lists"
        elif any(field["type"] == "str" for field in fields):
            category = "Strings"
        else:
            category = "Integers"
        types[category] += 1
        volumes[str(case["source_id"])] = sum(1 for _ in all_inputs(case))
    data_output = PROJECT / "results/apps-data-profile"
    data_output.mkdir(parents=True, exist_ok=True)
    write(data_output / "profile.json", {"dataset_id": manifest["dataset_id"], "split_counts_from_manifest": manifest["splits"],
                                         "train_source_input_types": dict(types), "train_domain_sizes": volumes,
                                         "test_content_inspected": False, "model_test_evaluation": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor="white")
    splits = ("train", "validation", "test")
    counts = [manifest["splits"][key]["problems"] for key in splits]
    bars = axes[0].bar(["Train", "Validation", "Held-out Test"], counts, color=["#47735b", "#93adbb", "#c9c8c0"])
    for bar, value in zip(bars, counts, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + .3, str(value), ha="center")
    axes[0].set_ylim(0, 20)
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_xlabel("Development splits (within APPS Train)")
    axes[0].set_ylabel("Source problems")
    axes[0].set_title("Disjoint problem-level splits", fontsize=11)
    axes[1].barh(list(types), list(types.values()), color="#93adbb")
    axes[1].set_xlabel("Train problems")
    axes[1].set_ylabel("Input type")
    axes[1].set_title("Train input types", fontsize=11)
    axes[2].bar(list(volumes), list(volumes.values()), color="#a5b395")
    axes[2].set_xlabel("APPS Train problem ID")
    axes[2].set_ylabel("Valid inputs in the finite domain")
    axes[2].set_title("Input domain size by Train problem", fontsize=11)
    axes[2].tick_params(axis="x", labelrotation=90, labelsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(data_output / "dataset-profile.png", dpi=180, facecolor="white")
    plt.close(fig)
    write_result_readme((data_output / 'README.md').parent)

    fig, ax = plt.subplots(figsize=(13, 6.2), facecolor="white")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    def box(x, y, w, h, text, color="#f2f5ee"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", facecolor=color, edgecolor="#98a69b", linewidth=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    def arrow(start, end, text=None):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=13, color="#58685d"))
        if text:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + .13, text, ha="center", fontsize=9)
    box(.25, 4.9, 2.1, .8, "APPS Introductory\n29 function-call problems")
    box(3.1, 4.9, 2.15, .8, "Problem-level split\nTrain / Val / Test")
    box(6, 4.9, 2.1, .8, "SFT 64 on Train\nadapter warm-up")
    box(9, 4.9, 3.2, .8, "Fixed GRPO\n2 rounds × 48 steps")
    box(9, 3.1, 3.2, .8, "Adversarial GRPO\n2 rounds × 48 steps", "#e3eee3")
    box(.25, 2.5, 2.5, 1, "Current policy\ngenerates Train tests")
    box(3.4, 2.5, 2.4, 1, "Base local Qwen\ngenerates code, frozen weights")
    box(6.4, 2.5, 1.8, 1, "Filter faulty code\nagainst current tests")
    box(9, 1.25, 3.2, .8, "Same Validation split\ncorrect expected + bug detection")
    box(4.1, .35, 3.7, .8, "Lock configuration → Test\nnot yet run in this snapshot", "#f5f0e6")
    arrow((2.5, 5.3), (2.95, 5.3))
    arrow((5.4, 5.3), (5.85, 5.3), "Train")
    arrow((8.25, 5.3), (8.85, 5.3))
    arrow((8.1, 4.95), (8.9, 3.8))
    arrow((2.9, 3), (3.25, 3))
    arrow((5.95, 3), (6.25, 3))
    arrow((8.35, 3), (8.85, 3.45))
    ax.plot([12.35, 12.65, 12.65], [5.3, 5.3, 1.65], color="#58685d", linewidth=1.2)
    arrow((12.65, 1.65), (12.35, 1.65))
    arrow((10.6, 2.95), (10.6, 2.2))
    arrow((8.85, 1.65), (7.6, .95))
    ax.plot([8.85, 8.65, 8.65, 1.5], [3.5, 3.5, 4.2, 4.2], color="#58685d", linewidth=1.2)
    arrow((1.5, 4.2), (1.5, 3.65))
    ax.text(4.7, 4.32, "Refresh curriculum between rounds", ha="center", fontsize=9)
    ax.text(.25, 1.1, "References are used only for scoring.\nOut-of-domain inputs, unsupported code\nand truncated outputs are recorded.", fontsize=10, color="#59685d")
    ax.text(.25, 5.95, "APPS pipeline · historical 29-problem snapshot", fontsize=15, fontweight="bold", color="#29372f")
    fig.savefig(data_output / "pipeline.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved error analysis, Train EDA and pipeline figures")


if __name__ == "__main__":
    main()
