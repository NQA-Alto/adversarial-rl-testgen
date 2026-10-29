import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from apps_core import canonical, digest, read, write
from apps_model_runner import PROJECT
from audit_apps_run import audit
from result_docs import write_result_readme


FIELDS = ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")
LABELS = {"attention-qv-v1": "LoRA 6 khối", "hybrid-qkv-v1": "LoRA 24 khối"}


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(root):
    plan, result = read(root / "plan.json"), read(root / "result.json")
    if result["status"] != "completed" or digest(root / "plan.json") != result["plan_sha256"]:
        raise ValueError("A complete trial with its original plan is required")
    indexed, table, errors, examples, costs = {}, [], [], [], []
    for profile in plan["arms"]:
        arm = result["arms"][profile]
        keys = {(row["stage"], row["generation_seed"]) for row in arm["evaluations"]}
        if len(arm["evaluations"]) != 6 or keys != {(stage, seed) for stage in ("sft", "grpo") for seed in plan["generation_seeds"]}:
            raise ValueError("Missing or duplicated planned evaluation")
        for stage in ("sft", "grpo"):
            folder = Path(arm[stage])
            if digest(folder / "result.json") != arm[stage + "_result_sha256"]:
                raise ValueError("Training result changed")
            audit(folder)
            info = read(folder / "result.json")
            seconds = info["training_seconds"] if stage == "sft" else next(row["train_runtime"] for row in records(folder / "trainer-metrics.jsonl") if "train_runtime" in row)
            costs.append({"profile": profile, "stage": stage, "seconds": seconds, "peak_pytorch_allocated_gib": info["peak_vram_gib"],
                          "trainable_parameters": info["trainable_parameters"], "steps": info["optimizer_steps"]})
        for row in arm["evaluations"]:
            folder = Path(row["path"])
            if digest(folder / "result.json") != row["result_sha256"]:
                raise ValueError("Evaluation result changed")
            audit(folder)
            info = read(folder / "result.json")
            if info["metrics"] != row["metrics"] or info["model_test_evaluation"] or info["split"] != "validation":
                raise ValueError("Invalid evaluation evidence")
            stage, seed = row["stage"], row["generation_seed"]
            indexed[profile, stage, seed] = row["metrics"]
            table.append({"profile": profile, "stage": stage, "generation_seed": seed, **{field: row["metrics"][field] for field in FIELDS}})
            counts, chosen = Counter(), Counter()
            for prediction in records(folder / "predictions.jsonl"):
                for number, sample in enumerate(prediction["samples"], 1):
                    category = "bug_detected" if sample["bug_detected"] else "correct_no_bug" if sample["correct_expected"] else "wrong_expected" if sample["valid_format"] else "invalid_or_truncated"
                    counts[category] += 1
                    if sample["valid_format"]:
                        input_hit = "buggy_error" in sample or canonical(sample["gold_output"]) != canonical(sample.get("buggy_output"))
                        counts["input_only_opportunity"] += input_hit
                        counts["lost_due_to_expected"] += input_hit and not sample["correct_expected"]
                    if category in ("wrong_expected", "bug_detected") and chosen[category] < 1:
                        examples.append({"profile": profile, "stage": stage, "generation_seed": seed, "case_id": prediction["case_id"],
                                         "sample_number": number, "category": category, "raw_output": sample["raw_output"],
                                         "gold_output": sample.get("gold_output"), "buggy_output": sample.get("buggy_output"), "buggy_error": sample.get("buggy_error")})
                        chosen[category] += 1
            if sum(counts[key] for key in ("bug_detected", "correct_no_bug", "wrong_expected", "invalid_or_truncated")) != 60:
                raise ValueError("Error counts do not cover all samples")
            errors.append({"profile": profile, "stage": stage, "generation_seed": seed, **dict(counts)})
    summary = {profile: {stage: {field: {"mean": statistics.mean(indexed[profile, stage, seed][field] for seed in plan["generation_seeds"]),
                                       "sample_sd": statistics.stdev(indexed[profile, stage, seed][field] for seed in plan["generation_seeds"])} for field in FIELDS}
                         for stage in ("sft", "grpo")} for profile in plan["arms"]}
    a, b = plan["arms"]
    control, candidate = summary[a]["grpo"], summary[b]["grpo"]
    eligible = candidate["correct_expected"]["mean"] >= control["correct_expected"]["mean"] + 3 and all(candidate[field]["mean"] >= control[field]["mean"] for field in ("valid_tests", "bug_detecting_tests", "cases_detected"))
    paired = [{"stage": stage, "generation_seed": seed, **{field: indexed[b, stage, seed][field] - indexed[a, stage, seed][field] for field in FIELDS}}
              for stage in ("sft", "grpo") for seed in plan["generation_seeds"]]
    output = PROJECT / "results/apps-lora-coverage"
    output.mkdir(parents=True, exist_ok=True)
    write(output / "metrics.json", {"root": str(root), "trial_result_sha256": digest(root / "result.json"), "plan": plan, "summary": summary,
                                    "runs": result["arms"], "hybrid_minus_narrow": paired, "eligible_for_adaptive_followup": eligible,
                                    "training_seeds": 1, "sampling_seeds": 3, "model_test_evaluation": False})
    write(output / "coverage.json", read(root / "coverage.json"))
    write(output / "error-examples.json", {"selection": "First wrong-expected and first bug-detecting sample in saved case/sample order per evaluation, when present; illustrative only.", "examples": examples})
    for name, rows in (("metrics.csv", table), ("costs.csv", costs), ("errors.csv", errors), ("paired-deltas.csv", paired)):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    write_result_readme((output / 'README.md').parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor="white")
    labels = [LABELS[p] + "\n" + stage.upper() for p in plan["arms"] for stage in ("sft", "grpo")]
    for ax, field, title, denominator in zip(axes.flat, FIELDS, ("Test hợp lệ", "Test có expected đúng", "Test bắt được lỗi", "Code phát hiện với tối đa 4 test"), (60, 60, 60, 15), strict=True):
        values = [summary[p][stage][field] for p in plan["arms"] for stage in ("sft", "grpo")]
        means = [item["mean"] for item in values]
        bars = ax.bar(range(4), means, yerr=[item["sample_sd"] for item in values], capsize=4, color=("#b5b7a9", "#869481", "#7c9b9a", "#38654e"))
        for bar, mean in zip(bars, means, strict=True):
            ax.annotate(f'{mean:.2f}', (bar.get_x() + bar.get_width()/2, mean), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
        ax.set(xticks=range(4), xticklabels=labels, title=title, ylabel=f'Số lượng /{denominator}', ylim=(0, denominator))
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180, facecolor="white")
    plt.close(fig)
    print(json.dumps({"output":str(output), "eligible_for_adaptive_followup":eligible, "summary":summary}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
