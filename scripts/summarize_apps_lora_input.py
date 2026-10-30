import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from apps_core import digest, read, write
from apps_lora_input_trial import FIELDS, OUTPUT, summarize
from result_docs import write_result_readme


LABELS = {"attention-qv-v1": "LoRA 6 khối", "hybrid-qkv-v1": "LoRA 24 khối"}


def main(root):
    summarize(root)
    data = read(OUTPUT / "metrics.json")
    result = read(root / "result.json")
    costs = []
    for variant, folder in (("attention-qv-v1", Path(data["plan"]["control_training"])), ("hybrid-qkv-v1", Path(result["training"]))):
        trained = read(folder / "result.json")
        costs.append({"variant": variant, "training_seconds": trained["training_seconds"],
                      "peak_pytorch_allocated_gib": trained["peak_vram_gib"], "trainable_parameters": trained["trainable_parameters"]})
    write(OUTPUT / "costs.json", {"rows": costs, "scope": "Recorded training/probe phase times, excluding model loading; runs occurred in separate sessions. Allocated PyTorch VRAM is not total device memory."})
    grouped = defaultdict(lambda: defaultdict(int))
    probes = [("train", "attention-qv-v1", data["plan"]["control_probe"]), ("train", "hybrid-qkv-v1", result["probe"])]
    if data["summary"]:
        probes += [("validation", "attention-qv-v1", run) for run in data["plan"]["control_validation"]]
        probes += [("validation", "hybrid-qkv-v1", run) for run in result["evaluations"]]
    for split, variant, run in probes:
        info = read(Path(run["path"]) / "result.json")
        expected_hash = data["plan"]["control_adapter_sha256"] if variant == "attention-qv-v1" else digest(Path(result["training"]) / "adapter/adapter_model.safetensors")
        if info["adapter_sha256"] != expected_hash or info["dataset_id"] != data["plan"]["dataset_id"] or info["split"] != split:
            raise ValueError("Evaluation does not match its adapter or dataset")
        for line in (Path(run["path"]) / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            counts = grouped[split, variant, info["seed"], row["source_id"]]
            counts["codes"] += 1
            counts["cases_detected"] += any(sample["bug_detected"] for sample in row["samples"])
            for sample in row["samples"]:
                counts["samples"] += 1
                counts["invalid"] += not sample["valid_format"]
                counts["wrong_expected"] += sample["valid_format"] and not sample["correct_expected"]
                counts["correct_expected"] += sample["correct_expected"]
                counts["bug_tests"] += sample["bug_detected"]
    rows = [{"split": key[0], "variant": key[1], "generation_seed": key[2], "source_id": key[3], **dict(values)} for key, values in sorted(grouped.items())]
    with (OUTPUT / "per-problem.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def comparison(rows, totals, name, title, error_bars=False):
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.8), facecolor="white")
        for ax, field, heading, maximum in zip(axes, FIELDS, ("Test hợp lệ", "Expected đúng", "Test bắt lỗi", "Code phát hiện @4"), totals, strict=True):
            values = [row[field]["mean"] if error_bars else row[field] for row in rows]
            errors = [row[field]["sample_sd"] for row in rows] if error_bars else None
            bars = ax.bar([LABELS[row["variant"]] for row in rows], values, yerr=errors, capsize=4, color=("#9aaa9e", "#38654e"))
            for index, (bar, value) in enumerate(zip(bars, values, strict=True)):
                top = value + errors[index] if error_bars else value
                ax.annotate(f'{value:.2f}' if error_bars else str(value), (bar.get_x() + bar.get_width()/2, top),
                            xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9,
                            fontweight="bold" if value == max(values) else "normal")
            ax.set(title=heading, ylabel=f'Số lượng /{maximum}', xlabel="Phạm vi LoRA", ylim=(0, maximum))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.15)
            ax.set_axisbelow(True)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(OUTPUT / name, dpi=180, facecolor="white")
        plt.close(fig)

    comparison(data["train_rows"], (164, 164, 164, 41), "train-comparison.png", "SFT256 · đánh giá trên Train, seed sinh 42")
    if data["summary"]:
        rows = [{"variant": name, **values} for name, values in data["summary"].items()]
        comparison(rows, (60, 60, 60, 15), "validation-comparison.png", "SFT256 · Validation, trung bình và SD qua ba seed sinh", True)
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
    for variant, folder, color in (("attention-qv-v1", Path(data["plan"]["control_training"]), "#9aaa9e"),
                                    ("hybrid-qkv-v1", Path(result["training"]), "#38654e")):
        steps = [json.loads(line) for line in (folder / "sft-steps.jsonl").read_text(encoding="utf-8").splitlines()]
        loss = [sum(row["loss"] for row in steps[max(0, i-7):i+1]) / len(steps[max(0, i-7):i+1]) for i in range(len(steps))]
        ax.plot([r["step"] for r in steps], loss, label=LABELS[variant], color=color)
    ax.set(title="Loss trên cùng thứ tự minibatch Train", xlabel="Bước cập nhật SFT · 4 ví dụ/bước", ylabel="Cross-entropy · trung bình trượt 8 bước")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "learning-curve.png", dpi=180, facecolor="white")
    plt.close(fig)
    write_result_readme(OUTPUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
