import argparse
import csv
import json
from pathlib import Path

from apps_core import digest, read, write
from apps_model_runner import PROJECT
from audit_apps_run import audit
from result_docs import write_result_readme


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(folder, publish=False):
    trial = read(folder / "result.json")
    if trial["status"] != "completed" or not trial["comparison_prompt_and_seed_match"]:
        raise ValueError("Completed controlled budget trial required")
    old_train = Path(trial["plan"]["baseline_training"])
    new_train = Path(trial["training_run"])
    old_val = Path(trial["plan"]["baseline_validation"])
    new_val = Path(trial["validation_run"])
    train_eval = Path(trial["train_evaluation"])
    for path in (old_train, new_train, old_val, new_val, train_eval):
        audit(path)
    old_result, new_result = read(old_train / "result.json"), read(new_train / "result.json")
    historical, b = read(old_val / "result.json"), read(new_val / "result.json")
    paired_path = folder / "paired64-result.json"
    paired = read(paired_path) if paired_path.exists() else None
    control_val = Path(paired["control_validation"]) if paired else old_val
    if paired:
        if paired["status"] != "completed":
            raise ValueError("Same-trajectory control is incomplete")
        audit(control_val)
    a = read(control_val / "result.json")
    if digest(old_train / "training-order.json") != digest(new_train / "training-order.json") or old_result["target_sha256"] != new_result["target_sha256"]:
        raise ValueError("Training data or order mismatch")
    summary = {"trial_folder": str(folder), "trial_result_sha256": digest(folder / "result.json"), "trial": trial,
               "sft64_training": old_result, "sft256_training": new_result,
               "validation": {"sft64_historical": historical, "sft64_control": a, "sft256": b},
               "paired_control": paired, "train_evaluation": read(train_eval / "result.json"),
               "model_test_evaluation": False, "method": "SFT_NOT_RL",
               "limits": "One training seed, five Validation problems. Training budgets differ by design. Final 256-step candidate and same-trajectory 64-step control; checkpoint128 is not evaluated. No checkpoint selection from intermediate Validation." if paired else "One training seed, five Validation problems. Training budgets differ by design. Final 256-step candidate only; no intermediate Validation selection."}
    output = PROJECT / "results/apps-sft-budget"
    output.mkdir(parents=True, exist_ok=True)
    write(output / "metrics.json", summary)
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "steps", "example_exposures", "valid_tests", "correct_expected", "bug_tests", "cases_detected", "generated_tests", "cases_total"])
        records = [("SFT64_historical", 64, 256, historical), ("SFT256", 256, 1024, b)]
        if paired:
            records.insert(1, ("SFT64_same_trajectory", 64, 256, a))
        for label, steps, exposures, validation in records:
            m = validation["metrics"]
            writer.writerow([label, steps, exposures, m["valid_tests"], m["correct_expected"], m["bug_detecting_tests"], m["cases_detected"], m["generated_tests"], m["evaluated_cases"]])
    per_case = []
    by_variant = {"SFT64_historical": rows(old_val / "predictions.jsonl"), "SFT64_control": rows(control_val / "predictions.jsonl"), "SFT256": rows(new_val / "predictions.jsonl")}
    for label, predictions in by_variant.items():
        for row in predictions:
            samples = row["samples"]
            per_case.append([label, row["case_id"], row["source_id"], sum(s["valid_format"] for s in samples), sum(s["correct_expected"] for s in samples), sum(s["bug_detected"] for s in samples)])
    with (output / "per-case.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "case_id", "apps_source_id", "valid_of_4", "correct_expected_of_4", "bug_tests_of_4"])
        writer.writerows(per_case)
    fields = ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")
    maximum = {field: max(historical["metrics"][field], a["metrics"][field], b["metrics"][field]) for field in fields}
    write_result_readme((output / 'README.md').parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#a5b29d", "#47735b"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), facecolor="white")
    for ax, field, title in zip(axes, ("valid_rate", "io_accuracy", "case_detection_at_4"),
                               ("Test hợp lệ", "Expected đúng", "Code lỗi phát hiện với 4 test"), strict=True):
        values = [record["metrics"][field] * 100 for record in (a, b)]
        bars = ax.bar(["SFT 64 bước", "SFT 256 bước"], values, color=colors, width=.55)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f'{value:.1f}%', ha="center", fontweight="bold" if value == max(values) else "normal")
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Ngân sách khởi động SFT")
        ax.set_ylabel("Tỷ lệ (%)")
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Đối chiếu 64 và 256 bước trong cùng lượt học" if paired else "Đối chiếu ngân sách khởi động SFT", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output / "validation-comparison.png", dpi=180, facecolor="white")
    plt.close(fig)
    events = rows(new_train / "sft-steps.jsonl")
    steps = [event["step"] for event in events]
    losses = [event["loss"] for event in events]
    rolling = [sum(losses[max(0, i - 15):i + 1]) / min(16, i + 1) for i in range(len(losses))]
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor="white")
    ax.plot(steps, losses, color="#bec7be", linewidth=.7, label="Loss mỗi bước (4 ví dụ Train)")
    ax.plot(steps, rolling, color="#47735b", linewidth=1.6, label="Trung bình trượt 16 bước")
    ax.axvline(64, color="#9b8567", linestyle="--", linewidth=1.2, label="Ngân sách SFT cũ: 64 bước")
    ax.set_xlabel("Bước cập nhật SFT")
    ax.set_ylabel("NLL trung bình trên ví dụ Train")
    ax.set_title("Quá trình khởi động SFT · không phải loss Validation", loc="left", fontsize=12)
    ax.grid(axis="y", alpha=.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "learning-curve.png", dpi=180, facecolor="white")
    plt.close(fig)
    if publish:
        state_path = PROJECT / "APPS_STATE.json"
        state = read(state_path)
        state["sft_budget_trial_path"] = str(folder)
        state["sft_budget_training"] = {"path": str(new_train), "result_sha256": digest(new_train / "result.json"),
                                         "adapter_path": str(new_train / "adapter"), "adapter_sha256": digest(new_train / "adapter/adapter_model.safetensors")}
        state["sft_budget_validation"] = {"path": str(new_val), "result_sha256": digest(new_val / "result.json")}
        if paired:
            state["sft_budget_control_validation"] = {"path": str(control_val), "result_sha256": digest(control_val / "result.json")}
        write(state_path, state)
    print("Saved:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    summarize(args.folder, args.publish)
