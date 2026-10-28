import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from apps_core import canonical, digest, read, write
from apps_model_runner import PROJECT
from audit_apps_run import audit
from apps_rewards import PROFILES, training_reward


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reward_sensitivity(rollouts):
    levels = Counter()
    differences = []
    for row in rollouts:
        legacy = [training_reward(sample, "legacy-v1") for sample in row["samples"]]
        equal = [training_reward(sample, "equal-v1") for sample in row["samples"]]
        levels[str(len(set(legacy)))] += 1
        def normalized(values):
            mean, std = statistics.mean(values), statistics.stdev(values)
            return [(value - mean) / (std + 1e-4) for value in values]
        differences.append(max(abs(a - b) for a, b in zip(normalized(legacy), normalized(equal), strict=True)))
    return {"groups_by_distinct_reward_levels": dict(levels),
            "groups_with_three_or_more_levels": sum(count for key, count in levels.items() if int(key) >= 3),
            "mean_counterfactual_standardized_reward_difference": statistics.mean(differences),
            "max_counterfactual_standardized_reward_difference": max(differences),
            "interpretation": "Diagnostic only: recompute both profiles on the SAME stored completions, float64 sample standard deviation and epsilon=1e-4. This is not a second model run or exact logged trainer advantage. With only two reward levels, positive affine rescaling preserves relative standardized scores apart from epsilon effects."}


def summarize(folder):
    trial = read(folder / "result.json")
    if trial["status"] != "completed" or not trial["controlled_evaluation_verified"] or not trial["training_case_order_match"]:
        raise ValueError("Completed controlled reward trial required")
    labels = {"legacy-v1": "GRPO reward hiện tại", "equal-v1": "GRPO reward chia đều"}
    parent_path = Path(trial["plan"]["parent_validation"])
    audit(parent_path)
    parent = read(parent_path / "result.json")
    records = {"SFT64_parent": parent}
    predictions = {"SFT64_parent": rows(parent_path / "predictions.jsonl")}
    for profile, arm in trial["arms"].items():
        for key in ("training", "train", "validation"):
            path = Path(arm[key])
            if digest(path / "result.json") != arm[key + "_result_sha256"]:
                raise ValueError("Evidence changed: " + key)
            audit(path)
        trained = read(Path(arm["training"]) / "result.json")
        if (trained["reward_profile"] != profile or trained["reward_weights"] != PROFILES[profile]
                or trained["parent_adapter_sha256"] != trial["plan"]["parent_adapter_sha256"]
                or trained["train_sha256"] != trial["plan"]["train_sha256"]
                or trained["optimizer_steps"] != trial["plan"]["steps_per_arm"]):
            raise ValueError("Training arm does not match prespecified profile, parent or data")
        record = read(Path(arm["validation"]) / "result.json")
        if canonical(record["metrics"]) != canonical(arm["validation_metrics"]):
            raise ValueError("Trial metrics differ from evaluation")
        records[profile] = record
        predictions[profile] = rows(Path(arm["validation"]) / "predictions.jsonl")
    output = PROJECT / "results/apps-reward-ablation"
    output.mkdir(parents=True, exist_ok=True)
    fields = ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")
    maximum = {field: max(record["metrics"][field] for record in records.values()) for field in fields}
    columns = ["variant", "generated_tests", *fields, "cases_total"]
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for key, record in records.items():
            m = record["metrics"]
            writer.writerow([key, m["generated_tests"], *(m[field] for field in fields), m["evaluated_cases"]])
    per_case = []
    for key, samples in predictions.items():
        for row in samples:
            per_case.append([key, row["case_id"], row["source_id"],
                             sum(s["valid_format"] for s in row["samples"]),
                             sum(s["correct_expected"] for s in row["samples"]),
                             sum(s["bug_detected"] for s in row["samples"])])
    with (output / "per-case.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "case_id", "apps_source_id", "valid_of_4", "correct_of_4", "bug_tests_of_4"])
        writer.writerows(per_case)
    by_variant = {key: {row["case_id"]: row for row in values} for key, values in predictions.items()}
    case_sets = {key: {row["case_id"] for row in values if any(s["bug_detected"] for s in row["samples"])} for key, values in predictions.items()}
    transitions = {"equal_only_detected": sorted(case_sets["equal-v1"] - case_sets["legacy-v1"]),
                   "legacy_only_detected": sorted(case_sets["legacy-v1"] - case_sets["equal-v1"]),
                   "detected_by_both": sorted(case_sets["equal-v1"] & case_sets["legacy-v1"])}
    same_raw = sum(a["raw_output"] == b["raw_output"]
                   for identity, arow in by_variant["legacy-v1"].items()
                   for a, b in zip(arow["samples"], by_variant["equal-v1"][identity]["samples"], strict=True))
    summaries = {}
    for key, arm in trial["arms"].items():
        training = read(Path(arm["training"]) / "result.json")
        rollouts = rows(Path(arm["training"]) / "rollouts.jsonl")
        samples = [sample for row in rollouts for sample in row["samples"]]
        summaries[key] = {"optimizer_steps": training["optimizer_steps"], "generated_tests_during_training": len(samples),
                          "nonzero_gradient_steps": training["nonzero_gradient_steps"], "policy_signal_groups": training["policy_signal_groups"],
                          "changed_adapter_tensors": len(training["changed_adapter_tensors"]), "saved_adapter_exact_match": training["adapter"]["parameters_match"],
                          "peak_vram_gib": training["peak_vram_gib"], "parent_adapter_sha256": training["parent_adapter_sha256"],
                          "train_probe_metrics": arm["train_metrics"], "validation_metrics": arm["validation_metrics"]}
        summaries[key]["reward_sensitivity"] = reward_sensitivity(rollouts)
        logs = rows(Path(arm["training"]) / "trainer-metrics.jsonl")
        summaries[key]["trainer_summary"] = next((row for row in reversed(logs) if "train_runtime" in row), {})
    summary = {"trial_folder": str(folder), "trial_result_sha256": digest(folder / "result.json"), "trial": trial,
               "runs": summaries, "case_transitions": transitions, "identical_raw_outputs_of_60": same_raw,
               "model_test_evaluation": False, "active_checkpoint_promoted": False}
    write(output / "metrics.json", summary)
    examples = []
    for key in ("equal-v1", "legacy-v1"):
        for category in ("bug_detected", "wrong_expected", "invalid"):
            found = next(((row, sample) for row in predictions[key] for sample in row["samples"]
                          if (sample["bug_detected"] if category == "bug_detected" else
                              sample["valid_format"] and not sample["correct_expected"] if category == "wrong_expected" else
                              not sample["valid_format"])), None)
            if found:
                row, sample = found
                examples.append({"variant": key, "category": category, "case_id": row["case_id"],
                                 "source_id": row["source_id"], "sample": sample})
    write(output / "error-examples.json", {"selection": "First stored sample per category and arm, not representative prevalence. Full predictions stay in evidence.", "examples": examples})
    lines = ["# APPS: kiểm chứng tỷ trọng phần thưởng GRPO", "",
             "Hai lượt mới cùng xuất phát từ SFT64 lịch sử, cùng 41 code Train, seed 42, prompt v2, LoRA rank 4, optimizer mới và 96 bước/nhánh. Thứ tự code và ngân sách rollout được kiểm tra khớp. Khác biệt chủ đích duy nhất là tỷ trọng reward; không thay curriculum đối kháng trong phép thử này.", "",
             "| Thành phần | Hiện tại · legacy-v1 | Chia đều · equal-v1 |", "|---|---:|---:|",
             "| Định dạng JSON và input đúng miền | 0,1 | 1/3 |", "| Expected đúng | 0,2 | 1/3 |", "| Expected đúng và bắt lỗi | 0,7 | 1/3 |", "",
             "Reward là tổng các thành phần thỏa mãn. Vì vậy bốn mức ở nhánh hiện tại là 0 / 0,1 / 0,3 / 1; nhánh chia đều là 0 / 1/3 / 2/3 / 1. Attack luôn bị chặn khi expected sai. Đầu ra bị cắt và test không hợp lệ nhận 0.", "",
             "[ATGen §4.1 và §5.1](https://arxiv.org/html/2510.14635v1) dùng tổng ba thành phần với trọng số bằng nhau. Phép thử này tham khảo tỷ trọng đó và chuẩn hóa tổng tối đa về 1; tiêu chí định dạng vẫn là JSON/miền input của đồ án, khác thẻ thinking của bài báo. Đây không phải tái lập đầy đủ reward và benchmark gốc.", "",
             "## Validation", "", "| Mốc | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện @4 /15 |", "|---|---:|---:|---:|---:|"]
    for key, record in records.items():
        m = record["metrics"]
        values = [f'**{m[field]}**' if m[field] == maximum[field] else str(m[field]) for field in fields]
        lines.append("| " + labels.get(key, "SFT64 trước RL") + " | " + " | ".join(values) + " |")
    lines += ["", "Chữ đậm là giá trị lớn nhất từng cột, kể cả khi hòa. Mỗi code sinh bốn test, cùng giới hạn 96 token và seed theo case. Mẫu sai/bị cắt vẫn ở mẫu số; expected không được sửa bằng oracle. Chỉ đánh giá checkpoint cuối sau khi cả hai nhánh đã học xong.", "",
              "## Kiểm chứng việc học", "", "| Nhánh | Bước GRPO | Gradient khác 0 | Nhóm có tín hiệu | Tensor LoRA đổi | Đỉnh VRAM PyTorch (GiB) |", "|---|---:|---:|---:|---:|---:|"]
    for key, item in summaries.items():
        lines.append(f'| {labels[key]} | {item["optimizer_steps"]} | {item["nonzero_gradient_steps"]} | {item["policy_signal_groups"]} | {item["changed_adapter_tensors"]} | {item["peak_vram_gib"]} |')
    lines += ["", "Trọng số nền được đóng băng; chỉ LoRA có gradient. Cả hai adapter đã lưu đều khớp tensor sau học. Chấm lại toàn bộ rollout theo đúng profile, đồng thời kiểm tra các cờ chấm chính bằng bộ chấm giữ nguyên. Có cập nhật tham số chưa tự chứng minh hiệu quả tìm lỗi.", "",
              "## Đọc kết quả", "",
              f'Ứng viên chia đều qua tiêu chí đặt trước để cân nhắc so sánh đối kháng tiếp theo: **{trial["eligible_for_adversarial_followup"]}**. Tiêu chí yêu cầu expected đúng tăng ít nhất 3/60 so với đối chứng mới và không giảm hợp lệ, test bắt lỗi hay code phát hiện so với đối chứng hoặc parent. Không tự thay checkpoint đang dùng.', "",
              f'Hai nhánh có {same_raw}/60 đầu ra trùng nguyên văn. Code chỉ nhánh chia đều phát hiện: {", ".join(transitions["equal_only_detected"]) or "không có"}. Code chỉ nhánh hiện tại phát hiện: {", ".join(transitions["legacy_only_detected"]) or "không có"}.', "",
              "Kết quả trên Train chỉ đo mức khớp tập học, tách riêng trong metrics.json. Reward trung bình của hai profile dùng thang thành phần khác nhau nên không được dùng để kết luận nhánh nào tốt hơn; hình quá trình học bên dưới dùng cùng cờ expected/attack đã kiểm chứng.", "",
              "![Kết quả Validation](validation-comparison.png)", "", "![Diễn biến học trên Train](training-components.png)", "",
              "Một seed train, một seed sinh test và năm bài Validation; các code trong cùng bài có tương quan. Chưa đủ kết luận ổn định hay ý nghĩa thống kê. CUDA nondeterminism chưa loại trừ. Đây là ablation trên dữ liệu cố định, không đo lợi ích của adaptive curriculum. Cả Test phát triển và Test chính thức tiếp tục được giữ lại.", "",
              "## Tín hiệu nhóm và giới hạn khi đổi reward", "",
              "GRPO hiện chuẩn hóa reward trong từng nhóm bốn mẫu. Nếu một nhóm chỉ có hai mức kết quả, đổi khoảng cách dương giữa hai mức gần như không đổi điểm chuẩn hóa tương đối; epsilon và tính toán số có thể tạo sai khác nhỏ. Nhóm từ ba mức trở lên mới làm tỷ trọng giữa các mục tiêu thể hiện rõ hơn. Đây là nhận xét về cơ chế, không tự chứng minh nguyên nhân chất lượng model.", "",
              "metrics.json lưu chẩn đoán đổi profile trên chính các mẫu đã lưu của mỗi nhánh. Chẩn đoán dùng số thực float64, độ lệch chuẩn mẫu và epsilon 1e-4; không gọi đó là advantage được lấy trực tiếp từ trainer hoặc kết quả của một lượt train khác. error-examples.json giữ mẫu đầu tiên theo mỗi loại thành công, sai expected và không hợp lệ để chuẩn bị giải thích; tần suất phải lấy từ toàn bộ kết quả.", "",
              "## Chạy lại và bằng chứng", "",
              "Kế hoạch có timestamp, nguồn đóng băng, hash parent, raw rollout, gradient và adapter nằm trong trial_folder ở metrics.json. Chạy scripts/apps_reward_trial.py để tạo một trial mới; script không công bố đè lên các mốc chính. Sau khi hoàn tất, dùng scripts/summarize_apps_rewards.py với đường dẫn trial để dựng lại bảng/hình."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#c4c5b1", "#95aa97", "#38654e"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), facecolor="white")
    xlabels = ["Trước RL", "Reward\nhiện tại", "Reward\nchia đều"]
    for ax, field, title in zip(axes, ("io_accuracy", "attack_rate", "case_detection_at_4"),
                               ("Expected đúng", "Test thực sự bắt lỗi", "Code phát hiện với tối đa 4 test"), strict=True):
        values = [record["metrics"][field] * 100 for record in records.values()]
        bars = ax.bar(xlabels, values, color=colors, width=.6)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f'{value:.1f}%', ha="center", fontweight="bold" if value == max(values) else "normal")
        ax.set_ylim(0, max(40, max(values) * 1.3))
        ax.set_xlabel("Mốc / tỷ trọng reward")
        ax.set_ylabel("Tỷ lệ (%)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Kiểm chứng tỷ trọng reward · cùng parent và 96 bước GRPO", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(output / "validation-comparison.png", dpi=180, facecolor="white")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="white")
    for profile, color in zip(labels, colors[1:], strict=True):
        rollouts = rows(Path(trial["arms"][profile]["training"]) / "rollouts.jsonl")
        for ax, field, title in zip(axes, ("correct_expected", "bug_detected"), ("Expected đúng khi học", "Test bắt lỗi khi học"), strict=True):
            steps = [row["optimizer_step_before"] + 1 for row in rollouts]
            values = [sum(sample[field] for sample in row["samples"]) / 4 * 100 for row in rollouts]
            moving = [sum(values[max(0, i - 15):i + 1]) / min(16, i + 1) for i in range(len(values))]
            ax.plot(steps, moving, color=color, label=labels[profile], linewidth=1.7)
            ax.set_title(title, loc="left", fontsize=11)
            ax.set_xlabel("Bước cập nhật GRPO")
            ax.set_ylabel("Tỷ lệ (%) · trung bình trượt 16 nhóm")
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "training-components.png", dpi=180, facecolor="white")
    plt.close(fig)
    print("Saved:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
