import json
from collections import Counter, defaultdict
from pathlib import Path

from apps_core import digest, read, write
from apps_model_runner import PROJECT


def analyze():
    state = read(PROJECT / "APPS_STATE.json")
    training = Path(state["sft_training"]["path"])
    targets = [json.loads(line) for line in (training / "targets.jsonl").read_text(encoding="utf-8").splitlines()]
    first = dict(Counter(next(iter(json.loads(row["target"]))) for row in targets))
    if digest(training / "targets.jsonl") != read(training / "result.json")["target_sha256"]:
        raise ValueError("Historical supervised targets changed")
    sampling = read(PROJECT / "results/apps-adversarial-review/metrics.json")
    summaries = {}
    for variant in ("base", "sft", "control", "adaptive"):
        counts = defaultdict(lambda: Counter(samples=0, valid=0, correct_expected=0, bug_detected=0))
        for row in sampling["runs"]:
            if row["variant"] != variant:
                continue
            folder = Path(row["path"])
            if digest(folder / "result.json") != row["result_sha256"]:
                raise ValueError("Saved evaluation changed")
            for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
                for sample in json.loads(line)["samples"]:
                    try:
                        value = json.loads(sample["raw_output"])
                        key = next(iter(value), "empty_object") if type(value) is dict else "not_object"
                    except (ValueError, TypeError):
                        key = "parse_error"
                    counts[key].update(samples=1, valid=int(sample["valid_format"]), correct_expected=int(sample["correct_expected"]), bug_detected=int(sample["bug_detected"]))
        summaries[variant] = {key: dict(value) for key, value in counts.items()}
    output = PROJECT / "results/apps-serialization-diagnosis"
    output.mkdir(parents=True, exist_ok=True)
    report = {"target_path": str(training / "targets.jsonl"), "target_sha256": digest(training / "targets.jsonl"),
              "supervised_targets": len(targets), "first_field_in_targets": first, "target_example": targets[0]["target"],
              "validation_first_field": summaries, "evaluations_per_variant": 3, "model_test_evaluation": False,
              "interpretation": "Observed order shift, not a causal demonstration that ordering alone explains low accuracy. Proposed experiment changes only top-level training-target field order; strict JSON scoring is order invariant."}
    write(output / "analysis.json", report)
    lines = ["# Chẩn đoán thứ tự trường trong nhãn SFT", "", f'{len(targets)} nhãn SFT lịch sử đều bắt đầu bằng expected. Nguyên nhân là hàm canonical sắp xếp tên khóa khi tuần tự hóa. JSON vẫn hợp lệ và bộ chấm không phụ thuộc thứ tự khóa.', "", "Ví dụ nhãn gốc:", "", "```json", targets[0]["target"], "```", "",
             "## Đầu ra đã lưu trên ba seed sinh test", "", "Mỗi phiên bản có 180 completion trên Validation. Bảng phân loại theo khóa đầu tiên trong JSON đọc được; parse_error là không đọc được JSON.", "",
             "| Phiên bản | Trường đầu | Số mẫu | Input/JSON hợp lệ | Expected đúng | Test bắt lỗi |", "|---|---|---:|---:|---:|---:|"]
    for variant, groups in summaries.items():
        for key, c in groups.items():
            lines.append(f'| {variant} | {key} | {c["samples"]} | {c["valid"]} | {c["correct_expected"]} | {c["bug_detected"]} |')
    lines += ["", "## Ý nghĩa và kiểm chứng cần làm", "", "Model tự hồi quy sinh token từ trái sang phải. Nếu nhãn đặt expected trước input, nó được dạy trả đáp án trước khi hoàn tất input cụ thể trong văn bản. Dữ liệu cho thấy thứ tự đầu ra sau SFT thay đổi rõ rệt, nhưng chưa chứng minh đổi thứ tự sẽ cải thiện accuracy: còn nhiều yếu tố học khác cùng thay đổi.", "",
              "Phép thử tiếp theo cần giữ nguyên profile LoRA, target input/expected, thứ tự bài, learning rate và ngân sách; chỉ đổi thứ tự hai khóa ngoài cùng sang input rồi expected. So riêng SFT và phần GRPO thêm vào; giữ parser và metric cũ. Không sắp xếp lại đầu ra model khi chấm hoặc dùng oracle sửa expected.", "",
              "Đây là vấn đề biểu diễn nhãn được phát hiện khi rà mã, chưa phải bằng chứng toàn bộ kết quả cũ bị chấm sai. Test chưa được model đánh giá. Kết quả ba seed có tương quan và không phải nhiều seed train."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print({"output": str(output), "target_first_fields": first, "saved_output_first_fields": summaries})


if __name__ == "__main__":
    analyze()
