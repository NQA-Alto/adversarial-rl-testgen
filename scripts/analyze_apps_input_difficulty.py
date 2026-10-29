import csv

from apps_core import all_inputs, canonical, digest, load_cases, valid_input, write
from apps_model_runner import PROJECT
from restricted_python import RestrictedCodeError, RestrictedFunction


def analyze():
    cases, manifest = load_cases("validation")
    rows = []
    for case in cases:
        reference = RestrictedFunction(case["reference"], case["arguments"])
        buggy = RestrictedFunction(case["buggy_code"], case["arguments"])
        total, different = 0, 0
        for inputs in all_inputs(case):
            if not valid_input(inputs, case):
                raise ValueError("Enumerator produced an invalid input")
            total += 1
            gold = canonical(reference(**inputs))
            try:
                different += canonical(buggy(**inputs)) != gold
            except RestrictedCodeError:
                raise
            except (ArithmeticError, IndexError, TypeError, ValueError):
                different += 1
        if not 0 < different < total:
            raise ValueError("Expected a nontrivial curated mutant")
        probability = different / total
        rows.append({"case_id": case["case_id"], "source_id": case["source_id"], "domain_inputs": total,
                     "bug_exposing_inputs": different, "uniform_input_attack_probability": probability,
                     "uniform_oracle_case_detection_at_4": 1 - (1 - probability) ** 4})
    output = PROJECT / "results/apps-input-difficulty"
    output.mkdir(parents=True, exist_ok=True)
    report = {"dataset_id": manifest["dataset_id"], "split": "validation", "split_sha256": manifest["splits"]["validation"]["sha256"],
              "rows": rows, "expected_detected_cases_with_oracle": sum(row["uniform_oracle_case_detection_at_4"] for row in rows),
              "assumptions": "Four independent uniform draws WITH replacement in each declared finite domain; every expected answer supplied by a perfect reference oracle. Analytic diagnostic only, not model-generated tests or measured performance.",
              "limitations": "Oracle supplies information unavailable to the test-generator policy. Not a fair end-to-end baseline, not official APPS, and not a result on Test.",
              "source_sha256": digest(__file__), "model_loaded": False, "model_test_evaluation": False}
    write(output / "analysis.json", report)
    with (output / "per-case.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Độ khó của việc chọn input trong miền hữu hạn", "", "Đây là chẩn đoán bằng chương trình tham chiếu trên Validation, không gọi hay huấn luyện model. Vét cạn tất cả input đã khai báo cho từng mutant để đếm bao nhiêu input có thể làm lộ lỗi.", "",
             "Với p là tỷ lệ input làm lộ lỗi, bốn lần chọn đều ngẫu nhiên độc lập, có hoàn lại, sẽ có xác suất phát hiện code là 1 − (1 − p)^4. Phép tính giả sử đáp án expected luôn do oracle hoàn hảo cung cấp.", "",
             "| Code | Input trong miền | Input làm lộ lỗi | Tỷ lệ input làm lộ lỗi | Xác suất phát hiện với 4 input + oracle |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f'| {row["case_id"]} | {row["domain_inputs"]} | {row["bug_exposing_inputs"]} | {row["uniform_input_attack_probability"]*100:.2f}% | {row["uniform_oracle_case_detection_at_4"]*100:.2f}% |')
    lines += ["", f'Tổng số code kỳ vọng được phát hiện theo giả định trên: **{report["expected_detected_cases_with_oracle"]:.2f}/15**. Đây là kỳ vọng tính bằng công thức, không phải số code của một lượt sinh test thực tế.', "",
              "## Cách dùng chẩn đoán", "", "Nếu nhiều input trong miền đã làm lộ lỗi mà model vẫn có ít test dùng được, cần xem expected sai và input không hợp lệ trước khi chỉ tăng mức đối kháng. Kết quả này không chứng minh riêng nguyên nhân nào, vì model có thể lấy mẫu input khác phân bố đều và bốn completion có thể trùng nhau.", "",
              "Oracle biết lời giải đúng, còn model trong đồ án phải tự dự đoán expected từ đề. Vì vậy không dùng con số này để tuyên bố thuật toán ngẫu nhiên tốt hơn model trong cùng điều kiện, không thay expected do model sinh bằng đáp án oracle để tăng điểm. Đây cũng không phải trần tối ưu của mọi phương pháp chọn input.", "",
              "Miền input nhỏ và mutant AST là giới hạn của dataset đồ án; kết quả không suy rộng sang toàn bộ APPS. Tập Test phát triển chưa được model đánh giá. Chạy lại bằng scripts/analyze_apps_input_difficulty.py."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print({"output": str(output), "validation_codes": len(rows), "finite_inputs_checked": sum(row["domain_inputs"] for row in rows), "oracle_diagnostic_expected_cases": report["expected_detected_cases_with_oracle"], "model_test_evaluation": False})


if __name__ == "__main__":
    analyze()
