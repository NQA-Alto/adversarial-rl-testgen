import csv
import json
from collections import Counter
from pathlib import Path

from apps_core import canonical, digest, load_cases, read, score, write
from apps_model_runner import PROJECT
from audit_apps_run import audit
from restricted_python import RestrictedFunction


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def analyze():
    state = read(PROJECT / "APPS_STATE.json")
    cases, manifest = load_cases("validation")
    by_id = {case["case_id"]: case for case in cases}
    names = {"base": "baseline_validation", "sft": "sft_validation", "control": "paired_control_validation", "adaptive": "adversarial_validation"}
    report = {"dataset_id": manifest["dataset_id"], "split": "validation", "model_test_evaluation": False,
              "diagnostic_only": "Input-only attack uses oracle expected solely to diagnose lost opportunities; it is not a usable model test or the primary Attack Rate.",
              "runs": {}}
    sample_rows, case_rows = [], []
    for variant, key in names.items():
        folder = Path(state[key]["path"])
        if digest(folder / "result.json") != state[key]["result_sha256"]:
            raise ValueError("Saved run changed")
        audit(folder)
        counts, detected, potential_cases, wrong_examples = Counter(), [], [], []
        signatures = {}
        for row in rows(folder / "predictions.jsonl"):
            case = by_id[row["case_id"]]
            reference = RestrictedFunction(case["reference"], case["arguments"])
            buggy = RestrictedFunction(case["buggy_code"], case["arguments"])
            seen, true_hits, potential_hits = set(), 0, 0
            for index, sample in enumerate(row["samples"], 1):
                input_hit, matches_buggy, duplicate = False, False, False
                if sample["valid_format"]:
                    test = sample["test"]
                    oracle = reference(**test["input"])
                    diagnostic = score(canonical({"input": test["input"], "expected": oracle}), True, case, reference, buggy)
                    if not diagnostic["correct_expected"] or diagnostic.get("error") == "verifier_limit":
                        raise ValueError("Oracle diagnostic failed")
                    input_hit = diagnostic["bug_detected"]
                    input_signature = canonical(test["input"])
                    duplicate = input_signature in seen
                    seen.add(input_signature)
                    matches_buggy = not sample["correct_expected"] and "buggy_output" in sample and canonical(test["expected"]) == canonical(sample["buggy_output"])
                category = "bug_detected" if sample["bug_detected"] else "correct_expected_no_bug" if sample["correct_expected"] else "wrong_expected" if sample["valid_format"] else "truncated" if sample.get("error") == "truncated" else "invalid_input_or_json"
                counts[category] += 1
                counts["input_only_attack"] += input_hit
                counts["attack_lost_due_to_wrong_expected"] += input_hit and not sample["correct_expected"]
                counts["wrong_expected_matches_buggy_output"] += matches_buggy
                counts["duplicate_valid_input_within_case"] += duplicate
                true_hits += sample["bug_detected"]
                potential_hits += input_hit
                sample_rows.append([variant, row["case_id"], index, category, input_hit, duplicate, matches_buggy])
                if sample["valid_format"] and not sample["correct_expected"] and len(wrong_examples) < 8:
                    wrong_examples.append({"case_id": row["case_id"], "title": case["title"], "model_test": sample["test"],
                                           "gold_output": sample["gold_output"], "buggy_output": sample.get("buggy_output"),
                                           "input_would_expose_bug_with_oracle_expected": input_hit,
                                           "model_expected_matches_buggy_output": matches_buggy})
            if true_hits:
                detected.append(row["case_id"])
            if potential_hits:
                potential_cases.append(row["case_id"])
            signatures[row["case_id"]] = [sample["raw_output"] for sample in row["samples"]]
            case_rows.append([variant, row["case_id"], case["source_id"], len(seen), true_hits, potential_hits])
        report["runs"][variant] = {"path": str(folder), "predictions_sha256": digest(folder / "predictions.jsonl"),
                                    "metrics": read(folder / "result.json")["metrics"], "counts": dict(counts),
                                    "detected_case_ids": detected, "input_only_detected_case_ids": potential_cases,
                                    "wrong_expected_examples": wrong_examples, "raw_signatures": signatures}
    control, adaptive = report["runs"]["control"], report["runs"]["adaptive"]
    control_ids, adaptive_ids = set(control["detected_case_ids"]), set(adaptive["detected_case_ids"])
    report["paired_comparison"] = {"both_detect": sorted(control_ids & adaptive_ids), "only_control": sorted(control_ids - adaptive_ids),
                                    "only_adaptive": sorted(adaptive_ids - control_ids), "neither_detect": sorted(set(by_id) - control_ids - adaptive_ids),
                                    "identical_raw_outputs": sum(a == b for identity in by_id for a, b in zip(control["raw_signatures"][identity], adaptive["raw_signatures"][identity], strict=True))}
    for entry in report["runs"].values():
        entry.pop("raw_signatures")
    output = PROJECT / "results/apps-error-analysis"
    output.mkdir(exist_ok=True, parents=True)
    write(output / "analysis.json", report)
    for name, header, data in (("samples.csv", ["variant", "case_id", "sample", "category", "input_only_attack", "duplicate_valid_input", "expected_matches_buggy"], sample_rows),
                               ("per-case.csv", ["variant", "case_id", "apps_source_id", "distinct_valid_inputs", "usable_bug_tests", "input_only_bug_tests"], case_rows)):
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(data)
    lines = ["# Phân tích lỗi APPS", "", "Chấm lại các bản ghi đã có, không gọi model hay cập nhật trọng số. Validation gồm 15 code / 60 mẫu sinh cho mỗi phiên bản. Test chưa được dùng.", "",
             "| Phiên bản | Input/JSON sai hoặc bị cắt | Expected sai | Đúng nhưng chưa bắt lỗi | Test bắt lỗi | Input có khả năng bắt lỗi* | Cơ hội mất do expected sai* |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for variant, entry in report["runs"].items():
        c = entry["counts"]
        lines.append(f'| {variant} | {c.get("invalid_input_or_json", 0) + c.get("truncated", 0)} | {c.get("wrong_expected", 0)} | {c.get("correct_expected_no_bug", 0)} | {c.get("bug_detected", 0)} | {c.get("input_only_attack", 0)} | {c.get("attack_lost_due_to_wrong_expected", 0)} |')
    pair = report["paired_comparison"]
    lines += ["", "*Hai cột cuối là chẩn đoán: giữ input mà model đã sinh và dùng reference để xem input đó có làm lộ lỗi không. Expected do reference cung cấp không được tính thành test đúng của model; các chỉ số chính không thay đổi. Đây là cách xem xét nút thắt input/expected, tương ứng ý tưởng Input Attack Rate ở ATGen §5.3.", "",
              "![Phân tích các nhóm thất bại](failure-breakdown.png)", "", "## Hai nhánh có phát hiện cùng code không?", "",
              f'Cùng phát hiện: {", ".join(pair["both_detect"])}. Chỉ cố định phát hiện: {len(pair["only_control"])}; chỉ đối kháng phát hiện: {len(pair["only_adaptive"])}; cả hai bỏ sót: {len(pair["neither_detect"])}. Có {pair["identical_raw_outputs"]}/60 đầu ra giống hệt nhau.', "",
              "Cùng số code phát hiện chưa có nghĩa là cùng tập code; bảng phân tích ở đây kiểm tra bằng ID. Đầu ra trùng nhau cũng không tự chứng minh adapter không học: tham số cập nhật và chất lượng đầu ra là hai kiểm chứng khác nhau.", "",
              "## Giới hạn", "", "Một seed và năm bài Validation chưa đủ kết luận tổng quát. Các ví dụ expected trùng đầu ra buggy chỉ là dấu hiệu quan sát, không chứng minh model đã sao chép cách tính của code lỗi. Không dùng Test để tìm nguyên nhân hay chọn cấu hình."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(output), "paired_comparison": report["paired_comparison"], "counts": {key: value["counts"] for key, value in report["runs"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    analyze()
