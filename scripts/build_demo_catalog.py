import hashlib
import json
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ASSETS = Path(r"D:\DA\TTNT_De19_Assets")
NAMES = {
    "p01-nonnegative": ("Số không âm", "Trả về true khi x ≥ 0, bao gồm x = 0."),
    "p03-closed-interval": ("Khoảng đóng", "Trả về true khi −2 ≤ x ≤ 3; hai đầu mút đều được tính."),
    "p08-series-sum": ("Tổng từ 1 đến n", "Tính tổng các số nguyên từ 1 đến n, bao gồm n. Khi n = 0, trả về 0."),
    "p10-gcd": ("Ước chung lớn nhất", "Tìm ước chung lớn nhất của hai số nguyên không âm a và b. Quy ước gcd(0, 0) = 0."),
    "p13-count-positive": ("Đếm số dương", "Đếm các phần tử lớn hơn 0 trong danh sách; phần tử lặp vẫn được tính. Danh sách rỗng trả về 0."),
    "p15-sum-positive": ("Tổng số dương", "Cộng tất cả phần tử lớn hơn 0 trong danh sách, kể cả phần tử lặp. Danh sách rỗng trả về 0."),
    "p16-unique-sorted": ("Loại trùng và sắp xếp", "Trả về danh sách các giá trị phân biệt, theo thứ tự tăng dần. Danh sách rỗng trả về []."),
    "p17-reverse": ("Đảo ngược danh sách", "Đảo ngược thứ tự toàn bộ phần tử; giữ nguyên số lượng và các phần tử lặp."),
    "p21-palindrome": ("Chuỗi đối xứng", "Kiểm tra chuỗi có giống chính nó khi đọc ngược hay không. Phân biệt chữ hoa, chữ thường và khoảng trắng; chuỗi rỗng là đối xứng."),
    "p22-vowels": ("Đếm nguyên âm", "Đếm mọi lần xuất hiện của a, e, i, o, u; không phân biệt chữ hoa và chữ thường."),
    "p28-leading-a": ("Dãy a ở đầu chuỗi", "Đếm số ký tự a liên tiếp ở đầu chuỗi, dừng ở ký tự b đầu tiên. Chuỗi rỗng trả về 0."),
}
VARIANTS = {"parent": "Sau SFT", "control": "GRPO cố định", "adversarial": "GRPO đối kháng"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def metrics(records):
    samples = [sample for record in records for sample in record["samples"]]
    total = len(samples)
    result = {"generated_tests": total, "evaluated_cases": len(records),
              "valid_tests": sum(bool(s["valid_format"]) for s in samples),
              "correct_expected": sum(bool(s["correct_expected"]) for s in samples),
              "bug_detecting_tests": sum(bool(s["bug_detected"]) for s in samples),
              "cases_detected_with_up_to_4_tests": sum(any(s["bug_detected"] for s in r["samples"]) for r in records)}
    for rate, count in (("valid_rate", "valid_tests"), ("io_accuracy", "correct_expected"), ("attack_rate", "bug_detecting_tests")):
        result[rate] = result[count] / total
    result["empirical_case_detection_at_4"] = result["cases_detected_with_up_to_4_tests"] / len(records)
    return result


def main():
    latest = read(PROJECT / "LATEST_POST_WARMUP_RL.json")
    warmup = read(PROJECT / "LATEST_WARMUP_BASELINE.json")
    root = ASSETS / "evidence" / ("demo-catalog-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    root.mkdir()
    experiment, sensitivity, pool = (Path(latest[key]) for key in ("primary_experiment", "decoding_sensitivity", "pool_diagnostic"))
    parent_predictions = ASSETS / "evidence/warmup-sft-v2-20261026-100202-498108/validation/predictions.jsonl"
    catalog = {"schema": 1, "created_at": datetime.now().astimezone().isoformat(), "variants": {}, "sources": {},
               "model_path": str(ASSETS / "models/Qwen3.5-2B"), "cases": {}, "predictions": {}, "metrics": {},
               "experiment": latest, "warmup": warmup, "training_seed_replicates": 1, "decoding_seeds": [42, 43, 44],
               "test_set_model_evaluation": False}

    def source(path):
        catalog["sources"][str(path)] = digest(path)
        return path

    for variant, key in (("parent", "warmup_parent"), ("control", "control"), ("adversarial", "adversarial")):
        item = latest["variants"][key]
        adapter = Path(item["path"])
        if digest(adapter / "adapter_model.safetensors") != item["adapter_sha256"]:
            raise ValueError("Adapter hash changed: " + variant)
        catalog["variants"][variant] = {"label": VARIANTS[variant], "adapter_path": str(adapter), "adapter_sha256": item["adapter_sha256"]}
    case_sources = {"validation": ASSETS / "data/internal-v1/validation.jsonl", "pool": pool / "probe-cases.jsonl"}
    for subset, path in case_sources.items():
        for case in rows(source(path)):
            title, translation = NAMES[case["problem_id"]]
            key = subset + "|" + case["case_id"]
            catalog["cases"][key] = dict(case, subset=subset, title=title, specification_vi=translation)
    for subset, seeds in (("validation", (42, 43, 44)), ("pool", (42,))):
        for seed in seeds:
            for variant in VARIANTS:
                if subset == "pool":
                    path = pool / variant / "predictions.jsonl"
                elif seed != 42:
                    path = sensitivity / f"{seed}-{variant}" / "predictions.jsonl"
                else:
                    path = parent_predictions if variant == "parent" else experiment / ("evaluate-" + variant) / "predictions.jsonl"
                records = rows(source(path))
                assert len(records) == (18 if subset == "validation" else 12)
                for record in records:
                    key = subset + "|" + record["case_id"]
                    assert key in catalog["cases"] and len(record["samples"]) == 4
                    catalog["predictions"].setdefault(key, {}).setdefault(str(seed), {})[variant] = dict(record, source_path=str(path))
                value = metrics(records)
                catalog["metrics"].setdefault(subset, {}).setdefault(str(seed), {})[variant] = value
                result_path = source(path.with_name("result.json"))
                expected = read(result_path)
                expected_metrics = expected.get("metrics") or expected["subsets"]["adversarial_pool"]["metrics"]
                for name in ("generated_tests", "valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected_with_up_to_4_tests"):
                    assert value[name] == expected_metrics[name], (path, name)
    for name in ("dataset_tasks.py", "restricted_python.py", "probe_round1_learning.py"):
        source(PROJECT / "scripts" / name)
    source(PROJECT / "LATEST_POST_WARMUP_RL.json")
    for key in catalog["cases"]:
        assert set(catalog["predictions"][key]) == ({"42", "43", "44"} if key.startswith("validation|") else {"42"})
    target = root / "catalog.json"
    write(target, catalog)
    write(PROJECT / "DEMO_CATALOG.json", {"path": str(target), "sha256": digest(target), "cases": len(catalog["cases"]),
          "source_files": len(catalog["sources"]), "status": "verified_saved_records_no_new_model_evaluation"})
    print("Catalog verified:", target)
    print("30 programs; 792 saved outputs; all 4 outputs retained per program/model/seed; no Test evaluation")


if __name__ == "__main__":
    main()
