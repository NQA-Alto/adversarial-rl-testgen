import hashlib
import json
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ASSETS = Path(r"D:\DA\TTNT_De19_Assets")
NAMES = {
    "p01-nonnegative": ("Nonnegative number", "Return true when x ≥ 0, including x = 0."),
    "p03-closed-interval": ("Closed interval", "Return true when −2 ≤ x ≤ 3, including both endpoints."),
    "p08-series-sum": ("Sum from 1 to n", "Sum integers from 1 through n. Return 0 when n = 0."),
    "p10-gcd": ("Greatest common divisor", "Find the greatest common divisor of nonnegative integers a and b, with gcd(0, 0) = 0."),
    "p13-count-positive": ("Count positive numbers", "Count list elements greater than 0, including duplicates. Return 0 for an empty list."),
    "p15-sum-positive": ("Sum positive numbers", "Sum all list elements greater than 0, including duplicates. Return 0 for an empty list."),
    "p16-unique-sorted": ("Sort unique values", "Return distinct values in ascending order. Return [] for an empty list."),
    "p17-reverse": ("Reverse a list", "Reverse all elements, preserving length and duplicates."),
    "p21-palindrome": ("Palindrome", "Check whether a string reads the same backwards. Case and whitespace are significant; the empty string is a palindrome."),
    "p22-vowels": ("Count vowels", "Count all occurrences of a, e, i, o and u, ignoring case."),
    "p28-leading-a": ("Leading a characters", "Count consecutive a characters at the start, stopping at the first b. Return 0 for an empty string."),
}
VARIANTS = {"parent": "After SFT", "control": "Fixed GRPO", "adversarial": "Adversarial GRPO"}


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
            catalog["cases"][key] = dict(case, subset=subset, title=title, display_specification=translation)
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
