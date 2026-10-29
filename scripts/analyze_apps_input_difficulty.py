import csv

from apps_core import all_inputs, canonical, digest, load_cases, valid_input, write
from apps_model_runner import PROJECT
from restricted_python import RestrictedCodeError, RestrictedFunction
from result_docs import write_result_readme


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
    write_result_readme((output / 'README.md').parent)
    print({"output": str(output), "validation_codes": len(rows), "finite_inputs_checked": sum(row["domain_inputs"] for row in rows), "oracle_diagnostic_expected_cases": report["expected_detected_cases_with_oracle"], "model_test_evaluation": False})


if __name__ == "__main__":
    analyze()
