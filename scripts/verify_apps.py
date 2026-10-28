import json
import sys
from collections import Counter
from pathlib import Path

from apps_core import DATA, SOURCE, all_inputs, canonical, digest, read, score, write
from prepare_apps import references
from restricted_python import RestrictedCodeError, RestrictedFunction


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    manifest = read(DATA / "manifest.json")
    rows = {row["id"]: row for row in [json.loads(line) for line in (SOURCE / "train.jsonl").read_text(encoding="utf-8").splitlines()]}
    checks, total, known = Counter(), 0, {}
    for split, spec in manifest["splits"].items():
        path = DATA / (split + ".jsonl")
        assert digest(path) == spec["sha256"]
        cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(cases) == spec["cases"]
        for case in cases:
            assert case["split"] == split and case["source_split"] == "train"
            key = case["problem_id"]
            if key in known:
                assert known[key] == split
            known[key] = split
            source = rows[case["source_id"]]
            assert __import__("hashlib").sha256(canonical(source).encode()).hexdigest() == case["source_record_sha256"]
            assert source["question"] == case["specification"]
            io, refs = references(source)
            assert case["reference"] == refs[0]["code"]
            reference = RestrictedFunction(case["reference"], case["arguments"])
            buggy = RestrictedFunction(case["buggy_code"], case["arguments"])
            hits, misses = 0, 0
            for inputs in all_inputs(case):
                gold = reference(**inputs)
                other = RestrictedFunction(refs[1]["code"], refs[1]["arguments"])(**dict(zip(refs[1]["arguments"], [inputs[name] for name in case["arguments"]], strict=True)))
                assert canonical(gold) == canonical(other)
                different = canonical(buggy(**inputs)) != canonical(gold)
                hits += different
                misses += not different
                sample = score(canonical({"input": inputs, "expected": gold}), True, case, reference, buggy)
                assert sample["valid_format"] and sample["correct_expected"] and sample["bug_detected"] == different
                assert sample["reward"] == (1.0 if different else 0.3)
                total += 1
            assert hits and misses
            checks["cases_verified"] += 1
            assert score('{"input":{},"expected":0}', True, case)["reward"] == 0
            assert score('{}', False, case)["reward"] == 0
            assert score('{"input":{},"input":{},"expected":0}', True, case)["reward"] == 0
    for code in ["def solve(x):\n    return __import__('os').system('echo nope')\n", "def solve(x):\n    while True:\n        pass\n", "def solve(x):\n    return x.__class__\n"]:
        try:
            RestrictedFunction(code, ["x"])
        except RestrictedCodeError:
            checks["unsafe_or_unsupported_construct_rejected"] += 1
        else:
            raise AssertionError("Interpreter accepted a prohibited construct")
    for label, code, value, expected in [
        ("rooms", "def solve(rooms):\n    return rooms[0]\n", ["O"], "O"),
        ("arithmetic", "def solve(x):\n    return x * x - 1\n", 4, 15),
        ("slices", "def solve(x):\n    return x[::-1]\n", "abA", "Aba"),
    ]:
        assert RestrictedFunction(code, ["rooms" if label == "rooms" else "x"])(**{"rooms" if label == "rooms" else "x": value}) == expected
    result = {"status": "passed", "dataset_id": manifest["dataset_id"], "checks": dict(checks),
              "exhaustive_case_input_checks": total, "problem_count": len(known),
              "model_loaded": False, "model_test_evaluation": False,
              "note": "Dataset/reference/verifier checks only. No model quality claim."}
    write(DATA / "verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
