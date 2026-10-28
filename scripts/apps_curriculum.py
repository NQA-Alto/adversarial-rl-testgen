import json
from collections import Counter
from pathlib import Path

from apps_core import all_inputs, canonical, digest, load_cases, score
from restricted_python import RestrictedCodeError, RestrictedFunction


def training_cases(path=None, expected_sha256=None):
    originals, manifest = load_cases("train")
    if path is None:
        return originals, manifest, {"static_curated": len(originals)}
    path = Path(path)
    if not expected_sha256 or digest(path) != expected_sha256:
        raise ValueError("Curriculum hash mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_id = {case["case_id"]: case for case in originals}
    if len(rows) != len(originals) or len({row["case_id"] for row in rows}) != len(rows) or {row["case_id"] for row in rows} != set(by_id):
        raise ValueError("Curriculum must retain the same Train case slots")
    for row in rows:
        original = by_id[row["case_id"]]
        for key, value in original.items():
            if key != "buggy_code" and canonical(row.get(key)) != canonical(value):
                raise ValueError("Curriculum changed protected task metadata: " + key)
        if row["buggy_code"] != original["buggy_code"]:
            reference = RestrictedFunction(row["reference"], row["arguments"])
            candidate = RestrictedFunction(row["buggy_code"], row["arguments"])
            hits, misses = 0, 0
            for inputs in all_inputs(row):
                checked = score(canonical({"input": inputs, "expected": reference(**inputs)}), True, row, reference, candidate)
                if checked.get("error") == "verifier_limit":
                    raise RestrictedCodeError("Candidate exceeded verifier limits")
                if not checked["correct_expected"]:
                    raise ValueError("Candidate verifier failed")
                hits += checked["bug_detected"]
                misses += not checked["bug_detected"]
            if not hits or not misses:
                raise ValueError("Candidate must be incorrect but nontrivial on the declared domain")
    return rows, manifest, dict(Counter(row.get("curriculum_source", "static_curated") for row in rows))
