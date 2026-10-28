from apps_adversary import assess, observed, signature
import tempfile
from pathlib import Path

from apps_core import ASSETS, canonical, digest, load_cases
from apps_curriculum import training_cases
from restricted_python import RestrictedCodeError, RestrictedFunction


def main():
    case = {"arguments": ["x"]}
    inputs = [{"x": i} for i in range(4)]
    reference = RestrictedFunction("def solve(x):\n    return x\n", ["x"])
    context = {"case": case, "inputs": inputs, "gold": [observed(reference, item) for item in inputs],
               "positions": {canonical(item): i for i, item in enumerate(inputs)},
               "known": set(), "suites": {"original-case": [{"input": {"x": 0}, "expected": 0}]}}
    source = "def solve(x):\n    return x if x < 3 else 0\n"
    good = assess(source, context, set())
    assert good["accepted"] and good["wrong_inputs"] == 1
    assert assess(source, context, {good["behavior_sha256"]})["reason"] == "duplicate_behavior"
    assert assess("def solve(x):\n    return x\n", context, set())["reason"] == "correct_on_declared_domain"
    assert assess("def solve(x):\n    return x + 1\n", context, set())["reason"] == "wrong_on_every_input"
    assert assess("def solve(x):\n    return 1 if x == 0 else x\n", context, set())["reason"] == "caught_by_current_policy"
    assert assess("def solve(x):\n    import os\n    return x\n", context, set())["reason"] == "unsupported_code"
    empty = {**context, "suites": {"empty": []}}
    assert assess(source, empty, set())["reason"] == "caught_by_current_policy"
    context["known"].add(good["behavior_sha256"])
    assert assess(source, context, set())["reason"] == "duplicate_behavior"
    try:
        observed(RestrictedFunction("def solve(x):\n    return 999999999 * 999999999\n", ["x"]), {"x": 1})
    except RestrictedCodeError:
        pass
    else:
        raise AssertionError("Verifier bounds must not become program failures")
    cases, manifest, counts = training_cases()
    assert len(cases) == 41 and counts == {"static_curated": 41}
    with tempfile.TemporaryDirectory(prefix="apps-curriculum-check-", dir=ASSETS / "temp") as temporary:
        path = Path(temporary) / "cases.jsonl"
        def save(data):
            path.write_text("".join(canonical(row) + "\n" for row in data), encoding="utf-8", newline="\n")
            return digest(path)
        expected = save(cases)
        assert len(training_cases(path, expected)[0]) == 41
        for changed, checksum in ((cases, "wrong-hash"),
                                  ([{**cases[0], "reference": "def solve(x): return 0\n"}, *cases[1:]], None),
                                  ([{**cases[0], "case_id": "validation-case"}, *cases[1:]], None),
                                  ([{**cases[0], "buggy_code": "def solve(distance_to_pump, mpg, fuel_left):\n    return None\n"}, *cases[1:]], None)):
            actual = save(changed)
            try:
                training_cases(path, checksum or actual)
            except ValueError:
                pass
            else:
                raise AssertionError("Tampered curriculum accepted")
    try:
        load_cases("test")
    except ValueError:
        pass
    else:
        raise AssertionError("Test must stay reserved")
    print("APPS adversary gate and curriculum: 15 positive/negative checks passed; Test stays reserved")


if __name__ == "__main__":
    main()
