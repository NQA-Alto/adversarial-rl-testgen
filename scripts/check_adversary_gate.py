import argparse
import ast
import copy
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from build_adversarial_pool import assess_candidate, behavior_signature, observed
from dataset_tasks import all_inputs, canonical, compile_trusted, problems, SPLIT_GROUPS
from restricted_python import RestrictedCodeError, RestrictedFunction


def run_checks():
    checks = 0
    programs = 0
    for task in problems():
        if task["split_group"] not in SPLIT_GROUPS["train"]:
            continue
        for source in (task["reference"], *task["mutants"]):
            trusted = compile_trusted(source)
            restricted = RestrictedFunction(source, task["domain"])
            for inputs in all_inputs(task["domain"]):
                assert observed(trusted, inputs) == observed(restricted, inputs), (task["problem_id"], inputs)
                checks += 1
            programs += 1
    blocked = [
        ("import os\ndef solve(x):\n    return 0", {"x": 1}),
        ("def solve(x):\n    return open('sample.txt')", {"x": 1}),
        ("def solve(x):\n    return eval('1')", {"x": 1}),
        ("def solve(x):\n    return x.__class__", {"x": 1}),
        ("def solve(x):\n    return solve(x)", {"x": 1}),
        ("def solve(x):\n    while True:\n        pass", {"x": 1}),
        ("def solve(x):\n    return 2 ** 1000000", {"x": 1}),
        ("def solve(x):\n    return list(range(1000000))", {"x": 1}),
        ("def solve(x):\n    return 'a' * 1000000", {"x": 1}),
        ("def solve(x):\n    return [x] * 1000000", {"x": 1}),
        ("def solve(x):\n    x.append(1)\n    return x", {"x": []}),
        ("def solve(x):\n    y = x\n    x += [1]\n    return y", {"x": []}),
        ("def solve(x):\n    return len(v for v in x)", {"x": [1]}),
        ("def solve(x):\n    a = [x] * 128\n    b = [a] * 128\n    return [b] * 128", {"x": 1}),
        ("def solve(x):\n    a = 0\n    for i in range(128):\n        for j in range(128):\n            a += 1\n    return a", {"x": 1}),
    ]
    for source, inputs in blocked:
        try:
            RestrictedFunction(source, inputs)(**inputs)
        except RestrictedCodeError:
            continue
        raise AssertionError("Unsupported or excessive code was not rejected: " + source)

    context = {"task": {"domain": {"x": {"type": "int", "min": -1, "max": 1}}},
               "inputs": [{"x": -1}, {"x": 0}, {"x": 1}],
               "gold": [canonical({"output": False}), canonical({"output": False}), canonical({"output": True})],
               "input_positions": {canonical({"x": x}): x + 1 for x in (-1, 0, 1)},
               "known_signatures": set(),
               "policy_suites": {"fixture/m1": [{"input": {"x": -1}, "expected": False}, {"input": {"x": 1}, "expected": True}]}}
    incorrect = "def solve(x):\n    return x >= 0\n"
    verdict = assess_candidate(incorrect, context, set())
    assert verdict["accepted"] and verdict["wrong_input_count"] == 1
    assert verdict["passes_policy_suites_for_cases"] == ["fixture/m1"]
    assert verdict["witness_for_audit_only"]["input"] == {"x": 0}
    assert assess_candidate("def solve(x):\n    return x > 0", context, set())["reason"] == "correct_on_finite_domain"
    strict = copy.deepcopy(context)
    strict["policy_suites"]["fixture/m1"].append({"input": {"x": 0}, "expected": False})
    assert not assess_candidate(incorrect, strict, set())["accepted"]
    empty = copy.deepcopy(context)
    empty["policy_suites"] = {"fixture/m1": []}
    assert not assess_candidate(incorrect, empty, set())["accepted"]
    assert assess_candidate(incorrect, context, {verdict["behavior_sha256"]})["reason"] == "behavior_duplicate"
    known = copy.deepcopy(context)
    known["known_signatures"].add(verdict["behavior_sha256"])
    assert assess_candidate(incorrect, known, set())["reason"] == "behavior_duplicate"
    assert assess_candidate("def solve(x):\n    return open('sample.txt')", context, set())["reason"] == "restricted_python_rejected"
    crashes = assess_candidate("def solve(x):\n    return x // x > 0", context, set())
    assert not crashes["accepted"]
    crash_context = copy.deepcopy(context)
    crash_context["policy_suites"] = {"fixture/m1": [{"input": {"x": 1}, "expected": True}]}
    crashes = assess_candidate("def solve(x):\n    return x // x > 0", crash_context, set())
    assert crashes["accepted"]
    for name in ("restricted_python.py", "build_adversarial_pool.py"):
        source = Path(__file__).with_name(name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval", "compile"} for node in ast.walk(tree))
    return {"status": "adversary_gate_checks_passed", "synthetic_only": True, "model_loaded": False,
            "train_curated_programs": programs, "trusted_vs_restricted_input_checks": checks,
            "blocked_unsupported_or_excessive_programs": len(blocked), "gate_fixture_checks": 9,
            "scope": "Interpreter agreement on frozen curated Train only plus synthetic gate fixtures. No generated adversary candidates, model quality results or Test evaluation."}


def main():
    parser = argparse.ArgumentParser(description="Check restricted interpreter semantics and adversary acceptance criteria without loading an AI model")
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    args = parser.parse_args()
    result = run_checks()
    folder = args.assets / "evidence" / ("adversary-gate-check-" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    folder.mkdir(parents=True, exist_ok=False)
    result["source_hashes"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in (Path(__file__).name, "restricted_python.py", "build_adversarial_pool.py", "dataset_tasks.py")}
    (folder / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("Saved evidence:", folder)


if __name__ == "__main__":
    main()
