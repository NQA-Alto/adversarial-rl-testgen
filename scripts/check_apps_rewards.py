import math

from apps_core import canonical, score
from apps_rewards import training_reward


def check():
    case = {"arguments": ["x"], "domain": {"x": {"type": "int", "min": 0, "max": 2}},
            "reference": "def solve(x):\n    return x >= 1\n",
            "buggy_code": "def solve(x):\n    return x > 1\n"}
    controls = [
        (canonical({"input": {"x": 1}, "expected": True}), True, (True, True, True), 1.0, 1.0),
        (canonical({"input": {"x": 2}, "expected": True}), True, (True, True, False), .3, 2 / 3),
        (canonical({"input": {"x": 1}, "expected": False}), True, (True, False, False), .1, 1 / 3),
        (canonical({"input": {"x": 3}, "expected": True}), True, (False, False, False), 0, 0),
        (canonical({"input": {"x": 1}, "expected": True}), False, (False, False, False), 0, 0),
        ('{"input":{"x":1},"expected":true,"expected":false}', True, (False, False, False), 0, 0),
        (canonical({"input": {"x": 1}, "expected": 1}), True, (True, False, False), .1, 1 / 3),
    ]
    for raw, terminated, flags, legacy, equal in controls:
        sample = score(raw, terminated, case)
        assert tuple(sample[key] for key in ("valid_format", "correct_expected", "bug_detected")) == flags
        assert training_reward(sample) == sample["reward"] == legacy
        assert math.isclose(training_reward(sample, "equal-v1"), equal, abs_tol=1e-15)
    invalid = {"valid_format": True, "correct_expected": False, "bug_detected": True, "terminated": True}
    try:
        training_reward(invalid, "equal-v1")
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong expected cannot earn attack reward")
    return {"status": "passed", "verifier_integration_controls": len(controls), "inconsistent_attack_rejected": True,
            "legacy_rewards_unchanged": True, "model_test_evaluation": False}


if __name__ == "__main__":
    print(canonical(check()))
