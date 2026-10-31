import itertools
import json
from unittest.mock import patch

from restricted_python import RestrictedCodeError, RestrictedFunction


def check():
    source = "def solve(numbers):\n    largest = 0\n    for i in range(len(numbers) - 1):\n        for j in numbers[i + 1:]:\n            pair = numbers[i] + j\n            if pair > largest:\n                largest = pair\n    return largest\n"
    ordinary = RestrictedFunction(source, ["numbers"])
    clock = itertools.chain([0.0], itertools.repeat(0.156))
    with patch("restricted_python.time.monotonic", side_effect=clock):
        assert ordinary(numbers=[1, 3, 0, 3]) == 6
    assert ordinary.steps < 4000
    timed = RestrictedFunction("def solve(x):\n    return x + 1\n", ["x"])
    with patch("restricted_python.time.monotonic", side_effect=[0.0, 1.01]):
        try:
            timed(x=1)
            raise AssertionError("Wall-time budget was not enforced")
        except RestrictedCodeError as error:
            assert str(error) == "Execution time limit"
    long = RestrictedFunction("def solve(x):\n    total = 0\n    for i in range(128):\n        for j in range(128):\n            total += 1\n    return total\n", ["x"])
    with patch("restricted_python.time.monotonic", return_value=0.0):
        try:
            long(x=1)
            raise AssertionError("Instruction budget was not enforced")
        except RestrictedCodeError as error:
            assert str(error) == "Execution step limit"
            assert long.steps == 4001
    try:
        RestrictedFunction("def solve(x):\n    return x.__class__\n", ["x"])
        raise AssertionError("Unsupported attribute was accepted")
    except RestrictedCodeError:
        pass
    return {"status": "passed", "scheduling_delay_156ms_accepted": True,
            "wall_time_limit_enforced": True, "instruction_limit": 4000,
            "unsupported_attribute_rejected": True}


if __name__ == "__main__":
    print(json.dumps(check()))
