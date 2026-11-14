import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from apps_core import canonical, score


def main():
    case = {
        "arguments": ["x"],
        "domain": {"x": {"type": "int", "min": 0, "max": 2}},
        "reference": "def solve(x):\n    return x >= 1\n",
        "buggy_code": "def solve(x):\n    return x > 1\n",
    }
    examples = [
        ("Detects the boundary bug", {"input": {"x": 1}, "expected": True}),
        ("Correct test, misses the bug", {"input": {"x": 2}, "expected": True}),
        ("Wrong expected output", {"input": {"x": 1}, "expected": False}),
    ]
    for label, test in examples:
        result = score(canonical(test), True, case)
        print(label)
        print(canonical({key: result[key] for key in
                         ("valid_format", "correct_expected", "bug_detected")}))


if __name__ == "__main__":
    main()
