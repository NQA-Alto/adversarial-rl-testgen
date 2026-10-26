import json


THRESHOLDS = (5, 10, 18, 21)


def task_prompt(threshold):
    return (
        "Find one test that exposes the bug in this Python function.\n"
        f"Specification: x is an integer from 0 to 40. Return true if x is at least {threshold}, otherwise false.\n"
        f"Buggy code:\ndef allowed(x):\n    return x > {threshold}\n"
        'Output only JSON: {"input":{"x":INTEGER},"expected":BOOLEAN}. '
        "The expected boolean must follow the specification. No markdown or explanation."
    )


def score_test(raw, threshold):
    result = {"valid_format": False, "correct_expected": False, "bug_detected": False, "reward": 0.0}
    try:
        test = json.loads(raw)
        if type(test) is not dict or set(test) != {"input", "expected"}:
            raise ValueError("Expected exactly input and expected")
        if type(test["input"]) is not dict or set(test["input"]) != {"x"}:
            raise ValueError("Expected exactly input.x")
        x = test["input"]["x"]
        if type(x) is not int or not 0 <= x <= 40 or type(test["expected"]) is not bool:
            raise ValueError("Input domain or expected type is invalid")
        gold = x >= threshold
        buggy = x > threshold
        correct = test["expected"] == gold
        attack = correct and buggy != gold
        result.update(
            valid_format=True, correct_expected=correct, bug_detected=attack,
            test=test, gold_output=gold, buggy_output=buggy,
            reward=round(0.1 + 0.2 * correct + 0.7 * attack, 6),
        )
    except (ValueError, TypeError, KeyError) as error:
        result["error"] = str(error)
    return result


def build_rows(tokenizer):
    rows = []
    for threshold in THRESHOLDS:
        messages = [
            {"role": "system", "content": "Generate correct tests that expose bugs. Output only JSON."},
            {"role": "user", "content": task_prompt(threshold)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        rows.append({"task_id": f"threshold-{threshold}", "threshold": threshold, "prompt": prompt})
    return rows
