from apps_core import all_inputs, load_cases, model_prompt
from apps_output_contracts import EXPECTED_KINDS
from restricted_python import RestrictedFunction


def matches(value, kind):
    if kind == "boolean":
        return type(value) is bool
    if kind == "integer":
        return type(value) is int
    if kind == "string":
        return type(value) is str
    item_type = int if kind == "integer_array" else str
    return type(value) is list and all(type(item) is item_type for item in value)


def main():
    checked, problems = 0, 0
    for split in ("train", "validation"):
        cases, _ = load_cases(split)
        unique = {case["source_id"]: case for case in cases}
        for case in unique.values():
            kind = EXPECTED_KINDS[case["source_id"]]
            reference = RestrictedFunction(case["reference"], case["arguments"])
            for inputs in all_inputs(case):
                assert matches(reference(**inputs), kind)
                checked += 1
            prompt = model_prompt(case, "apps-call-v3")
            assert len(prompt) == 2 and "expected field must be" in prompt[1]["content"]
            old, narrowed = model_prompt(case, "apps-call-v2"), model_prompt(case, "apps-call-v4")
            assert old[1] == narrowed[1] and narrowed[0]["content"].startswith(old[0]["content"])
            problems += 1
    print("Output-interface annotations checked:", problems, "Train/Validation tasks;", checked, "finite-domain reference outputs; Test not inspected")


if __name__ == "__main__":
    main()
