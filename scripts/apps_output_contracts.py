from apps_core import canonical, domain_text


EXPECTED_KINDS = {
    2667: "boolean", 2703: "integer", 2820: "integer_array", 2831: "integer", 2859: "integer",
    2865: "string", 2899: "integer", 2910: "boolean", 2914: "integer_array", 2948: "integer_array",
    2961: "integer_array", 2962: "integer_array", 2987: "boolean", 2991: "string", 3009: "integer",
    3019: "integer", 3024: "string_array", 2834: "integer_array", 2868: "boolean", 2947: "integer",
    2968: "string", 3014: "string",
}
TYPE_TEXT = {
    "boolean": "a JSON boolean, exactly true or false; never 0, 1, or a quoted string",
    "integer": "a JSON integer, not a quoted number or boolean",
    "string": "a JSON string, enclosed in double quotes",
    "integer_array": "a flat JSON array of integers; an empty array is allowed when required by the specification",
    "string_array": "a flat JSON array of strings; an empty array is allowed when required by the specification",
}


def typed_prompt(case):
    kind = EXPECTED_KINDS[case["source_id"]]
    return [{"role": "system", "content": "Generate ONE test for a Python function. Return only one JSON object with exactly the keys input and expected. No markdown, no explanations. Compute expected from the specification, not from the implementation. Follow the finite input contract and the required JSON output type."},
            {"role": "user", "content": "ORIGINAL TASK\n" + case["specification"]
             + "\n\nIMPLEMENTATION TO TEST (may contain a bug)\n" + case["buggy_code"]
             + "\n\nALLOWED INPUTS FOR THIS EXPERIMENT\n" + domain_text(case)
             + "\nThese finite input ranges override the larger ranges and examples in the original task. Do not copy numbers or characters outside these ranges."
             + "\nThe input object must contain exactly these argument names: " + canonical(case["arguments"])
             + ".\nThe expected field must be " + TYPE_TEXT[kind] + "."
             + "\nChoose a concrete allowed input that exposes an error, and calculate its correct output according to the specification. Return concrete JSON values, never schema descriptions or placeholders."}]
