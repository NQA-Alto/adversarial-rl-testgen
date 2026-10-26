import ast
import itertools
import json
import textwrap


def integer(low, high):
    return {"type": "int", "min": low, "max": high}


def sequence(alphabet=(-2, 0, 1, 3), minimum=0, maximum=4):
    return {"type": "list", "alphabet": list(alphabet), "min_length": minimum, "max_length": maximum}


def text(alphabet="abA ", maximum=4):
    return {"type": "str", "alphabet": list(alphabet), "min_length": 0, "max_length": maximum}


def problem(identity, group, spec, domain, reference, mutants, checks):
    return {
        "problem_id": identity, "split_group": group, "specification": spec, "domain": domain,
        "reference": textwrap.dedent(reference).strip() + "\n",
        "mutants": [textwrap.dedent(code).strip() + "\n" for code in mutants],
        "manual_checks": [{"input": inputs, "expected": expected} for inputs, expected in checks],
    }


def problems():
    return [
        problem("p01-nonnegative", "scalar-boundaries", "Return whether integer x is nonnegative, including zero.", {"x": integer(-5, 5)},
            "def solve(x):\n    return x >= 0", [
                "def solve(x):\n    return x > 0", "def solve(x):\n    return x <= 0", "def solve(x):\n    return x >= -1"],
            [({"x": -1}, False), ({"x": 0}, True), ({"x": 2}, True)]),
        problem("p02-positive", "scalar-boundaries", "Return whether integer x is strictly positive. Zero is not positive.", {"x": integer(-5, 5)},
            "def solve(x):\n    return x > 0", [
                "def solve(x):\n    return x >= 0", "def solve(x):\n    return x < 0", "def solve(x):\n    return x > 1"],
            [({"x": -1}, False), ({"x": 0}, False), ({"x": 1}, True)]),
        problem("p03-closed-interval", "scalar-boundaries", "Return whether x belongs to the closed interval [-2, 3], including both endpoints.", {"x": integer(-5, 5)},
            "def solve(x):\n    return -2 <= x <= 3", [
                "def solve(x):\n    return -2 < x <= 3", "def solve(x):\n    return -2 <= x < 3", "def solve(x):\n    return x >= -2 or x <= 3"],
            [({"x": -3}, False), ({"x": -2}, True), ({"x": 3}, True), ({"x": 4}, False)]),
        problem("p04-even", "parity", "Return whether integer x is even. Zero and negative even integers count as even.", {"x": integer(-5, 5)},
            "def solve(x):\n    return x % 2 == 0", [
                "def solve(x):\n    return x % 2 == 1", "def solve(x):\n    return x > 0 and x % 2 == 0", "def solve(x):\n    return x % 2 == 0 and x != 0"],
            [({"x": -2}, True), ({"x": 0}, True), ({"x": 3}, False)]),
        problem("p05-absolute", "absolute-distance", "Return the nonnegative absolute value of integer x.", {"x": integer(-5, 5)},
            "def solve(x):\n    return abs(x)", [
                "def solve(x):\n    return x", "def solve(x):\n    return -x", "def solve(x):\n    return abs(x) + 1"],
            [({"x": -3}, 3), ({"x": 0}, 0), ({"x": 2}, 2)]),
        problem("p06-clamp", "scalar-boundaries", "Clamp integer x to [-2, 3]: return -2 below the interval, 3 above it, and x inside it.", {"x": integer(-5, 5)},
            "def solve(x):\n    return min(3, max(-2, x))", [
                "def solve(x):\n    return min(3, x)", "def solve(x):\n    return max(-2, x)", "def solve(x):\n    return min(2, max(-2, x))"],
            [({"x": -5}, -2), ({"x": 0}, 0), ({"x": 3}, 3), ({"x": 5}, 3)]),
        problem("p07-sign", "scalar-boundaries", "Return -1 for negative x, 0 for zero, and 1 for positive x.", {"x": integer(-5, 5)},
            "def solve(x):\n    return (x > 0) - (x < 0)", [
                "def solve(x):\n    return 1 if x >= 0 else -1", "def solve(x):\n    return (x < 0) - (x > 0)", "def solve(x):\n    return 1 if x > 0 else 0"],
            [({"x": -3}, -1), ({"x": 0}, 0), ({"x": 2}, 1)]),
        problem("p08-series-sum", "series-sum", "Return the sum of all integers from 1 through n, inclusive. Return 0 when n is 0.", {"n": integer(0, 8)},
            "def solve(n):\n    return n * (n + 1) // 2", [
                "def solve(n):\n    return n * (n - 1) // 2", "def solve(n):\n    return n * (n + 1)", "def solve(n):\n    return n * (n + 1) // 2 + 1"],
            [({"n": 0}, 0), ({"n": 1}, 1), ({"n": 4}, 10)]),
        problem("p09-factorial", "factorial", "Return n factorial. In particular, 0 factorial is 1.", {"n": integer(0, 6)},
            "def solve(n):\n    answer = 1\n    for i in range(2, n + 1):\n        answer *= i\n    return answer", [
                "def solve(n):\n    answer = 1\n    for i in range(2, n):\n        answer *= i\n    return answer",
                "def solve(n):\n    answer = 0\n    for i in range(2, n + 1):\n        answer *= i\n    return answer",
                "def solve(n):\n    answer = 1\n    for i in range(2, n + 1):\n        answer += i\n    return answer"],
            [({"n": 0}, 1), ({"n": 1}, 1), ({"n": 4}, 24)]),
        problem("p10-gcd", "gcd", "Return the greatest common divisor of nonnegative integers a and b. Define gcd(0,0)=0.", {"a": integer(0, 6), "b": integer(0, 6)},
            "def solve(a, b):\n    answer = 0\n    for d in range(1, max(a, b) + 1):\n        if a % d == 0 and b % d == 0:\n            answer = d\n    return answer", [
                "def solve(a, b):\n    answer = 0\n    for d in range(1, max(a, b)):\n        if a % d == 0 and b % d == 0:\n            answer = d\n    return answer",
                "def solve(a, b):\n    return min(a, b)",
                "def solve(a, b):\n    answer = 0\n    for d in range(1, max(a, b) + 1):\n        if a % d == 0 or b % d == 0:\n            answer = d\n    return answer"],
            [({"a": 0, "b": 0}, 0), ({"a": 0, "b": 4}, 4), ({"a": 4, "b": 6}, 2), ({"a": 3, "b": 3}, 3)]),
        problem("p11-max-pair", "maximum", "Return the larger of integers a and b. Equal inputs return that same value.", {"a": integer(-3, 3), "b": integer(-3, 3)},
            "def solve(a, b):\n    return max(a, b)", [
                "def solve(a, b):\n    return min(a, b)", "def solve(a, b):\n    return a", "def solve(a, b):\n    return 0 if a == b else max(a, b)"],
            [({"a": -2, "b": -1}, -1), ({"a": 2, "b": 2}, 2), ({"a": 1, "b": 3}, 3)]),
        problem("p12-manhattan", "absolute-distance", "Return the Manhattan distance of point (x,y) from the origin: |x|+|y|.", {"x": integer(-3, 3), "y": integer(-3, 3)},
            "def solve(x, y):\n    return abs(x) + abs(y)", [
                "def solve(x, y):\n    return abs(x + y)", "def solve(x, y):\n    return max(abs(x), abs(y))", "def solve(x, y):\n    return abs(x) - abs(y)"],
            [({"x": 0, "y": 0}, 0), ({"x": -2, "y": 3}, 5), ({"x": 2, "y": 2}, 4)]),
        problem("p13-count-positive", "positive-aggregation", "Count entries strictly greater than zero in values. Count duplicates separately; empty input returns 0.", {"values": sequence()},
            "def solve(values):\n    return sum(1 for value in values if value > 0)", [
                "def solve(values):\n    return sum(1 for value in values if value >= 0)", "def solve(values):\n    return sum(1 for value in values if value < 0)", "def solve(values):\n    return sum(1 for value in set(values) if value > 0)"],
            [({"values": []}, 0), ({"values": [0, 1, 1, -2]}, 2), ({"values": [-2, 0]}, 0)]),
        problem("p14-count-even", "parity", "Count even entries in values, including zero and negative even entries. Count duplicates separately.", {"values": sequence()},
            "def solve(values):\n    return sum(1 for value in values if value % 2 == 0)", [
                "def solve(values):\n    return sum(1 for value in values if value % 2 != 0)", "def solve(values):\n    return sum(1 for value in values if value % 2 == 0 and value != 0)", "def solve(values):\n    return sum(1 for value in set(values) if value % 2 == 0)"],
            [({"values": []}, 0), ({"values": [-2, 0, 1, 3]}, 2), ({"values": [0, 0]}, 2)]),
        problem("p15-sum-positive", "positive-aggregation", "Sum only the strictly positive entries in values. Count duplicates separately; empty input returns 0.", {"values": sequence()},
            "def solve(values):\n    return sum(value for value in values if value > 0)", [
                "def solve(values):\n    return sum(values)", "def solve(values):\n    return sum(1 for value in values if value > 0)", "def solve(values):\n    return sum(value for value in set(values) if value > 0)"],
            [({"values": []}, 0), ({"values": [-2, 0, 1, 3]}, 4), ({"values": [3, 3]}, 6)]),
        problem("p16-unique-sorted", "list-order", "Return all distinct values as a list sorted in ascending numeric order. Empty input returns an empty list.", {"values": sequence()},
            "def solve(values):\n    return sorted(set(values))", [
                "def solve(values):\n    return sorted(values)", "def solve(values):\n    return sorted(set(values), reverse=True)", "def solve(values):\n    return sorted(set(values))[1:]"],
            [({"values": []}, []), ({"values": [3, -2, 3, 0]}, [-2, 0, 3]), ({"values": [1, 1]}, [1])]),
        problem("p17-reverse", "list-order", "Return a new list with all entries in reverse order. Preserve duplicates and every entry.", {"values": sequence()},
            "def solve(values):\n    return values[::-1]", [
                "def solve(values):\n    return values", "def solve(values):\n    return sorted(values, reverse=True)", "def solve(values):\n    return values[:0:-1]"],
            [({"values": []}, []), ({"values": [1]}, [1]), ({"values": [1, 3, 0]}, [0, 3, 1])]),
        problem("p18-prefix-sums", "prefix-sums", "Return the inclusive prefix sums: output[i] is sum of values[0] through values[i]. Empty input returns an empty list.", {"values": sequence()},
            "def solve(values):\n    return [sum(values[:i + 1]) for i in range(len(values))]", [
                "def solve(values):\n    return [sum(values[:i]) for i in range(len(values))]", "def solve(values):\n    return [sum(values) for i in range(len(values))]", "def solve(values):\n    return [sum(values[:i + 1]) for i in range(1, len(values))]"],
            [({"values": []}, []), ({"values": [3]}, [3]), ({"values": [1, 3, -2]}, [1, 4, 2])]),
        problem("p19-first-positive", "first-search", "Return the zero-based index of the first strictly positive entry. If there is none, including empty input, return -1.", {"values": sequence()},
            "def solve(values):\n    for i in range(len(values)):\n        if values[i] > 0:\n            return i\n    return -1", [
                "def solve(values):\n    for i in range(len(values)):\n        if values[i] >= 0:\n            return i\n    return -1",
                "def solve(values):\n    for i in range(len(values) - 1, -1, -1):\n        if values[i] > 0:\n            return i\n    return -1",
                "def solve(values):\n    for i in range(len(values)):\n        if values[i] > 0:\n            return i + 1\n    return -1"],
            [({"values": []}, -1), ({"values": [0, -2]}, -1), ({"values": [0, 1, 3]}, 1)]),
        problem("p20-max-list", "maximum", "Return the largest entry of the nonempty list values. Entries may be negative or zero.", {"values": sequence(minimum=1)},
            "def solve(values):\n    return max(values)", [
                "def solve(values):\n    return max(0, max(values))", "def solve(values):\n    return min(values)", "def solve(values):\n    return values[-1]"],
            [({"values": [-2]}, -2), ({"values": [3, 1, 0]}, 3), ({"values": [0, 1]}, 1)]),
        problem("p21-palindrome", "palindrome", "Return whether text equals its exact character-by-character reverse. Case and spaces matter. The empty string is a palindrome.", {"text": text()},
            "def solve(text):\n    return text == text[::-1]", [
                "def solve(text):\n    return len(text) > 0 and text == text[::-1]", "def solve(text):\n    return text.lower() == text.lower()[::-1]", "def solve(text):\n    return text == text[:0:-1]"],
            [({"text": ""}, True), ({"text": "aba"}, True), ({"text": "aA"}, False), ({"text": "ab"}, False)]),
        problem("p22-vowels", "vowels", "Count vowels a,e,i,o,u without case sensitivity. Count every occurrence, not only distinct vowels.", {"text": text("abEi")},
            "def solve(text):\n    return sum(1 for c in text.lower() if c in 'aeiou')", [
                "def solve(text):\n    return sum(1 for c in text if c in 'aeiou')", "def solve(text):\n    return sum(1 for c in text.lower() if c in 'aeou')", "def solve(text):\n    return sum(1 for c in set(text.lower()) if c in 'aeiou')"],
            [({"text": ""}, 0), ({"text": "E"}, 1), ({"text": "aai"}, 3), ({"text": "b"}, 0)]),
        problem("p23-word-count", "word-count", "Count nonempty words separated by one or more spaces. Leading and trailing spaces do not add words. Empty or all-space text has 0 words.", {"text": text("ab ", maximum=5)},
            "def solve(text):\n    return len(text.split())", [
                "def solve(text):\n    return len(text.split(' '))", "def solve(text):\n    return text.count(' ')", "def solve(text):\n    return len(text.split()) + 1"],
            [({"text": ""}, 0), ({"text": "   "}, 0), ({"text": "a  b"}, 2), ({"text": " a "}, 1)]),
        problem("p24-first-character", "first-search", "Return the zero-based index of the first occurrence of character ch in text. Return -1 when ch is absent, including in empty text.", {"text": text("ab"), "ch": {"type": "str", "enum": ["a", "b", "c"]}},
            "def solve(text, ch):\n    return text.find(ch)", [
                "def solve(text, ch):\n    return text.rfind(ch)", "def solve(text, ch):\n    return text.find(ch) + 1", "def solve(text, ch):\n    return max(0, text.find(ch))"],
            [({"text": "", "ch": "a"}, -1), ({"text": "aba", "ch": "a"}, 0), ({"text": "ab", "ch": "b"}, 1), ({"text": "aa", "ch": "c"}, -1)]),
    ]


SPLIT_GROUPS = {
    "train": {"scalar-boundaries", "parity", "positive-aggregation", "maximum", "factorial"},
    "validation": {"series-sum", "gcd", "list-order", "palindrome", "vowels"},
    "test": {"absolute-distance", "prefix-sums", "first-search", "word-count"},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def compile_trusted(source):
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != "solve":
        raise ValueError("Expected exactly one solve function")
    forbidden = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.While, ast.Global, ast.Nonlocal)
    for node in ast.walk(tree):
        if isinstance(node, forbidden) or isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Unexpected code in curated dataset")
        if isinstance(node, ast.Attribute) and node.attr not in {"lower", "split", "find", "rfind", "count"}:
            raise ValueError("Unexpected method in curated dataset")
    namespace = {"__builtins__": {name: value for name, value in {
        "abs": abs, "min": min, "max": max, "sum": sum, "range": range,
        "len": len, "sorted": sorted, "set": set,
    }.items()}}
    exec(compile(tree, "<curated-dataset>", "exec"), namespace)
    return namespace["solve"]


def values_for(field):
    if "enum" in field:
        return field["enum"]
    if field["type"] == "int":
        return range(field["min"], field["max"] + 1)
    return [
        list(items) if field["type"] == "list" else "".join(items)
        for length in range(field["min_length"], field["max_length"] + 1)
        for items in itertools.product(field["alphabet"], repeat=length)
    ]


def all_inputs(domain):
    names = list(domain)
    for values in itertools.product(*(values_for(domain[name]) for name in names)):
        yield dict(zip(names, values, strict=True))


def input_is_valid(inputs, domain):
    if type(inputs) is not dict or set(inputs) != set(domain):
        return False
    for name, field in domain.items():
        value = inputs[name]
        expected_type = {"int": int, "str": str, "list": list}[field["type"]]
        if type(value) is not expected_type:
            return False
        if "enum" in field:
            if value not in field["enum"]:
                return False
        elif field["type"] == "int":
            if not field["min"] <= value <= field["max"]:
                return False
        else:
            if not field["min_length"] <= len(value) <= field["max_length"]:
                return False
            if field["type"] == "list" and any(type(item) is not int for item in value):
                return False
            if any(item not in field["alphabet"] for item in value):
                return False
    return True


def evaluate_output(raw, case, reference=None, buggy=None):
    result = {"valid_format": False, "correct_expected": False, "bug_detected": False, "reward": 0.0}
    try:
        test = json.loads(raw)
        if type(test) is not dict or set(test) != {"input", "expected"} or not input_is_valid(test["input"], case["domain"]):
            raise ValueError("Invalid schema or input domain")
        result["valid_format"] = True
        reference = reference or compile_trusted(case["reference"])
        buggy = buggy or compile_trusted(case["buggy_code"])
        gold = reference(**test["input"])
        correct = canonical(test["expected"]) == canonical(gold)
        try:
            observed = buggy(**test["input"])
            differs = canonical(observed) != canonical(gold)
            result["buggy_output"] = observed
        except Exception as error:
            differs = True
            result["buggy_error"] = type(error).__name__
        attack = correct and differs
        result.update(test=test, gold_output=gold, correct_expected=correct, bug_detected=attack,
                      reward=round(0.1 + 0.2 * correct + 0.7 * attack, 6))
    except (ValueError, TypeError, KeyError) as error:
        result["error"] = str(error)
    return result


PROMPT_VERSIONS = ("schema-v1", "plain-v2")


def expected_type_name(case):
    if case["problem_id"] in {"p01-nonnegative", "p02-positive", "p03-closed-interval", "p04-even", "p21-palindrome"}:
        return "a JSON boolean: true or false, without quotation marks"
    if case["problem_id"] in {"p16-unique-sorted", "p17-reverse", "p18-prefix-sums"}:
        return "a JSON array of integers, with numeric entries rather than strings"
    return "a JSON integer, without quotation marks"


def readable_domain(domain):
    lines = []
    for name, field in domain.items():
        if "enum" in field:
            rule = "one of these JSON strings: " + ", ".join(canonical(value) for value in field["enum"])
        elif field["type"] == "int":
            rule = f'a JSON integer between {field["min"]} and {field["max"]}, inclusive'
        elif field["type"] == "list":
            rule = (
                f'an actual JSON array of {field["min_length"]} to {field["max_length"]} integers; '
                f'each entry must be chosen from {canonical(field["alphabet"])}. '
                'Repeated entries are allowed. Supply the array itself, never a dictionary describing an array'
            )
        else:
            rule = (
                f'a JSON string of {field["min_length"]} to {field["max_length"]} characters; '
                'each character must be one of ' + ", ".join(canonical(value) for value in field["alphabet"])
            )
        lines.append(f'- "{name}": {rule}.')
    return "\n".join(lines)


def model_prompt(case, version="plain-v2"):
    if version == "plain-v2":
        return [
            {"role": "system", "content": (
                "Generate exactly one test case as a JSON object with two keys: input and expected. "
                "Use concrete input values, never schema descriptions. Compute expected from the specification, "
                "not from the buggy implementation. Return no other text."
            )},
            {"role": "user", "content": (
                f'Specification: {case["specification"]}\n'
                'Arguments and allowed values:\n' + readable_domain(case["domain"]) + '\n'
                f'Buggy Python implementation:\n{case["buggy_code"]}\n'
                'Find one allowed input that makes the buggy implementation disagree with the specification.\n'
                'Output requirements:\n'
                '- "input" must be an object containing exactly these argument keys: '
                + ", ".join(canonical(name) for name in case["domain"]) + '.\n'
                '- "expected" must be ' + expected_type_name(case) + '.\n'
                '- Both fields must contain actual values. Do not copy domain rules or add keys such as '
                'type, alphabet, min_length, max_length, or FUNCTION_ARGUMENTS.\n'
                '- Return only the JSON test, without markdown or explanations.'
            )},
        ]
    if version != "schema-v1":
        raise ValueError(f"Unknown prompt version: {version}")
    domain = canonical(case["domain"])
    return [
        {"role": "system", "content": "Generate correct tests that expose bugs. Output only JSON."},
        {"role": "user", "content": (
            f'Specification: {case["specification"]}\n'
            f'Allowed input domain: {domain}\n'
            f'Buggy Python code:\n{case["buggy_code"]}\n'
            'Find one valid input that exposes the bug and compute its expected output from the specification. '
            'Return exactly {"input":{FUNCTION_ARGUMENTS},"expected":EXPECTED_VALUE}. '
            'Use the function argument names as input keys. No markdown or explanation.'
        )},
    ]
