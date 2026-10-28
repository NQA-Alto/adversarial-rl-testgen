import ast
import copy
import hashlib
import itertools
import json
from pathlib import Path

from restricted_python import RestrictedCodeError, RestrictedFunction


ASSETS = Path(r"D:\DA\TTNT_De19_Assets")
REVISION = "21e74ddf8de1a21436da12e3e653065c5213e9d1"
SOURCE = ASSETS / "data/apps-source" / REVISION
DATA = ASSETS / "data/apps-introductory-v2"
PROMPT_VERSION = "apps-call-v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def normalized(source, name):
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            functions += [child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == name]
    if len(functions) != 1:
        raise ValueError("Expected a single entry point")
    fn = copy.deepcopy(functions[0])
    if fn.args.args and fn.args.args[0].arg == "self":
        fn.args.args.pop(0)
    fn.name, fn.returns = "solve", None
    for arg in fn.args.args:
        arg.annotation = None
    code = ast.unparse(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))) + "\n"
    arguments = [arg.arg for arg in fn.args.args]
    RestrictedFunction(code, arguments)
    return code, arguments


def official_matches(observed, expected):
    return canonical(observed) == canonical(expected) or (
        type(expected) is list and len(expected) == 1 and canonical(observed) == canonical(expected[0]))


def field_values(field):
    if "enum" in field:
        return field["enum"]
    if field["type"] == "int":
        return range(field["min"], field["max"] + 1)
    result = []
    for length in range(field["min_length"], field["max_length"] + 1):
        for items in itertools.product(field["alphabet"], repeat=length):
            result.append("".join(items) if field["type"] == "str" else list(items))
    return result


def valid_input(inputs, case):
    domain = case["domain"]
    if type(inputs) is not dict or set(inputs) != set(domain):
        return False
    for name, field in domain.items():
        value = inputs[name]
        if type(value) is not {"int": int, "str": str, "list": list}[field["type"]]:
            return False
        if "enum" in field:
            if canonical(value) not in {canonical(item) for item in field["enum"]}:
                return False
        elif field["type"] == "int":
            if not field["min"] <= value <= field["max"]:
                return False
        else:
            if not field["min_length"] <= len(value) <= field["max_length"]:
                return False
            if any(canonical(item) not in {canonical(a) for a in field["alphabet"]} for item in value):
                return False
    for constraint in case.get("constraints", []):
        kind, *names = constraint.split(":")
        if kind == "lt" and not inputs[names[0]] < inputs[names[1]]:
            return False
        if kind == "same_length" and len(inputs[names[0]]) != len(inputs[names[1]]):
            return False
        if kind == "not_all_zero" and not any(inputs[names[0]]):
            return False
    return True


def all_inputs(case):
    names = list(case["domain"])
    for values in itertools.product(*(field_values(field) for field in case["domain"].values())):
        inputs = dict(zip(names, values, strict=True))
        if valid_input(inputs, case):
            yield inputs


def domain_text(case):
    lines = []
    for name, field in case["domain"].items():
        if "enum" in field:
            rule = "one of " + canonical(field["enum"])
        elif field["type"] == "int":
            rule = f'integer from {field["min"]} through {field["max"]}, inclusive'
        else:
            kind = "string" if field["type"] == "str" else "array"
            rule = f'{kind}, length {field["min_length"]} through {field["max_length"]}; entries from {canonical(field["alphabet"])}'
        lines.append(f'"{name}": {rule}.')
    for constraint in case.get("constraints", []):
        kind, *names = constraint.split(":")
        if kind == "lt":
            lines.append(f'{names[0]} must be strictly less than {names[1]}.')
        elif kind == "same_length":
            lines.append(f'{names[0]} and {names[1]} must have the same length.')
        elif kind == "not_all_zero":
            lines.append(f'{names[0]} must contain at least one nonzero entry.')
    return "\n".join(lines)


def model_prompt(case, version=PROMPT_VERSION):
    if version == "apps-call-v2":
        return [{"role": "system", "content": "You generate executable test data for Python functions. Follow the input contract exactly. Return ONE JSON object, no markdown, no explanation. The only top-level keys are input and expected. Compute expected from the problem specification; the implementation may contain a bug."},
                {"role": "user", "content": "Output format example for an unrelated addition function: {\"input\":{\"a\":2,\"b\":3},\"expected\":5}. Do not reuse its field names or numbers.\n\nPROBLEM SPECIFICATION\n" + case["specification"]
                 + "\n\nBUGGY FUNCTION TO TEST\n" + case["buggy_code"]
                 + "\nINPUT CONTRACT FOR THIS EXPERIMENT (overrides larger input ranges or examples above)\n" + domain_text(case)
                 + "\nThe input object must contain exactly these argument names: " + canonical(case["arguments"])
                 + ". Every array element and every character must be from its stated allowed set. Do not use other numbers or letters from the original examples.\n\nReturn exactly {\"input\":{...concrete allowed argument values...},\"expected\":...correct concrete JSON value...}. Fill in the values; do not output placeholders. Try an allowed input that makes the buggy function disagree with the specification."}]
    if version != "apps-call-v1":
        raise ValueError("Unknown APPS prompt version")
    return [{"role": "system", "content": "Generate exactly one test that exposes a bug. Return only a JSON object with keys input and expected. Compute expected from the specification, not from the buggy implementation."},
            {"role": "user", "content": "Original APPS problem:\n" + case["specification"]
             + "\n\nThis experiment evaluates only the following valid bounded subset of inputs:\n" + domain_text(case)
             + "\n\nPython implementation to test (entry point adapted to solve):\n" + case["buggy_code"]
             + "\nReturn one concrete allowed input and its correct expected output. Input must be an object with exactly these keys: "
             + ", ".join(case["domain"]) + ". Expected must be a concrete JSON value. No explanations or markdown."}]


def score(raw, terminated, case, reference=None, buggy=None):
    result = {"raw_output": raw, "terminated": bool(terminated), "valid_format": False,
              "correct_expected": False, "bug_detected": False, "reward": 0.0}
    if not terminated:
        result["error"] = "truncated"
        return result
    def unique(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("Duplicate JSON key")
            output[key] = value
        return output
    try:
        test = json.loads(raw, object_pairs_hook=unique, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if type(test) is not dict or set(test) != {"input", "expected"} or not valid_input(test["input"], case):
            raise ValueError("schema_or_domain")
        result.update(valid_format=True, test=test, reward=0.1)
        reference = reference or RestrictedFunction(case["reference"], case["arguments"])
        buggy = buggy or RestrictedFunction(case["buggy_code"], case["arguments"])
        gold = reference(**test["input"])
        result["gold_output"] = gold
        correct = canonical(gold) == canonical(test["expected"])
        try:
            observed = buggy(**test["input"])
            result["buggy_output"] = observed
            different = canonical(observed) != canonical(gold)
        except RestrictedCodeError:
            raise
        except (ArithmeticError, IndexError, TypeError, ValueError) as error:
            different = True
            result["buggy_error"] = type(error).__name__
        result.update(correct_expected=correct, bug_detected=correct and different,
                      reward=1.0 if correct and different else 0.3 if correct else 0.1)
    except RestrictedCodeError as error:
        result.update(valid_format=False, correct_expected=False, bug_detected=False, reward=0.0,
                      error="verifier_limit", detail=str(error))
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        result["error"] = str(error)
    return result


def load_cases(split, allow_test=False):
    if split == "test" and not allow_test:
        raise ValueError("Development Test is reserved for the final evaluation")
    manifest = read(DATA / "manifest.json")
    path = DATA / (split + ".jsonl")
    if digest(path) != manifest["splits"][split]["sha256"]:
        raise ValueError("Dataset hash changed")
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return cases, manifest


def metrics(predictions):
    samples = [sample for row in predictions for sample in row["samples"]]
    n = len(samples)
    return {"evaluated_cases": len(predictions), "generated_tests": n,
            "valid_tests": sum(s["valid_format"] for s in samples),
            "correct_expected": sum(s["correct_expected"] for s in samples),
            "bug_detecting_tests": sum(s["bug_detected"] for s in samples),
            "cases_detected": sum(any(s["bug_detected"] for s in row["samples"]) for row in predictions),
            "valid_rate": sum(s["valid_format"] for s in samples) / n if n else 0,
            "io_accuracy": sum(s["correct_expected"] for s in samples) / n if n else 0,
            "attack_rate": sum(s["bug_detected"] for s in samples) / n if n else 0,
            "case_detection_at_4": sum(any(s["bug_detected"] for s in row["samples"]) for row in predictions) / len(predictions) if predictions else 0,
            "denominator": "All generated tests, including invalid and truncated outputs; case rate uses all evaluated buggy programs"}
