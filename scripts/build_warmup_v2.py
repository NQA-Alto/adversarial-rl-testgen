import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from dataset_tasks import SPLIT_GROUPS, all_inputs, canonical, compile_trusted, input_is_valid, model_prompt, problem, text
from restricted_python import RestrictedFunction


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def string_tasks():
    return [
        problem("p25-text-length", "character-length", "Return the number of characters in text. Count spaces and repeated characters separately. Empty text has length 0.", {"text": text("abA ")},
            "def solve(text):\n    return len(text)", [
                "def solve(text):\n    return len(text) + 1",
                "def solve(text):\n    return len(text[:-1])",
                "def solve(text):\n    return len(set(text))"],
            [({"text": ""}, 0), ({"text": "a A"}, 3), ({"text": "aa"}, 2)]),
        problem("p26-uppercase-count", "uppercase-count", "Count every uppercase A or B in text. Lowercase a and b and spaces do not count. Count repeated uppercase characters separately.", {"text": text("aAbB ")},
            "def solve(text):\n    return sum(1 for c in text if c in 'AB')", [
                "def solve(text):\n    return sum(1 for c in text if c in 'ab')",
                "def solve(text):\n    return sum(1 for c in text if c != ' ')",
                "def solve(text):\n    return len(set(c for c in text if c in 'AB'))"],
            [({"text": ""}, 0), ({"text": "Aa B"}, 2), ({"text": "AA"}, 2), ({"text": "ab"}, 0)]),
        problem("p27-adjacent-changes", "adjacent-transitions", "Count indices i from 1 to len(text)-1 for which text[i] differs from text[i-1]. Text with fewer than two characters returns 0.", {"text": text("ab")},
            "def solve(text):\n    return sum(1 for i in range(1, len(text)) if text[i] != text[i-1])", [
                "def solve(text):\n    return sum(1 for i in range(1, len(text)) if text[i] == text[i-1])",
                "def solve(text):\n    return sum(1 for i in range(1, len(text)-1) if text[i] != text[i-1])",
                "def solve(text):\n    return max(0, len(set(text))-1)"],
            [({"text": ""}, 0), ({"text": "a"}, 0), ({"text": "aa"}, 0), ({"text": "abba"}, 2), ({"text": "abab"}, 3)]),
        problem("p28-leading-a", "leading-run-length", "Return the length of the initial consecutive run of lowercase a characters in text. Stop at the first b. Empty text and text starting with b return 0.", {"text": text("ab")},
            "def solve(text):\n    count = 0\n    for c in text:\n        if c != 'a':\n            break\n        count += 1\n    return count", [
                "def solve(text):\n    return text.count('a')",
                "def solve(text):\n    return len(text) if text and text[0] == 'a' else 0",
                "def solve(text):\n    count = 0\n    for c in text:\n        if c != 'a':\n            break\n        count += 1\n    return max(0, count-1)"],
            [({"text": ""}, 0), ({"text": "ba"}, 0), ({"text": "aaba"}, 2), ({"text": "aaaa"}, 4)]),
    ]


def behavior(function, inputs):
    try:
        return canonical(function(**inputs))
    except Exception as error:
        return "exception:" + type(error).__name__


def build(assets):
    frozen = assets / "data/internal-v1"
    manifest = read(frozen / "manifest.json")
    for name, sha in manifest["files"].items():
        if digest(frozen / name) != sha:
            raise ValueError("Frozen v1 data changed: " + name)
    cases = copy.deepcopy(rows(frozen / "train.jsonl"))
    forbidden = {case["problem_id"] for case in rows(frozen / "validation.jsonl")}
    forbidden_groups = {case["split_group"] for case in rows(frozen / "validation.jsonl")}
    forbidden_groups |= SPLIT_GROUPS["test"]
    additions = string_tasks()
    checks, manual_count, new_audits = 0, 0, []
    for task in additions:
        if task["problem_id"] in forbidden or task["split_group"] in forbidden_groups:
            raise ValueError("Added Train task overlaps Validation")
        oracle = compile_trusted(task["reference"])
        restricted_oracle = RestrictedFunction(task["reference"], task["domain"])
        for sample in task["manual_checks"]:
            if not input_is_valid(sample["input"], task["domain"]) or canonical(oracle(**sample["input"])) != canonical(sample["expected"]):
                raise ValueError("Manual reference check failed: " + task["problem_id"])
            manual_count += 1
        probes = list(all_inputs(task["domain"]))
        gold = [behavior(oracle, inputs) for inputs in probes]
        if any(behavior(restricted_oracle, inputs) != expected for inputs, expected in zip(probes, gold, strict=True)):
            raise ValueError("Restricted reference differs from Python")
        signatures = set()
        for index, code in enumerate(task["mutants"], 1):
            fn = compile_trusted(code)
            restricted = RestrictedFunction(code, task["domain"])
            outputs = [behavior(fn, inputs) for inputs in probes]
            for inputs, expected in zip(probes, outputs, strict=True):
                if behavior(restricted, inputs) != expected:
                    raise ValueError("Restricted mutant differs from Python")
                checks += 1
            signature = hashlib.sha256(canonical(outputs).encode()).hexdigest()
            if outputs == gold or signature in signatures:
                raise ValueError("Equivalent or duplicate new mutant")
            signatures.add(signature)
            case = {key: task[key] for key in ("problem_id", "split_group", "specification", "domain", "reference")}
            case.update(case_id=task["problem_id"] + "/m" + str(index), split="train", buggy_code=code)
            cases.append(case)
            new_audits.append({"case_id": case["case_id"], "domain_inputs": len(probes), "bug_revealing_inputs": sum(a != b for a, b in zip(outputs, gold, strict=True))})
    if len(cases) != 48 or len({case["problem_id"] for case in cases}) != 16:
        raise ValueError("Expected 16 Train problems / 48 cases")
    examples = {"sft_train": [], "sft_dev": []}
    problem_partitions = {}
    for case in cases:
        pid = case["problem_id"]
        if pid not in problem_partitions:
            candidates = sorted(all_inputs(case["domain"]), key=lambda value: hashlib.sha256((pid + canonical(value) + "42").encode()).hexdigest())
            problem_partitions[pid] = {"sft_dev": candidates[:2], "sft_train": candidates[2:]}
        reference = compile_trusted(case["reference"])
        buggy = compile_trusted(case["buggy_code"])
        for split in examples:
            available = problem_partitions[pid][split]
            count = 2 if split == "sft_dev" else min(6, len(available))
            if split == "sft_train":
                kills = [inputs for inputs in available if behavior(reference, inputs) != behavior(buggy, inputs)]
                nonkills = [inputs for inputs in available if behavior(reference, inputs) == behavior(buggy, inputs)]
                chosen = kills[:3] + nonkills[:3]
                identities = {canonical(inputs) for inputs in chosen}
                chosen += [inputs for inputs in available if canonical(inputs) not in identities][:count - len(chosen)]
            else:
                chosen = available[:count]
            for inputs in chosen:
                expected = reference(**inputs)
                target = json.dumps({"input": inputs, "expected": expected}, ensure_ascii=False, separators=(",", ":"))
                examples[split].append({"sample_id": case["case_id"] + "/" + split + "/" + hashlib.sha256(canonical(inputs).encode()).hexdigest()[:12],
                    "case_id": case["case_id"], "problem_id": pid, "input": inputs, "expected": expected,
                    "messages": model_prompt(case, version="plain-v2"), "target": target,
                    "bug_revealing": behavior(reference, inputs) != behavior(buggy, inputs)})
    for split, values in examples.items():
        if len(values) != len({value["sample_id"] for value in values}):
            raise ValueError("Duplicate SFT sample")
        for value in values:
            if value["problem_id"] in forbidden or "def solve" in value["target"]:
                raise ValueError("Unexpected held-out data or reference in target")
    train_inputs = {(value["problem_id"], canonical(value["input"])) for value in examples["sft_train"]}
    dev_inputs = {(value["problem_id"], canonical(value["input"])) for value in examples["sft_dev"]}
    if train_inputs & dev_inputs:
        raise ValueError("SFT Train/Dev inputs overlap across sibling mutants")
    identity = hashlib.sha256(canonical({"cases": cases, "examples": examples}).encode()).hexdigest()
    audit = {"dataset_id": "warmup-v2-" + identity[:12], "parent_dataset_id": manifest["dataset_id"], "train_problems": 16,
        "train_cases": 48, "added_string_problems": 4, "manual_new_reference_checks": manual_count, "exhaustive_new_mutant_input_checks": checks,
        "new_mutants": new_audits, "sft_train_samples": len(examples["sft_train"]), "sft_dev_samples": len(examples["sft_dev"]),
        "train_input_types": sorted({field["type"] for case in cases for field in case["domain"].values()}),
        "train_dev_problem_input_overlap": 0, "validation_file_sha256": manifest["files"]["validation.jsonl"],
        "test_file_sha256_unchanged": manifest["files"]["test.jsonl"], "test_contents_used_for_new_tasks_or_targets": False,
        "scope": "Locally authored synthetic Train extension. SFT Dev has new inputs on seen Train problems; it is not held-out model Validation.",
        "source": "Reference-oracle targets from Train only, no external model/API or generated-code targets"}
    return cases, examples, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cases, examples, audit = build(args.assets)
    if args.check_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return
    output = args.assets / "data/warmup-v2-input-first"
    payloads = {"train-cases.jsonl": cases, **{name + ".jsonl": values for name, values in examples.items()}}
    content = {name: "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values) for name, values in payloads.items()}
    if output.exists():
        old = read(output / "manifest.json")
        if old["dataset_id"] != audit["dataset_id"] or any(digest(output / name) != sha for name, sha in old["files"].items()):
            raise ValueError("Existing warmup dataset differs; do not overwrite")
        print("Existing audited warmup dataset retained:", output)
        return
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in content.items():
        (output / name).write_text(payload, encoding="utf-8", newline="\n")
    source_dir = output / "source"
    source_dir.mkdir()
    source_hashes = {}
    for name in (Path(__file__).name, "dataset_tasks.py", "restricted_python.py"):
        current = Path(__file__).with_name(name)
        shutil.copy2(current, source_dir / name)
        source_hashes[name] = digest(current)
    write(output / "audit.json", audit)
    write(output / "manifest.json", {"created_at": datetime.now().astimezone().isoformat(), "dataset_id": audit["dataset_id"],
        "files": {name: digest(output / name) for name in content}, "source_hashes": source_hashes,
        "frozen_validation_source": str(args.assets / "data/internal-v1/validation.jsonl"),
        "validation_sha256": audit["validation_file_sha256"], "test_set_model_evaluation": False})
    print("WARMUP V2 DATASET AUDIT PASSED:", json.dumps(audit, ensure_ascii=False, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()
