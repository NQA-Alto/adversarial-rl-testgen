import argparse
import ast
import copy
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

from apps_core import DATA, REVISION, SOURCE, all_inputs, canonical, digest, normalized, official_matches, write
from apps_profiles import PROFILES, split_for
from restricted_python import RestrictedCodeError, RestrictedFunction


def mutations(code):
    tree = ast.parse(code)
    seen = {code}
    for index, node in enumerate(ast.walk(tree)):
        edits = []
        if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) < 100:
            edits = [("value", node.value - 1), ("value", node.value + 1)]
        elif isinstance(node, ast.Compare) and len(node.ops) == 1:
            swaps = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In}
            if type(node.ops[0]) in swaps:
                edits = [("ops", [swaps[type(node.ops[0])]()] )]
        elif isinstance(node, ast.BinOp):
            swaps = {ast.Add: [ast.Sub], ast.Sub: [ast.Add], ast.Mult: [ast.Add, ast.Sub], ast.Mod: [ast.FloorDiv]}
            edits = [("op", kind()) for kind in swaps.get(type(node.op), [])]
        elif isinstance(node, ast.BoolOp):
            edits = [("op", ast.Or() if isinstance(node.op, ast.And) else ast.And())]
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            edits = [("op", ast.UAdd() if isinstance(node.op, ast.USub) else ast.USub())]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"min", "max", "all", "any"}:
            swap = {"min": "max", "max": "min", "all": "any", "any": "all"}
            edits = [("func", ast.Name(id=swap[node.func.id], ctx=ast.Load()))]
        for field, value in edits:
            edited = copy.deepcopy(tree)
            target = list(ast.walk(edited))[index]
            setattr(target, field, value)
            candidate = ast.unparse(ast.fix_missing_locations(edited)) + "\n"
            if candidate not in seen:
                seen.add(candidate)
                yield candidate, {"node_index": index, "node_type": type(node).__name__, "field": field}


def references(row):
    io = json.loads(row["input_output"])
    if not io.get("fn_name") or len(io["inputs"]) < 3:
        raise ValueError("Expected a call-based task and at least three source tests")
    accepted = []
    for index, source in enumerate(json.loads(row["solutions"])[:12]):
        try:
            code, arguments = normalized(source, io["fn_name"])
            function = RestrictedFunction(code, arguments)
            for inputs, expected in zip(io["inputs"], io["outputs"], strict=True):
                if type(inputs) is not list or len(inputs) != len(arguments):
                    raise ValueError("Source argument shape")
                if not official_matches(function(**dict(zip(arguments, inputs, strict=True))), expected):
                    raise ValueError("Source test mismatch")
            accepted.append({"solution_index": index, "original_source": source, "code": code, "arguments": arguments})
            if len(accepted) == 2:
                break
        except Exception:
            continue
    if len(accepted) != 2:
        raise ValueError("Two independently stored compatible source solutions are required")
    return io, accepted


def build():
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    source_manifest = json.loads((SOURCE / "source-manifest.json").read_text(encoding="utf-8"))
    for item in source_manifest["files"]:
        if digest(SOURCE / item["name"]) != item["sha256"]:
            raise ValueError("Raw source hash mismatch")
    if DATA.exists():
        raise FileExistsError("Dataset version already exists; do not overwrite a frozen dataset")
    rows = [json.loads(line) for line in (SOURCE / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = {name: [] for name in ("train", "validation", "test")}
    audits, excluded, problems = [], [], []
    for identity, (title, fields, relations) in PROFILES.items():
        row = next(row for row in rows if row["id"] == identity)
        try:
            if row["difficulty"] != "introductory":
                raise ValueError("Wrong difficulty")
            io, refs = references(row)
            arguments = refs[0]["arguments"]
            if len(fields) != len(arguments):
                raise ValueError("Domain arity")
            constraints = [":".join([relation[0], *(arguments[i] for i in relation[1:])]) for relation in relations]
            problem = {"problem_id": f"apps-train-{identity}", "source_id": identity, "source_split": "train",
                       "source_revision": REVISION, "source_url": row["url"], "source_entry_point": io["fn_name"],
                       "title": title, "specification": row["question"], "split": split_for(identity),
                       "domain": dict(zip(arguments, fields, strict=True)), "arguments": arguments, "constraints": constraints,
                       "reference": refs[0]["code"], "references": refs,
                       "source_record_sha256": __import__("hashlib").sha256(canonical(row).encode()).hexdigest(),
                       "source_tests": io, "scope": "Explicit finite valid subdomain of a public APPS call-based problem"}
            inputs = list(all_inputs(problem))
            runners = [RestrictedFunction(ref["code"], ref["arguments"]) for ref in refs]
            outputs = []
            for value in inputs:
                positions = list(value.values())
                observed = [fn(**dict(zip(ref["arguments"], positions, strict=True))) for fn, ref in zip(runners, refs, strict=True)]
                if canonical(observed[0]) != canonical(observed[1]):
                    raise ValueError("Source references disagree on the declared finite domain")
                outputs.append(observed[0])
            behavior_seen = {canonical(outputs)}
            retained, rejected = [], Counter()
            for code, change in mutations(problem["reference"]):
                try:
                    mutant = RestrictedFunction(code, arguments)
                    observed = [mutant(**value) for value in inputs]
                    vector = canonical(observed)
                    if vector in behavior_seen:
                        rejected["duplicate_or_equivalent_on_domain"] += 1
                        continue
                    passed = [canonical(a) == canonical(b) for a, b in zip(observed, outputs, strict=True)]
                    if not any(passed):
                        rejected["fails_every_input"] += 1
                        continue
                    witness = next(i for i, same in enumerate(passed) if not same)
                    case = {key: value for key, value in problem.items() if key not in {"references", "source_tests"}}
                    case.update(case_id=problem["problem_id"] + f"/m{len(retained) + 1}", buggy_code=code,
                                bug_origin="single_AST_mutation_of_public_reference", mutation=change)
                    retained.append(case)
                    audits.append({"case_id": case["case_id"], "domain_inputs": len(inputs),
                                   "different_inputs": len(passed) - sum(passed), "same_inputs": sum(passed),
                                   "witness": {"input": inputs[witness], "expected": outputs[witness], "buggy_output": observed[witness]}})
                    behavior_seen.add(vector)
                    if len(retained) == 3:
                        break
                except Exception as error:
                    rejected[type(error).__name__] += 1
            if not retained:
                raise ValueError("No nontrivial supported mutation on the declared domain")
            splits[problem["split"]].extend(retained)
            problem["audit"] = {"source_tests_passed_per_reference": len(io["inputs"]), "references": 2,
                                "finite_domain_inputs": len(inputs), "mutants": len(retained), "rejected_mutations": dict(rejected)}
            problems.append(problem)
            print("ACCEPT", identity, problem["split"], len(inputs), "inputs", len(retained), "mutants", flush=True)
        except Exception as error:
            excluded.append({"id": identity, "title": title, "reason": str(error)})
            print("EXCLUDE", identity, str(error), flush=True)
    if min(len(splits[name]) for name in splits) < 3:
        raise RuntimeError("Insufficient data in one split")
    groups = [{case["problem_id"] for case in rows} for rows in splits.values()]
    if any(a & b for index, a in enumerate(groups) for b in groups[index + 1:]):
        raise RuntimeError("Problem leakage")
    urls = [{case["source_url"] for case in rows} for rows in splits.values()]
    if any(a & b for index, a in enumerate(urls) for b in urls[index + 1:]):
        raise RuntimeError("Source URL leakage")
    DATA.mkdir(parents=True)
    manifest = {"dataset": "APPS Introductory, audited bounded call-based subset", "source": "codeparrot/apps",
                "source_revision": REVISION, "source_manifest_sha256": digest(SOURCE / "source-manifest.json"),
                "selection": "35 prespecified problem IDs with explicit finite contracts; at least 3 source tests, 2 compatible source references, agreement over all declared inputs, 1-3 nontrivial AST mutants. No selection by model scores.",
                "official_test_downloaded": False, "official_test_evaluated": False,
                "split_policy": "Problem-disjoint development Train/Validation/Test partition within the official APPS Train split. Official APPS Test is separate and untouched.",
                "limits": "Call-based functions only; finite subdomains; restricted AST interpreter. Not the full APPS Introductory benchmark or ATGen replication.",
                "planned_problem_count": len(PROFILES), "accepted_problem_count": len(problems), "excluded": excluded,
                "splits": {}}
    for split, cases in splits.items():
        path = DATA / (split + ".jsonl")
        path.write_text("".join(canonical(case) + "\n" for case in cases), encoding="utf-8", newline="\n")
        manifest["splits"][split] = {"cases": len(cases), "problems": len({case["problem_id"] for case in cases}), "sha256": digest(path)}
    manifest["dataset_id"] = "apps-intro-call-v2-" + __import__("hashlib").sha256(canonical(manifest).encode()).hexdigest()[:12]
    write(DATA / "problems.json", problems)
    write(DATA / "audit.json", {"cases": audits, "excluded": excluded, "total_finite_case_checks": sum(item["domain_inputs"] for item in audits)})
    write(DATA / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser().parse_args()
    build()
