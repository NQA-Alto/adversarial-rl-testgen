import argparse
import ast
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from dataset_tasks import SPLIT_GROUPS, all_inputs, canonical, compile_trusted, input_is_valid, problems


def build_records():
    splits = {name: [] for name in SPLIT_GROUPS}
    audit = []
    problem_ids = set()
    reference_splits = {}
    group_membership = {}
    for split, groups in SPLIT_GROUPS.items():
        for group in groups:
            if group in group_membership:
                raise ValueError("A split group appears in multiple splits")
            group_membership[group] = split
    for task in problems():
        identity = task["problem_id"]
        if identity in problem_ids:
            raise ValueError("Duplicate problem identifier")
        problem_ids.add(identity)
        split = group_membership[task["split_group"]]
        reference = compile_trusted(task["reference"])
        signature = ast.dump(ast.parse(task["reference"]), include_attributes=False)
        if signature in reference_splits and reference_splits[signature] != split:
            raise ValueError("Identical reference code across splits")
        reference_splits[signature] = split
        for check in task["manual_checks"]:
            if not input_is_valid(check["input"], task["domain"]):
                raise ValueError(f"Invalid manual check for {identity}")
            if canonical(reference(**check["input"])) != canonical(check["expected"]):
                raise ValueError(f"Reference fails manual check for {identity}: {check}")
        probes = list(all_inputs(task["domain"]))
        expected = [canonical(reference(**inputs)) for inputs in probes]
        mutant_signatures = set()
        for index, code in enumerate(task["mutants"], 1):
            buggy = compile_trusted(code)
            outputs = []
            witness = None
            killed = 0
            for inputs, gold in zip(probes, expected, strict=True):
                try:
                    observed = canonical(buggy(**inputs))
                except Exception as error:
                    observed = f"exception:{type(error).__name__}"
                outputs.append(observed)
                if observed != gold:
                    killed += 1
                    if witness is None:
                        witness = {"input": inputs, "expected": json.loads(gold)}
            if not killed:
                raise ValueError(f"Mutant is equivalent to reference: {identity}/m{index}")
            signature = hashlib.sha256(canonical(outputs).encode()).hexdigest()
            if signature in mutant_signatures:
                raise ValueError(f"Behaviorally duplicate mutants: {identity}/m{index}")
            mutant_signatures.add(signature)
            record = {
                "case_id": f"{identity}/m{index}", "problem_id": identity,
                "split_group": task["split_group"], "split": split,
                "specification": task["specification"], "domain": task["domain"],
                "reference": task["reference"], "buggy_code": code,
            }
            splits[split].append(record)
            audit.append({"case_id": record["case_id"], "split": split, "domain_inputs": len(probes),
                          "bug_revealing_inputs": killed, "witness_for_audit_only": witness})
    counts = {
        name: {"problems": len({row["problem_id"] for row in rows}), "cases": len(rows),
               "groups": sorted({row["split_group"] for row in rows})}
        for name, rows in splits.items()
    }
    if [(counts[name]["problems"], counts[name]["cases"]) for name in ("train", "validation", "test")] != [(12, 36), (6, 18), (6, 18)]:
        raise ValueError(f"Unexpected dataset counts: {counts}")
    digest = hashlib.sha256(canonical(splits).encode("utf-8")).hexdigest()
    return splits, {"dataset_id": f"internal-v1-{digest[:12]}", "content_sha256": digest,
                    "counts": counts, "case_audits": audit, "total_mutant_input_checks": sum(row["domain_inputs"] for row in audit)}


def plot_overview(out, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    figure, axes = pyplot.subplots(1, 2, figsize=(9, 3.6), facecolor="white")
    names = ["Train", "Validation", "Test"]
    colors = ["#3984c6", "#4ca68b", "#e5a13a"]
    for axis, key, title in zip(axes, ["problems", "cases"], ["Distinct problems", "Buggy-code cases"], strict=True):
        values = [counts[name][key] for name in ("train", "validation", "test")]
        bars = axis.bar(names, values, color=colors, width=0.6)
        axis.bar_label(bars, padding=3)
        axis.set_title(title)
        axis.set_ylim(0, max(values) * 1.25)
        axis.set_ylabel("Count")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Internal dataset v1 - groups remain in one split")
    figure.tight_layout()
    figure.savefig(out / "dataset-overview.png", dpi=160, facecolor="white")
    pyplot.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Build and audit a small, locally curated Python bug dataset")
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    os.environ["MPLCONFIGDIR"] = str(args.assets / "cache" / "matplotlib")
    splits, audit = build_records()
    print("DATASET AUDIT PASSED:", audit["dataset_id"])
    for name, count in audit["counts"].items():
        print(f'{name}: {count["problems"]} problems | {count["cases"]} cases')
    print("Reference manual checks: passed")
    print("Every mutant differs from its reference: passed")
    print("Sibling mutants have distinct behaviors: passed")
    print("Split groups and problem IDs do not overlap: passed")
    print("Exhaustive mutant/input checks:", audit["total_mutant_input_checks"])
    if args.check_only:
        print("CHECK ONLY: dataset files were not written.")
        return
    out = args.assets / "data" / "internal-v1"
    payloads = {f"{split}.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
                for split, rows in splits.items()}
    if out.exists():
        for filename, payload in payloads.items():
            path = out / filename
            if not path.is_file() or path.read_text(encoding="utf-8") != payload:
                raise RuntimeError("Existing dataset differs; preserve it and investigate before rebuilding")
        existing_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if existing_manifest["dataset_id"] != audit["dataset_id"]:
            raise RuntimeError("Existing manifest has a different dataset identifier")
        for filename in payloads:
            if hashlib.sha256((out / filename).read_bytes()).hexdigest() != existing_manifest["files"][filename]:
                raise RuntimeError("Existing dataset file hash does not match its manifest")
        print("Existing dataset matches; no records overwritten.")
        print("Dataset:", out)
        return
    out.mkdir(parents=True, exist_ok=False)
    for filename, payload in payloads.items():
        (out / filename).write_text(payload, encoding="utf-8", newline="\n")
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset_id": audit["dataset_id"], "content_sha256": audit["content_sha256"],
        "source": "Locally authored internal dataset; not APPS, Codeforces, MBPP or paper-reported data",
        "split_strategy": "Fixed disjoint problem/logic groups; all mutants of a problem remain together",
        "counts": audit["counts"],
        "files": {name: hashlib.sha256(payload.encode("utf-8")).hexdigest() for name, payload in payloads.items()},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task_source_sha256": hashlib.sha256(Path(__file__).with_name("dataset_tasks.py").read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# Internal dataset v1\n\nLocally authored controlled Python functions and three distinct bugs per problem. "
        "This is an internal experimental dataset, not the original paper benchmark.\n\n"
        "All logic groups and all mutants of a problem stay in one split. Test is reserved for final evaluation. "
        "Audit witnesses and reference code must never be placed into the model prompt.\n\n"
        "The builder checks manual reference examples and exhaustively checks every mutant over the declared finite input domain. "
        "These checks do not prove that every natural-language specification is perfect.\n",
        encoding="utf-8",
    )
    plot_overview(out, audit["counts"])
    print("Dataset and overview chart:", out)
    print("Next: evaluate the BASE model on validation only. Keep test untouched for model evaluation.")


if __name__ == "__main__":
    main()
