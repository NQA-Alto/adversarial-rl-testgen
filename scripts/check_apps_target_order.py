import json
import hashlib
from pathlib import Path

from apps_core import canonical, digest, read
from apps_model_runner import PROJECT
from apps_sft import build_examples, target_text


def check():
    old, cases, _ = build_examples(42, "sorted-v1")
    new, _, _ = build_examples(42, "input-first-v1")
    state = read(PROJECT / "APPS_STATE.json")
    study = read(PROJECT / "results/apps-target-order/metrics.json")
    control = study["runs"]["sorted-v1"]
    control_path = Path(control["sft"])
    if digest(control_path / "result.json") != control["sft_result_sha256"]:
        raise AssertionError("Historical control evidence changed")
    original = read(control_path / "result.json")
    payload = "".join(canonical(row) + "\n" for row in old).encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != original["target_sha256"] or len(old) != 319:
        raise AssertionError("Historical target bytes changed")
    for a, b in zip(old, new, strict=True):
        if {k: v for k, v in a.items() if k != "target"} != {k: v for k, v in b.items() if k != "target"}:
            raise AssertionError("Sample identity or label changed")
        if canonical(json.loads(a["target"])) != canonical(json.loads(b["target"])):
            raise AssertionError("Input or expected value changed")
        if next(iter(json.loads(a["target"]))) != "expected" or next(iter(json.loads(b["target"]))) != "input":
            raise AssertionError("Outer field order not as declared")
        value = json.loads(a["target"])
        if target_text(value["input"], value["expected"]) != a["target"]:
            raise AssertionError("Default serialization changed")
    active = read(Path(state["sft_training"]["path"]) / "result.json")
    active_order = active["configuration"].get("target_order", "sorted-v1")
    active_rows = new if active_order == "input-first-v1" else old
    if hashlib.sha256("".join(canonical(row) + "\n" for row in active_rows).encode("utf-8")).hexdigest() != active["target_sha256"]:
        raise AssertionError("Active target pool no longer matches its declared order")
    return {"status": "passed", "targets": len(old), "train_cases": len(cases), "historical_target_bytes_unchanged": True,
            "candidate_semantic_labels_unchanged": True, "only_outer_field_order_changes": True, "active_target_order": active_order, "model_loaded": False}


if __name__ == "__main__":
    print(check())
