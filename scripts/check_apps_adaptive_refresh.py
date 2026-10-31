import json
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from apps_adversary import contexts, observed, signature
from apps_adversarial_round import curriculum
from apps_core import canonical, digest, score, write
from restricted_python import RestrictedFunction


def main():
    original = {"case_id": "train/a", "source_id": 1, "problem_id": "train", "split": "train",
                "arguments": ["x"], "domain": {"x": {"type": "int", "min": 0, "max": 4}}, "constraints": [],
                "specification": "Return x", "reference": "def solve(x):\n    return x\n",
                "buggy_code": "def solve(x):\n    return 0 if x == 4 else x\n"}
    second = {**original, "case_id": "train/b", "buggy_code": "def solve(x):\n    return 1 if x == 0 else x\n"}
    current = {**original, "buggy_code": "def solve(x):\n    return 0 if x >= 2 else x\n",
               "curriculum_source": "local_adversary", "candidate_id": "r1-candidate", "pool_sha256": "old"}
    static = {**second, "curriculum_source": "static_curated"}
    manifest = {"dataset_id": "synthetic-refresh-check"}
    fake_load = lambda split: ([original, second], manifest)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        parent = root / "parent"
        parent.mkdir()
        (parent / "adapter_model.safetensors").write_bytes(b"synthetic-hash-only")
        current_path = root / "current.jsonl"
        current_path.write_text(canonical(current) + "\n" + canonical(static) + "\n", encoding="utf-8")
        probe = root / "probe"
        probe.mkdir()
        (probe / "evaluation-cases.jsonl").write_bytes(current_path.read_bytes())
        write(probe / "result.json", {"split": "train", "complete_split": True, "prompt_version": "apps-call-v2",
              "adapter_sha256": digest(parent / "adapter_model.safetensors"), "curriculum_path": str(current_path),
              "curriculum_sha256": digest(current_path)})
        predicted = []
        for case in (current, static):
            sample = score(canonical({"input": {"x": 2}, "expected": 2}), True, case)
            predicted.append({"case_id": case["case_id"], "samples": [sample] * 4})
        (probe / "predictions.jsonl").write_text("".join(canonical(row) + "\n" for row in predicted), encoding="utf-8")
        with patch("apps_adversary.audit"), patch("apps_adversary.load_cases", side_effect=fake_load), \
             patch("apps_curriculum.load_cases", side_effect=fake_load):
            context, cases, _ = contexts(probe, parent)
        assert cases[0]["buggy_code"] == current["buggy_code"]
        assert set(context["1"]["suites"]) == {"train/a"}
        for case in (original, current, second):
            function = RestrictedFunction(case["buggy_code"], case["arguments"])
            behavior = signature([observed(function, inputs) for inputs in context["1"]["inputs"]])
            assert behavior in context["1"]["known"]
        pool = root / "pool"
        pool.mkdir()
        (pool / "adversarial-cases.jsonl").write_text("", encoding="utf-8")
        with patch("apps_adversarial_round.load_cases", side_effect=fake_load), \
             patch("apps_curriculum.load_cases", side_effect=fake_load), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=SimpleNamespace(encode=lambda *a, **k: [1])):
            paths = curriculum(root, 2, pool, current_path)
        adaptive = [json.loads(line) for line in Path(paths["adaptive"]["path"]).read_text(encoding="utf-8").splitlines()]
        control = [json.loads(line) for line in Path(paths["control"]["path"]).read_text(encoding="utf-8").splitlines()]
        assert adaptive[0] == current
        assert adaptive[1] == static
        assert control[0]["buggy_code"] == original["buggy_code"]
        assert paths["adaptive"]["counts"] == {"local_adversary": 1, "static_curated": 1}
    print("PASS: probe current code, reject known original/current behavior, retain unresolved adversary, static control unchanged")


if __name__ == "__main__":
    main()
