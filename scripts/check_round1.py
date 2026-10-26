import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from restricted_python import RestrictedCodeError
from round1_worker import RoundRewards
from run_round1_experiment import compare


def main():
    with tempfile.TemporaryDirectory(prefix="round1-guard-check-", dir=r"D:\DA\TTNT_De19_Assets\temp") as location:
        root = Path(location)
        case = {"case_id": "guard", "problem_id": "guard", "domain": {"x": {"type": "int", "min": 0, "max": 40}},
                "reference": "def solve(x):\n    return x >= 5\n", "buggy_code": "def solve(x):\n    return x > 5\n"}
        audit = RoundRewards([case], root, [1])
        outputs = ['{"input":{"x":5},"expected":true}', '{"input":{"x":4},"expected":false}',
                   '{"input":{"x":5},"expected":false}', 'not-json']
        values = audit.reward(outputs, ["guard"] * 4, [[1]] * 4, SimpleNamespace(global_step=0))
        if values != [1.0, 0.3, 0.1, 0.0]:
            raise AssertionError("Reward semantics changed")

        def resource_failure(**kwargs):
            raise RestrictedCodeError("Synthetic resource-limit fixture")

        audit.compiled["guard"] = (audit.compiled["guard"][0], resource_failure)
        try:
            audit.reward(outputs, ["guard"] * 4, [[1]] * 4, SimpleNamespace(global_step=0))
        except RuntimeError as error:
            if "abort before" not in str(error):
                raise
        else:
            raise AssertionError("Restricted interpreter limit was incorrectly rewarded")
        generation = {"seed": 42}
        metrics = {"complete_split": True, "generated_tests": 72, "valid_rate": 0.5,
                   "io_accuracy": 0.3, "attack_rate": 0.2, "empirical_case_detection_at_4": 0.4}
        item = {"configuration": {"steps": 72}, "optimizer_steps": 72, "groups": 72,
                "parent_adapter_exact_match": True, "reward_driven_update_verified": True,
                "parent_adapter_sha256": "parent", "origin_case_exposures": {str(i): 2 for i in range(36)},
                "final_adapter": {"adapter_sha256": "adapter"}}
        training = {arm: copy.deepcopy(item) for arm in ("control", "adversarial")}
        validation = {arm: {"generation_config": generation, "dataset_split_sha256": "validation",
                           "adapter_sha256": "adapter", "metrics": metrics} for arm in training}
        spec = {"parent_adapter_sha256": "parent", "generation_config": generation, "validation_split_sha256": "validation"}
        payload = "".join(json.dumps({"case_id": "slot" + str(i)}) + "\n" for i in range(72))
        for arm in training:
            directory = root / ("train-" + arm)
            directory.mkdir()
            (directory / "rollouts.jsonl").write_text(payload, encoding="utf-8", newline="\n")
        result = compare(root, spec, {"metrics": metrics}, training, validation)
        if any(value != 0 for delta in result["delta_percentage_points"].values() for value in delta.values()):
            raise AssertionError("Equal scores must produce zero deltas")
        rejected = 0
        for key, value in (("optimizer_steps", 71), ("parent_adapter_sha256", "wrong"), ("origin_case_exposures", {"one": 72})):
            changed = copy.deepcopy(training)
            changed["adversarial"][key] = value
            try:
                compare(root, spec, {"metrics": metrics}, changed, validation)
            except RuntimeError:
                rejected += 1
            else:
                raise AssertionError("Comparison accepted invalid " + key)
        changed = copy.deepcopy(validation)
        changed["adversarial"]["generation_config"] = {"seed": 43}
        try:
            compare(root, spec, {"metrics": metrics}, training, changed)
        except RuntimeError:
            rejected += 1
        else:
            raise AssertionError("Comparison accepted unequal generation settings")
        (root / "train-adversarial/rollouts.jsonl").write_text(payload.replace('"slot0"', '"slot1"'), encoding="utf-8", newline="\n")
        try:
            compare(root, spec, {"metrics": metrics}, training, validation)
        except RuntimeError:
            rejected += 1
        else:
            raise AssertionError("Comparison accepted a repeated training slot")
        print("ROUND1 GUARD CHECK PASSED: reward levels, resource-limit rejection, equal-score comparison,", rejected, "invalid comparisons rejected")


if __name__ == "__main__":
    main()
