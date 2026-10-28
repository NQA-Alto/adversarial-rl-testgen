import argparse
import json
import sys
from pathlib import Path

from apps_core import canonical, digest, load_cases, metrics, read, score, write


def audit(folder):
    result = read(folder / "result.json")
    if result["status"] == "evaluation_completed":
        cases, manifest = load_cases(result["split"])
        by_id = {case["case_id"]: case for case in cases}
        predictions = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        identities = [row["case_id"] for row in predictions]
        if len(identities) != len(set(identities)) or result["dataset_id"] != manifest["dataset_id"]:
            raise ValueError("Dataset or case identity mismatch")
        if result["complete_split"] and set(identities) != set(by_id):
            raise ValueError("Incomplete split")
        count = 0
        for row in predictions:
            if len(row["samples"]) != 4:
                raise ValueError("Unexpected test budget")
            for sample in row["samples"]:
                checked = score(sample["raw_output"], sample["terminated"], by_id[row["case_id"]])
                for key in ("valid_format", "correct_expected", "bug_detected", "reward"):
                    if sample[key] != checked[key]:
                        raise ValueError("Rescoring mismatch: " + key)
                count += 1
        if canonical(metrics(predictions)) != canonical(result["metrics"]):
            raise ValueError("Metrics mismatch")
        review = {"status": "passed", "predictions_rescored": count, "metrics_recomputed": True,
                  "predictions_sha256": digest(folder / "predictions.jsonl"), "model_test_evaluation": False}
    elif result["status"] in {"static_grpo_pilot_completed", "pilot_no_verified_policy_update"}:
        cases, manifest = load_cases("train")
        by_id = {case["case_id"]: case for case in cases}
        events = [json.loads(line) for line in (folder / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()]
        count = 0
        for event in events:
            rewards = []
            for sample in event["samples"]:
                checked = score(sample["raw_output"], sample["terminated"], by_id[sample["case_id"]])
                for key in ("valid_format", "correct_expected", "bug_detected", "reward"):
                    if checked[key] != sample[key]:
                        raise ValueError("Training reward mismatch")
                rewards.append(checked["reward"])
                count += 1
            if rewards != event["rewards"] or event["policy_signal"] != (max(rewards) - min(rewards) > 1e-8):
                raise ValueError("Group reward mismatch")
        if digest(Path(result["adapter"]["path"]) / "adapter_model.safetensors") != result["adapter"]["adapter_sha256"]:
            raise ValueError("Adapter hash mismatch")
        review = {"status": "passed", "train_samples_rescored": count, "groups": len(events),
                  "adapter_hash_verified": True, "policy_update_verified": result["update_verified"],
                  "quality_improvement_claim": False, "model_test_evaluation": False}
    elif result["status"] == "apps_supervised_warmup_completed":
        cases, manifest = load_cases("train")
        by_id = {case["case_id"]: case for case in cases}
        if digest(folder / "targets.jsonl") != result["target_sha256"]:
            raise ValueError("Supervised targets changed")
        targets = [json.loads(line) for line in (folder / "targets.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in targets:
            checked = score(row["target"], True, by_id[row["case_id"]])
            if not checked["correct_expected"] or checked["bug_detected"] != row["bug_detecting_target"]:
                raise ValueError("Supervised target does not match the verifier")
        events = [json.loads(line) for line in (folder / "sft-steps.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(events) != result["optimizer_steps"] or any(row["method"] != "SFT_NOT_RL" or row["gradient_norm"] <= 0 for row in events):
            raise ValueError("Incomplete supervised training evidence")
        if digest(Path(result["adapter"]["path"]) / "adapter_model.safetensors") != result["adapter"]["adapter_sha256"]:
            raise ValueError("Supervised adapter hash mismatch")
        review = {"status": "passed", "supervised_targets_verified": len(targets), "train_only": True,
                  "adapter_hash_verified": True, "method": "SFT_NOT_RL", "model_test_evaluation": False}
    else:
        raise ValueError("Run is not complete")
    write(folder / "review.json", review)
    return review


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.folder), indent=2))
