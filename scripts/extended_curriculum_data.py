import copy
from collections import Counter
from pathlib import Path

from round1_data import prepare as prepare_round1, read, rows


def prepare(assets, config_path):
    base_path = Path(__file__).resolve().parents[1] / "configs/adversarial-round1.json"
    base, parent, validation, _, audit = prepare_round1(assets, base_path)
    config = read(config_path)
    for key in ("parent_run", "parent_result_sha256", "parent_validation_run", "parent_validation_result_sha256", "pool_run", "pool_result_sha256", "pool_cases_sha256"):
        if config[key] != base[key]:
            raise ValueError("Extended experiment changed frozen provenance: " + key)
    if config["steps_per_arm"] != 144 or config["passes_per_original_case"] != 4 or config["checkpoint_interval"] != 36:
        raise ValueError("Expected preplanned 144-step, four-pass ablation")
    original = rows(assets / "data/internal-v1/train.jsonl")
    pool = rows(assets / "evidence" / config["pool_run"] / "adversarial-cases.jsonl")
    arms = {"control": [], "adversarial": []}
    usage = Counter()
    mappings = []
    for repeat in range(4):
        for case in original:
            identity = case["case_id"] + f"/extended-pass{repeat + 1}"
            control = copy.deepcopy(case)
            control.update(case_id=identity, origin_case_id=case["case_id"], curriculum_source="curated")
            options = [candidate for candidate in pool if case["case_id"] in candidate["origin_policy_cases"]]
            chosen = min(options, key=lambda candidate: (usage[candidate["case_id"]], candidate["behavior_sha256"])) if options else None
            treatment = copy.deepcopy(chosen if chosen else case)
            treatment.update(case_id=identity, origin_case_id=case["case_id"], curriculum_source="adversary" if chosen else "curated")
            if chosen:
                usage[chosen["case_id"]] += 1
                treatment["candidate_source_case_id"] = chosen["case_id"]
            arms["control"].append(control)
            arms["adversarial"].append(treatment)
            mappings.append({"slot_id": identity, "origin_case_id": case["case_id"], "replacement": chosen["case_id"] if chosen else None})
    audit.update(slots_per_arm=144, replaced_slots=sum(usage.values()), distinct_adversary_cases_used=len(usage), candidate_usage=dict(usage),
                 unused_pool_cases=[case["case_id"] for case in pool if case["case_id"] not in usage], mappings=mappings,
                 reason_for_unused="Fixed four-pass schedule; preserve every accepted pool record")
    return config, parent, validation, arms, audit
