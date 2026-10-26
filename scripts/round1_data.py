import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from build_adversarial_pool import assess_candidate, prepare_context
from dataset_tasks import canonical


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare(assets, config_path):
    config = read(config_path)
    evidence = assets / "evidence"
    references = (("parent_run", "parent_result_sha256"), ("parent_validation_run", "parent_validation_result_sha256"), ("pool_run", "pool_result_sha256"))
    for name, sha_key in references:
        if digest(evidence / config[name] / "result.json") != config[sha_key]:
            raise ValueError("Frozen result changed: " + name)
    parent = read(evidence / config["parent_run"] / "result.json")
    validation = read(evidence / config["parent_validation_run"] / "result.json")
    pool_dir = evidence / config["pool_run"]
    pool_result = read(pool_dir / "result.json")
    if pool_result["status"] != "adversary_pool_completed" or pool_result["preflight_only"]:
        raise ValueError("Expected a completed real adversary pool")
    if digest(pool_dir / "adversarial-cases.jsonl") != config["pool_cases_sha256"] or config["pool_cases_sha256"] != pool_result["accepted_cases_sha256"]:
        raise ValueError("Accepted adversary code changed")
    if validation["status"] != "evaluation_completed" or validation["split"] != "validation" or validation["training_run"] != str(evidence / config["parent_run"]):
        raise ValueError("Parent Validation provenance differs")
    adapter = Path(parent["final_adapter"]["path"])
    if digest(adapter / "adapter_model.safetensors") != parent["final_adapter"]["adapter_sha256"] or validation["adapter_sha256"] != parent["final_adapter"]["adapter_sha256"]:
        raise ValueError("Parent adapter changed")
    for name in ("dataset_tasks.py", "evaluate_model.py", "train_static_grpo.py"):
        if digest(Path(__file__).with_name(name)) != parent["source_hashes"][name]:
            raise ValueError("Frozen source changed: " + name)
    if digest(Path(__file__).with_name("restricted_python.py")) != pool_result["source_hashes"]["restricted_python.py"]:
        raise ValueError("Restricted interpreter changed after pool sampling")
    contexts, verified_parent, diagnostic = prepare_context(SimpleNamespace(assets=assets, diagnostic_run=Path(pool_result["diagnostic_run"])))
    if diagnostic["training_run"] != str(evidence / config["parent_run"]):
        raise ValueError("Adversary was not filtered against this parent policy")
    if digest(Path(pool_result["diagnostic_run"]) / "result.json") != pool_result["diagnostic_result_sha256"]:
        raise ValueError("Policy diagnostic changed")
    original = rows(assets / "data/internal-v1/train.jsonl")
    pool = rows(pool_dir / "adversarial-cases.jsonl")
    if len(pool) != pool_result["accepted_candidates"] or len({case["case_id"] for case in pool}) != len(pool):
        raise ValueError("Pool has missing or duplicate records")
    seen = {identity: set() for identity in contexts}
    audits = []
    for case in pool:
        if case["split"] != "train" or case["problem_id"] not in contexts:
            raise ValueError("Non-Train adversary code")
        context = contexts[case["problem_id"]]
        for key in ("reference", "specification", "domain", "split_group"):
            if canonical(case[key]) != canonical(context["task"][key]):
                raise ValueError("Adversary changed the original task: " + key)
        checked = assess_candidate(case["buggy_code"], context, seen[case["problem_id"]])
        if not checked["accepted"] or checked["behavior_sha256"] != case["behavior_sha256"] or sorted(checked["passes_policy_suites_for_cases"]) != sorted(case["origin_policy_cases"]):
            raise ValueError("Candidate fails re-audit: " + case["case_id"])
        seen[case["problem_id"]].add(checked["behavior_sha256"])
        audits.append({"case_id": case["case_id"], **checked})
    if config["passes_per_original_case"] != 2 or config["steps_per_arm"] != 72:
        raise ValueError("This round fixes two passes and 72 updates per arm")
    arms = {"control": [], "adversarial": []}
    use_counts = Counter()
    mappings = []
    for pass_index in range(2):
        for case in original:
            identity = case["case_id"] + f"/round1-pass{pass_index + 1}"
            control = copy.deepcopy(case)
            control.update(case_id=identity, origin_case_id=case["case_id"], curriculum_source="curated")
            options = [candidate for candidate in pool if case["case_id"] in candidate["origin_policy_cases"]]
            chosen = min(options, key=lambda value: (use_counts[value["case_id"]], value["behavior_sha256"])) if options else None
            treatment = copy.deepcopy(chosen if chosen else case)
            treatment.update(case_id=identity, origin_case_id=case["case_id"], curriculum_source="adversary" if chosen else "curated")
            if chosen:
                use_counts[chosen["case_id"]] += 1
                treatment["candidate_source_case_id"] = chosen["case_id"]
            arms["control"].append(control)
            arms["adversarial"].append(treatment)
            mappings.append({"slot_id": identity, "origin_case_id": case["case_id"], "replacement": chosen["case_id"] if chosen else None})
    assert [case["case_id"] for case in arms["control"]] == [case["case_id"] for case in arms["adversarial"]]
    assert len(arms["control"]) == len(arms["adversarial"]) == 72
    audit = {"pool_candidates_reverified": len(audits), "pool_problems": len({case["problem_id"] for case in pool}),
             "finite_domain_checks": sum(item["domain_input_count"] for item in audits),
             "slots_per_arm": 72, "replaced_slots": sum(bool(item["replacement"]) for item in mappings),
             "distinct_adversary_cases_used": len(use_counts), "candidate_usage": dict(use_counts),
             "unused_pool_cases": [case["case_id"] for case in pool if case["case_id"] not in use_counts],
             "reason_for_unused": "Fixed two-slot-per-parent budget and deterministic selection; all pool records remain preserved",
             "mappings": mappings, "candidate_audits_with_witnesses_not_for_model_prompt": audits}
    return config, parent, validation, arms, audit


def save_curriculum(folder, arms, audit):
    hashes = {}
    for arm, cases in arms.items():
        path = folder / (arm + "-curriculum.jsonl")
        payload = "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases)
        path.write_text(payload, encoding="utf-8", newline="\n")
        hashes[arm] = digest(path)
    (folder / "curriculum-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return hashes
