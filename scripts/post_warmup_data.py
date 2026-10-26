import copy
from collections import Counter
from pathlib import Path

from build_adversarial_pool import assess_candidate, behavior_signature, observed
from dataset_tasks import all_inputs, canonical, compile_trusted, evaluate_output
from evaluate_model import summarize
from restricted_python import RestrictedFunction
from round1_data import digest, read, rows


def baseline(assets, config_path):
    config = read(config_path)
    evidence = assets / "evidence"
    warmup_dir = evidence / config["parent_run"]
    probe = evidence / config["train_probe"]
    expected = ((warmup_dir / "result.json", "parent_result_sha256"), (warmup_dir / "validation/result.json", "parent_validation_result_sha256"),
                (probe / "result.json", "train_probe_result_sha256"), (probe / "parent/predictions.jsonl", "train_predictions_sha256"))
    for path, key in expected:
        if digest(path) != config[key]:
            raise ValueError("Frozen post-warmup provenance changed: " + key)
    completed = read(warmup_dir / "result.json")
    trained = read(warmup_dir / "train/result.json")
    validation = read(warmup_dir / "validation/result.json")
    spec = read(warmup_dir / "warmup-spec.json")
    if completed["status"] != "warmup_sft_experiment_completed" or trained["status"] != "supervised_warmup_completed" or trained["rl_training"]:
        raise ValueError("Expected the verified SFT warmup baseline")
    adapter = Path(trained["final_adapter"]["path"])
    if digest(adapter / "adapter_model.safetensors") != config["parent_adapter_sha256"] or validation["adapter_sha256"] != config["parent_adapter_sha256"]:
        raise ValueError("Warmup adapter differs")
    dataset = assets / "data" / config["dataset_folder"]
    manifest = read(dataset / "manifest.json")
    if manifest["dataset_id"] != config["dataset_id"] or digest(dataset / "train-cases.jsonl") != config["train_cases_sha256"]:
        raise ValueError("Train v2 data changed")
    for name, sha in manifest["files"].items():
        if digest(dataset / name) != sha:
            raise ValueError("Warmup dataset file changed: " + name)
    for name in ("dataset_tasks.py", "evaluate_model.py", "restricted_python.py", "train_static_grpo.py"):
        if digest(Path(__file__).with_name(name)) != spec["source_hashes"][name]:
            raise ValueError("Frozen verifier or trainer changed: " + name)
    if digest(assets / "data/internal-v1/validation.jsonl") != validation["dataset_split_sha256"]:
        raise ValueError("Validation changed")
    cases = rows(dataset / "train-cases.jsonl")
    predictions = rows(probe / "parent/predictions.jsonl")
    policy_result = read(probe / "parent/result.json")
    if len(cases) != 48 or len(predictions) != 48 or len({case["problem_id"] for case in cases}) != 16:
        raise ValueError("Expected 48 Train cases / 16 problems")
    if policy_result["status"] != "probe_evaluation_completed" or policy_result["adapter_sha256"] != config["parent_adapter_sha256"] or policy_result["generation_config"] != validation["generation_config"]:
        raise ValueError("Current policy probe does not match warmup")
    if policy_result["subsets"]["curated_train"]["metrics"] != summarize(predictions, True, 48):
        raise ValueError("Current policy metrics do not match records")
    if {record["case_id"] for record in predictions} != {case["case_id"] for case in cases} or any(case["split"] != "train" for case in cases):
        raise ValueError("Train case mismatch")
    contexts = {}
    lookup = {case["case_id"]: case for case in cases}
    for case in cases:
        pid = case["problem_id"]
        if pid not in contexts:
            inputs = list(all_inputs(case["domain"]))
            reference = compile_trusted(case["reference"])
            contexts[pid] = {"task": case, "inputs": inputs, "gold": [canonical({"output": reference(**value)}) for value in inputs],
                             "input_positions": {canonical(value): index for index, value in enumerate(inputs)}, "policy_suites": {}, "known_signatures": set()}
        context = contexts[pid]
        curated = compile_trusted(case["buggy_code"])
        context["known_signatures"].add(behavior_signature([observed(curated, value) for value in context["inputs"]]))
    for record in predictions:
        case = lookup[record["case_id"]]
        reference, buggy = compile_trusted(case["reference"]), RestrictedFunction(case["buggy_code"], case["domain"])
        suite = {}
        for sample in record["samples"]:
            scored = evaluate_output(sample["raw_output"].strip(), case, reference, buggy)
            if scored.get("buggy_error") == "RestrictedCodeError":
                raise ValueError("Resource-limited policy verification")
            if not sample["terminated"]:
                scored.update(valid_format=False, correct_expected=False, bug_detected=False, reward=0.0)
            if any(scored[key] != sample[key] for key in ("valid_format", "correct_expected", "bug_detected", "reward")):
                raise ValueError("Current policy scores changed")
            if scored["correct_expected"]:
                test = scored["test"]
                suite[canonical(test["input"])] = test
        contexts[case["problem_id"]]["policy_suites"][case["case_id"]] = list(suite.values())
    configuration = read(Path(__file__).resolve().parents[1] / "configs/static-grpo-v1.json")
    configuration.update(experiment="post-warmup-grpo-r1", dataset_id=config["dataset_id"], steps=96, checkpoint_interval=48,
                         baseline_run=config["parent_run"] + "/validation", baseline_result_sha256=config["parent_validation_result_sha256"],
                         selection_rule=config["selection_rule"])
    parent = {"model_path": spec["model_path"], "dataset_id": config["dataset_id"], "train_split_sha256": config["train_cases_sha256"],
              "final_adapter": trained["final_adapter"], "trainable_parameters": trained["trainable_parameters"], "configuration": configuration}
    diagnostic = {"training_run": str(warmup_dir / "train"), "adapter_sha256": config["parent_adapter_sha256"]}
    return config, parent, validation, cases, contexts, diagnostic


def prepare_context(args):
    config, parent, validation, cases, contexts, diagnostic = baseline(args.assets, args.config)
    if args.diagnostic_run.resolve() != (args.assets / "evidence" / config["train_probe"]).resolve():
        raise ValueError("Sampler diagnostic path differs from frozen plan")
    return contexts, parent, diagnostic


def prepare(assets, config_path):
    config, parent, validation, cases, contexts, _ = baseline(assets, config_path)
    pool_dir = assets / "evidence" / config["pool_run"]
    if digest(pool_dir / "result.json") != config["pool_result_sha256"] or digest(pool_dir / "adversarial-cases.jsonl") != config["pool_cases_sha256"]:
        raise ValueError("Post-warmup adversarial pool changed")
    pool_result = read(pool_dir / "result.json")
    if pool_result["status"] != "adversary_pool_completed" or pool_result["protagonist_adapter_sha256"] != config["parent_adapter_sha256"]:
        raise ValueError("Expected a real nonempty pool filtered against this warmup")
    if pool_result["generated_candidates"] != 256 or pool_result["configuration"]["samples_per_problem"] != 16:
        raise ValueError("Sampler budget differs from plan")
    if pool_result["source_hashes"]["restricted_python.py"] != digest(Path(__file__).with_name("restricted_python.py")):
        raise ValueError("Pool interpreter changed")
    pool = rows(pool_dir / "adversarial-cases.jsonl")
    if len(pool) != pool_result["accepted_candidates"] or not pool or len({case["case_id"] for case in pool}) != len(pool):
        raise ValueError("Empty, duplicated or incomplete pool")
    seen = {pid: set() for pid in contexts}
    audits = []
    for candidate in pool:
        if candidate["split"] != "train" or candidate["problem_id"] not in contexts:
            raise ValueError("Non-Train candidate")
        context = contexts[candidate["problem_id"]]
        for key in ("reference", "specification", "domain", "split_group"):
            if canonical(candidate[key]) != canonical(context["task"][key]):
                raise ValueError("Candidate changed task: " + key)
        verdict = assess_candidate(candidate["buggy_code"], context, seen[candidate["problem_id"]])
        if not verdict["accepted"] or verdict["behavior_sha256"] != candidate["behavior_sha256"] or sorted(verdict["passes_policy_suites_for_cases"]) != sorted(candidate["origin_policy_cases"]):
            raise ValueError("Candidate re-audit failed")
        seen[candidate["problem_id"]].add(verdict["behavior_sha256"])
        audits.append({"case_id": candidate["case_id"], **verdict})
    arms = {"control": [], "adversarial": []}
    usage, mappings = Counter(), []
    for repeat in range(2):
        for case in cases:
            identity = case["case_id"] + f"/post-warmup-pass{repeat + 1}"
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
    audit = {"pool_candidates_reverified": len(audits), "pool_problems": len({case["problem_id"] for case in pool}),
             "finite_domain_checks": sum(item["domain_input_count"] for item in audits), "slots_per_arm": 96,
             "replaced_slots": sum(usage.values()), "distinct_adversary_cases_used": len(usage), "candidate_usage": dict(usage),
             "unused_pool_cases": [case["case_id"] for case in pool if case["case_id"] not in usage],
             "mappings": mappings, "candidate_audits_with_witnesses_not_for_model_prompt": audits}
    return config, parent, validation, arms, audit
