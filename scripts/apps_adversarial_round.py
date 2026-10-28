import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from apps_adversary import audit_pool, rows
from apps_core import ASSETS, canonical, digest, load_cases, read, write
from apps_curriculum import training_cases
from apps_model_runner import PROJECT, render_prompt, run_folder
from apps_round import run
from audit_apps_run import audit


def curriculum(root, round_number, pool):
    from transformers import AutoTokenizer
    cases, _ = load_cases("train")
    tokenizer = AutoTokenizer.from_pretrained(ASSETS / "models/Qwen3.5-2B", local_files_only=True)
    accepted = rows(pool / "adversarial-cases.jsonl")
    adaptive, usage = [], Counter()
    for case in cases:
        choices = []
        for candidate in accepted:
            if case["case_id"] not in candidate["gate"]["passes_policy_cases"]:
                continue
            trial = {**case, "buggy_code": candidate["buggy_code"]}
            if len(tokenizer.encode(render_prompt(tokenizer, trial, "apps-call-v2"), add_special_tokens=False)) <= 1536:
                choices.append(candidate)
        row = {**case, "curriculum_source": "static_curated"}
        if choices:
            chosen = min(choices, key=lambda value: (usage[value["candidate_id"]], value["candidate_id"]))
            row.update(buggy_code=chosen["buggy_code"], curriculum_source="local_adversary",
                       candidate_id=chosen["candidate_id"], pool_sha256=digest(pool / "adversarial-cases.jsonl"))
            usage[chosen["candidate_id"]] += 1
        adaptive.append(row)
    output = {}
    for arm, data in (("control", [{**case, "curriculum_source": "static_curated"} for case in cases]), ("adaptive", adaptive)):
        path = root / f'round-{round_number}-{arm}-curriculum.jsonl'
        path.write_text("".join(canonical(row) + "\n" for row in data), encoding="utf-8", newline="\n")
        checked, _, counts = training_cases(path, digest(path))
        output[arm] = {"path": str(path), "sha256": digest(path), "counts": counts}
    return output


def record(folder, training=False):
    output = {"path": str(folder), "result_sha256": digest(folder / "result.json")}
    if training:
        output.update(adapter_path=str(folder / "adapter"), adapter_sha256=digest(folder / "adapter/adapter_model.safetensors"))
    return output


def main(args):
    root = run_folder("adaptive-rounds")
    print("EVIDENCE:", root, flush=True)
    os.environ.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                      HF_HOME=str(ASSETS / "cache/huggingface"), TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"))
    previous = read(PROJECT / "APPS_STATE.json")
    write(root / "previous-state.json", previous)
    cases, manifest = load_cases("train")
    parent = Path(previous["sft_training"]["adapter_path"])
    parent_validation = Path(previous["sft_validation"]["path"])
    for key in ("sft_training", "sft_validation"):
        if digest(Path(previous[key]["path"]) / "result.json") != previous[key]["result_sha256"]:
            raise ValueError("Parent evidence changed")
    source = root / "source"
    source.mkdir()
    names = ("apps_core.py", "apps_model_runner.py", "apps_adversary.py", "apps_adversarial_round.py", "apps_curriculum.py",
             "audit_apps_run.py", "apps_profiles.py", "restricted_python.py", "local_model_loader.py", "train_static_grpo.py", "check_lora_update.py", "apps_round.py")
    hashes = {}
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), source / name)
        hashes[name] = digest(source / name)
    plan = {"dataset_id": manifest["dataset_id"], "train_sha256": manifest["splits"]["train"]["sha256"],
            "rounds": 2, "steps_per_round_per_arm": args.steps_per_round, "seed": 42,
            "samples_per_problem": 16, "prompt_version": "apps-call-v2", "source_hashes": hashes,
            "parent_adapter_sha256": digest(parent / "adapter_model.safetensors"),
            "control": "Curated static code, same 41 case slots, two optimizer restarts",
            "adaptive": "Refresh locally sampled code between rounds, replace only original cases caught by the current policy",
            "adversary": "Frozen original local Qwen3.5-2B; samples correct solutions then filters naturally incorrect solutions",
            "selection": "Final checkpoint at the declared budget, no checkpoint selection by Validation, no Test model evaluation",
            "initial_probe": str(args.initial_probe), "initial_pool": str(args.initial_pool),
            "initial_pool_sha256": digest(args.initial_pool / "adversarial-cases.jsonl")}
    write(root / "plan.json", plan)
    result = {"status": "running", "plan": plan, "rounds": [], "model_test_evaluation": False}
    write(root / "result.json", result)
    python = str(ASSETS / "env/Scripts/python.exe")
    runner = [python, "-B", str(PROJECT / "scripts/apps_model_runner.py")]
    parents = {arm: parent for arm in ("control", "adaptive")}
    validations = {arm: parent_validation for arm in parents}
    try:
        for number in (1, 2):
            if number == 1:
                probe, pool = args.initial_probe, args.initial_pool
            else:
                probe = run(runner + ["evaluate", "--split", "train", "--adapter", str(parents["adaptive"]),
                                     "--prompt-version", "apps-call-v2", "--no-publish"], root, f'r{number}-train-probe')
                audit(probe)
                pool = run([python, "-B", str(PROJECT / "scripts/apps_adversary.py"), "--probe", str(probe),
                            "--parent-adapter", str(parents["adaptive"]), "--round", str(number), "--samples-per-problem", "16"], root, f'r{number}-adversary')
            audit_pool(pool)
            pool_result = read(pool / "result.json")
            if pool_result["parent_adapter_sha256"] != digest(parents["adaptive"] / "adapter_model.safetensors"):
                raise ValueError("Adversary pool was not filtered against this round's parent")
            curricula = curriculum(root, number, pool)
            if number == 1 and not curricula["adaptive"]["counts"].get("local_adversary", 0):
                raise RuntimeError("No eligible locally generated adversarial code; refusing to label static training adversarial")
            stage = {"round": number, "probe": str(probe), "pool": str(pool), "curricula": curricula, "arms": {}}
            for arm in ("control", "adaptive"):
                train = run(runner + ["train", "--steps", str(args.steps_per_round), "--init-adapter", str(parents[arm]),
                                     "--baseline-run", str(validations[arm]), "--prompt-version", "apps-call-v2", "--no-publish",
                                     "--curriculum", curricula[arm]["path"], "--curriculum-sha256", curricula[arm]["sha256"]], root, f'r{number}-{arm}-train')
                audit(train)
                trained = read(train / "result.json")
                if not trained["update_verified"]:
                    raise ValueError("No verified RL update")
                validation = run(runner + ["evaluate", "--adapter", str(train / "adapter"), "--prompt-version", "apps-call-v2", "--no-publish"], root, f'r{number}-{arm}-validation')
                audit(validation)
                rollouts = rows(train / "rollouts.jsonl")
                adaptive_ids = {row["case_id"] for row in rows(Path(curricula[arm]["path"])) if row["curriculum_source"] == "local_adversary"}
                exposure = sum(len(event["samples"]) for event in rollouts if event["case_id"] in adaptive_ids)
                stage["arms"][arm] = {"training": record(train, True), "validation": record(validation),
                                      "adversarial_generated_test_exposures": exposure,
                                      "metrics": read(validation / "result.json")["metrics"]}
                parents[arm], validations[arm] = train / "adapter", validation
                write(root / "progress.json", {"round": number, "arm_completed": arm, "stage": stage})
            comparison = []
            for folder in (parent_validation, *validations.values()):
                comparison.append({row["case_id"]: (row["prompt_sha256"], row["case_seed"]) for row in rows(folder / "predictions.jsonl")})
            if not comparison[0] == comparison[1] == comparison[2]:
                raise ValueError("Validation prompts or case seeds differ")
            result["rounds"].append(stage)
            write(root / "result.json", result)
        for name, expected in hashes.items():
            if digest(Path(__file__).with_name(name)) != expected:
                raise ValueError("Source changed during experiment: " + name)
        state = dict(previous)
        state["adversarial_round_path"] = str(root)
        for arm, prefix in (("control", "paired_control"), ("adaptive", "adversarial")):
            state[prefix + "_training"] = result["rounds"][-1]["arms"][arm]["training"]
            state[prefix + "_validation"] = result["rounds"][-1]["arms"][arm]["validation"]
        result.update(status="completed", comparison_prompt_and_seed_match=True,
                      total_steps_per_arm=2 * args.steps_per_round,
                      adversarial_generated_test_exposures=sum(stage["arms"]["adaptive"]["adversarial_generated_test_exposures"] for stage in result["rounds"]))
        write(root / "result.json", result)
        write(PROJECT / "APPS_STATE.json", state)
        print("ADAPTIVE ROUNDS COMPLETED:", canonical({arm: stage["metrics"] for arm, stage in result["rounds"][-1]["arms"].items()}), flush=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        write(root / "result.json", result)
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-probe", type=Path, required=True)
    parser.add_argument("--initial-pool", type=Path, required=True)
    parser.add_argument("--steps-per-round", type=int, default=48)
    args = parser.parse_args()
    if not 1 <= args.steps_per_round <= 96:
        parser.error("Each round must use 1..96 optimizer steps")
    main(args)
