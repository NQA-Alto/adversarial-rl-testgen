import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from check_local_model import Tee
from round1_data import digest, read, rows, save_curriculum
from extended_curriculum_data import prepare


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def launch(args, root, arm, phase):
    command = [sys.executable, "-B", str(Path(__file__).with_name("extended_curriculum_worker.py")), "--assets", str(args.assets),
               "--root", str(root), "--arm", arm, "--phase", phase]
    if args.preflight:
        command.append("--preflight")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    if code:
        raise RuntimeError(f"{phase}-{arm} failed with exit code {code}; see its console.txt")
    value = read(root / (phase + "-" + arm) / "result.json")
    expected = ("curriculum_preflight_passed_no_weights_loaded" if phase == "train" else "preflight_passed_no_model_weights_loaded") if args.preflight else ("curriculum_training_completed" if phase == "train" else "evaluation_completed")
    if value["status"] != expected:
        raise RuntimeError("Unexpected worker status: " + value["status"])
    return value


def compare(root, spec, parent_validation, training, validation):
    if training["control"]["configuration"] != training["adversarial"]["configuration"]:
        raise RuntimeError("Training configurations differ")
    group_orders = {}
    for arm in ("control", "adversarial"):
        item = training[arm]
        if item["optimizer_steps"] != 144 or item["groups"] != 144 or not item["parent_adapter_exact_match"] or not item["reward_driven_update_verified"]:
            raise RuntimeError("Training budget or starting adapter verification failed: " + arm)
        if item["parent_adapter_sha256"] != spec["parent_adapter_sha256"]:
            raise RuntimeError("Starting adapter hash differs")
        if item["origin_case_exposures"] != training["control"]["origin_case_exposures"]:
            raise RuntimeError("Unequal original-case exposure")
        group_orders[arm] = [group["case_id"] for group in rows(root / ("train-" + arm) / "rollouts.jsonl")]
        evaluation = validation[arm]
        if evaluation["generation_config"] != spec["generation_config"] or evaluation["dataset_split_sha256"] != spec["validation_split_sha256"]:
            raise RuntimeError("Unequal Validation protocol")
        if evaluation["adapter_sha256"] != item["final_adapter"]["adapter_sha256"]:
            raise RuntimeError("Validation used another adapter")
        if not evaluation["metrics"]["complete_split"] or evaluation["metrics"]["generated_tests"] != 72:
            raise RuntimeError("Incomplete Validation")
    if group_orders["control"] != group_orders["adversarial"] or len(set(group_orders["control"])) != 144:
        raise RuntimeError("Training slot order differs or contains repetitions")
    scores = {"parent_static_72": parent_validation["metrics"], **{arm: value["metrics"] for arm, value in validation.items()}}
    keys = ("valid_rate", "io_accuracy", "attack_rate", "empirical_case_detection_at_4")
    delta = {"adversarial_minus_" + name: {key: round(100 * (scores["adversarial"][key] - score[key]), 4) for key in keys}
             for name, score in scores.items() if name != "adversarial"}
    return {"metrics": scores, "delta_percentage_points": delta, "same_starting_adapter": True, "same_training_slot_order": True,
            "fresh_optimizer_in_both_arms": True, "steps_per_arm": 144, "generated_training_samples_per_arm": 576,
            "validation_cases": 18, "generated_validation_tests_per_arm": 72, "seed": 42,
            "interpretation": "Single-seed internal-data comparison, one curriculum refresh; no Test result or general improvement claim"}


def run(args, root, result):
    config_path = Path(__file__).resolve().parents[1] / "configs/adversarial-extended-v1.json"
    config, parent, parent_validation, arms, audit = prepare(args.assets, config_path)
    hashes = save_curriculum(root, arms, audit)
    sources = root / "source"
    sources.mkdir()
    names = ("run_extended_curriculum.py", "extended_curriculum_data.py", "round1_data.py", "extended_curriculum_worker.py", "build_adversarial_pool.py", "restricted_python.py",
             "dataset_tasks.py", "evaluate_model.py", "train_static_grpo.py", "check_local_model.py", "check_lora_update.py", "run_grpo_pilot.py", "pilot_tasks.py")
    source_hashes = {}
    for name in names:
        source = Path(__file__).with_name(name)
        shutil.copy2(source, sources / name)
        source_hashes[name] = digest(source)
    shutil.copy2(config_path, sources / config_path.name)
    spec = {"round_config": config, "training_config": parent["configuration"], "source_hashes": source_hashes,
            "parent_adapter_path": parent["final_adapter"]["path"], "parent_adapter_sha256": parent["final_adapter"]["adapter_sha256"],
            "parent_trainable_parameters": parent["trainable_parameters"], "model_path": parent["model_path"],
            "curriculum_hashes": hashes, "generation_config": parent_validation["generation_config"],
            "validation_split_sha256": parent_validation["dataset_split_sha256"]}
    write(root / "extended-spec.json", spec)
    result.update(source_hashes=source_hashes, config_sha256=digest(config_path), curriculum_hashes=hashes,
                  parent_run=config["parent_run"], parent_adapter_sha256=spec["parent_adapter_sha256"],
                  dataset_id=parent["dataset_id"], pool_run=config["pool_run"], pool_cases_sha256=config["pool_cases_sha256"],
                  curriculum_summary={key: audit[key] for key in ("pool_candidates_reverified", "pool_problems", "finite_domain_checks", "slots_per_arm", "replaced_slots", "distinct_adversary_cases_used", "unused_pool_cases")})
    print("POOL RE-AUDIT PASSED:", audit["pool_candidates_reverified"], "candidates |", audit["pool_problems"], "problems", flush=True)
    print("CURRICULUM: 144 slots/arm | replacements:", audit["replaced_slots"], "| distinct adversary cases used:", audit["distinct_adversary_cases_used"], flush=True)
    print("Same parent adapter; fresh optimizer; 144 GRPO steps per arm; Validation only", flush=True)
    training, validation = {}, {}
    result["completed_stages"] = []
    for arm in ("control", "adversarial"):
        for phase, target in (("train", training), ("evaluate", validation)):
            print("STAGE:", phase, arm, flush=True)
            target[arm] = launch(args, root, arm, phase)
            result["completed_stages"].append(phase + "-" + arm)
            write(root / "progress.json", result)
    if args.preflight:
        result["status"] = "extended_preflight_passed"
        print("PREFLIGHT PASSED: audit, both curricula, token budgets and Validation protocol; no model weights loaded", flush=True)
        return
    comparison = compare(root, spec, parent_validation, training, validation)
    result.update(status="extended_curriculum_experiment_completed", rl_training=True, extended_comparison=comparison)
    write(root / "extended-comparison.json", comparison)
    print("EXTENDED CURRICULUM COMPARISON", flush=True)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    for name, value in {"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "HF_HOME": str(args.assets / "cache/huggingface"),
                        "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "TEMP": str(args.assets / "temp"), "TMP": str(args.assets / "temp")}.items():
        os.environ[name] = value
    root = args.assets / "evidence" / (("extended-preflight-" if args.preflight else "extended-curriculum-") + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    root.mkdir(parents=True, exist_ok=False)
    result = {"started_at": datetime.now().astimezone().isoformat(), "status": "started", "rl_training": False,
              "preflight_only": args.preflight, "synthetic_cpu_self_check": False, "test_set_evaluation": False}
    old_out, old_err = sys.stdout, sys.stderr
    code = 0
    with (root / "console.txt").open("w", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(old_out, log), Tee(old_err, log)
        try:
            print("Evidence:", root, flush=True)
            run(args, root, result)
            print("STATUS:", result["status"], flush=True)
        except KeyboardInterrupt:
            result["status"], code = "interrupted", 130
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__, error=str(error))
            code = 1
            traceback.print_exc()
        finally:
            result["finished_at"] = datetime.now().astimezone().isoformat()
            write(root / "result.json", result)
            print("Saved evidence:", root, flush=True)
            sys.stdout, sys.stderr = old_out, old_err
    return code


if __name__ == "__main__":
    raise SystemExit(main())
