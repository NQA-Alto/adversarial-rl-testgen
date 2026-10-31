import argparse
import json
import os
import shutil
import statistics
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ASSETS = Path(r"D:\DA\TTNT_De19_Assets")
os.environ.update(TTNT_APPS_DATA_DIR=str(ASSETS / "data/apps-introductory-expanded-v4"), TTNT_MODEL_PROFILE="2b",
                  PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                  HF_HOME=str(ASSETS / "cache/huggingface"), TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"))
sys.path.insert(0, str(PROJECT / "scripts"))
from apps_adversarial_round import curriculum, record
from apps_adversary import audit_pool, rows
from apps_core import DATA, canonical, digest, load_cases, read, write
from apps_model_runner import run_folder
from apps_round import run as execute
from apps_sampling import make_schedule
from audit_apps_run import audit


def mean_metrics(records):
    metrics = [read(Path(item["path"]) / "result.json")["metrics"] for item in records]
    return {key: {"mean": statistics.mean(item[key] for item in metrics),
                  "sd": statistics.stdev(item[key] for item in metrics)}
            for key in ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")}


def main(args):
    old = args.parent_study
    previous = read(old / "result.json")
    parent = Path(previous["plan"]["parent_adapter"]) if "plan" in previous else Path(read(old / "plan.json")["parent_adapter"])
    original_plan = read(old / "plan.json")
    parent_validation = Path(original_plan["parent_validation"])
    state = read(PROJECT / "APPS_STATE.json")
    os.environ["TTNT_APPS_STATE_PATH"] = state["selected_local_model"]["state_path"]
    root = run_folder("balanced-current-curriculum")
    print("EVIDENCE:", root, flush=True)
    write(ASSETS / "evidence/APPS_BALANCED_STUDY.json", {"path": str(root)})
    cases, manifest = load_cases("train")
    guard_files = (DATA / "test.jsonl", ASSETS / "evidence/APPS_FINAL_TEST_LOCK.json", PROJECT / "APPS_STATE.json")
    guards = {str(path): digest(path) for path in guard_files}
    source = root / "source"
    source.mkdir()
    names = ("apps_model_runner.py", "apps_sampling.py", "apps_core.py", "apps_adversary.py", "apps_adversarial_round.py",
             "apps_curriculum.py", "apps_rewards.py", "restricted_python.py", "audit_apps_run.py", "model_profiles.py",
             "apps_lora.py", "local_model_loader.py", "train_static_grpo.py")
    for name in names:
        shutil.copy2(PROJECT / "scripts" / name, source / name)
    shutil.copy2(Path(__file__), source / Path(__file__).name)
    plan = {"dataset_id": manifest["dataset_id"], "train_sha256": manifest["splits"]["train"]["sha256"],
            "parent_adapter": str(parent), "parent_adapter_sha256": digest(parent / "adapter_model.safetensors"),
            "parent_validation": str(parent_validation), "model_profile": "2b", "prompt_version": "apps-call-v2",
            "training_seeds": [42, 43], "generation_seeds": [42, 43, 44], "rounds": 2, "steps_per_round": 64,
            "confirmation_rule": "Run seed43 only if seed42 adaptive adds at least1 mean detected code vs SFT with nondecreasing expected, or adds at least3 expected with nondecreasing code detection, or adds >=1/3 detected code vs matched control with nondecreasing expected. Otherwise stop the pilot and retain SFT.",
            "revision": "Probe the current curriculum; replace solved slots only and retain unsolved code.",
            "learning_rate": 5e-6, "reward_profile": "equal-v1", "group_size": 4,
            "sampling": "50% groups on eligible adversarial slots, 50% on remaining Train slots; exact same case-ID schedule in control",
            "adversary": "Frozen local Qwen3.5-2B; same audited SFT840 pool in round1, resample from each adaptive policy in round2",
            "comparison": "Same parent, slots, schedule, reward, steps and learning rate. Only the eligible buggy implementation differs.",
            "selection": "Keep SFT unless adaptive adds >=1 mean detected code/15 with nondecreasing IO or >=3 correct expected/60 with nondecreasing code detection. Validity may fall at most1/60. Compare adaptive against matched control separately.",
            "evaluation_scope": "Exploratory Validation after the previous final Test event. Old Test is not rerun or used for this study.",
            "guards": guards, "source_hashes": {name: digest(source / name) for name in names}}
    write(root / "plan.json", plan)
    progress = {"status": "running", "plan_sha256": digest(root / "plan.json"), "seeds": [], "model_test_evaluation": False}
    write(root / "result.json", progress)
    python = str(ASSETS / "env/Scripts/python.exe")
    runner = [python, "-B", str(PROJECT / "scripts/apps_model_runner.py")]
    def stage(command, label):
        return execute(command, root, label)
    def evaluate(adapter, label, seed=42, split="validation", curriculum_path=None):
        command = runner + ["evaluate", "--split", split, "--adapter", str(adapter), "--seed", str(seed),
                            "--prompt-version", "apps-call-v2", "--no-publish"]
        if curriculum_path:
            command += ["--curriculum", str(curriculum_path), "--curriculum-sha256", digest(curriculum_path)]
        folder = stage(command, label)
        audit(folder)
        return folder
    def guard():
        for path, expected in guards.items():
            if digest(Path(path)) != expected:
                raise RuntimeError("Protected final evaluation or active state changed: " + path)
        for name, expected in plan["source_hashes"].items():
            if digest(PROJECT / "scripts" / name) != expected:
                raise RuntimeError("Source changed during study: " + name)
    try:
        for seed in plan["training_seeds"]:
            seed_root = root / f"seed-{seed}"
            seed_root.mkdir()
            parents = {arm: parent for arm in ("control", "adaptive")}
            validations = {arm: parent_validation for arm in parents}
            seed_record = {"seed": seed, "rounds": []}
            active_curriculum = None
            for number in (1, 2):
                guard()
                if number == 1:
                    old_stages = read(old / "stages.json")
                    probe, pool = Path(old_stages["r1-train-probe"]), Path(old_stages["r1-adversary"])
                else:
                    probe = evaluate(parents["adaptive"], f"s{seed}-r{number}-train-probe", split="train", curriculum_path=active_curriculum)
                    pool = stage([python, "-B", str(PROJECT / "scripts/apps_adversary.py"), "--probe", str(probe),
                                  "--parent-adapter", str(parents["adaptive"]), "--round", str(number),
                                  "--samples-per-problem", "8"], f"s{seed}-r{number}-adversary")
                audit_pool(pool)
                if read(pool / "result.json")["parent_adapter_sha256"] != digest(parents["adaptive"] / "adapter_model.safetensors"):
                    raise ValueError("Pool belongs to another policy")
                curricula = curriculum(seed_root, number, pool, active_curriculum)
                adaptive_rows = rows(Path(curricula["adaptive"]["path"]))
                focused = [row["case_id"] for row in adaptive_rows if row["curriculum_source"] == "local_adversary"]
                schedule = make_schedule([row["case_id"] for row in cases], focused, 64, seed)
                schedule_path = seed_root / f"round-{number}-schedule.json"
                write(schedule_path, schedule)
                info = {"round": number, "probe": str(probe), "pool": str(pool), "focus_slots": len(focused),
                        "schedule_sha256": digest(schedule_path), "arms": {}}
                for arm in ("control", "adaptive"):
                    guard()
                    trained = stage(runner + ["train", "--steps", "64", "--seed", str(seed), "--learning-rate", "0.000005",
                        "--init-adapter", str(parents[arm]), "--baseline-run", str(validations[arm]),
                        "--prompt-version", "apps-call-v2", "--reward-profile", "equal-v1", "--no-publish",
                        "--curriculum", curricula[arm]["path"], "--curriculum-sha256", curricula[arm]["sha256"],
                        "--schedule", str(schedule_path), "--schedule-sha256", digest(schedule_path)], f"s{seed}-r{number}-{arm}-train")
                    review = audit(trained)
                    if not review["policy_update_verified"]:
                        raise RuntimeError("No verified parameter update")
                    validation = evaluate(trained / "adapter", f"s{seed}-r{number}-{arm}-validation42")
                    events, gradients = rows(trained / "rollouts.jsonl"), read(trained / "gradients.json")
                    exposure = sum(len(event["samples"]) for event in events if event["case_id"] in focused)
                    signal = sum(event["policy_signal"] and gradients[event["optimizer_step_before"]]["norm"] > 0
                                 for event in events if event["case_id"] in focused)
                    if exposure != 128:
                        raise RuntimeError("Balanced exposure not realized")
                    info["arms"][arm] = {"training": record(trained, True), "validation": record(validation),
                        "focus_test_exposures": exposure, "focus_groups_with_nonzero_update": signal,
                        "adversarial_test_exposures": exposure if arm == "adaptive" else 0}
                    parents[arm], validations[arm] = trained / "adapter", validation
                    write(root / "progress.json", {"seed": seed, "round": number, "arm_completed": arm, "stage": info})
                active_curriculum = Path(curricula["adaptive"]["path"])
                seed_record["rounds"].append(info)
                write(seed_root / "result.json", seed_record)
            seed_record["validation"] = {}
            for arm in parents:
                records = [record(validations[arm])] + [record(evaluate(parents[arm], f"s{seed}-{arm}-validation{value}", value))
                                                       for value in (43, 44)]
                seed_record["validation"][arm] = {"runs": records, "summary": mean_metrics(records)}
            progress["seeds"].append(seed_record)
            write(seed_root / "result.json", seed_record)
            write(root / "result.json", progress)
            if seed == 42:
                adaptive = seed_record["validation"]["adaptive"]["summary"]
                control = seed_record["validation"]["control"]["summary"]
                sft = previous["summary"]["sft"]
                a = lambda key: adaptive[key]["mean"]
                c = lambda key: control[key]["mean"]
                b = lambda key: sft[key]["mean"]
                promising = (a("cases_detected") >= b("cases_detected") + 1 - 1e-9 and a("correct_expected") >= b("correct_expected") - 1e-9 or
                             a("correct_expected") >= b("correct_expected") + 3 - 1e-9 and a("cases_detected") >= b("cases_detected") - 1e-9 or
                             a("cases_detected") >= c("cases_detected") + 1/3 - 1e-9 and a("correct_expected") >= c("correct_expected") - 1e-9)
                progress["pilot_promising_for_confirmation"] = promising
                if not promising:
                    progress["confirmation_skipped_reason"] = "Pilot has no predeclared meaningful adaptive advantage; retain SFT."
                    break
        guard()
        summary = {arm: mean_metrics([run for seed in progress["seeds"] for run in seed["validation"][arm]["runs"]])
                   for arm in ("control", "adaptive")}
        sft = read(old / "result.json")["summary"]["sft"]
        mean = lambda arm, key: summary[arm][key]["mean"]
        base = lambda key: sft[key]["mean"]
        eligible = mean("adaptive", "valid_tests") >= base("valid_tests") - 1 and (
            mean("adaptive", "cases_detected") >= base("cases_detected") + 1 - 1e-9 and mean("adaptive", "correct_expected") >= base("correct_expected") - 1e-9 or
            mean("adaptive", "correct_expected") >= base("correct_expected") + 3 - 1e-9 and mean("adaptive", "cases_detected") >= base("cases_detected") - 1e-9)
        progress.update(status="completed", summary=summary, sft_validation=sft, adaptive_passes_selection_gate=eligible,
                        selected_variant="adaptive" if eligible else "sft", total_adversarial_test_exposures=256 * len(progress["seeds"]),
                        final_test_unchanged=True, active_demo_unchanged=True)
        write(root / "result.json", progress)
        print("BALANCED STUDY COMPLETED:", root, canonical(summary), flush=True)
    except BaseException as error:
        progress.update(status="failed", error=repr(error))
        write(root / "result.json", progress)
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-study", type=Path, required=True)
    main(parser.parse_args())
