import argparse
import copy
import json
import os
import shutil
import statistics
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from check_local_model import Tee
from post_warmup_data import prepare
from round1_data import digest, read, rows, save_curriculum
from run_post_warmup_round import compare


PROJECT = Path(__file__).resolve().parents[1]
ASSETS = Path(r"D:\DA\TTNT_De19_Assets")
RATES = ("valid_rate", "io_accuracy", "attack_rate", "empirical_case_detection_at_4")


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temp, path)


def launch(root, arm, phase, preflight=False):
    command = [sys.executable, "-B", str(PROJECT / "scripts/replicate_grpo_worker.py"), "--assets", str(ASSETS),
               "--root", str(root), "--arm", arm, "--phase", phase]
    if preflight:
        command.append("--preflight")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError("Worker failed: " + str(root) + " / " + phase + " / " + arm)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    result = read(root / (phase + "-" + arm) / "result.json")
    expected = ("post_warmup_grpo_preflight_passed_no_weights_loaded" if phase == "train" else "preflight_passed_no_model_weights_loaded") if preflight else ("post_warmup_grpo_training_completed" if phase == "train" else "evaluation_completed")
    if result["status"] != expected:
        raise ValueError("Unexpected worker result: " + result["status"])
    return result


def verify_validation(folder):
    from dataset_tasks import canonical, compile_trusted
    from evaluate_model import summarize
    from probe_round1_learning import score
    from restricted_python import RestrictedFunction
    lookup = {case["case_id"]: case for case in rows(ASSETS / "data/internal-v1/validation.jsonl")}
    records = rows(folder / "predictions.jsonl")
    assert len(records) == 18 and {r["case_id"] for r in records} == set(lookup)
    for record in records:
        case = lookup[record["case_id"]]
        reference, buggy = compile_trusted(case["reference"]), RestrictedFunction(case["buggy_code"], case["domain"])
        assert len(record["samples"]) == 4
        for sample in record["samples"]:
            actual = score(sample["raw_output"], sample["terminated"], case, reference, buggy)
            for key in ("valid_format", "correct_expected", "bug_detected", "reward"):
                assert actual[key] == sample[key], (record["case_id"], key)
            for key in ("gold_output", "buggy_output", "buggy_error"):
                assert canonical(actual.get(key)) == canonical(sample.get(key))
    expected = read(folder / "result.json")["metrics"]
    assert summarize(records, True, 18) == expected
    return {"outputs_rescored": 72, "metrics_match": True}


def run(root, result, preflight_only):
    latest = read(PROJECT / "LATEST_POST_WARMUP_RL.json")
    original = Path(latest["primary_experiment"])
    original_spec = read(original / "post-warmup-spec.json")
    assert read(original / "result.json")["status"] == "post_warmup_grpo_experiment_completed"
    for name, sha in original_spec["source_hashes"].items():
        assert digest(PROJECT / "scripts" / name) == sha, name
    config, parent, parent_validation, arms, audit = prepare(ASSETS, PROJECT / "configs/post-warmup-training-r1.json")
    sources = root / "source"
    sources.mkdir()
    names = list(original_spec["source_hashes"]) + [Path(__file__).name, "replicate_grpo_worker.py", "replicate_evaluator.py", "local_model_loader.py"]
    hashes = {}
    for name in dict.fromkeys(names):
        path = PROJECT / "scripts" / name
        hashes[name] = digest(path)
        shutil.copy2(path, sources / name)
    plan = {"created_at": datetime.now().astimezone().isoformat(), "new_training_seeds": [43, 44], "reused_training_seed": 42,
            "decoding_seed": 42, "steps_per_arm": 96, "arms": ["control", "adversarial"], "parent_adapter_sha256": original_spec["parent_adapter_sha256"],
            "parent_and_pool_fixed": True, "pool_cases_sha256": config["pool_cases_sha256"], "checkpoint_selection": "final only",
            "new_optimizer_in_every_arm": True, "source_experiment": str(original), "source_result_sha256": digest(original / "result.json"),
            "source_hashes": hashes, "change_scope": "GRPO torch/trainer/data seeds plus mmap loader; unchanged reward, curriculum, budget and held-out evaluator",
            "interpretation": "Conditional GRPO training-seed sensitivity from one fixed SFT parent and one frozen adversarial pool; not independent full-pipeline reproductions",
            "test_set_model_evaluation": False}
    write(root / "plan.json", plan)
    data_hashes = {name: digest(ASSETS / "data/internal-v1" / (name + ".jsonl")) for name in ("train", "validation", "test")}
    result.update(plan=plan, dataset_hashes_before=data_hashes, completed_stages=[], paired_comparisons={})

    def prepare_seed(folder, seed):
        folder.mkdir()
        curricula = save_curriculum(folder, arms, audit)
        assert curricula == original_spec["curriculum_hashes"]
        spec = copy.deepcopy(original_spec)
        spec["source_hashes"] = hashes
        spec["training_config"]["seed"] = seed
        spec["round_config"]["seed"] = seed
        spec["replication_scope"] = plan["interpretation"]
        old_control = read(original / "train-control/result.json")["final_adapter"]
        spec["loader_equivalence_adapter_path"] = old_control["path"]
        spec["loader_equivalence_adapter_sha256"] = old_control["adapter_sha256"]
        write(folder / "post-warmup-spec.json", spec)
        return spec

    preflight = root / "preflight"
    prepare_seed(preflight, 43)
    for arm, phase in (("control", "train"), ("adversarial", "train"), ("control", "evaluate")):
        print("STAGE: preflight", phase, arm, flush=True)
        launch(preflight, arm, phase, True)
    result["preflight_passed"] = True
    write(root / "progress.json", result)
    if preflight_only:
        result["status"] = "replicate_preflight_passed_no_training"
        return
    print("STAGE: verify mmap loader against all 72 saved Validation outputs", flush=True)
    launch(preflight, "control", "recheck")
    old_records = rows(original / "evaluate-control/predictions.jsonl")
    rechecked = rows(preflight / "recheck-control/predictions.jsonl")
    assert [r["case_id"] for r in old_records] == [r["case_id"] for r in rechecked]
    assert [[s["raw_output"] for s in r["samples"]] for r in old_records] == [[s["raw_output"] for s in r["samples"]] for r in rechecked]
    result["mmap_loader_equivalence"] = {"same_adapter_sha256": True, "identical_raw_outputs": 72, "fresh_training": False}
    write(root / "loader-equivalence.json", result["mmap_loader_equivalence"])
    for seed in (43, 44):
        folder = root / ("seed-" + str(seed))
        spec = prepare_seed(folder, seed)
        training, validation = {}, {}
        for arm in ("control", "adversarial"):
            for phase, target in (("train", training), ("evaluate", validation)):
                print(f"STAGE: seed {seed} | {phase} {arm}", flush=True)
                result["active_stage"] = {"training_seed": seed, "phase": phase, "arm": arm}
                write(root / "progress.json", result)
                target[arm] = launch(folder, arm, phase)
                result["completed_stages"].append(f"seed-{seed}/{phase}-{arm}")
                if phase == "evaluate":
                    verification = verify_validation(folder / ("evaluate-" + arm))
                    write(folder / ("evaluate-" + arm) / "independent-score-check.json", verification)
                else:
                    assert target[arm]["configuration"]["seed"] == seed
                    assert target[arm]["parent_adapter_exact_match"] and target[arm]["reward_driven_update_verified"]
                    adapter = Path(target[arm]["final_adapter"]["path"])
                    assert digest(adapter / "adapter_model.safetensors") == target[arm]["final_adapter"]["adapter_sha256"]
                write(root / "progress.json", result)
        comparison = compare(folder, spec, parent_validation, training, validation)
        comparison.pop("seed", None)
        comparison.update(training_seed=seed, decoding_seed=42, interpretation=plan["interpretation"])
        write(folder / "comparison.json", comparison)
        result["paired_comparisons"][str(seed)] = comparison
    all_metrics = {"42": latest["primary_comparison"]["metrics"]}
    for seed, comparison in result["paired_comparisons"].items():
        all_metrics[seed] = comparison["metrics"]
    aggregate = {}
    for arm in ("control", "adversarial"):
        aggregate[arm] = {key: {"mean": statistics.mean(values := [all_metrics[str(s)][arm][key] for s in (42, 43, 44)]),
                               "sample_std": statistics.stdev(values), "per_training_seed": values} for key in RATES}
    paired = {key: [100 * (all_metrics[str(s)]["adversarial"][key] - all_metrics[str(s)]["control"][key]) for s in (42, 43, 44)] for key in RATES}
    data_after = {name: digest(ASSETS / "data/internal-v1" / (name + ".jsonl")) for name in data_hashes}
    assert data_after == data_hashes
    for name, sha in hashes.items():
        assert digest(PROJECT / "scripts" / name) == sha, name
    result.update(status="grpo_training_seed_replicates_completed", rl_training=True, training_seeds=[42, 43, 44], decoding_seed=42,
                  aggregate=aggregate, paired_differences_pp=paired, all_metrics=all_metrics,
                  new_optimizer_steps=384, new_generated_training_samples=1536, new_validation_outputs_rescored=288,
                  dataset_hashes_after=data_after, active_stage=None)
    write(root / "comparison.json", {key: result[key] for key in ("training_seeds", "decoding_seed", "aggregate", "paired_differences_pp", "all_metrics")})
    write(PROJECT / "LATEST_GRPO_REPLICATES.json", {"path": str(root), "status": result["status"], "scope": plan["interpretation"],
          "training_seeds": [42, 43, 44], "decoding_seed": 42, "test_set_model_evaluation": False, "demo_model_selection_changed": False})
    print("TRAINING SEED REPLICATION COMPLETED:", json.dumps(aggregate, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    for key, value in {"HF_HOME": str(ASSETS / "cache/huggingface"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "TEMP": str(ASSETS / "temp"), "TMP": str(ASSETS / "temp"),
        "WANDB_MODE": "disabled", "TORCHINDUCTOR_CACHE_DIR": str(ASSETS / "cache/torchinductor")}.items():
        os.environ[key] = value
    root = ASSETS / "evidence" / (("grpo-replicates-preflight-" if args.preflight else "grpo-training-seeds-") + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    root.mkdir()
    result = {"status": "started", "started_at": datetime.now().astimezone().isoformat(), "rl_training": False, "test_set_model_evaluation": False, "preflight_only": args.preflight}
    out, err, code = sys.stdout, sys.stderr, 0
    with (root / "console.txt").open("w", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(out, log), Tee(err, log)
        try:
            print("Evidence:", root, flush=True)
            run(root, result, args.preflight)
        except BaseException as error:
            result.update(status="failed", error_type=type(error).__name__, error=str(error))
            traceback.print_exc()
            code = 1
        finally:
            result["finished_at"] = datetime.now().astimezone().isoformat()
            write(root / "result.json", result)
            print("Saved evidence:", root, flush=True)
            sys.stdout, sys.stderr = out, err
    return code


if __name__ == "__main__":
    raise SystemExit(main())
