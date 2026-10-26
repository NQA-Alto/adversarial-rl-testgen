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
from round1_data import digest, read


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def invoke(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError("Warmup child process failed; inspect its console.txt")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        raise


def evaluate(args, folder, result):
    import evaluate_model
    spec = read(args.root / "warmup-spec.json")
    for name, sha in spec["source_hashes"].items():
        if digest(Path(__file__).with_name(name)) != sha:
            raise ValueError("Warmup source changed")
    trained = read(args.root / "train/result.json")
    adapter = None if args.preflight else Path(trained["final_adapter"]["path"])
    if not args.preflight and (trained["status"] != "supervised_warmup_completed" or trained["rl_training"]):
        raise ValueError("Expected actual supervised warmup, not RL or smoke test")
    if adapter and digest(adapter / "adapter_model.safetensors") != trained["final_adapter"]["adapter_sha256"]:
        raise ValueError("SFT adapter changed")
    result.update(model_path=spec["model_path"], training_run=str(args.root / "train"), method="SFT_NOT_RL")
    evaluate_model.run(argparse.Namespace(assets=args.assets, model=Path(spec["model_path"]), adapter=adapter,
        split="validation", seed=42, prompt_version="plain-v2", limit=None, preflight=args.preflight), folder, result)
    if not args.preflight and (result["generation_config"] != spec["generation_config"] or result["dataset_split_sha256"] != spec["validation_sha256"]):
        raise ValueError("Validation protocol changed")


def orchestrate(args, folder, result):
    config_path = Path(__file__).resolve().parents[1] / "configs/warmup-sft-v2.json"
    config = read(config_path)
    evidence = args.assets / "evidence"
    parent_result_path = evidence / config["parent_run"] / "result.json"
    validation_result_path = evidence / config["parent_validation_run"] / "result.json"
    if digest(parent_result_path) != config["parent_result_sha256"] or digest(validation_result_path) != config["parent_validation_result_sha256"]:
        raise ValueError("Frozen parent results changed")
    parent, validation = read(parent_result_path), read(validation_result_path)
    parent_adapter = Path(parent["final_adapter"]["path"])
    if digest(parent_adapter / "adapter_model.safetensors") != parent["final_adapter"]["adapter_sha256"]:
        raise ValueError("Frozen parent adapter changed")
    dataset = args.assets / "data" / config["dataset_folder"]
    manifest = read(dataset / "manifest.json")
    audit = read(dataset / "audit.json")
    if manifest["dataset_id"] != config["dataset_id"] or manifest["validation_sha256"] != validation["dataset_split_sha256"]:
        raise ValueError("Warmup dataset or Validation differs")
    if config["epochs"] != 1 or config["gradient_accumulation"] != 4 or audit["sft_train_samples"] != 285 or audit["sft_dev_samples"] != 96:
        raise ValueError("Unexpected preplanned warmup budget")
    for name, sha in manifest["files"].items():
        if digest(dataset / name) != sha:
            raise ValueError("Warmup dataset changed: " + name)
    for name in ("dataset_tasks.py", "evaluate_model.py", "check_lora_update.py", "train_static_grpo.py"):
        if digest(Path(__file__).with_name(name)) != parent["source_hashes"][name]:
            raise ValueError("Frozen helper changed: " + name)
    source_dir = folder / "source"
    source_dir.mkdir()
    names = (Path(__file__).name, "warmup_sft_worker.py", "build_warmup_v2.py", "round1_data.py", "build_adversarial_pool.py",
             "restricted_python.py", "dataset_tasks.py", "evaluate_model.py", "check_local_model.py", "check_lora_update.py",
             "train_static_grpo.py", "run_grpo_pilot.py", "pilot_tasks.py")
    hashes = {}
    for name in names:
        source = Path(__file__).with_name(name)
        shutil.copy2(source, source_dir / name)
        hashes[name] = digest(source)
    shutil.copy2(config_path, source_dir / config_path.name)
    for name in ("manifest.json", "audit.json"):
        shutil.copy2(dataset / name, folder / ("dataset-" + name))
    spec = {"configuration": config, "dataset_path": str(dataset), "dataset_id": manifest["dataset_id"],
            "dataset_file_hashes": manifest["files"], "source_hashes": hashes,
            "parent_adapter_path": str(parent_adapter), "parent_adapter_sha256": parent["final_adapter"]["adapter_sha256"],
            "model_path": parent["model_path"], "generation_config": validation["generation_config"], "validation_sha256": validation["dataset_split_sha256"]}
    write(folder / "warmup-spec.json", spec)
    result.update(configuration=config, dataset_id=manifest["dataset_id"], source_hashes=hashes,
                  config_sha256=digest(config_path), parent_run=config["parent_run"], method="SFT_NOT_RL")
    train_command = [sys.executable, "-B", str(Path(__file__).with_name("warmup_sft_worker.py")), "--assets", str(args.assets), "--root", str(folder)]
    evaluation_command = [sys.executable, "-B", str(Path(__file__).resolve()), "--assets", str(args.assets), "--root", str(folder), "--evaluate-worker"]
    if args.preflight:
        train_command.append("--preflight")
        evaluation_command.append("--preflight")
    print("SUPERVISED WARMUP: 285 Train targets | 96 SFT Dev targets | 1 pass | 72 updates | NOT RL", flush=True)
    if args.reuse_trained_run:
        previous_spec = read(args.reuse_trained_run / "warmup-spec.json")
        trained = read(args.reuse_trained_run / "train/result.json")
        for key in ("configuration", "dataset_id", "dataset_file_hashes", "parent_adapter_sha256", "model_path"):
            if previous_spec[key] != spec[key]:
                raise ValueError("Cannot reuse a different training protocol: " + key)
        for name, sha in spec["source_hashes"].items():
            if name != Path(__file__).name and previous_spec["source_hashes"][name] != sha:
                raise ValueError("Training implementation changed: " + name)
        if trained["status"] != "supervised_warmup_completed" or trained["optimizer_steps"] != 72 or trained["examples_seen"] != 285:
            raise ValueError("Cannot reuse incomplete SFT")
        if digest(Path(trained["final_adapter"]["path"]) / "adapter_model.safetensors") != trained["final_adapter"]["adapter_sha256"]:
            raise ValueError("Reused adapter changed")
        shutil.copytree(args.reuse_trained_run / "train", folder / "train")
        write(folder / "training-reuse-provenance.json", {"source_run": str(args.reuse_trained_run),
              "source_spec_sha256": digest(args.reuse_trained_run / "warmup-spec.json"),
              "source_training_result_sha256": digest(args.reuse_trained_run / "train/result.json"),
              "training_rerun": False, "reason": "Repeat evaluation after fixing serialization of model loading metadata; training and evaluator algorithms unchanged"})
        result["training_reused_from"] = str(args.reuse_trained_run)
        print("REUSE completed SFT adapter; no new optimizer steps", flush=True)
    else:
        invoke(train_command)
    invoke(evaluation_command)
    if args.preflight:
        if read(folder / "train/result.json")["status"] != "sft_preflight_passed_no_weights_loaded" or read(folder / "validation/result.json")["status"] != "preflight_passed_no_model_weights_loaded":
            raise ValueError("Preflight did not complete")
        result["status"] = "warmup_preflight_passed"
        return
    trained, evaluated = read(folder / "train/result.json"), read(folder / "validation/result.json")
    if trained["status"] != "supervised_warmup_completed" or trained["optimizer_steps"] != 72 or trained["examples_seen"] != 285:
        raise ValueError("Supervised training budget differs")
    if evaluated["status"] != "evaluation_completed" or not evaluated["metrics"]["complete_split"]:
        raise ValueError("Validation incomplete")
    metrics = {"parent_static_72": validation["metrics"], "after_sft_new_train": evaluated["metrics"]}
    delta = {key: round(100 * (evaluated["metrics"][key] - validation["metrics"][key]), 4) for key in
             ("valid_rate", "io_accuracy", "attack_rate", "empirical_case_detection_at_4")}
    comparison = {"metrics": metrics, "delta_percentage_points": delta, "same_validation_protocol": True,
                  "sft_dev_loss": {"before": trained["dev_before"], "after": trained["dev_after"]},
                  "attribution": "This run changes training method and Train coverage together: new-data SFT warmup. Not an RL gain and not an isolated estimate of adding string tasks.",
                  "next_rl_baseline": trained["final_adapter"], "single_seed": True, "test_set_evaluation": False}
    write(folder / "warmup-comparison.json", comparison)
    result.update(status="warmup_sft_experiment_completed", warmup_comparison=comparison)
    print("WARMUP COMPARISON:", json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--evaluate-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reuse-trained-run", type=Path)
    args = parser.parse_args()
    if bool(args.root) != args.evaluate_worker:
        parser.error("root is for the evaluation worker only")
    if args.reuse_trained_run and (args.preflight or args.evaluate_worker):
        parser.error("Training reuse is for actual orchestration only")
    if args.reuse_trained_run and not args.reuse_trained_run.resolve().is_relative_to((args.assets / "evidence").resolve()):
        raise ValueError("Expected reused training inside evidence")
    for key, value in {"HF_HOME": str(args.assets / "cache/huggingface"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "TOKENIZERS_PARALLELISM": "false",
        "TEMP": str(args.assets / "temp"), "TMP": str(args.assets / "temp")}.items():
        os.environ[key] = value
    if args.evaluate_worker and not args.root.resolve().is_relative_to((args.assets / "evidence").resolve()):
        raise ValueError("Expected evidence root")
    folder = args.root / "validation" if args.evaluate_worker else args.assets / "evidence" / (("warmup-preflight-" if args.preflight else "warmup-sft-v2-") + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    folder.mkdir(parents=True, exist_ok=False)
    result = {"started_at": datetime.now().astimezone().isoformat(), "status": "started", "training": False, "rl_training": False,
              "test_set_evaluation": False, "preflight_only": args.preflight}
    out, err = sys.stdout, sys.stderr
    code = 0
    with (folder / "console.txt").open("w", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(out, log), Tee(err, log)
        try:
            print("Evidence:", folder, flush=True)
            (evaluate if args.evaluate_worker else orchestrate)(args, folder, result)
            print("STATUS:", result["status"], flush=True)
        except KeyboardInterrupt:
            result["status"], code = "interrupted", 130
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__, error=str(error))
            traceback.print_exc()
            code = 1
        finally:
            result["finished_at"] = datetime.now().astimezone().isoformat()
            write(folder / "result.json", result)
            print("Saved evidence:", folder, flush=True)
            sys.stdout, sys.stderr = out, err
    return code


if __name__ == "__main__":
    raise SystemExit(main())
