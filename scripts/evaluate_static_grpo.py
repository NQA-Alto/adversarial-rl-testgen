import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from check_local_model import Tee
import evaluate_model


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_training(args):
    evidence = args.assets / "evidence"
    if args.run is None:
        pointer = json.loads((evidence / "latest-static-training.json").read_text(encoding="utf-8"))
        args.run = Path(pointer["run_dir"])
        if digest(args.run / "result.json") != pointer["result_sha256"]:
            raise ValueError("Latest training result hash differs from the saved pointer")
    run = args.run.resolve()
    if not run.is_relative_to(evidence.resolve()):
        raise ValueError("Select a training run inside the project's evidence directory")
    trained = json.loads((run / "result.json").read_text(encoding="utf-8"))
    if trained["status"] != "static_grpo_completed" or not trained["rl_training"] or trained["synthetic_cpu_self_check"]:
        raise ValueError("Select a completed, verified real static GRPO run")
    adapter = Path(trained["final_adapter"]["path"])
    if digest(adapter / "adapter_model.safetensors") != trained["final_adapter"]["adapter_sha256"]:
        raise ValueError("Adapter changed after training")
    baseline_path = Path(trained["baseline_result_path"])
    if digest(baseline_path) != trained["configuration"]["baseline_result_sha256"]:
        raise ValueError("Frozen baseline result changed")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    source = Path(__file__).with_name("dataset_tasks.py")
    if digest(source) != baseline["task_source_sha256"]:
        raise ValueError("Verifier or prompt source differs from the frozen baseline")
    return trained, adapter, baseline


def main():
    parser = argparse.ArgumentParser(description="Evaluate the completed static GRPO adapter with the frozen baseline protocol")
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    for key, value in {
        "HF_HOME": str(args.assets / "cache" / "huggingface"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(args.assets / "cache" / "torchinductor"),
        "TEMP": str(args.assets / "temp"), "TMP": str(args.assets / "temp"),
    }.items():
        os.environ[key] = value
    (args.assets / "temp").mkdir(parents=True, exist_ok=True)
    run_dir = args.assets / "evidence" / ("static-validation-" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {"started_at": datetime.now().astimezone().isoformat(), "status": "started", "training": False,
              "test_set_evaluation": False, "preflight_only": args.preflight, "prompt_version": "plain-v2",
              "wrapper_sha256": digest(Path(__file__)), "evaluator_sha256": digest(Path(__file__).with_name("evaluate_model.py")),
              "task_source_sha256": digest(Path(__file__).with_name("dataset_tasks.py"))}
    stdout, stderr = sys.stdout, sys.stderr
    exit_code = 0
    with (run_dir / "console.txt").open("w", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        try:
            trained, adapter, baseline = check_training(args)
            result["training_run"] = str(args.run)
            print("Training run:", args.run, flush=True)
            print("Adapter:", adapter, flush=True)
            evaluation_args = argparse.Namespace(
                assets=args.assets, model=Path(trained["model_path"]), adapter=adapter,
                split="validation", seed=baseline["generation_config"]["seed"],
                prompt_version=trained["prompt_version"], limit=None, preflight=args.preflight,
            )
            evaluate_model.run(evaluation_args, run_dir, result)
            if not args.preflight:
                if result["dataset_id"] != baseline["dataset_id"] or result["dataset_split_sha256"] != baseline["dataset_split_sha256"]:
                    raise ValueError("Validation data differs from baseline")
                if result["generation_config"] != baseline["generation_config"]:
                    raise ValueError("Generation budget/config differs from baseline; do not compare as an identical protocol")
                if not result["metrics"]["complete_split"]:
                    raise ValueError("Validation evaluation is incomplete")
                keys = ("valid_rate", "io_accuracy", "attack_rate", "empirical_case_detection_at_4")
                comparison = {"baseline_run": trained["configuration"]["baseline_run"], "training_run": str(args.run),
                              "protocol_matches": True, "seed": evaluation_args.seed,
                              "baseline": baseline["metrics"], "after_static_grpo": result["metrics"],
                              "delta_percentage_points": {key: round((result["metrics"][key] - baseline["metrics"][key]) * 100, 4) for key in keys},
                              "scope": "One seed on internal Validation; not final Test results or proof of general improvement"}
                (run_dir / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
                result["comparison"] = comparison
                print("BASELINE VS STATIC GRPO (percentage points):", flush=True)
                print(json.dumps(comparison["delta_percentage_points"], indent=2))
        except KeyboardInterrupt:
            result["status"] = "interrupted"; exit_code = 130
            print("Interrupted; completed predictions remain available.")
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__, error=str(error)); exit_code = 1
            traceback.print_exc()
        finally:
            result["finished_at"] = datetime.now().astimezone().isoformat()
            (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print("Saved evidence:", run_dir, flush=True)
            sys.stdout, sys.stderr = stdout, stderr
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
