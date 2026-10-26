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


RATES = ("valid_rate", "io_accuracy", "attack_rate", "empirical_case_detection_at_4")


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def worker(args, folder, result):
    import evaluate_model
    spec = read(args.root / "sensitivity-spec.json")
    for name, expected in spec["source_hashes"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("Sensitivity evaluator changed")
    variant = spec["variants"][args.variant]
    adapter = Path(variant["adapter_path"])
    if digest(adapter / "adapter_model.safetensors") != variant["adapter_sha256"]:
        raise ValueError("Sensitivity adapter changed")
    result.update(variant=args.variant, model_path=spec["model_path"], seed=args.seed, method="decoding_sensitivity_not_training_replicates")
    evaluate_model.run(argparse.Namespace(assets=args.assets, model=Path(spec["model_path"]), adapter=adapter,
        split="validation", seed=args.seed, prompt_version="plain-v2", limit=None, preflight=False), folder, result)
    expected = {**spec["generation_config"], "seed": args.seed}
    if result["generation_config"] != expected or result["dataset_split_sha256"] != spec["validation_sha256"]:
        raise ValueError("Secondary evaluation changed another protocol setting")


def orchestrate(args, folder, result):
    source = read(args.run / "result.json")
    training_spec = read(args.run / "post-warmup-spec.json")
    plan = read(args.run / "preplanned-decoding-sensitivity.json")
    if source["status"] != "post_warmup_grpo_experiment_completed" or plan["secondary_decoding_seeds"] != [43, 44]:
        raise ValueError("Expected completed experiment and preplanned seeds")
    config = training_spec["round_config"]
    parent_validation = args.assets / "evidence" / config["parent_validation_run"] / "result.json"
    if digest(parent_validation) != config["parent_validation_result_sha256"]:
        raise ValueError("Primary warmup baseline changed")
    variants = {"parent": {"adapter_path": training_spec["parent_adapter_path"], "adapter_sha256": training_spec["parent_adapter_sha256"], "primary_result": str(parent_validation)}}
    for arm in ("control", "adversarial"):
        trained = read(args.run / ("train-" + arm) / "result.json")
        if trained["status"] != "post_warmup_grpo_training_completed" or trained["optimizer_steps"] != 96:
            raise ValueError("Expected preplanned final GRPO adapter")
        variants[arm] = {"adapter_path": trained["final_adapter"]["path"], "adapter_sha256": trained["final_adapter"]["adapter_sha256"],
                         "primary_result": str(args.run / ("evaluate-" + arm) / "result.json")}
    source_dir = folder / "source"
    source_dir.mkdir()
    hashes = {}
    for name in (Path(__file__).name, "evaluate_model.py", "dataset_tasks.py", "check_local_model.py"):
        current = Path(__file__).with_name(name)
        shutil.copy2(current, source_dir / name)
        hashes[name] = digest(current)
    spec = {"variants": variants, "source_hashes": hashes, "model_path": training_spec["model_path"],
            "generation_config": training_spec["generation_config"], "validation_sha256": training_spec["validation_split_sha256"],
            "source_experiment_sha256": digest(args.run / "result.json"), "preplanned_decoding_plan_sha256": digest(args.run / "preplanned-decoding-sensitivity.json")}
    write(folder / "sensitivity-spec.json", spec)
    result.update(source_experiment=str(args.run), seeds=[42, 43, 44], primary_seed=42, results={})
    for variant, details in variants.items():
        primary = read(Path(details["primary_result"]))
        if primary["generation_config"] != spec["generation_config"] or primary["adapter_sha256"] != details["adapter_sha256"]:
            raise ValueError("Primary result is not comparable")
        result["results"][variant] = {"42": primary["metrics"]}
    for seed in plan["secondary_decoding_seeds"]:
        for variant in variants:
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--assets", str(args.assets), "--root", str(folder),
                       "--variant", variant, "--seed", str(seed)]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                if process.wait():
                    raise RuntimeError("Secondary decoding evaluation failed")
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                raise
            evaluated = read(folder / (str(seed) + "-" + variant) / "result.json")
            if evaluated["status"] != "evaluation_completed" or not evaluated["metrics"]["complete_split"]:
                raise ValueError("Secondary decoding evaluation incomplete")
            result["results"][variant][str(seed)] = evaluated["metrics"]
    result["mean_per_seed_rates"] = {variant: {key: sum(m[key] for m in values.values()) / 3 for key in RATES} for variant, values in result["results"].items()}
    result["adversarial_minus_control_pp_by_seed"] = {str(seed): {key: round(100 * (result["results"]["adversarial"][str(seed)][key] - result["results"]["control"][str(seed)][key]), 4) for key in RATES} for seed in (42, 43, 44)}
    result["status"] = "post_warmup_decoding_sensitivity_completed"
    write(folder / "sensitivity-comparison.json", {key: result[key] for key in ("results", "mean_per_seed_rates", "adversarial_minus_control_pp_by_seed")})
    print("DECODING SENSITIVITY:", json.dumps(result["mean_per_seed_rates"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--variant", choices=("parent", "control", "adversarial"))
    parser.add_argument("--seed", type=int, choices=(43, 44))
    args = parser.parse_args()
    if args.root and not (args.variant and args.seed) or not args.root and not args.run:
        parser.error("Select main run or complete worker arguments")
    for path in (args.root, args.run):
        if path and not path.resolve().is_relative_to((args.assets / "evidence").resolve()):
            raise ValueError("Expected local experiment evidence")
    for key, value in {"HF_HOME": str(args.assets / "cache/huggingface"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1",
        "TEMP": str(args.assets / "temp"), "TMP": str(args.assets / "temp")}.items():
        os.environ[key] = value
    folder = args.root / (str(args.seed) + "-" + args.variant) if args.root else args.assets / "evidence" / ("post-warmup-decoding-" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    folder.mkdir(parents=True, exist_ok=False)
    result = {"started_at": datetime.now().astimezone().isoformat(), "status": "started", "training": False, "rl_training": False,
              "test_set_evaluation": False, "scope": "Secondary decoding-seed sensitivity on Validation; model training was run with a single seed"}
    out, err = sys.stdout, sys.stderr
    code = 0
    with (folder / "console.txt").open("w", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(out, log), Tee(err, log)
        try:
            print("Evidence:", folder, flush=True)
            (worker if args.root else orchestrate)(args, folder, result)
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
