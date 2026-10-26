import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

from check_local_model import Tee
from evaluate_static_grpo import check_training
import evaluate_model


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def check_inputs(args):
    trained, adapter, baseline = check_training(args)
    args.run = args.run.resolve()
    source = Path(__file__).parent
    for name in ("dataset_tasks.py", "evaluate_model.py"):
        if digest(source / name) != trained["source_hashes"][name]:
            raise ValueError("Frozen experiment source changed: " + name)
    if trained["model_path"] != baseline["model_path"] or trained["prompt_version"] != "plain-v2":
        raise ValueError("Original model or prompt does not match the frozen experiment")
    probe = argparse.Namespace(assets=args.assets, split="train", preflight=False)
    cases, manifest, split_hash = evaluate_model.read_cases(probe)
    if split_hash != trained["train_split_sha256"] or manifest["dataset_id"] != trained["dataset_id"]:
        raise ValueError("Train data changed since training")
    if len(cases) != 36 or len({case["problem_id"] for case in cases}) != 12:
        raise ValueError("Expected the complete frozen Train split: 12 problems, 36 cases")
    return trained, adapter, baseline


def error_categories(records):
    counts = Counter()
    for row in records:
        for sample in row["samples"]:
            if not sample["terminated"]:
                category = "truncated"
            elif not sample["valid_format"]:
                category = "invalid_json_schema_or_domain"
            elif not sample["correct_expected"]:
                category = "wrong_expected"
            elif not sample["bug_detected"]:
                category = "correct_but_no_bug_detected"
            else:
                category = "bug_detected"
            counts[category] += 1
    return dict(counts)


def compare_records(left, right, before, after, expected_generation):
    if before["status"] != "evaluation_completed" or after["status"] != "evaluation_completed":
        raise ValueError("Both evaluations must complete")
    for result in (before, after):
        if result["split"] != "train" or result["generation_config"] != expected_generation:
            raise ValueError("Expected Train with the frozen generation configuration")
        if result["metrics"] != evaluate_model.summarize(left if result is before else right, True, 36):
            raise ValueError("Saved metrics differ from prediction records")
    for key in ("dataset_id", "dataset_split_sha256", "prompt_version", "model_path", "evaluator_sha256", "task_source_sha256"):
        if before[key] != after[key]:
            raise ValueError("Comparison protocol differs: " + key)
    if before["model_variant"] != "original_base" or after["model_variant"] != "base_plus_adapter":
        raise ValueError("Expected original base versus trained adapter")
    if len(left) != 36 or len(right) != 36 or len({row["case_id"] for row in left}) != 36:
        raise ValueError("Expected all 36 unique Train cases")
    for a, b in zip(left, right, strict=True):
        for key in ("case_id", "problem_id", "case_seed", "prompt_sha256", "prompt_messages"):
            if a[key] != b[key]:
                raise ValueError("Case protocol differs: " + key)
        if len(a["samples"]) != 4 or len(b["samples"]) != 4:
            raise ValueError("Expected four tests per case")
    keys = ("valid_rate", "io_accuracy", "attack_rate", "empirical_case_detection_at_4")
    problems = sorted({row["problem_id"] for row in left})
    per_problem = []
    for problem in problems:
        a = [row for row in left if row["problem_id"] == problem]
        b = [row for row in right if row["problem_id"] == problem]
        per_problem.append({"problem_id": problem, "baseline": evaluate_model.summarize(a, True, len(a)),
                            "after_static_grpo": evaluate_model.summarize(b, True, len(b))})
    return {
        "protocol_matches": True, "split": "train", "baseline": before["metrics"], "after_static_grpo": after["metrics"],
        "delta_percentage_points": {key: round((after["metrics"][key] - before["metrics"][key]) * 100, 4) for key in keys},
        "error_categories": {"baseline": error_categories(left), "after_static_grpo": error_categories(right)},
        "per_problem": per_problem,
        "scope": "One-seed diagnostic on previously seen Train problems. Not held-out quality or Test results. Train/Validation differ in problem families; their raw gap alone does not prove overfitting.",
    }


def worker(args, output, result):
    trained, adapter, baseline = check_inputs(args)
    evaluation_args = argparse.Namespace(
        assets=args.assets, model=Path(trained["model_path"]),
        adapter=adapter if args.worker == "trained" else None,
        split="train", seed=baseline["generation_config"]["seed"],
        prompt_version=trained["prompt_version"], limit=None, preflight=args.preflight,
    )
    result.update(model_path=trained["model_path"], training_run=str(args.run),
                  evaluator_sha256=digest(Path(__file__).with_name("evaluate_model.py")),
                  task_source_sha256=digest(Path(__file__).with_name("dataset_tasks.py")))
    print("Evaluation stage:", args.worker, "| split: Train | no optimizer or training", flush=True)
    evaluate_model.run(evaluation_args, output, result)


def launch_worker(args, stage, output):
    command = [sys.executable, str(Path(__file__).resolve()), "--assets", str(args.assets), "--run", str(args.run),
               "--worker", stage, "--output", str(output)]
    if args.preflight:
        command.append("--preflight")
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            if process.wait() != 0:
                raise RuntimeError(stage + " evaluation failed; inspect its console.txt")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise


def main():
    parser = argparse.ArgumentParser(description="Compare original and trained local model on Train in separate processes")
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", choices=("original", "trained"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if bool(args.worker) != bool(args.output):
        parser.error("Worker and output must be supplied together")
    args.assets = args.assets.resolve()
    for key, value in {
        "HF_HOME": str(args.assets / "cache/huggingface"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(args.assets / "cache/torchinductor"),
        "TEMP": str(args.assets / "temp"), "TMP": str(args.assets / "temp"),
    }.items():
        os.environ[key] = value
    (args.assets / "temp").mkdir(parents=True, exist_ok=True)
    output = args.output or args.assets / "evidence" / ("train-diagnostic-" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True, exist_ok=False)
    result = {"started_at": datetime.now().astimezone().isoformat(), "status": "started", "training": False,
              "test_set_evaluation": False, "preflight_only": args.preflight,
              "script_sha256": digest(Path(__file__)), "prompt_version": "plain-v2"}
    original_stdout, original_stderr = sys.stdout, sys.stderr
    code = 0
    with (output / "console.txt").open("w", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(original_stdout, log), Tee(original_stderr, log)
        try:
            if args.worker:
                worker(args, output, result)
            else:
                trained, adapter, baseline = check_inputs(args)
                result.update(training_run=str(args.run), adapter_path=str(adapter),
                              adapter_sha256=digest(adapter / "adapter_model.safetensors"),
                              frozen_training_result_sha256=digest(args.run / "result.json"))
                source_dir = output / "source"
                source_dir.mkdir()
                for name in (Path(__file__).name, "evaluate_model.py", "evaluate_static_grpo.py", "dataset_tasks.py", "check_local_model.py"):
                    shutil.copy2(Path(__file__).with_name(name), source_dir / name)
                print("Evidence:", output, flush=True)
                print("Training run:", args.run, flush=True)
                print("[1/2] Evaluate original base on 36 Train cases", flush=True)
                launch_worker(args, "original", output / "original")
                print("[2/2] Evaluate trained adapter on the same 36 Train cases", flush=True)
                launch_worker(args, "trained", output / "trained")
                if args.preflight:
                    result["status"] = "preflight_passed_no_model_weights_loaded"
                    print("PREFLIGHT PASSED: both stages; no model weights loaded or predictions generated.", flush=True)
                else:
                    before, after = read_json(output / "original/result.json"), read_json(output / "trained/result.json")
                    if after["adapter_sha256"] != result["adapter_sha256"]:
                        raise ValueError("Evaluated adapter differs from the verified training adapter")
                    comparison = compare_records(read_rows(output / "original/predictions.jsonl"), read_rows(output / "trained/predictions.jsonl"), before, after, baseline["generation_config"])
                    comparison.update(training_run=str(args.run), seed=baseline["generation_config"]["seed"])
                    write_json(output / "comparison.json", comparison)
                    with (output / "per-problem.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(("problem_id", "variant", "generated_tests", "valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected_with_up_to_4_tests"))
                        for row in comparison["per_problem"]:
                            for variant in ("baseline", "after_static_grpo"):
                                m = row[variant]
                                writer.writerow((row["problem_id"], variant, m["generated_tests"], m["valid_tests"], m["correct_expected"], m["bug_detecting_tests"], m["cases_detected_with_up_to_4_tests"]))
                    result.update(status="train_diagnostic_completed", comparison=comparison)
                    print("TRAIN BEFORE VS AFTER GRPO:", flush=True)
                    print(json.dumps({key: comparison[key] for key in ("baseline", "after_static_grpo", "delta_percentage_points", "error_categories")}, indent=2), flush=True)
                    print("Seen Train data only. No Test-set model evaluation. No parameter updates.", flush=True)
                print("STATUS:", result["status"], flush=True)
        except KeyboardInterrupt:
            result["status"] = "interrupted"
            code = 130
            print("Interrupted; partial evidence is preserved.", flush=True)
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__, error=str(error))
            code = 1
            traceback.print_exc()
        finally:
            result["finished_at"] = datetime.now().astimezone().isoformat()
            write_json(output / "result.json", result)
            print("Saved evidence:", output, flush=True)
            sys.stdout, sys.stderr = original_stdout, original_stderr
    return code


if __name__ == "__main__":
    raise SystemExit(main())
