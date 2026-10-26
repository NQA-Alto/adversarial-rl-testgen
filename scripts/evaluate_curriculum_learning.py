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
from round1_data import digest, read, rows


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args, output, result):
    trained_result = read(args.run / "result.json")
    if trained_result["status"] != "extended_curriculum_experiment_completed":
        raise ValueError("Complete the matched extended experiment first")
    spec = read(args.run / "extended-spec.json")
    before = read(args.baseline_probe / "result.json")
    before_spec = read(args.baseline_probe / "probe-spec.json")
    if before["status"] != "round1_learning_probe_completed" or before_spec["generation_config"] != spec["generation_config"]:
        raise ValueError("Expected completed baseline probe with matching decoding")
    case_path = args.baseline_probe / "probe-cases.jsonl"
    if digest(case_path) != before_spec["cases_sha256"]:
        raise ValueError("Baseline probe cases changed")
    candidates = rows(case_path)
    pool = args.assets / "evidence" / spec["round_config"]["pool_run"] / "adversarial-cases.jsonl"
    if digest(pool) != spec["round_config"]["pool_cases_sha256"]:
        raise ValueError("Adaptive pool changed")
    if {case["case_id"] for case in candidates if case["probe_subset"] == "adversarial_pool"} != {case["case_id"] for case in rows(pool)}:
        raise ValueError("Diagnostic pool differs from training pool")
    variants = {"parent": before_spec["variants"]["parent"]}
    if variants["parent"]["adapter_sha256"] != spec["parent_adapter_sha256"]:
        raise ValueError("Parent models differ")
    for arm in ("control", "adversarial"):
        trained = read(args.run / ("train-" + arm) / "result.json")
        if trained["status"] != "curriculum_training_completed" or not trained["rl_training"]:
            raise ValueError("Invalid training result")
        adapter = trained["final_adapter"]
        if digest(Path(adapter["path"]) / "adapter_model.safetensors") != adapter["adapter_sha256"]:
            raise ValueError("Trained adapter changed")
        variants[arm] = {"adapter_path": adapter["path"], "adapter_sha256": adapter["adapter_sha256"]}
    shutil.copy2(case_path, output / "probe-cases.jsonl")
    source_dir = output / "source"
    source_dir.mkdir()
    for name, sha in before_spec["source_hashes"].items():
        current = Path(__file__).with_name(name)
        if digest(current) != sha:
            raise ValueError("Probe implementation changed: " + name)
        shutil.copy2(current, source_dir / name)
    shutil.copy2(Path(__file__), source_dir / Path(__file__).name)
    new_spec = {**before_spec, "variants": variants, "source_round_result_sha256": digest(args.run / "result.json")}
    write(output / "probe-spec.json", new_spec)
    parent_folder = output / "parent"
    parent_folder.mkdir()
    for name in ("predictions.jsonl", "result.json"):
        shutil.copy2(args.baseline_probe / "parent" / name, parent_folder / name)
    write(parent_folder / "reuse-provenance.json", {"source_folder": str(args.baseline_probe / "parent"),
        "source_result_sha256": digest(args.baseline_probe / "parent/result.json"), "predictions_sha256": digest(parent_folder / "predictions.jsonl"),
        "same_cases_sha256": new_spec["cases_sha256"], "same_generation_config": True, "same_parent_adapter": True,
        "note": "Reuse exactly the same parent predictions, no new inference for this variant"})
    result.update(source_experiment=str(args.run), baseline_probe=str(args.baseline_probe), parent_predictions_reused=True,
                  executor_sha256=digest(Path(__file__)), results={"parent": read(parent_folder / "result.json")})
    for arm in ("control", "adversarial"):
        command = [sys.executable, "-B", str(Path(__file__).with_name("probe_round1_learning.py")), "--assets", str(args.assets),
                   "--root", str(output), "--variant", arm]
        print("COMMAND:", json.dumps(command), flush=True)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            if process.wait():
                raise RuntimeError("Diagnostic worker failed: " + arm)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise
        value = read(output / arm / "result.json")
        if value["status"] != "probe_evaluation_completed" or value["generation_config"] != new_spec["generation_config"]:
            raise ValueError("Diagnostic incomplete or protocol changed")
        if value["adapter_sha256"] != variants[arm]["adapter_sha256"]:
            raise ValueError("Diagnostic used another adapter")
        result["results"][arm] = value
    compact = {arm: value["subsets"] for arm, value in result["results"].items()}
    write(output / "probe-comparison.json", compact)
    result["status"] = "extended_learning_probe_completed"
    print("EXTENDED LEARNING COMPARISON:", json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline-probe", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.run.resolve(), args.baseline_probe.resolve()):
        if not path.is_relative_to((args.assets / "evidence").resolve()):
            raise ValueError("Expected paths inside project evidence")
    for key, value in {"HF_HOME": str(args.assets / "cache/huggingface"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1",
        "TEMP": str(args.assets / "temp"), "TMP": str(args.assets / "temp")}.items():
        os.environ[key] = value
    output = args.assets / "evidence" / ("extended-learning-probe-" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True, exist_ok=False)
    result = {"started_at": datetime.now().astimezone().isoformat(), "status": "started", "training": False,
              "test_set_evaluation": False, "scope": "Diagnostic on seen Train and adaptive Train pool, not held-out quality"}
    out, err = sys.stdout, sys.stderr
    code = 0
    with (output / "console.txt").open("w", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(out, log), Tee(err, log)
        try:
            print("Evidence:", output, flush=True)
            run(args, output, result)
            print("STATUS:", result["status"], flush=True)
        except KeyboardInterrupt:
            result["status"], code = "interrupted", 130
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__, error=str(error))
            traceback.print_exc()
            code = 1
        finally:
            result["finished_at"] = datetime.now().astimezone().isoformat()
            write(output / "result.json", result)
            print("Saved evidence:", output, flush=True)
            sys.stdout, sys.stderr = out, err
    return code


if __name__ == "__main__":
    raise SystemExit(main())
