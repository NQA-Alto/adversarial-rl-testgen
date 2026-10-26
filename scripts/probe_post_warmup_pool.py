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
from evaluate_model import summarize
from round1_data import digest, read, rows


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run(args, folder, result):
    spec = read(args.run / "post-warmup-spec.json")
    completed = read(args.run / "result.json")
    if completed["status"] != "post_warmup_grpo_experiment_completed":
        raise ValueError("Complete paired post-warmup GRPO first")
    config = spec["round_config"]
    pool = args.assets / "evidence" / config["pool_run"]
    if digest(pool / "result.json") != config["pool_result_sha256"] or digest(pool / "adversarial-cases.jsonl") != config["pool_cases_sha256"]:
        raise ValueError("Adaptive pool changed")
    values = [dict(case, probe_subset="adversarial_pool") for case in rows(pool / "adversarial-cases.jsonl")]
    if not values or any(case["split"] != "train" for case in values):
        raise ValueError("Expected a nonempty Train-only pool")
    cases_path = folder / "probe-cases.jsonl"
    cases_path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8", newline="\n")
    variants = {"parent": {"adapter_path": spec["parent_adapter_path"], "adapter_sha256": spec["parent_adapter_sha256"]}}
    for arm in ("control", "adversarial"):
        trained = read(args.run / ("train-" + arm) / "result.json")
        if trained["status"] != "post_warmup_grpo_training_completed" or not trained["rl_training"]:
            raise ValueError("Expected completed real GRPO")
        variants[arm] = {"adapter_path": trained["final_adapter"]["path"], "adapter_sha256": trained["final_adapter"]["adapter_sha256"]}
    sources = folder / "source"
    sources.mkdir()
    names = (Path(__file__).name, "probe_round1_learning.py", "dataset_tasks.py", "evaluate_model.py", "check_local_model.py", "restricted_python.py", "round1_data.py", "build_adversarial_pool.py")
    hashes = {}
    for name in names:
        source = Path(__file__).with_name(name)
        hashes[name] = digest(source)
        shutil.copy2(source, sources / name)
    for value in variants.values():
        if digest(Path(value["adapter_path"]) / "adapter_model.safetensors") != value["adapter_sha256"]:
            raise ValueError("Diagnostic adapter changed")
    write(folder / "probe-spec.json", {"model_path": spec["model_path"], "variants": variants, "cases_sha256": digest(cases_path),
          "generation_config": spec["generation_config"], "source_hashes": hashes, "source_round_result_sha256": digest(args.run / "result.json"),
          "model_role": "SFT baseline versus matched GRPO arms on adaptive Train pool; not held-out quality"})
    usage = read(args.run / "curriculum-audit.json")["candidate_usage"]
    result.update(source_experiment=str(args.run), pool_run=str(pool), source_hashes=hashes, variants=variants, results={})
    for variant in variants:
        command = [sys.executable, "-B", str(Path(__file__).with_name("probe_round1_learning.py")), "--assets", str(args.assets),
                   "--root", str(folder), "--variant", variant]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            if process.wait():
                raise RuntimeError("Adaptive pool probe failed: " + variant)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise
        evaluated = read(folder / variant / "result.json")
        if evaluated["status"] != "probe_evaluation_completed" or evaluated["adapter_sha256"] != variants[variant]["adapter_sha256"] or evaluated["generation_config"] != spec["generation_config"]:
            raise ValueError("Probe protocol differs")
        records = rows(folder / variant / "predictions.jsonl")
        used = [record for record in records if record["case_id"] in usage]
        unused = [record for record in records if record["case_id"] not in usage]
        result["results"][variant] = {"all_adaptive_train_pool": summarize(records, True, len(values)),
            "used_in_adversarial_curriculum": summarize(used, True, len(usage)),
            "unused_adaptive_pool": summarize(unused, True, len(values) - len(usage)),
            "failure_categories": evaluated["subsets"]["adversarial_pool"]["failures"]}
    result["status"] = "post_warmup_pool_probe_completed"
    write(folder / "pool-comparison.json", result["results"])
    print("ADAPTIVE TRAIN POOL COMPARISON:", json.dumps(result["results"], ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if not args.run.resolve().is_relative_to((args.assets / "evidence").resolve()):
        raise ValueError("Expected experiment inside evidence")
    for key, value in {"HF_HOME": str(args.assets / "cache/huggingface"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1",
        "TEMP": str(args.assets / "temp"), "TMP": str(args.assets / "temp")}.items():
        os.environ[key] = value
    folder = args.assets / "evidence" / ("post-warmup-pool-probe-" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    folder.mkdir(parents=True, exist_ok=False)
    result = {"started_at": datetime.now().astimezone().isoformat(), "status": "started", "training": False, "rl_training": False,
              "test_set_evaluation": False, "scope": "Adaptive Train-pool diagnostic, not a generalization or Test result"}
    out, err = sys.stdout, sys.stderr
    code = 0
    with (folder / "console.txt").open("w", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(out, log), Tee(err, log)
        try:
            print("Evidence:", folder, flush=True)
            run(args, folder, result)
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
