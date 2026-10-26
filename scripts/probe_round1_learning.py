import argparse
import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from check_local_model import Tee
from dataset_tasks import canonical, compile_trusted, evaluate_output, input_is_valid, model_prompt
from evaluate_model import summarize
from restricted_python import RestrictedFunction
from round1_data import digest, prepare, read, rows


ASSETS = Path(r"D:\DA\TTNT_De19_Assets")
ROUND = "round1-20261026-081733-163920"


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def failure_reason(sample, case):
    if not sample["terminated"]:
        return "truncated"
    try:
        value = json.loads(sample["raw_output"])
    except (ValueError, TypeError):
        return "invalid_json"
    if not isinstance(value, dict) or set(value) != {"input", "expected"}:
        return "invalid_schema"
    if not input_is_valid(value["input"], case["domain"]):
        return "input_outside_domain"
    if not sample["correct_expected"]:
        return "wrong_expected"
    return "bug_detected" if sample["bug_detected"] else "correct_but_no_bug"


def score(raw, terminated, case, reference, buggy):
    value = evaluate_output(raw, case, reference, buggy)
    if value.get("buggy_error") == "RestrictedCodeError":
        raise RuntimeError("Restricted verifier limit; do not count as detected bug")
    value.update(raw_output=raw, terminated=terminated)
    if not terminated:
        value.update(valid_format=False, correct_expected=False, bug_detected=False, reward=0.0)
    value["failure_category"] = failure_reason(value, case)
    return value


def worker(args, folder, result):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    spec = read(args.root / "probe-spec.json")
    for name, sha in spec["source_hashes"].items():
        if digest(Path(__file__).with_name(name)) != sha:
            raise ValueError("Probe source changed: " + name)
    cases_path = args.root / "probe-cases.jsonl"
    if digest(cases_path) != spec["cases_sha256"]:
        raise ValueError("Probe cases changed")
    cases = rows(cases_path)
    selected = spec["variants"][args.variant]
    adapter = Path(selected["adapter_path"])
    if digest(adapter / "adapter_model.safetensors") != selected["adapter_sha256"]:
        raise ValueError("Probe adapter changed")
    result.update(adapter_path=str(adapter), adapter_sha256=selected["adapter_sha256"], model_path=spec["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(spec["model_path"], local_files_only=True, padding_side="left")
    prompts = [tokenizer.apply_chat_template(model_prompt(case, version="plain-v2"), tokenize=False,
                add_generation_prompt=True, enable_thinking=False) for case in cases]
    lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    if max(lengths) > 768:
        raise RuntimeError("Prompt budget exceeded")
    result["prompt_token_range"] = [min(lengths), max(lengths)]
    if args.preflight:
        result["status"] = "probe_preflight_passed_no_weights_loaded"
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    torch.cuda.reset_peak_memory_stats()
    base, loading = Qwen3_5ForConditionalGeneration.from_pretrained(spec["model_path"], local_files_only=True,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa", output_loading_info=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        raise RuntimeError("Unexpected base loading result")
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True)
    model.eval()
    declared = model.generation_config.eos_token_id
    eos = sorted({tokenizer.eos_token_id, *(declared if isinstance(declared, list) else [declared])} - {None})
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    generation = {"tests_per_case": 4, "max_new_tokens": 64, "temperature": 1.0, "top_p": 0.95,
                  "top_k": 20, "seed": 42, "seed_strategy": "seed plus SHA-256 prefix of case_id; reset for each case",
                  "eos_token_ids": eos, "dtype": "bfloat16", "thinking": False}
    if generation != spec["generation_config"]:
        raise RuntimeError("Probe generation protocol changed")
    result.update(generation_config=generation, gpu=torch.cuda.get_device_name(0))
    records = []
    start = time.perf_counter()
    for index, (case, prompt) in enumerate(zip(cases, prompts, strict=True), 1):
        seed = (42 + int(hashlib.sha256(case["case_id"].encode()).hexdigest()[:8], 16)) % 2**32
        torch.manual_seed(seed)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        with torch.inference_mode():
            output = model.generate(**inputs, num_return_sequences=4, max_new_tokens=64, max_time=120,
                do_sample=True, temperature=1.0, top_p=0.95, top_k=20, use_cache=True, pad_token_id=pad, eos_token_id=eos)
        reference = compile_trusted(case["reference"])
        buggy = RestrictedFunction(case["buggy_code"], case["domain"])
        samples = []
        for tokens in output[:, inputs["input_ids"].shape[1]:].tolist():
            end = next((i for i, token in enumerate(tokens) if token in eos), None)
            actual = tokens[:end + 1] if end is not None else tokens
            raw = tokenizer.decode(actual, skip_special_tokens=True).strip()
            value = score(raw, end is not None, case, reference, buggy)
            value["generated_tokens"] = len(actual)
            samples.append(value)
        record = {"case_id": case["case_id"], "problem_id": case["problem_id"], "probe_subset": case["probe_subset"],
                  "case_seed": seed, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "samples": samples}
        records.append(record)
        with (folder / "predictions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f'PROBE {args.variant} {index}/{len(cases)} | {case["case_id"]} | hits={sum(s["bug_detected"] for s in samples)}/4', flush=True)
    result.update(status="probe_evaluation_completed", evaluation_seconds=round(time.perf_counter() - start, 3),
                  peak_torch_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3), subsets={})
    for subset in ("curated_train", "adversarial_pool"):
        selected_records = [record for record in records if record["probe_subset"] == subset]
        result["subsets"][subset] = {"metrics": summarize(selected_records, True, len(selected_records)),
            "failures": dict(collections.Counter(sample["failure_category"] for record in selected_records for sample in record["samples"]))}


def orchestrate(args, folder, result):
    config, parent, validation, arms, audit = prepare(args.assets, Path(__file__).resolve().parents[1] / "configs/adversarial-round1.json")
    round_folder = args.assets / "evidence" / ROUND
    completed = read(round_folder / "result.json")
    if completed["status"] != "round1_experiment_completed":
        raise ValueError("Round 1 incomplete")
    train_path = args.assets / "data/internal-v1/train.jsonl"
    cases = [dict(case, probe_subset="curated_train") for case in rows(train_path)]
    cases += [dict(case, probe_subset="adversarial_pool") for case in rows(args.assets / "evidence" / config["pool_run"] / "adversarial-cases.jsonl")]
    variants = {"parent": {"adapter_path": parent["final_adapter"]["path"], "adapter_sha256": parent["final_adapter"]["adapter_sha256"]}}
    for arm in ("control", "adversarial"):
        trained = read(round_folder / ("train-" + arm) / "result.json")
        if trained["status"] != "round1_training_completed" or not trained["rl_training"]:
            raise ValueError("Invalid trained adapter")
        variants[arm] = {"adapter_path": trained["final_adapter"]["path"], "adapter_sha256": trained["final_adapter"]["adapter_sha256"]}
    source_folder = folder / "source"
    source_folder.mkdir()
    names = (Path(__file__).name, "dataset_tasks.py", "evaluate_model.py", "check_local_model.py", "restricted_python.py", "round1_data.py", "build_adversarial_pool.py")
    hashes = {}
    for name in names:
        path = Path(__file__).with_name(name)
        shutil.copy2(path, source_folder / name)
        hashes[name] = digest(path)
    case_path = folder / "probe-cases.jsonl"
    case_path.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8", newline="\n")
    write(folder / "probe-spec.json", {"model_path": parent["model_path"], "variants": variants, "cases_sha256": digest(case_path),
          "generation_config": validation["generation_config"], "source_hashes": hashes, "source_round_result_sha256": digest(round_folder / "result.json")})
    result.update(source_round=str(round_folder), source_hashes=hashes, variants=variants, results={})
    for variant in variants:
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--assets", str(args.assets), "--root", str(folder), "--variant", variant]
        if args.preflight:
            command.append("--preflight")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            if process.wait():
                raise RuntimeError("Probe failed: " + variant)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise
        value = read(folder / variant / "result.json")
        result["results"][variant] = value
    result["status"] = "probe_preflight_passed" if args.preflight else "round1_learning_probe_completed"
    if not args.preflight:
        compact = {name: value["subsets"] for name, value in result["results"].items()}
        write(folder / "probe-comparison.json", compact)
        print("LEARNING PROBE COMPARISON:", json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--variant", choices=("parent", "control", "adversarial"))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if bool(args.root) != bool(args.variant):
        parser.error("root and variant required together")
    for key, value in {"HF_HOME": str(args.assets / "cache/huggingface"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1",
        "TEMP": str(args.assets / "temp"), "TMP": str(args.assets / "temp")}.items():
        os.environ[key] = value
    folder = args.root / args.variant if args.variant else args.assets / "evidence" / (("probe-preflight-" if args.preflight else "round1-learning-probe-") + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    folder.mkdir(parents=True, exist_ok=False)
    result = {"started_at": datetime.now().astimezone().isoformat(), "status": "started", "training": False,
              "test_set_evaluation": False, "preflight_only": args.preflight, "scope": "Seen Train problems and adaptive Train pool only; not held-out quality"}
    out, err = sys.stdout, sys.stderr
    code = 0
    with (folder / "console.txt").open("w", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(out, log), Tee(err, log)
        try:
            print("Evidence:", folder, flush=True)
            (worker if args.variant else orchestrate)(args, folder, result)
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
