import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from check_local_model import Tee
from local_model_loader import load_qwen_local
from dataset_tasks import PROMPT_VERSIONS, canonical, compile_trusted, evaluate_output, model_prompt


def read_cases(args):
    root = args.assets / "data" / "internal-v1"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        if args.preflight:
            from build_dataset import build_records
            splits, audit = build_records()
            return splits[args.split], {"dataset_id": audit["dataset_id"], "files": {}}, "in-memory preflight only"
        raise FileNotFoundError("Build the dataset first using scripts/build_dataset.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = root / f"{args.split}.jsonl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest["files"][path.name]:
        raise RuntimeError("Dataset split hash differs from its manifest")
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({case["case_id"] for case in cases}) != len(cases) or any(case["split"] != args.split for case in cases):
        raise RuntimeError("Duplicate cases or incorrect split labels")
    return cases, manifest, digest


def summarize(records, complete_cases, available_cases):
    samples = [sample for row in records for sample in row["samples"]]
    total = len(samples)
    valid = sum(sample["valid_format"] for sample in samples)
    correct = sum(sample["correct_expected"] for sample in samples)
    attacks = sum(sample["bug_detected"] for sample in samples)
    detected = sum(any(sample["bug_detected"] for sample in row["samples"]) for row in records)
    return {
        "evaluated_cases": len(records), "available_cases": available_cases,
        "complete_split": complete_cases,
        "generated_tests": total, "valid_tests": valid, "correct_expected": correct,
        "bug_detecting_tests": attacks, "cases_detected_with_up_to_4_tests": detected,
        "valid_rate": valid / total if total else None,
        "io_accuracy": correct / total if total else None,
        "attack_rate": attacks / total if total else None,
        "empirical_case_detection_at_4": detected / len(records) if records else None,
        "metric_denominator": "All generated tests, including invalid and truncated tests, for per-test rates",
    }


def run(args, run_dir, result):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    cases, manifest, split_hash = read_cases(args)
    available = len(cases)
    if args.limit is not None:
        cases = cases[:args.limit]
    result.update(dataset_id=manifest["dataset_id"], split=args.split, dataset_split_sha256=split_hash,
                  available_cases=available, requested_cases=len(cases))
    result["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")}
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left")
    prompts = []
    for case in cases:
        messages = model_prompt(case, version=args.prompt_version)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
        if tokens > 768:
            raise RuntimeError(f'Prompt too long: {case["case_id"]}, {tokens} tokens')
        prompts.append((messages, rendered, tokens))
    result["prompt_token_range"] = [min(item[2] for item in prompts), max(item[2] for item in prompts)]
    if args.preflight:
        result["status"] = "preflight_passed_no_model_weights_loaded"
        print("PREFLIGHT OK:", len(cases), "cases; prompt tokens:", result["prompt_token_range"], flush=True)
        print("No model weights loaded, no generated tests, no Test-set model evaluation.")
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA and BF16 are required")
    result["gpu"] = torch.cuda.get_device_name(0)
    torch.cuda.reset_peak_memory_stats()
    print("GPU:", result["gpu"], flush=True)
    print("Load original model for baseline; no optimizer and no training", flush=True)
    started = time.perf_counter()
    model, loading = load_qwen_local(args.model, output_loading_info=True)
    result["loader"] = "local_mmap_state_dict"
    result["loading_info"] = loading
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise RuntimeError("Missing or mismatched original model weights")
    result["model_variant"] = "original_base"
    if args.adapter:
        from peft import PeftModel
        if (args.adapter.parent / "SMOKE_TEST_ONLY.txt").exists():
            raise RuntimeError("Do not evaluate the arithmetic smoke-test adapter as a test-generation model")
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False, local_files_only=True)
        result["model_variant"] = "base_plus_adapter"
        result["adapter_path"] = str(args.adapter)
        result["adapter_sha256"] = hashlib.sha256((args.adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    model.eval()
    torch.cuda.synchronize()
    result["load_seconds"] = round(time.perf_counter() - started, 3)
    declared_eos = model.generation_config.eos_token_id
    eos_ids = sorted({tokenizer.eos_token_id, *(declared_eos if isinstance(declared_eos, list) else [declared_eos])} - {None})
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    result["generation_config"] = {
        "tests_per_case": 4, "max_new_tokens": 64, "temperature": 1.0,
        "top_p": 0.95, "top_k": 20, "seed": args.seed,
        "seed_strategy": "seed plus SHA-256 prefix of case_id; reset for each case",
        "eos_token_ids": eos_ids, "dtype": "bfloat16", "thinking": False,
    }
    records = []
    evaluation_started = time.perf_counter()
    for index, (case, (messages, prompt, _)) in enumerate(zip(cases, prompts, strict=True), 1):
        print(f'CASE {index}/{len(cases)} | {case["case_id"]}', flush=True)
        seed = (args.seed + int(hashlib.sha256(case["case_id"].encode()).hexdigest()[:8], 16)) % 2**32
        torch.manual_seed(seed)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs, num_return_sequences=4, max_new_tokens=64, max_time=120,
                do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
                use_cache=True, pad_token_id=pad_id, eos_token_id=eos_ids,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        reference, buggy = compile_trusted(case["reference"]), compile_trusted(case["buggy_code"])
        samples = []
        for tokens in output[:, inputs["input_ids"].shape[1]:].tolist():
            end = next((position for position, token in enumerate(tokens) if token in eos_ids), None)
            terminated = end is not None
            actual_tokens = tokens[:end + 1] if terminated else tokens
            raw = tokenizer.decode(actual_tokens, skip_special_tokens=True).strip()
            scored = evaluate_output(raw, case, reference, buggy)
            scored.update(raw_output=raw, terminated=terminated, generated_tokens=len(actual_tokens))
            if not terminated:
                scored["unmasked_verification"] = {key: scored[key] for key in ("valid_format", "correct_expected", "bug_detected")}
                scored.update(valid_format=False, correct_expected=False, bug_detected=False, reward=0.0)
            samples.append(scored)
            print("  MODEL:", raw, "| valid:", scored["valid_format"], "| correct:", scored["correct_expected"], "| bug:", scored["bug_detected"], flush=True)
        record = {
            "case_id": case["case_id"], "problem_id": case["problem_id"],
            "split_group": case["split_group"], "generation_seconds": round(elapsed, 3),
            "case_seed": seed, "prompt_messages": messages, "samples": samples,
            "prompt_version": args.prompt_version,
            "prompt_sha256": hashlib.sha256(canonical(messages).encode("utf-8")).hexdigest(),
        }
        records.append(record)
        with (run_dir / "predictions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        partial = summarize(records, len(records) == available, available)
        (run_dir / "partial-metrics.json").write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
    result["evaluation_seconds"] = round(time.perf_counter() - evaluation_started, 3)
    result["metrics"] = summarize(records, len(records) == available, available)
    result["peak_torch_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    result["peak_torch_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    result["status"] = "evaluation_completed"
    print("VALIDATION EVALUATION COMPLETED" if args.split == "validation" else "TRAIN-SPLIT EVALUATION COMPLETED", flush=True)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print("Generation/evaluation time (s):", result["evaluation_seconds"], flush=True)
    print("Peak PyTorch allocated VRAM (GiB):", result["peak_torch_allocated_gib"], flush=True)
    print("No Test-set model evaluation was performed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate original local model or a saved adapter on the frozen internal dataset")
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-version", choices=PROMPT_VERSIONS, default="plain-v2")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    args.model = args.model or args.assets / "models" / "Qwen3.5-2B"
    for key, value in {
        "HF_HOME": str(args.assets / "cache" / "huggingface"), "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1", "TORCHINDUCTOR_CACHE_DIR": str(args.assets / "cache" / "torchinductor"),
        "TEMP": str(args.assets / "temp"), "TMP": str(args.assets / "temp"),
    }.items():
        os.environ[key] = value
    (args.assets / "temp").mkdir(parents=True, exist_ok=True)
    tag = "adapter" if args.adapter else "baseline"
    run_dir = args.assets / "evidence" / (f"{tag}-{args.split}-" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "started_at": datetime.now().astimezone().isoformat(), "status": "started",
        "training": False, "test_set_evaluation": False, "preflight_only": args.preflight,
        "model_path": str(args.model),
        "prompt_version": args.prompt_version,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task_source_sha256": hashlib.sha256(Path(__file__).with_name("dataset_tasks.py").read_bytes()).hexdigest(),
    }
    stdout, stderr = sys.stdout, sys.stderr
    exit_code = 0
    with (run_dir / "console.txt").open("w", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        try:
            print("Evidence:", run_dir, flush=True)
            run(args, run_dir, result)
        except KeyboardInterrupt:
            result["status"] = "interrupted"
            exit_code = 130
            print("Interrupted; completed predictions and partial metrics were preserved.")
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__, error=str(error))
            exit_code = 1
            traceback.print_exc()
        finally:
            result["finished_at"] = datetime.now().astimezone().isoformat()
            (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print("Saved evidence:", run_dir, flush=True)
            sys.stdout, sys.stderr = stdout, stderr
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
