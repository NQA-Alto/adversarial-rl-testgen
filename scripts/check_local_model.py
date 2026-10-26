import argparse
import importlib.metadata
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


PROMPT = """Generate one test case for this Python function.
Specification: age is an integer from 0 to 120 inclusive. A person can register
if and only if age is at least 18. The function must return a boolean.
Buggy implementation:
def can_register(age):
    return age > 18

Find an input that exposes the bug. Compute the expected output from the
specification, not from the buggy implementation.
Return only one JSON object with exactly these keys:
{"input": {"age": INTEGER}, "expected": BOOLEAN}
Do not output Python code, markdown, explanations, or additional test cases.
"""


class Tee:
    def __init__(self, console, log):
        self.console = console
        self.log = log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return False


def verify_test(raw):
    result = {"valid_format": False, "correct_expected": False, "bug_detected": False}
    try:
        test = json.loads(raw)
        if type(test) is not dict or set(test) != {"input", "expected"}:
            raise ValueError("Expected exactly input and expected keys")
        if type(test["input"]) is not dict or set(test["input"]) != {"age"}:
            raise ValueError("Input must contain exactly age")
        age = test["input"]["age"]
        if type(age) is not int or not 0 <= age <= 120:
            raise ValueError("Age must be an integer from 0 to 120")
        if type(test["expected"]) is not bool:
            raise ValueError("Expected output must be a JSON boolean")
        correct = age >= 18
        buggy = age > 18
        result.update(
            valid_format=True,
            test=test,
            gold_output=correct,
            buggy_output=buggy,
            correct_expected=test["expected"] == correct,
            bug_detected=test["expected"] == correct and buggy != correct,
        )
    except (ValueError, TypeError, KeyError) as error:
        result["error"] = str(error)
    return result


def run(args, result):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, TextStreamer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Stop and check the PyTorch installation.")
    result["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "trl", "peft", "accelerate")
    }
    free, total = torch.cuda.mem_get_info()
    result.update(
        gpu=torch.cuda.get_device_name(0),
        initial_free_vram_gib=round(free / 2**30, 3),
        total_vram_gib=round(total / 2**30, 3),
    )
    print("GPU:", result["gpu"])
    print("Free / total VRAM (GiB):", result["initial_free_vram_gib"], "/", result["total_vram_gib"])
    print("[1/4] Read local tokenizer and prepare prompt", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    messages = [
        {"role": "system", "content": "You generate correct test cases. Output only valid JSON."},
        {"role": "user", "content": PROMPT},
    ]
    chat = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(chat, return_tensors="pt", add_special_tokens=False)
    result["prompt"] = messages
    result["prompt_tokens"] = inputs["input_ids"].shape[1]
    if args.preflight:
        result["status"] = "preflight_passed_no_model_weights_loaded"
        print("PREFLIGHT OK: model class imported and prompt tokenized. Weights NOT loaded.")
        return
    print("[2/4] Load original weights onto GPU; this can take some time", flush=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    result["dtype"] = str(dtype)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, loading_info = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=dtype,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    torch.cuda.synchronize()
    result["load_seconds"] = round(time.perf_counter() - started, 3)
    result["loading_info"] = loading_info
    if loading_info.get("missing_keys") or loading_info.get("mismatched_keys"):
        raise RuntimeError("Missing or mismatched checkpoint weights; inspect result.json")
    model.eval()
    result["weights_loaded_on_gpu"] = True
    print("MODEL LOADED ON GPU. Load time (s):", result["load_seconds"], flush=True)
    inputs = {name: value.to("cuda:0") for name, value in inputs.items()}
    print("[3/4] Generate one test (no training, no parameter updates)", flush=True)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            max_time=180,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True),
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    generated = output[0, result["prompt_tokens"]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    result.update(
        inference_completed=True,
        raw_output=raw,
        generated_tokens=len(generated),
        generation_seconds=round(seconds, 3),
        tokens_per_second=round(len(generated) / seconds, 3) if seconds else None,
        peak_torch_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
        peak_torch_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 3),
        output_reached_token_limit=len(generated) >= args.max_new_tokens,
    )
    print("[4/4] Verify model-generated JSON using fixed reference logic", flush=True)
    result["verification"] = verify_test(raw)
    result["status"] = "inference_passed" if raw else "empty_output"
    print(json.dumps(result["verification"], ensure_ascii=False, indent=2))
    print("Generation time (s):", result["generation_seconds"])
    print("Peak PyTorch allocated VRAM (GiB):", result["peak_torch_allocated_gib"])
    print("INFERENCE COMPLETED. This is not an RL training run.")


def main():
    parser = argparse.ArgumentParser(description="Offline GPU inference check for Qwen3.5-2B")
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 256:
        parser.error("--max-new-tokens must be from 1 to 256 for this small check")
    args.model = args.model or args.assets / "models" / "Qwen3.5-2B"
    if not (args.model / "config.json").is_file():
        parser.error(f"Missing model config: {args.model}")
    for key, value in {
        "HF_HOME": str(args.assets / "cache" / "huggingface"),
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TORCHINDUCTOR_CACHE_DIR": str(args.assets / "cache" / "torchinductor"),
        "TEMP": str(args.assets / "temp"),
        "TMP": str(args.assets / "temp"),
    }.items():
        os.environ[key] = value
    (args.assets / "temp").mkdir(parents=True, exist_ok=True)
    run_dir = args.assets / "evidence" / ("model-check-" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "started_at": datetime.now().astimezone().isoformat(),
        "model_path": str(args.model),
        "status": "started",
        "weights_loaded_on_gpu": False,
        "inference_completed": False,
        "rl_training": False,
        "max_new_tokens": args.max_new_tokens,
        "script_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    stdout, stderr = sys.stdout, sys.stderr
    exit_code = 0
    with (run_dir / "console.txt").open("w", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        try:
            print("Evidence:", run_dir, flush=True)
            run(args, result)
        except KeyboardInterrupt:
            result["status"] = "interrupted"
            exit_code = 130
            print("Interrupted. Partial evidence saved.")
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__, error=str(error))
            exit_code = 1
            traceback.print_exc()
        finally:
            result["finished_at"] = datetime.now().astimezone().isoformat()
            (run_dir / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            print("Saved evidence:", run_dir, flush=True)
            sys.stdout, sys.stderr = stdout, stderr
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
