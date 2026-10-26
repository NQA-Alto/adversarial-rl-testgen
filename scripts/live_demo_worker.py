import argparse
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


ASSETS = Path(r"D:\DA\TTNT_De19_Assets")
VARIANTS = ("parent", "control", "adversarial")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temp, path)


def worker(root, variant, request, catalog):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig, Qwen3_5ForConditionalGeneration
    from dataset_tasks import compile_trusted, model_prompt
    from probe_round1_learning import score
    from restricted_python import RestrictedFunction

    case = catalog["cases"][request["case_key"]]
    variants = VARIANTS if variant == "all" else (variant,)
    for name in variants:
        selected = catalog["variants"][name]
        if digest(Path(selected["adapter_path"]) / "adapter_model.safetensors") != selected["adapter_sha256"]:
            raise ValueError("Adapter hash changed: " + name)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(catalog["model_path"], local_files_only=True, padding_side="left")
    messages = model_prompt(case, version="plain-v2")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if len(tokenizer.encode(prompt, add_special_tokens=False)) > 768:
        raise RuntimeError("Prompt budget exceeded")
    print("BASE MODEL: loading once; local mmap checkpoint; sequential frozen adapters", flush=True)
    model_path = Path(catalog["model_path"])
    index = model_path / "model.safetensors.index.json"
    files = sorted(set(read(index)["weight_map"].values())) if index.is_file() else ["model.safetensors"]
    weights = {}
    for name in files:
        weights.update(load_file(model_path / name, device="cpu", backend="mmap"))
    base, loading = Qwen3_5ForConditionalGeneration.from_pretrained(None,
        config=AutoConfig.from_pretrained(model_path, local_files_only=True), state_dict=weights,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa", output_loading_info=True)
    del weights
    if (model_path / "generation_config.json").is_file():
        base.generation_config = GenerationConfig.from_pretrained(model_path, local_files_only=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        raise RuntimeError("Unexpected base loading result")
    first = variants[0]
    model = PeftModel.from_pretrained(base, catalog["variants"][first]["adapter_path"], adapter_name=first, is_trainable=False, local_files_only=True)
    for name in variants[1:]:
        model.load_adapter(catalog["variants"][name]["adapter_path"], adapter_name=name, is_trainable=False, local_files_only=True)
    model.eval()
    assert not any(parameter.requires_grad for parameter in model.parameters())
    declared = model.generation_config.eos_token_id
    eos = sorted({tokenizer.eos_token_id, *(declared if isinstance(declared, list) else [declared])} - {None})
    if eos != [248044, 248046]:
        raise RuntimeError("EOS protocol changed")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    reference = compile_trusted(case["reference"])
    buggy = RestrictedFunction(case["buggy_code"], case["domain"])
    for name in variants:
        print("MODEL:", name, "| frozen adapter inference", flush=True)
        model.set_adapter(name, inference_mode=True)
        model.requires_grad_(False)
        model.eval()
        assert not any(parameter.requires_grad for parameter in model.parameters())
        target = root / name
        target.mkdir(exist_ok=False)
        seed = (request["seed"] + int(hashlib.sha256(case["case_id"].encode()).hexdigest()[:8], 16)) % 2**32
        torch.manual_seed(seed)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        generated_at = datetime.now().astimezone().isoformat()
        gen_start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, num_return_sequences=4, max_new_tokens=64, max_time=120,
                do_sample=True, temperature=1.0, top_p=0.95, top_k=20, use_cache=True, pad_token_id=pad, eos_token_id=eos)
        samples = []
        for tokens in output[:, inputs["input_ids"].shape[1]:].tolist():
            end = next((i for i, token in enumerate(tokens) if token in eos), None)
            actual = tokens[:end + 1] if end is not None else tokens
            raw = tokenizer.decode(actual, skip_special_tokens=True).strip()
            sample = score(raw, end is not None, case, reference, buggy)
            sample["generated_tokens"] = len(actual)
            samples.append(sample)
        assert len(samples) == 4
        prediction = {"case_id": case["case_id"], "variant": name, "samples": samples, "case_seed": seed,
                      "prompt_messages": messages, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_version": "plain-v2",
                      "generated_at": generated_at, "generation_seconds": round(time.perf_counter() - gen_start, 3)}
        write(target / "prediction.json", prediction)
        write(target / "result.json", {"status": "completed", "training": False, "test_set_evaluation": False,
              "loader": "one_base_mmap_three_frozen_adapters", "all_parameters_frozen": True, "adapter": catalog["variants"][name],
              "gpu": torch.cuda.get_device_name(0), "peak_torch_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
              "total_seconds_including_shared_load": round(time.perf_counter() - start, 3), "generation_config": {"seed": request["seed"], "tests_per_case": 4,
              "max_new_tokens": 64, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "eos_token_ids": eos, "dtype": "bfloat16", "thinking": False}})
        print("MODEL COMPLETED:", name, "| hits:", sum(s["bug_detected"] for s in samples), "/4", flush=True)
        del inputs, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variant", choices=(*VARIANTS, "all"))
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to((ASSETS / "evidence/ui-jobs").resolve()):
        raise ValueError("Demo run must be inside ui-jobs")
    request = read(root / "request.json")
    catalog_path = root / "catalog.json"
    if digest(catalog_path) != request["catalog_sha256"]:
        raise ValueError("Demo catalog hash changed")
    catalog = read(catalog_path)
    if request["seed"] not in (42, 43, 44) or request["case_key"] not in catalog["cases"]:
        raise ValueError("Invalid demo selection")
    for key, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HOME": str(ASSETS / "cache/huggingface"), "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false", "TEMP": str(ASSETS / "temp"), "TMP": str(ASSETS / "temp")}.items():
        os.environ[key] = value
    for path, sha in catalog["sources"].items():
        if digest(Path(path)) != sha:
            raise ValueError("Catalog source changed: " + path)
    if args.variant:
        worker(root, args.variant, request, catalog)
        return 0
    result = {"status": "started", "started_at": datetime.now().astimezone().isoformat(), "training": False,
              "test_set_evaluation": False, "method": "live_demo_inference_not_full_benchmark", "case_key": request["case_key"], "seed": request["seed"]}
    write(root / "result.json", result)
    code = 0
    try:
        sources = root / "source"
        sources.mkdir()
        for name in ("live_demo_worker.py", "dataset_tasks.py", "restricted_python.py", "probe_round1_learning.py"):
            shutil.copy2(Path(__file__).with_name(name), sources / name)
        result["source_hashes"] = {p.name: digest(p) for p in sources.iterdir()}
        print("Evidence:", root, flush=True)
        for variant in ("all",):
            print("DEMO: 3 adapters / 1 shared base", flush=True)
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--root", str(root), "--variant", variant]
            memory_error = False
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    memory_error = memory_error or "1455" in line or "paging file" in line or "out of memory" in line.lower()
                if process.wait():
                    if memory_error:
                        result["error_category"] = "memory"
                    raise RuntimeError("Live inference failed: " + variant)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error=str(error), error_type=type(error).__name__)
        traceback.print_exc()
        code = 1
    finally:
        result["finished_at"] = datetime.now().astimezone().isoformat()
        write(root / "result.json", result)
        print("Saved evidence:", root, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
