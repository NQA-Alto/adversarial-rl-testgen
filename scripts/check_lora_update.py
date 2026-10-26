import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from check_local_model import Tee


TARGET_MODULES = r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
QUESTION = "Add the two integers a=2 and b=3. Return only JSON with the key result."
ANSWER = '{"result":5}'


def completion_loss(model, inputs, targets):
    import torch.nn.functional as functional

    output = model(**inputs, use_cache=False, logits_to_keep=targets.shape[1] + 1)
    predictions = output.logits[:, :-1, :].float()
    return functional.cross_entropy(predictions.reshape(-1, predictions.shape[-1]), targets.reshape(-1))


def update_and_reload(base, inputs, targets, adapter_dir, result):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from peft.utils import get_peft_model_state_dict

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(base, config)
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
    frozen = [value for value in model.parameters() if not value.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Expected only LoRA parameters to be trainable")
    count = sum(value.numel() for value in trainable.values())
    total = sum(value.numel() for value in model.parameters())
    result["lora"] = {
        "rank": 4, "alpha": 8, "dropout": 0.0,
        "target_modules": TARGET_MODULES,
        "trainable_parameters": count,
        "total_parameters": total,
        "trainable_percent": round(count / total * 100, 6),
        "learning_rate": 1e-4,
        "optimizer": "AdamW", "optimizer_steps": 1,
        "gradient_checkpointing": True,
    }
    print("TRAINABLE PARAMS:", count, "/", total, flush=True)
    before = {name: value.detach().cpu().clone() for name, value in trainable.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=1e-4, weight_decay=0.0, foreach=False)
    print("[3/5] Forward + backward on one fixed engineering sample (NOT RL)", flush=True)
    started = time.perf_counter()
    model.train()
    loss = completion_loss(model, inputs, targets)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss before update")
    result["loss_before_update"] = float(loss.detach().cpu())
    loss.backward()
    grads = [value.grad for value in trainable.values() if value.grad is not None]
    if not grads or not all(torch.isfinite(value).all().item() for value in grads):
        raise RuntimeError("Missing or non-finite adapter gradients")
    norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), max_norm=1.0)
    result["gradient_norm_before_clipping"] = float(norm.detach().cpu())
    result["nonzero_gradient"] = result["gradient_norm_before_clipping"] > 0
    result["frozen_parameters_have_no_grad"] = all(value.grad is None for value in frozen)
    if not result["nonzero_gradient"] or not result["frozen_parameters_have_no_grad"]:
        raise RuntimeError("Adapter gradient or frozen-parameter check failed")
    print("NONZERO GRADIENT:", result["nonzero_gradient"], flush=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    changed = []
    maximum_delta = 0.0
    for name, value in trainable.items():
        delta = (value.detach().cpu().float() - before[name].float()).abs()
        maximum_delta = max(maximum_delta, float(delta.max()))
        if torch.count_nonzero(delta).item():
            changed.append(name)
    result["adapter_updated"] = bool(changed)
    result["changed_adapter_tensors"] = changed
    result["max_parameter_delta"] = maximum_delta
    if not changed:
        raise RuntimeError("Optimizer step did not change any adapter parameters")
    print("ADAPTER UPDATED:", result["adapter_updated"], flush=True)
    del loss, grads, before, optimizer, trainable, frozen
    model.eval()
    with torch.no_grad():
        result["loss_after_update"] = float(completion_loss(model, inputs, targets).cpu())
    if not math.isfinite(result["loss_after_update"]):
        raise RuntimeError("Non-finite loss after update")
    if inputs["input_ids"].is_cuda:
        torch.cuda.synchronize()
    result["update_and_loss_check_seconds"] = round(time.perf_counter() - started, 3)
    print("Loss before / after:", result["loss_before_update"], "/", result["loss_after_update"], flush=True)
    print("[4/5] Save adapter, remove it from memory, then reload from disk", flush=True)
    model.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
    expected = {
        name: value.detach().cpu().clone()
        for name, value in get_peft_model_state_dict(model).items()
    }
    base = model.unload()
    base.gradient_checkpointing_disable()
    del model
    gc.collect()
    if inputs["input_ids"].is_cuda:
        torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False, local_files_only=True)
    actual = get_peft_model_state_dict(reloaded)
    matched = set(expected) == set(actual) and all(
        torch.equal(expected[name], actual[name].detach().cpu()) for name in expected
    )
    result["adapter_reloaded_exactly"] = matched
    result["reload_method"] = "Remove LoRA without merging; reload saved adapter onto the same frozen base in memory"
    if not matched:
        raise RuntimeError("Reloaded adapter differs from saved parameters")
    reloaded.eval()
    with torch.no_grad():
        result["loss_after_reload"] = float(completion_loss(reloaded, inputs, targets).cpu())
    result["reload_loss_matches"] = math.isclose(
        result["loss_after_update"], result["loss_after_reload"], rel_tol=1e-4, abs_tol=1e-4
    )
    if not result["reload_loss_matches"]:
        raise RuntimeError("Loss changed unexpectedly after adapter reload")
    result["adapter_files"] = [
        {"name": path.name, "bytes": path.stat().st_size,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(adapter_dir.iterdir()) if path.is_file()
    ]
    print("ADAPTER RELOAD MATCH:", matched, flush=True)
    print("LOSS AFTER RELOAD MATCH:", result["reload_loss_matches"], flush=True)
    if inputs["input_ids"].is_cuda:
        result["peak_torch_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        result["peak_torch_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        print("Peak PyTorch allocated VRAM (GiB):", result["peak_torch_allocated_gib"], flush=True)


def run(args, run_dir, result):
    import torch
    from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForConditionalGeneration

    torch.manual_seed(42)
    result["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "trl", "accelerate")
    }
    if args.self_check:
        print("SYNTHETIC CPU SELF-CHECK: no pretrained weights, no GPU training", flush=True)
        config = Qwen3_5Config(
            text_config={
                "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 4, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 16,
                "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                "layer_types": ["linear_attention"] * 3 + ["full_attention"],
                "max_position_embeddings": 128, "eos_token_id": 2,
                "rope_parameters": {
                    "rope_type": "default", "rope_theta": 10000,
                    "partial_rotary_factor": 0.5,
                    "mrope_section": [2, 1, 1], "mrope_interleaved": True,
                },
            },
            vision_config={
                "depth": 1, "hidden_size": 32, "intermediate_size": 64,
                "num_heads": 2, "out_hidden_size": 32, "patch_size": 2,
                "temporal_patch_size": 1, "spatial_merge_size": 1,
                "num_position_embeddings": 16,
            },
        )
        base = Qwen3_5ForConditionalGeneration(config)
        ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 2]])
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        targets = ids[:, -4:]
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; stop and check the environment")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This check requires BF16 support; do not change dtype automatically")
        result["gpu"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        result["initial_free_vram_gib"] = round(free / 2**30, 3)
        result["total_vram_gib"] = round(total / 2**30, 3)
        print("GPU:", result["gpu"], flush=True)
        print("Free VRAM (GiB):", result["initial_free_vram_gib"], flush=True)
        print("[1/5] Prepare one separate engineering sample, not the age test", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": QUESTION}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        suffix = tokenizer.encode(ANSWER, add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(prefix) + len(suffix) > 128:
            raise RuntimeError("Engineering sample is unexpectedly longer than 128 tokens")
        ids = torch.tensor([prefix + suffix], device="cuda:0")
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        targets = ids[:, -len(suffix):]
        result.update(
            engineering_question=QUESTION, fixed_training_answer=ANSWER,
            sequence_tokens=len(prefix) + len(suffix), completion_tokens=len(suffix),
            dtype="torch.bfloat16", gradient_sample="fixed labelled engineering sample; not reward-driven",
        )
        if args.preflight:
            from peft import LoraConfig
            LoraConfig(r=4, target_modules=TARGET_MODULES)
            result["status"] = "preflight_passed_no_model_weights_loaded"
            print("PREFLIGHT OK. Sequence tokens:", result["sequence_tokens"], flush=True)
            print("No model weights loaded; no training performed.")
            return
        print("[2/5] Load frozen base model in BF16", flush=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        base, loading_info = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation="sdpa", output_loading_info=True,
        )
        torch.cuda.synchronize()
        result["load_seconds"] = round(time.perf_counter() - started, 3)
        result["loading_info"] = loading_info
        if loading_info.get("missing_keys") or loading_info.get("mismatched_keys"):
            raise RuntimeError("Checkpoint has missing or mismatched weights")
    adapter_dir = run_dir / "adapter"
    update_and_reload(base, inputs, targets, adapter_dir, result)
    (run_dir / "SMOKE_TEST_ONLY.txt").write_text(
        "Engineering LoRA update check only. Not RL. Not a final trained model. "
        "Do not use this arithmetic adapter as a test-generation model.\n", encoding="utf-8"
    )
    result["status"] = "synthetic_cpu_self_check_passed" if args.self_check else "lora_update_check_passed"
    print("[5/5] LORA UPDATE CHECK PASSED. This is NOT RL or proof of quality improvement.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="One local LoRA optimizer-step and adapter round-trip check; NOT RL")
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check and args.preflight:
        parser.error("Choose only one of --self-check and --preflight")
    args.model = args.model or args.assets / "models" / "Qwen3.5-2B"
    if not args.self_check and not (args.model / "config.json").is_file():
        parser.error("Local model config not found")
    for key, value in {
        "HF_HOME": str(args.assets / "cache" / "huggingface"),
        "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(args.assets / "cache" / "torchinductor"),
        "TEMP": str(args.assets / "temp"), "TMP": str(args.assets / "temp"),
    }.items():
        os.environ[key] = value
    (args.assets / "temp").mkdir(parents=True, exist_ok=True)
    prefix = "lora-self-check-" if args.self_check else "lora-check-"
    run_dir = args.assets / "evidence" / (prefix + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "started_at": datetime.now().astimezone().isoformat(),
        "status": "started", "rl_training": False, "quality_evaluation": False,
        "synthetic_cpu_self_check": args.self_check, "preflight_only": args.preflight,
        "model_path": None if args.self_check else str(args.model),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
