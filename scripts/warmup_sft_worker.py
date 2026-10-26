import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from check_local_model import Tee
from check_lora_update import TARGET_MODULES, completion_loss
from round1_data import digest, read, rows
from train_static_grpo import save_adapter_checked


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def encode(tokenizer, values, config):
    result = []
    for value in values:
        prompt = tokenizer.apply_chat_template(value["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(value["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(prompt_ids) + len(target_ids) > config["max_total_tokens"] or len(target_ids) > config["max_target_tokens"]:
            raise ValueError("SFT token budget exceeded")
        result.append({"sample_id": value["sample_id"], "input_ids": prompt_ids + target_ids, "target_ids": target_ids})
    return result


def tensors(value, device):
    import torch
    ids = torch.tensor([value["input_ids"]], dtype=torch.long, device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, torch.tensor([value["target_ids"]], dtype=torch.long, device=device)


def dev_loss(model, values, device):
    import torch
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for value in values:
            inputs, targets = tensors(value, device)
            loss = completion_loss(model, inputs, targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite SFT Dev loss")
            total += float(loss.cpu()) * targets.numel()
            count += targets.numel()
    return {"completion_token_nll": total / count, "target_tokens": count, "examples": len(values)}


def train(args, folder, result):
    import torch
    import torch.nn.functional as functional
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    torch.manual_seed(42)
    result["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "trl", "accelerate")}
    if args.self_check:
        from run_grpo_pilot import tiny_self_check
        base, tokenizer, _, _ = tiny_self_check()
        temporary = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, lora_dropout=0.0, bias="none", target_modules=TARGET_MODULES))
        with torch.no_grad():
            for name, value in temporary.named_parameters():
                if "lora_B" in name:
                    value.fill_(0.001)
        save_adapter_checked(temporary, folder / "tiny-parent", 0)
        parent = folder / "tiny-parent"
        model = PeftModel.from_pretrained(temporary.unload(), parent, is_trainable=True, local_files_only=True)
        training = [{"sample_id": f"tiny-{index}", "input_ids": [3, 4, 1], "target_ids": [4, 1]} for index in range(8)]
        development = training[:2]
        config = {"gradient_accumulation": 4, "learning_rate": 0.001, "seed": 42, "checkpoint_steps": 1}
        device = "cpu"
    else:
        spec = read(args.root / "warmup-spec.json")
        for name, sha in spec["source_hashes"].items():
            if digest(Path(__file__).with_name(name)) != sha:
                raise ValueError("Warmup source changed: " + name)
        config = spec["configuration"]
        dataset = Path(spec["dataset_path"])
        for name, sha in spec["dataset_file_hashes"].items():
            if digest(dataset / name) != sha:
                raise ValueError("SFT data changed: " + name)
        parent = Path(spec["parent_adapter_path"])
        if digest(parent / "adapter_model.safetensors") != spec["parent_adapter_sha256"]:
            raise ValueError("Parent adapter changed")
        tokenizer = AutoTokenizer.from_pretrained(spec["model_path"], local_files_only=True, padding_side="left")
        training = encode(tokenizer, rows(dataset / "sft_train.jsonl"), config)
        development = encode(tokenizer, rows(dataset / "sft_dev.jsonl"), config)
        result.update(dataset_id=spec["dataset_id"], parent_adapter_sha256=spec["parent_adapter_sha256"],
                      configuration=config, training_samples=len(training), sft_dev_samples=len(development),
                      prompt_plus_target_token_range=[min(len(value["input_ids"]) for value in training), max(len(value["input_ids"]) for value in training)],
                      target_token_range=[min(len(value["target_ids"]) for value in training), max(len(value["target_ids"]) for value in training)],
                      target_eos_token_id=tokenizer.eos_token_id)
        if args.preflight:
            result["status"] = "sft_preflight_passed_no_weights_loaded"
            return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 required")
        torch.cuda.reset_peak_memory_stats()
        result["gpu"] = torch.cuda.get_device_name(0)
        base, loading = Qwen3_5ForConditionalGeneration.from_pretrained(spec["model_path"], local_files_only=True,
            dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa", output_loading_info=True)
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
            raise RuntimeError("Base loading mismatch")
        model = PeftModel.from_pretrained(base, parent, is_trainable=True, local_files_only=True)
        device = "cuda:0"
    initial_state = get_peft_model_state_dict(model)
    expected = load_file(str(parent / "adapter_model.safetensors"), device="cpu")
    if set(expected) != set(initial_state) or not all(torch.equal(value.detach().cpu(), expected[name]) for name, value in initial_state.items()):
        raise ValueError("Initial adapter differs from parent")
    result["parent_adapter_exact_match"] = True
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model.eval()
    sample_inputs, sample_targets = tensors(training[0], device)
    with torch.no_grad():
        selected = completion_loss(model, sample_inputs, sample_targets)
        full = model(**sample_inputs, use_cache=False).logits[:, -(sample_targets.shape[1] + 1):-1, :].float()
        explicit = functional.cross_entropy(full.reshape(-1, full.shape[-1]), sample_targets.reshape(-1))
        if not torch.allclose(selected, explicit, rtol=1e-5, atol=1e-5):
            raise ValueError("Completion-only loss alignment check failed")
    del full, selected, explicit, sample_inputs, sample_targets
    result["completion_only_loss_alignment_verified"] = True
    result["dev_before"] = dev_loss(model, development, device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("Only LoRA parameters may train")
    result["trainable_parameters"] = sum(parameter.numel() for parameter in trainable.values())
    if not args.self_check and result["trainable_parameters"] != 208896:
        raise ValueError("Unexpected LoRA parameter count")
    initial = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=config["learning_rate"], weight_decay=0.0, foreach=False)
    order = list(range(len(training)))
    random.Random(config["seed"]).shuffle(order)
    write(folder / "training-order.json", [training[index]["sample_id"] for index in order])
    gradients, checkpoints = [], []
    start = time.perf_counter()
    model.train()
    accumulation = config["gradient_accumulation"]
    for offset in range(0, len(order), accumulation):
        batch = order[offset:offset + accumulation]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index in batch:
            inputs, targets = tensors(training[index], device)
            loss = completion_loss(model, inputs, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite supervised loss")
            losses.append(float(loss.detach().cpu()))
            (loss / len(batch)).backward()
        values = [parameter.grad for parameter in trainable.values() if parameter.grad is not None]
        if not values or not all(torch.isfinite(value).all() for value in values):
            raise RuntimeError("Missing or non-finite SFT gradient")
        if any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
            raise RuntimeError("Gradient on frozen base parameter")
        norm = float(torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0).detach().cpu())
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError("No finite supervised gradient")
        optimizer.step()
        event = {"step": len(gradients) + 1, "loss": sum(losses) / len(losses), "gradient_norm": norm,
                 "examples_in_step": len(batch), "method": "SFT_NOT_RL", "frozen_parameters_have_no_grad": True}
        gradients.append(event)
        with (folder / "sft-steps.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        print("SFT STEP", event["step"], "/", math.ceil(len(order) / accumulation), "| loss:", round(event["loss"], 5), "| grad:", round(norm, 5), flush=True)
        if event["step"] % config["checkpoint_steps"] == 0:
            checkpoints.append(save_adapter_checked(model, folder / ("checkpoint-" + str(event["step"])), event["step"]))
    optimizer.zero_grad(set_to_none=True)
    result["training_seconds"] = round(time.perf_counter() - start, 3)
    model.gradient_checkpointing_disable()
    result["dev_after"] = dev_loss(model, development, device)
    changes = {name: float((value.detach().cpu().float() - initial[name].float()).abs().max()) for name, value in trainable.items()}
    if max(changes.values()) <= 0:
        raise RuntimeError("SFT did not change adapter")
    result.update(optimizer_steps=len(gradients), examples_seen=sum(value["examples_in_step"] for value in gradients),
                  nonzero_gradient_steps=len(gradients), max_parameter_delta=max(changes.values()),
                  changed_adapter_tensors=[name for name, delta in changes.items() if delta > 0], checkpoints=checkpoints,
                  final_adapter=save_adapter_checked(model, folder / "adapter", len(gradients)),
                  status="synthetic_sft_check_passed" if args.self_check else "supervised_warmup_completed")
    if not args.self_check:
        result["peak_torch_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        result["peak_torch_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    print("SUPERVISED WARMUP COMPLETE. This is SFT, not RL.", flush=True)
    print("SFT DEV BEFORE / AFTER:", result["dev_before"], result["dev_after"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.root and not args.self_check:
        parser.error("root required")
    if args.self_check and args.preflight:
        parser.error("Separate self-check from preflight")
    if args.self_check:
        args.root = args.assets / "evidence" / ("sft-self-check-" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    if not args.root.resolve().is_relative_to((args.assets / "evidence").resolve()):
        raise ValueError("Expected evidence root")
    for key, value in {"HF_HOME": str(args.assets / "cache/huggingface"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false", "TEMP": str(args.assets / "temp"), "TMP": str(args.assets / "temp")}.items():
        os.environ[key] = value
    folder = args.root / "train"
    folder.mkdir(parents=True, exist_ok=False)
    result = {"started_at": datetime.now().astimezone().isoformat(), "status": "started", "method": "SFT_NOT_RL", "rl_training": False,
              "training": not args.preflight and not args.self_check, "synthetic_cpu_self_check": args.self_check,
              "preflight_only": args.preflight, "test_set_evaluation": False}
    out, err = sys.stdout, sys.stderr
    code = 0
    with (folder / "console.txt").open("w", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(out, log), Tee(err, log)
        try:
            print("Evidence:", folder, flush=True)
            train(args, folder, result)
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
