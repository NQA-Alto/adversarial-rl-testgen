import argparse
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
from check_lora_update import TARGET_MODULES
from pilot_tasks import build_rows, score_test


class RewardAudit:
    def __init__(self, run_dir, eos_ids, group_size, synthetic=False):
        self.run_dir = run_dir
        self.eos_ids = set(eos_ids)
        self.group_size = group_size
        self.synthetic = synthetic
        self.groups = []

    def reward(self, completions, threshold, task_id, completion_ids, trainer_state, **kwargs):
        records = []
        for raw, cut, identity, ids in zip(completions, threshold, task_id, completion_ids, strict=True):
            if not isinstance(raw, str):
                raise TypeError("This pilot expects non-conversational completion strings")
            item = score_test(raw.strip(), int(cut))
            item.update(
                task_id=identity, threshold=int(cut), raw_output=raw,
                completion_tokens=len(ids), terminated=bool(ids) and ids[-1] in self.eos_ids,
            )
            if not item["terminated"]:
                item["reward"] = 0.0
            records.append(item)
        if len(records) % self.group_size:
            raise RuntimeError("Incomplete GRPO reward group")
        rewards = []
        for start in range(0, len(records), self.group_size):
            group = records[start:start + self.group_size]
            if len({item["task_id"] for item in group}) != 1:
                raise RuntimeError("GRPO group contains different tasks")
            values = [item["reward"] for item in group]
            mean = sum(values) / len(values)
            mixed = max(values) - min(values) > 1e-8
            signal = mixed and any(item["terminated"] and abs(item["reward"] - mean) > 1e-8 for item in group)
            event = {
                "group": len(self.groups) + 1, "optimizer_step_before_group": trainer_state.global_step,
                "sample_source": "scripted_cpu_self_check" if self.synthetic else "model_generated",
                "task_id": group[0]["task_id"], "rewards": values,
                "different_rewards": mixed, "potential_policy_signal": signal, "samples": group,
            }
            self.groups.append(event)
            with (self.run_dir / "rollouts.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(f'GROUP {event["group"]} | {event["task_id"]} | rewards={values} | signal={signal}', flush=True)
            for item in group:
                label = "SCRIPTED CHECK:" if self.synthetic else "MODEL:"
                print(" ", label, item["raw_output"].strip(), "| reward:", item["reward"], flush=True)
            rewards.extend(values)
        return rewards


def trainer_config(run_dir, steps, cpu=False):
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=str(run_dir / "trainer"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_steps=steps,
        max_completion_length=48,
        learning_rate=5e-6,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        max_grad_norm=1.0,
        beta=0.0,
        entropy_coef=0.0,
        epsilon=0.2,
        loss_type="grpo",
        scale_rewards="group",
        num_iterations=1,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        chat_template_kwargs={"enable_thinking": False},
        mask_truncated_completions=True,
        use_vllm=False,
        bf16=not cpu,
        fp16=False,
        use_cpu=cpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        disable_dropout=True,
        optim="adamw_torch",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        push_to_hub=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=42,
        data_seed=42,
        disable_tqdm=True,
        log_completions=False,
    )


def tiny_self_check():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5ForConditionalGeneration

    samples = [
        '{"input":{"x":5},"expected":true}',
        '{"input":{"x":4},"expected":false}',
        '{"input":{"x":5},"expected":false}',
        'not-json',
    ]
    vocab = {token: index for index, token in enumerate(["[PAD]", "[EOS]", "[UNK]", "prompt", *samples])}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        model_max_length=128,
    )
    config = Qwen3_5Config(
        text_config={
            "vocab_size": len(vocab), "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 16, "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "max_position_embeddings": 128, "eos_token_id": 1, "pad_token_id": 0,
            "rope_parameters": {
                "rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.5,
                "mrope_section": [2, 1, 1], "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
            "out_hidden_size": 32, "patch_size": 2, "temporal_patch_size": 1,
            "spatial_merge_size": 1, "num_position_embeddings": 16,
        },
    )
    model = Qwen3_5ForConditionalGeneration(config)
    rows = [{"prompt": "prompt", "threshold": 5, "task_id": "synthetic-threshold-5"}] * 4

    def scripted_rollouts(prompts, trainer):
        return {
            "prompt_ids": [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts],
            "completion_ids": [[tokenizer.convert_tokens_to_ids(samples[index % 4]), 1] for index in range(len(prompts))],
            "logprobs": None,
        }

    return model, tokenizer, rows, scripted_rollouts


def run(args, run_dir, result):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, TrainerCallback
    from trl import GRPOTrainer

    torch.manual_seed(42)
    result["versions"] = {
        name: importlib.metadata.version(name) for name in ("torch", "transformers", "trl", "peft", "accelerate")
    }
    result["pilot_only"] = True
    steps = 1 if args.self_check else args.steps
    config = trainer_config(run_dir, steps, cpu=args.self_check)
    if args.self_check:
        print("SYNTHETIC CPU SELF-CHECK: scripted rollouts; not evidence of pretrained-model RL", flush=True)
        model, tokenizer, rows, rollout_func = tiny_self_check()
    else:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA and BF16 support are required for this pilot")
        result["gpu"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        result.update(initial_free_vram_gib=round(free / 2**30, 3), total_vram_gib=round(total / 2**30, 3))
        print("GPU:", result["gpu"], "| free VRAM (GiB):", result["initial_free_vram_gib"], flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left")
        rows = build_rows(tokenizer)
        lengths = [len(tokenizer.encode(row["prompt"], add_special_tokens=False)) for row in rows]
        if max(lengths) > 256:
            raise RuntimeError("Pilot prompt unexpectedly exceeds 256 tokens")
        result["prompt_token_lengths"] = lengths
        if args.preflight:
            result["status"] = "preflight_passed_no_weights_loaded"
            print("PREFLIGHT OK: GRPO config and task prompts prepared. No weights loaded.")
            return
        print("[1/4] Load fresh original model; arithmetic adapter is not used", flush=True)
        torch.cuda.reset_peak_memory_stats()
        model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation="sdpa", output_loading_info=True,
        )
        result["loading_info"] = info
        if info.get("missing_keys") or info.get("mismatched_keys"):
            raise RuntimeError("Missing or mismatched checkpoint weights")
        rollout_func = None
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    audit = RewardAudit(run_dir, [tokenizer.eos_token_id, tokenizer.pad_token_id], 4, synthetic=args.self_check)
    steps_audit = []

    class GradientAudit(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
            if not gradients or not all(torch.isfinite(value).all().item() for value in gradients):
                raise RuntimeError("Missing or non-finite policy gradients")
            norm_sq = sum(float(value.detach().float().square().sum().cpu()) for value in gradients)
            frozen_ok = all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
            if not frozen_ok:
                raise RuntimeError("Unexpected gradient on frozen base weights")
            event = {"optimizer_step": state.global_step + 1, "gradient_norm_after_clipping": math.sqrt(norm_sq), "frozen_parameters_have_no_grad": frozen_ok}
            steps_audit.append(event)
            print("POLICY UPDATE", event["optimizer_step"], "| gradient norm:", event["gradient_norm_after_clipping"], flush=True)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
        lora_dropout=0.0, bias="none", target_modules=TARGET_MODULES,
    )
    print("[2/4] Construct GRPOTrainer: 4 samples/group, no external AI, no labelled target answers", flush=True)
    trainer = GRPOTrainer(
        model=model, args=config, reward_funcs=audit.reward, train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer, peft_config=peft_config,
        callbacks=[GradientAudit()], rollout_func=rollout_func,
    )
    trainable = {name: parameter for name, parameter in trainer.model.named_parameters() if parameter.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Only LoRA parameters must be trainable")
    before = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}
    result["trainable_parameters"] = sum(parameter.numel() for parameter in trainable.values())
    result["configuration"] = {
        "steps": steps, "group_size": 4, "microbatch_size": 1, "gradient_accumulation": 4,
        "max_completion_tokens": 48, "learning_rate": 5e-6, "beta": 0.0,
        "entropy_coef": 0.0, "weight_decay": 0.0, "loss_type": "grpo",
        "scale_rewards": "group", "seed": 42, "temperature": 1.0,
        "top_p": 0.95, "top_k": 20, "use_vllm": False,
        "lora_rank": 4, "lora_alpha": 8, "target_modules": TARGET_MODULES,
        "reward": {"valid_format_and_domain": 0.1, "correct_expected": 0.2, "valid_bug_detection": 0.7},
        "training_data": "4 toy threshold problems; no held-out evaluation in this pilot",
    }
    (run_dir / "pilot-tasks.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TRAINABLE PARAMS:", result["trainable_parameters"], flush=True)
    print("[3/4] Run the small reward-driven training loop", flush=True)
    started = time.perf_counter()
    trained = trainer.train()
    result["training_seconds"] = round(time.perf_counter() - started, 3)
    result["trainer_metrics"] = trained.metrics
    result["optimizer_steps"] = trainer.state.global_step
    result["gradient_audit"] = steps_audit
    result["reward_groups"] = len(audit.groups)
    result["groups_with_policy_signal"] = sum(group["potential_policy_signal"] for group in audit.groups)
    deltas = {
        name: float((parameter.detach().cpu().float() - before[name].float()).abs().max())
        for name, parameter in trainable.items()
    }
    result["max_parameter_delta"] = max(deltas.values())
    result["changed_adapter_tensors"] = [name for name, delta in deltas.items() if delta > 0]
    result["adapter_updated"] = bool(result["changed_adapter_tensors"])
    print("[4/4] Save pilot adapter and verify saved parameters", flush=True)
    adapter_dir = run_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
    expected = get_peft_model_state_dict(trainer.model)
    saved = load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
    result["saved_adapter_matches"] = set(expected) == set(saved) and all(torch.equal(value.detach().cpu(), saved[name]) for name, value in expected.items())
    if not result["saved_adapter_matches"]:
        raise RuntimeError("Saved adapter parameters do not match the trained adapter")
    (run_dir / "trainer-log-history.json").write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2), encoding="utf-8")
    real_signal = result["groups_with_policy_signal"] > 0 and any(item["gradient_norm_after_clipping"] > 0 for item in steps_audit)
    verified = real_signal and result["adapter_updated"]
    result["reward_driven_update_verified"] = verified
    result["rl_training"] = bool(verified and not args.self_check)
    if not args.self_check:
        result["peak_torch_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        result["peak_torch_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        print("Peak PyTorch allocated VRAM (GiB):", result["peak_torch_allocated_gib"], flush=True)
    result["status"] = ("synthetic_trainer_self_check_passed" if args.self_check else "rl_pilot_update_verified") if verified else "completed_without_verified_learning_signal"
    print("GROUPS WITH POLICY SIGNAL:", result["groups_with_policy_signal"], flush=True)
    print("ADAPTER UPDATED:", result["adapter_updated"], flush=True)
    print("SAVED ADAPTER MATCH:", result["saved_adapter_matches"], flush=True)
    print("REWARD-DRIVEN UPDATE VERIFIED:", verified, flush=True)
    if args.self_check and not verified:
        raise RuntimeError("Synthetic GRPO trainer check did not produce the expected update")
    print("STATUS:", result["status"], flush=True)
    print("PILOT ONLY: no claim of quality improvement or complete adversarial training.")


def main():
    parser = argparse.ArgumentParser(description="Small genuine GRPO test-generation pilot using local model weights")
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 8 or args.self_check and args.preflight:
        parser.error("Use 1–8 steps and choose at most one check mode")
    args.model = args.model or args.assets / "models" / "Qwen3.5-2B"
    if not args.self_check and not (args.model / "config.json").is_file():
        parser.error("Local model config is missing")
    for key, value in {
        "HF_HOME": str(args.assets / "cache" / "huggingface"), "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(args.assets / "cache" / "torchinductor"),
        "TEMP": str(args.assets / "temp"), "TMP": str(args.assets / "temp"),
        "WANDB_MODE": "disabled", "WANDB_DISABLED": "true",
    }.items():
        os.environ[key] = value
    (args.assets / "temp").mkdir(parents=True, exist_ok=True)
    prefix = "grpo-self-check-" if args.self_check else "grpo-pilot-"
    run_dir = args.assets / "evidence" / (prefix + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "started_at": datetime.now().astimezone().isoformat(), "status": "started",
        "rl_training": False, "quality_evaluation": False,
        "synthetic_cpu_self_check": args.self_check, "preflight_only": args.preflight,
        "model_path": None if args.self_check else str(args.model),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task_source_sha256": hashlib.sha256(Path(__file__).with_name("pilot_tasks.py").read_bytes()).hexdigest(),
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
            print("Interrupted; partial evidence preserved.")
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
