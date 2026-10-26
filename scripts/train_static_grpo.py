import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from check_local_model import Tee
from check_lora_update import TARGET_MODULES
from dataset_tasks import canonical, compile_trusted, evaluate_output, model_prompt
from evaluate_model import read_cases


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["prompt_version"] != "plain-v2" or config["group_size"] != 4:
        raise ValueError("This first static experiment fixes plain-v2 and group size 4")
    if config["microbatch_size"] != 1 or config["gradient_accumulation_steps"] != 4:
        raise ValueError("Unexpected microbatch/accumulation configuration")
    if not 1 <= config["steps"] <= 144 or config["checkpoint_interval"] < 1:
        raise ValueError("Invalid training budget")
    if file_hash(Path(__file__).with_name("dataset_tasks.py")) != config["task_source_sha256"]:
        raise ValueError("Task/prompt source changed after baseline; review before changing the frozen experiment")
    baseline_path = args.assets / "evidence" / config["baseline_run"] / "result.json"
    if file_hash(baseline_path) != config["baseline_result_sha256"]:
        raise ValueError("Frozen baseline result hash changed")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline["status"] != "evaluation_completed" or baseline["prompt_version"] != config["prompt_version"]:
        raise ValueError("Missing completed baseline with matching prompt")
    if Path(baseline["model_path"]).resolve() != args.model.resolve():
        raise ValueError("Original model path differs from the frozen baseline")
    cases, manifest, split_hash = read_cases(SimpleNamespace(assets=args.assets, split="train", preflight=False))
    if manifest["dataset_id"] != config["dataset_id"] or baseline["dataset_id"] != config["dataset_id"]:
        raise ValueError("Dataset identifier changed")
    if len(cases) != 36 or len({row["problem_id"] for row in cases}) != 12:
        raise ValueError("Expected 36 Train cases from 12 problems")
    return config, cases, split_hash, baseline_path


def save_adapter_checked(model, path, step):
    import torch
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file

    path.mkdir(parents=True, exist_ok=False)
    expected = get_peft_model_state_dict(model)
    if not all(torch.isfinite(value).all().item() for value in expected.values()):
        raise RuntimeError("Adapter contains non-finite parameters")
    model.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
    saved = load_file(str(path / "adapter_model.safetensors"), device="cpu")
    match = set(saved) == set(expected) and all(torch.equal(saved[name], value.detach().cpu()) for name, value in expected.items())
    if not match:
        raise RuntimeError("Saved adapter differs from live parameters")
    info = {"path": str(path), "optimizer_step": step, "parameters_match": True,
            "adapter_sha256": file_hash(path / "adapter_model.safetensors")}
    (path / "checkpoint-info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


class TrainingRewards:
    def __init__(self, cases, run_dir, eos_ids, synthetic=False):
        self.cases = {case["case_id"]: case for case in cases}
        self.compiled = {identity: (compile_trusted(case["reference"]), compile_trusted(case["buggy_code"]))
                         for identity, case in self.cases.items()}
        self.run_dir = run_dir
        self.eos_ids = set(eos_ids)
        self.synthetic = synthetic
        self.groups = []

    def reward(self, completions, case_id, completion_ids, trainer_state, **kwargs):
        samples = []
        for raw, identity, tokens in zip(completions, case_id, completion_ids, strict=True):
            if not isinstance(raw, str) or identity not in self.cases:
                raise ValueError("Unexpected completion or a non-Train case")
            reference, buggy = self.compiled[identity]
            item = evaluate_output(raw.strip(), self.cases[identity], reference, buggy)
            item.update(case_id=identity, raw_output=raw, tokens=len(tokens),
                        terminated=bool(tokens) and tokens[-1] in self.eos_ids)
            if not item["terminated"]:
                item["unmasked_verification"] = {key: item[key] for key in ("valid_format", "correct_expected", "bug_detected")}
                item.update(valid_format=False, correct_expected=False, bug_detected=False, reward=0.0)
            samples.append(item)
        if len(samples) % 4:
            raise RuntimeError("Incomplete reward group")
        rewards = []
        for start in range(0, len(samples), 4):
            group = samples[start:start + 4]
            if len({sample["case_id"] for sample in group}) != 1:
                raise RuntimeError("A GRPO group mixes different cases")
            values = [sample["reward"] for sample in group]
            mean = sum(values) / 4
            signal = max(values) - min(values) > 1e-8 and any(
                sample["terminated"] and abs(sample["reward"] - mean) > 1e-8 for sample in group
            )
            event = {"group": len(self.groups) + 1, "optimizer_step_before": trainer_state.global_step,
                     "case_id": group[0]["case_id"], "rewards": values, "policy_signal": signal,
                     "sample_source": "scripted_cpu_self_check" if self.synthetic else "model_generated",
                     "samples": group}
            self.groups.append(event)
            with (self.run_dir / "rollouts.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(f'GROUP {event["group"]} | {event["case_id"]} | rewards={values} | signal={signal}', flush=True)
            if len(self.groups) <= 2:
                for sample in group:
                    print("  SAMPLE:", sample["raw_output"].strip(), "| reward:", sample["reward"], flush=True)
            rewards.extend(values)
        return rewards


def make_config(run_dir, config, synthetic):
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=str(run_dir / "trainer"), max_steps=2 if synthetic else config["steps"],
        per_device_train_batch_size=1, gradient_accumulation_steps=4, num_generations=4,
        max_completion_length=config["max_completion_tokens"],
        learning_rate=config["learning_rate"], lr_scheduler_type="constant", weight_decay=0.0,
        beta=config["beta"], entropy_coef=0.0, max_grad_norm=1.0, epsilon=0.2,
        loss_type="grpo", scale_rewards="group", num_iterations=1,
        temperature=config["temperature"], top_p=config["top_p"], top_k=config["top_k"],
        chat_template_kwargs={"enable_thinking": False}, mask_truncated_completions=True,
        use_vllm=False, use_cpu=synthetic, bf16=not synthetic, fp16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        disable_dropout=True, optim="adamw_torch", logging_steps=1 if synthetic else 6,
        logging_first_step=True, save_strategy="no", eval_strategy="no",
        report_to="none", push_to_hub=False, dataloader_num_workers=0,
        dataloader_pin_memory=False, remove_unused_columns=False,
        seed=config["seed"], data_seed=config["seed"], disable_tqdm=True, log_completions=False,
    )


def run(args, run_dir, result):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, TrainerCallback
    from trl import GRPOTrainer

    config, cases, split_hash, baseline_path = validate_inputs(args)
    result.update(configuration=config, dataset_id=config["dataset_id"], train_split_sha256=split_hash,
                  baseline_result_path=str(baseline_path), prompt_version=config["prompt_version"],
                  versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "trl", "peft", "accelerate")})
    torch.manual_seed(config["seed"])
    if args.self_check:
        from run_grpo_pilot import tiny_self_check
        model, tokenizer, _, rollout_func = tiny_self_check()
        cases = [{"case_id": "synthetic-only/m1", "problem_id": "synthetic-only", "split": "train",
                  "domain": {"x": {"type": "int", "min": 0, "max": 40}},
                  "reference": "def solve(x):\n    return x >= 5\n",
                  "buggy_code": "def solve(x):\n    return x > 5\n"}]
        rows = [{"case_id": cases[0]["case_id"], "prompt": "prompt"}] * 4
        result["dataset_id"] = "synthetic-cpu-only"
        print("SYNTHETIC CPU CHECK: scripted samples, no pretrained model training", flush=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left")
        rows = [{"case_id": case["case_id"], "prompt": tokenizer.apply_chat_template(
            model_prompt(case, version=config["prompt_version"]), tokenize=False,
            add_generation_prompt=True, enable_thinking=False)} for case in cases]
        lengths = [len(tokenizer.encode(row["prompt"], add_special_tokens=False)) for row in rows]
        if max(lengths) > 768:
            raise RuntimeError("Training prompt exceeds the checked limit")
        result["prompt_token_range"] = [min(lengths), max(lengths)]
        training_args = make_config(run_dir, config, False)
        if args.preflight:
            result["status"] = "preflight_passed_no_weights_loaded"
            print("PREFLIGHT OK: 36 Train cases, plain-v2, frozen baseline and hashes match.")
            print("Planned steps:", training_args.max_steps, "| prompt tokens:", result["prompt_token_range"])
            print("No weights loaded. No training or Validation/Test model evaluation.")
            return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA and BF16 are required")
        result["gpu"] = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
        print("GPU:", result["gpu"], flush=True)
        print("[1/4] Load original base; previous arithmetic/pilot adapters are not used", flush=True)
        model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation="sdpa", output_loading_info=True,
        )
        result["loading_info"] = info
        if info.get("missing_keys") or info.get("mismatched_keys"):
            raise RuntimeError("Checkpoint weights missing or mismatched")
        rollout_func = None
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    declared = model.generation_config.eos_token_id
    eos = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    eos.update(declared if isinstance(declared, list) else [declared])
    audit = TrainingRewards(cases, run_dir, eos - {None}, synthetic=args.self_check)
    gradients = []
    checkpoints = []
    interval = 1 if args.self_check else config["checkpoint_interval"]

    class AuditCallback(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            values = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
            if not values or not all(torch.isfinite(value).all().item() for value in values):
                raise RuntimeError("Missing or non-finite policy gradients")
            norm = math.sqrt(sum(float(value.detach().float().square().sum().cpu()) for value in values))
            if not math.isfinite(norm):
                raise RuntimeError("Non-finite policy gradient norm")
            if any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
                raise RuntimeError("Unexpected gradient on frozen base")
            event = {"step": state.global_step + 1, "gradient_norm": norm, "frozen_parameters_have_no_grad": True}
            gradients.append(event)
            with (run_dir / "gradient-audit.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step % interval == 0:
                checkpoint = save_adapter_checked(model, run_dir / f"checkpoint-{state.global_step}", state.global_step)
                checkpoints.append(checkpoint)
                print("ADAPTER CHECKPOINT SAVED:", checkpoint["path"], flush=True)

    trainer = GRPOTrainer(
        model=model, args=make_config(run_dir, config, args.self_check), reward_funcs=audit.reward,
        train_dataset=Dataset.from_list(rows), processing_class=tokenizer,
        peft_config=LoraConfig(task_type=TaskType.CAUSAL_LM, r=config["lora_rank"], lora_alpha=config["lora_alpha"],
                               lora_dropout=0.0, bias="none", target_modules=TARGET_MODULES),
        callbacks=[AuditCallback()], rollout_func=rollout_func,
    )
    trainable = {name: parameter for name, parameter in trainer.model.named_parameters() if parameter.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Only adapter parameters should be trainable")
    initial = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}
    result["trainable_parameters"] = sum(parameter.numel() for parameter in trainable.values())
    print("[2/4] STATIC GRPO TRAINING | cases:", len(cases), "| steps:", trainer.args.max_steps, flush=True)
    print("TRAINABLE PARAMS:", result["trainable_parameters"], flush=True)
    started = time.perf_counter()
    trained = trainer.train()
    if trainer.state.global_step != trainer.args.max_steps:
        raise RuntimeError("Training ended before its preplanned step budget")
    result["training_seconds"] = round(time.perf_counter() - started, 3)
    result.update(optimizer_steps=trainer.state.global_step, trainer_metrics=trained.metrics,
                  groups=len(audit.groups), groups_with_policy_signal=sum(group["policy_signal"] for group in audit.groups),
                  nonzero_gradient_steps=sum(item["gradient_norm"] > 0 for item in gradients), checkpoints=checkpoints)
    delta = {name: float((parameter.detach().cpu().float() - initial[name].float()).abs().max()) for name, parameter in trainable.items()}
    result["max_parameter_delta"] = max(delta.values())
    result["changed_adapter_tensors"] = [name for name, change in delta.items() if change > 0]
    print("[3/4] Save final adapter and verify exact parameter match", flush=True)
    result["final_adapter"] = save_adapter_checked(trainer.model, run_dir / "adapter", trainer.state.global_step)
    verified = bool(result["changed_adapter_tensors"]) and result["groups_with_policy_signal"] > 0 and result["nonzero_gradient_steps"] > 0
    result["reward_driven_update_verified"] = verified
    result["rl_training"] = verified and not args.self_check
    result["status"] = ("synthetic_static_self_check_passed" if args.self_check else "static_grpo_completed") if verified else "completed_without_verified_update"
    (run_dir / "trainer-log-history.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    if not args.self_check:
        result["peak_torch_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        result["peak_torch_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        print("Peak PyTorch allocated VRAM (GiB):", result["peak_torch_allocated_gib"], flush=True)
    print("GROUPS WITH POLICY SIGNAL:", result["groups_with_policy_signal"], flush=True)
    print("NONZERO GRADIENT STEPS:", result["nonzero_gradient_steps"], flush=True)
    print("REWARD-DRIVEN UPDATE VERIFIED:", verified, flush=True)
    print("STATUS:", result["status"], flush=True)
    print("[4/4] Training finished. Quality must be measured separately on Validation.", flush=True)
    if args.self_check and not verified:
        raise RuntimeError("Synthetic check did not verify an adapter update")


def main():
    parser = argparse.ArgumentParser(description="Static-data GRPO training on the frozen Train split")
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "static-grpo-v1.json")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check and args.preflight:
        parser.error("Choose only one preparation check")
    args.model = args.model or args.assets / "models" / "Qwen3.5-2B"
    for key, value in {
        "HF_HOME": str(args.assets / "cache" / "huggingface"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(args.assets / "cache" / "torchinductor"),
        "TEMP": str(args.assets / "temp"), "TMP": str(args.assets / "temp"), "WANDB_MODE": "disabled",
    }.items():
        os.environ[key] = value
    (args.assets / "temp").mkdir(parents=True, exist_ok=True)
    tag = "static-self-check-" if args.self_check else "static-grpo-"
    run_dir = args.assets / "evidence" / (tag + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {"started_at": datetime.now().astimezone().isoformat(), "status": "started", "rl_training": False,
              "quality_evaluation": False, "synthetic_cpu_self_check": args.self_check, "preflight_only": args.preflight,
              "model_path": str(args.model), "config_sha256": file_hash(args.config), "source_hashes": {}}
    source_dir = run_dir / "source"; source_dir.mkdir()
    for filename in ("train_static_grpo.py", "dataset_tasks.py", "evaluate_model.py", "check_local_model.py", "check_lora_update.py", "run_grpo_pilot.py", "pilot_tasks.py"):
        source = Path(__file__).with_name(filename)
        shutil.copy2(source, source_dir / filename)
        result["source_hashes"][filename] = file_hash(source)
    shutil.copy2(args.config, source_dir / "static-grpo-v1.json")
    stdout, stderr = sys.stdout, sys.stderr
    exit_code = 0
    with (run_dir / "console.txt").open("w", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        try:
            print("Evidence:", run_dir, flush=True)
            run(args, run_dir, result)
        except KeyboardInterrupt:
            result["status"] = "interrupted"; exit_code = 130
            print("Interrupted; prior adapter checkpoints and logs remain available.")
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__, error=str(error)); exit_code = 1
            traceback.print_exc()
        finally:
            result["finished_at"] = datetime.now().astimezone().isoformat()
            result_path = run_dir / "result.json"
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            if result["status"] == "static_grpo_completed":
                pointer = {"run_dir": str(run_dir), "result_sha256": file_hash(result_path),
                           "adapter_path": result["final_adapter"]["path"], "updated_at": result["finished_at"]}
                (args.assets / "evidence" / "latest-static-training.json").write_text(json.dumps(pointer, indent=2), encoding="utf-8")
            print("Saved evidence:", run_dir, flush=True)
            sys.stdout, sys.stderr = stdout, stderr
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
