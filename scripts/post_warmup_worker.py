import argparse
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
from dataset_tasks import compile_trusted, model_prompt
from restricted_python import RestrictedFunction
from round1_data import digest, read, rows
from train_static_grpo import TrainingRewards, make_config, save_adapter_checked


class RoundRewards(TrainingRewards):
    def __init__(self, cases, folder, eos):
        self.cases = {case["case_id"]: case for case in cases}
        self.compiled = {case["case_id"]: (compile_trusted(case["reference"]), RestrictedFunction(case["buggy_code"], case["domain"])) for case in cases}
        self.run_dir, self.eos_ids, self.synthetic, self.groups = folder, set(eos), False, []

    def reward(self, *args, **kwargs):
        start = len(self.groups)
        rewards = super().reward(*args, **kwargs)
        if any(sample.get("buggy_error") == "RestrictedCodeError" for group in self.groups[start:] for sample in group["samples"]):
            raise RuntimeError("Restricted verifier hit a limit; abort before using this group's rewards")
        return rewards


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def check_spec(args):
    if not args.root.resolve().is_relative_to((args.assets / "evidence").resolve()):
        raise ValueError("Experiment must stay inside evidence")
    spec = read(args.root / "post-warmup-spec.json")
    for name, expected in spec["source_hashes"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("Source changed since experiment preparation: " + name)
    if digest(Path(spec["parent_adapter_path"]) / "adapter_model.safetensors") != spec["parent_adapter_sha256"]:
        raise ValueError("Parent adapter changed")
    path = args.root / (args.arm + "-curriculum.jsonl")
    if digest(path) != spec["curriculum_hashes"][args.arm]:
        raise ValueError("Curriculum changed")
    cases = rows(path)
    if len(cases) != 96 or len({case["case_id"] for case in cases}) != 96 or any(case["split"] != "train" for case in cases):
        raise ValueError("Expected 96 unique Train slots")
    return spec, cases


def train(args, folder, result):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, TrainerCallback
    from trl import GRPOTrainer
    from check_lora_update import TARGET_MODULES

    spec, cases = check_spec(args) if not args.self_check else ({}, [])
    config = dict(spec["training_config"]) if spec else read(Path(__file__).resolve().parents[1] / "configs/static-grpo-v1.json")
    config.update(steps=96, checkpoint_interval=48)
    config.update(experiment="post-warmup-adversarial-r1", selection_rule="Preplanned final adapter; no checkpoint selection by Validation")
    result["configuration"] = config
    result["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "trl", "peft", "accelerate")}
    if args.self_check:
        from run_grpo_pilot import tiny_self_check
        torch.manual_seed(42)
        base, tokenizer, _, scripted = tiny_self_check()
        prepared = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, lora_dropout=0.0, bias="none", target_modules=TARGET_MODULES))
        with torch.no_grad():
            for name, parameter in prepared.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.001)
        parent_info = save_adapter_checked(prepared, folder / "synthetic-parent", 0)
        model = PeftModel.from_pretrained(prepared.unload(), folder / "synthetic-parent", is_trainable=True, local_files_only=True)
        parent_path = folder / "synthetic-parent"
        cases = [{"case_id": "synthetic-only", "problem_id": "synthetic-only", "origin_case_id": "synthetic-only", "split": "train",
                  "domain": {"x": {"type": "int", "min": 0, "max": 40}}, "reference": "def solve(x):\n    return x >= 5\n", "buggy_code": "def solve(x):\n    return x > 5\n"}]
        training_rows = [{"case_id": "synthetic-only", "prompt": "prompt"}] * 4
        print("SYNTHETIC CPU CHECK: tiny random model, scripted completions, nonzero parent adapter", flush=True)
    else:
        parent_path = Path(spec["parent_adapter_path"])
        tokenizer = AutoTokenizer.from_pretrained(spec["model_path"], local_files_only=True, padding_side="left")
        training_rows = [{"case_id": case["case_id"], "prompt": tokenizer.apply_chat_template(model_prompt(case, version="plain-v2"), tokenize=False, add_generation_prompt=True, enable_thinking=False)} for case in cases]
        lengths = [len(tokenizer.encode(row["prompt"], add_special_tokens=False)) for row in training_rows]
        if max(lengths) > 768:
            raise ValueError("Curriculum prompt exceeds the checked budget")
        result.update(parent_adapter_sha256=spec["parent_adapter_sha256"], parent_run=spec["round_config"]["parent_run"],
                      curriculum_sha256=spec["curriculum_hashes"][args.arm], prompt_token_range=[min(lengths), max(lengths)], model_path=spec["model_path"])
        for case in cases:
            RestrictedFunction(case["buggy_code"], case["domain"])
        if args.preflight:
            make_config(folder, config, False)
            result["status"] = "post_warmup_grpo_preflight_passed_no_weights_loaded"
            print("PREFLIGHT OK:", args.arm, "| 96 slots | prompt tokens:", result["prompt_token_range"], flush=True)
            return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA and BF16 are required")
        result["gpu"] = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
        print("Load original base + SAME frozen parent adapter | arm:", args.arm, flush=True)
        base, loading = Qwen3_5ForConditionalGeneration.from_pretrained(spec["model_path"], local_files_only=True, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation="sdpa", output_loading_info=True)
        if loading.get("missing_keys") or loading.get("unexpected_keys") or loading.get("mismatched_keys"):
            raise RuntimeError("Unexpected base checkpoint loading result")
        result["loading_info"] = loading
        model = PeftModel.from_pretrained(base, parent_path, is_trainable=True, local_files_only=True)
        scripted = None
    expected = load_file(str(parent_path / "adapter_model.safetensors"), device="cpu")
    live = get_peft_model_state_dict(model)
    if set(live) != set(expected) or not all(torch.equal(value.detach().cpu(), expected[name]) for name, value in live.items()):
        raise RuntimeError("Starting adapter differs from parent")
    result["parent_adapter_exact_match"] = True
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    declared = model.generation_config.eos_token_id
    eos = {tokenizer.eos_token_id, tokenizer.pad_token_id, *(declared if isinstance(declared, list) else [declared])} - {None}
    audit = RoundRewards(cases, folder, eos)
    audit.synthetic = args.self_check
    gradients, checkpoints = [], []

    class AuditCallback(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            values = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not values or not all(torch.isfinite(value).all().item() for value in values):
                raise RuntimeError("Missing or non-finite gradient")
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise RuntimeError("Gradient on frozen base")
            norm = math.sqrt(sum(float(value.detach().float().square().sum().cpu()) for value in values))
            event = {"step": state.global_step + 1, "gradient_norm": norm, "frozen_parameters_have_no_grad": True}
            gradients.append(event)
            with (folder / "gradient-audit.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step % (1 if args.use_cpu else 48) == 0:
                checkpoints.append(save_adapter_checked(model, folder / f"checkpoint-{state.global_step}", state.global_step))

    trainer = GRPOTrainer(model=model, args=make_config(folder, config, args.self_check), reward_funcs=audit.reward,
                          train_dataset=Dataset.from_list(training_rows), processing_class=tokenizer,
                          callbacks=[AuditCallback()], rollout_func=scripted)
    parameters = {name: p for name, p in trainer.model.named_parameters() if p.requires_grad}
    if not parameters or any("lora_" not in name for name in parameters):
        raise RuntimeError("Only LoRA parameters may train")
    initial = {name: p.detach().cpu().clone() for name, p in parameters.items()}
    result["trainable_parameters"] = sum(p.numel() for p in parameters.values())
    if not args.self_check and result["trainable_parameters"] != spec["parent_trainable_parameters"]:
        raise RuntimeError("Trainable parameter count differs from parent")
    torch.manual_seed(42)
    print("POST WARMUP GRPO | arm:", args.arm, "| steps:", trainer.args.max_steps, "| parent exact match: True", flush=True)
    start = time.perf_counter()
    trained = trainer.train()
    if trainer.state.global_step != trainer.args.max_steps:
        raise RuntimeError("Incomplete update budget")
    changes = {name: float((p.detach().cpu().float() - initial[name].float()).abs().max()) for name, p in parameters.items()}
    result.update(training_seconds=round(time.perf_counter() - start, 3), optimizer_steps=trainer.state.global_step,
                  trainer_metrics=trained.metrics, max_parameter_delta=max(changes.values()), changed_adapter_tensors=[name for name, delta in changes.items() if delta > 0],
                  groups=len(audit.groups), groups_with_policy_signal=sum(row["policy_signal"] for row in audit.groups),
                  nonzero_gradient_steps=sum(row["gradient_norm"] > 0 for row in gradients), checkpoints=checkpoints)
    result["final_adapter"] = save_adapter_checked(trainer.model, folder / "adapter", trainer.state.global_step)
    verified = bool(result["changed_adapter_tensors"]) and result["groups_with_policy_signal"] > 0 and result["nonzero_gradient_steps"] > 0
    result.update(reward_driven_update_verified=verified, rl_training=verified and not args.self_check,
                  status="synthetic_post_warmup_check_passed" if args.self_check and verified else "post_warmup_grpo_training_completed")
    if not verified:
        raise RuntimeError("Parent-continuation check did not verify a reward-driven update")
    if not args.self_check:
        from collections import Counter
        exposures = Counter(audit.cases[group["case_id"]]["origin_case_id"] for group in audit.groups)
        if len(exposures) != 48 or set(exposures.values()) != {2}:
            raise RuntimeError("Original-case exposure differs from the planned two slots per parent")
        result["origin_case_exposures"] = dict(exposures)
        result["peak_torch_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    write(folder / "trainer-log-history.json", trainer.state.log_history)
    print("PARENT ADAPTER EXACT MATCH: True", flush=True)
    print("REWARD-DRIVEN UPDATE VERIFIED:", verified, flush=True)
    print("OPTIMIZER STEPS:", result["optimizer_steps"], flush=True)


def evaluate(args, folder, result):
    import evaluate_model
    spec, cases = check_spec(args)
    training_folder = args.root / ("train-" + args.arm)
    training = read(training_folder / "result.json") if not args.preflight else None
    if training and (training["status"] != "post_warmup_grpo_training_completed" or training["synthetic_cpu_self_check"]):
        raise ValueError("Expected completed real training")
    adapter = Path(training["final_adapter"]["path"]) if training else None
    if adapter and digest(adapter / "adapter_model.safetensors") != training["final_adapter"]["adapter_sha256"]:
        raise ValueError("Final adapter changed")
    evaluation_args = argparse.Namespace(assets=args.assets, model=Path(spec["model_path"]), adapter=adapter,
        split="validation", seed=42, prompt_version="plain-v2", limit=None, preflight=args.preflight)
    result.update(model_path=spec["model_path"], training_run=str(training_folder), parent_run=spec["round_config"]["parent_run"],
                  evaluator_sha256=digest(Path(__file__).with_name("evaluate_model.py")), task_source_sha256=digest(Path(__file__).with_name("dataset_tasks.py")))
    evaluate_model.run(evaluation_args, folder, result)
    if not args.preflight and (result["generation_config"] != spec["generation_config"] or result["dataset_split_sha256"] != spec["validation_split_sha256"]):
        raise RuntimeError("Validation protocol differs from frozen parent")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--arm", choices=("control", "adversarial"), default="control")
    parser.add_argument("--phase", choices=("train", "evaluate"), default="train")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check and (args.preflight or args.phase != "train"):
        parser.error("Self-check is training-only and separate from preflight")
    if args.root is None and not args.self_check:
        parser.error("Experiment root is required")
    if args.self_check:
        args.root = args.assets / "evidence" / ("post-warmup-self-check-" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    for key, value in {"HF_HOME": str(args.assets / "cache/huggingface"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1", "TORCHINDUCTOR_CACHE_DIR": str(args.assets / "cache/torchinductor"),
        "TEMP": str(args.assets / "temp"), "TMP": str(args.assets / "temp"), "WANDB_MODE": "disabled"}.items():
        os.environ[key] = value
    folder = args.root / (args.phase + "-" + args.arm)
    folder.mkdir(parents=True, exist_ok=False)
    result = {"started_at": datetime.now().astimezone().isoformat(), "status": "started", "arm": args.arm, "phase": args.phase,
              "training": args.phase == "train" and not args.preflight and not args.self_check, "rl_training": False,
              "synthetic_cpu_self_check": args.self_check, "preflight_only": args.preflight, "test_set_evaluation": False, "prompt_version": "plain-v2"}
    original_stdout, original_stderr = sys.stdout, sys.stderr
    code = 0
    with (folder / "console.txt").open("w", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(original_stdout, log), Tee(original_stderr, log)
        try:
            print("Evidence:", folder, flush=True)
            (train if args.phase == "train" else evaluate)(args, folder, result)
            print("STATUS:", result["status"], flush=True)
        except KeyboardInterrupt:
            result["status"] = "interrupted"
            code = 130
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__, error=str(error))
            code = 1
            traceback.print_exc()
        finally:
            result["finished_at"] = datetime.now().astimezone().isoformat()
            write(folder / "result.json", result)
            print("Saved evidence:", folder, flush=True)
            sys.stdout, sys.stderr = original_stdout, original_stderr
    return code


if __name__ == "__main__":
    raise SystemExit(main())
