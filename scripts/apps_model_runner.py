import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from apps_core import ASSETS, DATA, PROMPT_VERSION, canonical, digest, load_cases, metrics, model_prompt, read, score, write
from local_model_loader import load_qwen_local
from restricted_python import RestrictedFunction


PROJECT = Path(__file__).resolve().parents[1]
STATE = PROJECT / "APPS_STATE.json"


def run_folder(label):
    stamp = datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d-%H%M%S-%f")
    folder = ASSETS / "evidence" / ("apps-" + label + "-" + stamp)
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def freeze_sources(folder):
    names = ("apps_core.py", "apps_profiles.py", "apps_model_runner.py", "restricted_python.py", "local_model_loader.py", "train_static_grpo.py", "check_lora_update.py", "apps_curriculum.py", "apps_output_contracts.py", "apps_rewards.py", "apps_lora.py")
    (folder / "source").mkdir()
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), folder / "source" / name)
    return {name: digest(Path(__file__).with_name(name)) for name in names}


def load_model(adapter=None, train=False, lora_profile="attention-qv-v1"):
    import torch
    from transformers import AutoTokenizer
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from apps_lora import PROFILES

    path = ASSETS / "models/Qwen3.5-2B"
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side="left")
    base, loading = load_qwen_local(path, output_loading_info=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        raise RuntimeError("Unexpected model loading result")
    if adapter:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=train, local_files_only=True)
    elif train:
        model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
                              lora_dropout=0, bias="none", target_modules=PROFILES[lora_profile]))
    else:
        model = base
    if not train:
        model.requires_grad_(False)
        model.eval()
    declared = model.generation_config.eos_token_id
    eos = sorted({tokenizer.eos_token_id, *(declared if isinstance(declared, list) else [declared])} - {None})
    model.generation_config.eos_token_id = eos
    model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    return model, tokenizer, eos


def render_prompt(tokenizer, case, version=PROMPT_VERSION):
    return tokenizer.apply_chat_template(model_prompt(case, version), tokenize=False, add_generation_prompt=True, enable_thinking=False)


def evaluate(args, folder, result):
    import torch
    cases, manifest = load_cases(args.split)
    if args.case:
        cases = [case for case in cases if case["case_id"] == args.case]
        if not cases:
            raise ValueError("Case not in the selected split")
    result.update(dataset_id=manifest["dataset_id"], split=args.split,
                  split_sha256=manifest["splits"][args.split]["sha256"], adapter=str(args.adapter) if args.adapter else None,
                  adapter_sha256=digest(args.adapter / "adapter_model.safetensors") if args.adapter else None,
                  seed=args.seed, tests_per_case=4, prompt_version=args.prompt_version,
                  model_test_evaluation=False, complete_split=not bool(args.case), training=False,
                  generation={"max_new_tokens": 96, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "thinking": False})
    model, tokenizer, eos = load_model(args.adapter)
    predictions = []
    started = time.perf_counter()
    for index, case in enumerate(cases):
        prompt = render_prompt(tokenizer, case, args.prompt_version)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda:0")
        if inputs.input_ids.shape[1] > 1536:
            raise RuntimeError("Prompt exceeds predeclared token budget")
        seed = (args.seed + int(hashlib.sha256(case["case_id"].encode()).hexdigest()[:8], 16)) % 2**32
        torch.manual_seed(seed)
        with torch.inference_mode():
            output = model.generate(**inputs, num_return_sequences=4, max_new_tokens=96, max_time=120,
                                    do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
                                    eos_token_id=eos, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, use_cache=True)
        reference = RestrictedFunction(case["reference"], case["arguments"])
        buggy = RestrictedFunction(case["buggy_code"], case["arguments"])
        samples = []
        for tokens in output[:, inputs.input_ids.shape[1]:].tolist():
            end = next((i for i, token in enumerate(tokens) if token in eos), None)
            actual = tokens[:end + 1] if end is not None else tokens
            sample = score(tokenizer.decode(actual, skip_special_tokens=True).strip(), end is not None, case, reference, buggy)
            if sample.get("error") == "verifier_limit":
                raise RuntimeError("Verifier limit during evaluation")
            sample["generated_tokens"] = len(actual)
            samples.append(sample)
        row = {"case_id": case["case_id"], "source_id": case["source_id"], "samples": samples,
               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "case_seed": seed}
        predictions.append(row)
        with (folder / "predictions.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical(row) + "\n")
        print(f'CASE {index + 1}/{len(cases)} | {case["case_id"]} | valid={sum(s["valid_format"] for s in samples)}/4 | hits={sum(s["bug_detected"] for s in samples)}/4', flush=True)
        result.update(status="running", metrics=metrics(predictions))
        write(folder / "result.json", result)
        del output, inputs
    result.update(status="evaluation_completed", metrics=metrics(predictions), elapsed_seconds=round(time.perf_counter() - started, 3),
                  peak_vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3), gpu=torch.cuda.get_device_name(0))
    write(folder / "result.json", result)
    if not args.no_publish and not args.case and args.split == "validation" and args.seed == 42:
        state = read(STATE) if STATE.exists() else {}
        key = "pilot_validation" if args.adapter else "baseline_validation"
        if args.adapter and result["adapter_sha256"] != state.get("pilot_training", {}).get("adapter_sha256"):
            return
        state.update(dataset_id=manifest["dataset_id"], dataset_path=str(DATA))
        state[key] = {"path": str(folder), "result_sha256": digest(folder / "result.json")}
        write(STATE, state)


def train(args, folder, result):
    import torch
    from datasets import Dataset
    from peft.utils import get_peft_model_state_dict
    from transformers import TrainerCallback
    from trl import GRPOTrainer
    from train_static_grpo import make_config, save_adapter_checked

    from apps_curriculum import training_cases
    from apps_rewards import PROFILES, training_reward
    reward_profile = getattr(args, "reward_profile", "legacy-v1")
    cases, manifest, counts = training_cases(args.curriculum, args.curriculum_sha256)
    state = read(STATE)
    baseline_path = args.baseline_run or Path(state["baseline_validation"]["path"])
    baseline = read(baseline_path / "result.json")
    if baseline["dataset_id"] != manifest["dataset_id"] or baseline["status"] != "evaluation_completed" or baseline["prompt_version"] != args.prompt_version:
        raise RuntimeError("Completed matching baseline required")
    config = {"steps": args.steps, "max_completion_tokens": 96, "learning_rate": 1e-5,
              "beta": 0.0, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": args.seed}
    result.update(status="preparing", dataset_id=manifest["dataset_id"], prompt_version=args.prompt_version,
                  train_sha256=manifest["splits"]["train"]["sha256"], baseline_path=str(baseline_path),
                  baseline_result_sha256=digest(baseline_path / "result.json"),
                  configuration=config, training=True, model_test_evaluation=False,
                  initialization="Declared APPS parent adapter; new GRPO optimizer" if args.init_adapter else "original Qwen3.5-2B plus new rank-4 LoRA; no previous adapters",
                  parent_adapter_sha256=digest(args.init_adapter / "adapter_model.safetensors") if args.init_adapter else None,
                  scope="GRPO on a frozen per-round curriculum" if args.curriculum else "Static APPS GRPO pilot; adversarial curriculum is not implemented in this run",
                  curriculum_path=str(args.curriculum) if args.curriculum else None,
                  curriculum_sha256=args.curriculum_sha256, curriculum_counts=counts,
                  reward_profile=reward_profile, reward_weights=PROFILES[reward_profile],
                  reward_format_scope="Strict JSON plus bounded input validity; this differs from ATGen thinking tags")
    if args.curriculum:
        shutil.copy2(args.curriculum, folder / "training-cases.jsonl")
    torch.manual_seed(args.seed)
    model, tokenizer, eos = load_model(args.init_adapter, train=True)
    initial = {name: value.detach().cpu().clone() for name, value in get_peft_model_state_dict(model).items()}
    if args.init_adapter:
        from safetensors.torch import load_file
        expected = load_file(str(args.init_adapter / "adapter_model.safetensors"), device="cpu")
        if set(expected) != set(initial) or not all(torch.equal(expected[name], value) for name, value in initial.items()):
            raise RuntimeError("Initial adapter differs from declared parent")
        if baseline["adapter_sha256"] != result["parent_adapter_sha256"]:
            raise RuntimeError("The GRPO baseline must evaluate the same parent adapter")
        result["parent_adapter_exact_match"] = True
    elif not all(torch.count_nonzero(value).item() == 0 for name, value in initial.items() if "lora_B" in name):
        raise RuntimeError("LoRA is not freshly initialized")
    rows = [{"case_id": case["case_id"], "prompt": render_prompt(tokenizer, case, args.prompt_version)} for case in cases]
    if max(len(tokenizer.encode(row["prompt"], add_special_tokens=False)) for row in rows) > 1536:
        raise RuntimeError("Training prompt too long")
    functions = {case["case_id"]: (RestrictedFunction(case["reference"], case["arguments"]), RestrictedFunction(case["buggy_code"], case["arguments"])) for case in cases}
    by_id = {case["case_id"]: case for case in cases}
    events, gradients = [], []
    def reward(completions, case_id, completion_ids, trainer_state, **kwargs):
        values = []
        batch = []
        for raw, identity, tokens in zip(completions, case_id, completion_ids, strict=True):
            case = by_id[identity]
            sample = score(raw.strip(), bool(tokens) and tokens[-1] in eos, case, *functions[identity])
            if sample.get("error") == "verifier_limit":
                raise RuntimeError("Verifier limit; refusing to train with that reward")
            sample.update(case_id=identity, generated_tokens=len(tokens),
                          training_reward=training_reward(sample, reward_profile))
            batch.append(sample)
            values.append(sample["training_reward"])
        for index in range(0, len(batch), 4):
            group = batch[index:index + 4]
            if len(group) != 4 or len({s["case_id"] for s in group}) != 1:
                raise RuntimeError("Invalid GRPO group")
            rewards = [s["training_reward"] for s in group]
            event = {"group": len(events) + 1, "optimizer_step_before": trainer_state.global_step,
                     "case_id": group[0]["case_id"], "rewards": rewards, "reward_profile": reward_profile,
                     "policy_signal": max(rewards) - min(rewards) > 1e-8, "samples": group}
            events.append(event)
            with (folder / "rollouts.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(canonical(event) + "\n")
            print("GROUP", len(events), event["case_id"], rewards, flush=True)
        return values
    class Audit(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            net = kwargs["model"]
            square = sum(float(p.grad.detach().float().square().sum().item()) for p in net.parameters() if p.grad is not None)
            if not __import__("math").isfinite(square):
                raise RuntimeError("Nonfinite gradient")
            if any(p.grad is not None for name, p in net.named_parameters() if "lora_" not in name):
                raise RuntimeError("Base parameters received gradients")
            gradients.append({"step": state.global_step + 1, "norm": square ** 0.5})
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                with (folder / "trainer-metrics.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(canonical({"step": state.global_step, **logs}) + "\n")
    trainer = GRPOTrainer(model=model, args=make_config(folder, config, False),
                          reward_funcs=reward, train_dataset=Dataset.from_list(rows), processing_class=tokenizer, callbacks=[Audit()])
    result["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write(folder / "plan.json", result)
    trainer.train()
    if trainer.state.global_step != args.steps:
        raise RuntimeError("Unexpected optimizer step count")
    current = get_peft_model_state_dict(model)
    changed = [name for name, value in current.items() if not torch.equal(initial[name], value.detach().cpu())]
    checkpoint = save_adapter_checked(model, folder / "adapter", trainer.state.global_step)
    write(folder / "gradients.json", gradients)
    result.update(status="static_grpo_pilot_completed", optimizer_steps=trainer.state.global_step,
                  changed_adapter_tensors=changed, nonzero_gradient_steps=sum(g["norm"] > 0 for g in gradients),
                  policy_signal_groups=sum(e["policy_signal"] for e in events), adapter=checkpoint,
                  peak_vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3), gpu=torch.cuda.get_device_name(0),
                  update_verified=bool(changed) and any(g["norm"] > 0 for g in gradients) and any(e["policy_signal"] for e in events))
    if not result["update_verified"]:
        result["status"] = "pilot_no_verified_policy_update"
    write(folder / "result.json", result)
    if not args.no_publish:
        state["pilot_training"] = {"path": str(folder), "result_sha256": digest(folder / "result.json"),
                                   "adapter_path": str(folder / "adapter"), "adapter_sha256": checkpoint["adapter_sha256"]}
        state.pop("pilot_validation", None)
        write(STATE, state)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for key, value in {"HF_HUB_OFFLINE": "1", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false",
                       "HF_HOME": str(ASSETS / "cache/huggingface"), "TEMP": str(ASSETS / "temp"), "TMP": str(ASSETS / "temp")}.items():
        os.environ[key] = value
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("evaluate", "train"))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=48)
    active = read(PROJECT / "configs/active-dataset.json")
    parser.add_argument("--prompt-version", choices=("apps-call-v1", "apps-call-v2", "apps-call-v3", "apps-call-v4"), default=active.get("prompt_version", PROMPT_VERSION))
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--curriculum", type=Path)
    parser.add_argument("--curriculum-sha256")
    from apps_rewards import PROFILES
    parser.add_argument("--reward-profile", choices=tuple(PROFILES), default="legacy-v1")
    args = parser.parse_args()
    if not 1 <= args.steps <= 96:
        parser.error("Pilot must use 1 to 96 steps")
    if args.action == "train" and args.reward_profile != "legacy-v1" and not args.no_publish:
        parser.error("Research reward profiles require --no-publish")
    folder = run_folder("grpo-pilot" if args.action == "train" else "adapter-validation" if args.adapter else "baseline-validation")
    print("EVIDENCE:", folder, flush=True)
    result = {"status": "started", "action": args.action, "source_hashes": freeze_sources(folder),
              "started_at": datetime.now(timezone.utc).isoformat(), "model_path": str(ASSETS / "models/Qwen3.5-2B"),
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "trl", "peft")}}
    write(folder / "result.json", result)
    try:
        (train if args.action == "train" else evaluate)(args, folder, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        write(folder / "result.json", result)
        raise


if __name__ == "__main__":
    main()
