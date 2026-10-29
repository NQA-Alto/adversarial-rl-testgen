import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

from apps_core import all_inputs, canonical, digest, load_cases, read, score, write
from apps_model_runner import freeze_sources, load_model, render_prompt, run_folder
from restricted_python import RestrictedFunction


def build_examples(seed=42):
    cases, manifest = load_cases("train")
    rng = random.Random(seed)
    rows = []
    for case in cases:
        reference = RestrictedFunction(case["reference"], case["arguments"])
        buggy = RestrictedFunction(case["buggy_code"], case["arguments"])
        groups = {True: [], False: []}
        for inputs in all_inputs(case):
            target = canonical({"input": inputs, "expected": reference(**inputs)})
            checked = score(target, True, case, reference, buggy)
            if not checked["valid_format"] or not checked["correct_expected"]:
                raise RuntimeError("Invalid supervised target")
            groups[checked["bug_detected"]].append(target)
        for hits in (True, False):
            choices = groups[hits]
            rng.shuffle(choices)
            for index, target in enumerate(choices[:4]):
                rows.append({"sample_id": case["case_id"] + "/" + str(hits) + "/" + str(index),
                             "case_id": case["case_id"], "target": target, "bug_detecting_target": hits})
    return rows, cases, manifest


def train(folder, steps, seed, prompt_version, checkpoint_steps=(), reference_adapter=None, lora_profile="attention-qv-v1"):
    import torch
    from peft.utils import get_peft_model_state_dict
    from check_lora_update import completion_loss
    from train_static_grpo import save_adapter_checked

    rows, cases, manifest = build_examples(seed)
    source = freeze_sources(folder)
    source["apps_sft.py"] = digest(Path(__file__))
    __import__("shutil").copy2(__file__, folder / "source/apps_sft.py")
    path = folder / "targets.jsonl"
    path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
    config = {"steps": steps, "gradient_accumulation": 4, "learning_rate": 5e-5, "seed": seed,
              "prompt_version": prompt_version, "max_total_tokens": 1664, "max_target_tokens": 96}
    if lora_profile != "attention-qv-v1":
        config["lora_profile"] = lora_profile
    if checkpoint_steps:
        config["checkpoint_steps"] = sorted(checkpoint_steps)
    result = {"status": "started", "method": "SFT_NOT_RL", "source_hashes": source, "configuration": config,
              "dataset_id": manifest["dataset_id"], "train_sha256": manifest["splits"]["train"]["sha256"],
              "target_sha256": digest(path), "training_examples": len(rows), "model_test_evaluation": False,
              "initialization": "Original Qwen3.5-2B and fresh rank-4 LoRA; APPS Train targets only"}
    write(folder / "plan.json", result)
    write(folder / "result.json", result)
    torch.manual_seed(seed)
    model, tokenizer, eos = load_model(train=True, lora_profile=lora_profile)
    from apps_lora import coverage
    result["lora_coverage"] = coverage(lora_profile)
    result["gpu"] = torch.cuda.get_device_name(0)
    result["trainable_parameters"] = sum(value.numel() for value in model.parameters() if value.requires_grad)
    if result["trainable_parameters"] != result["lora_coverage"]["trainable_parameters"]:
        raise RuntimeError("Live LoRA count differs from checkpoint coverage audit")
    initial = {name: value.detach().cpu().clone() for name, value in get_peft_model_state_dict(model).items()}
    if not all(torch.count_nonzero(value).item() == 0 for name, value in initial.items() if "lora_B" in name):
        raise RuntimeError("SFT adapter must start fresh")
    by_id = {case["case_id"]: case for case in cases}
    encoded = []
    for row in rows:
        prefix = tokenizer.encode(render_prompt(tokenizer, by_id[row["case_id"]], prompt_version), add_special_tokens=False)
        target = tokenizer.encode(row["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(prefix) + len(target) > config["max_total_tokens"] or len(target) > config["max_target_tokens"]:
            raise RuntimeError("SFT token budget exceeded")
        encoded.append({"sample_id": row["sample_id"], "tokens": prefix + target, "target": target})
    result["target_token_range"] = [min(len(row["target"]) for row in encoded), max(len(row["target"]) for row in encoded)]
    write(folder / "result.json", result)
    def loss_for(row):
        ids = torch.tensor([row["tokens"]], dtype=torch.long, device="cuda:0")
        targets = torch.tensor([row["target"]], dtype=torch.long, device="cuda:0")
        return completion_loss(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, targets)
    def probe():
        model.eval()
        with torch.no_grad():
            return sum(float(loss_for(row).item()) for row in encoded[:8]) / 8
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    before = probe()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if any("lora_" not in name for name in trainable):
        raise RuntimeError("Frozen base must not be trainable")
    optimizer = torch.optim.AdamW(trainable.values(), lr=config["learning_rate"], weight_decay=0, foreach=False)
    order = list(range(len(encoded)))
    random.Random(seed).shuffle(order)
    write(folder / "training-order.json", [encoded[index]["sample_id"] for index in order])
    model.train()
    started = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(4):
            row = encoded[order[(step * 4 + micro) % len(order)]]
            loss = loss_for(row)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite SFT loss")
            losses.append(float(loss.detach().item()))
            (loss / 4).backward()
        if any(value.grad is not None for name, value in model.named_parameters() if "lora_" not in name):
            raise RuntimeError("Base received SFT gradients")
        norm = float(torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1).item())
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError("No finite SFT gradient")
        optimizer.step()
        event = {"step": step + 1, "loss": sum(losses) / 4, "gradient_norm": norm, "examples_in_step": 4, "method": "SFT_NOT_RL"}
        with (folder / "sft-steps.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical(event) + "\n")
        print("SFT STEP", step + 1, "/", steps, "loss", round(event["loss"], 5), flush=True)
        if step + 1 in checkpoint_steps:
            checkpoint = save_adapter_checked(model, folder / ("checkpoint-" + str(step + 1)), step + 1)
            result.setdefault("checkpoints", {})[str(step + 1)] = checkpoint
            if reference_adapter and step + 1 == 64:
                from safetensors.torch import load_file
                saved = load_file(str(Path(reference_adapter) / "adapter_model.safetensors"), device="cpu")
                live = {name: value.detach().cpu() for name, value in get_peft_model_state_dict(model).items()}
                if set(saved) != set(live):
                    raise RuntimeError("Reference adapter keys differ")
                squared = sum(float((live[name].float() - saved[name].float()).square().sum()) for name in saved)
                baseline_squared = sum(float(value.float().square().sum()) for value in saved.values())
                replay = {"reference_adapter": str(reference_adapter), "exact_tensor_match": all(torch.equal(live[name], saved[name]) for name in saved),
                          "relative_l2_difference": math.sqrt(squared / max(baseline_squared, 1e-30)),
                          "max_absolute_difference": max(float((live[name].float() - saved[name].float()).abs().max()) for name in saved)}
                result["step64_replay"] = replay
                write(folder / "step64-replay.json", replay)
                print("SFT STEP 64 REPLAY:", replay, flush=True)
            write(folder / "result.json", result)
    optimizer.zero_grad(set_to_none=True)
    model.gradient_checkpointing_disable()
    after = probe()
    state = get_peft_model_state_dict(model)
    changed = [name for name, value in state.items() if not torch.equal(value.detach().cpu(), initial[name])]
    if not changed:
        raise RuntimeError("SFT adapter did not change")
    adapter = save_adapter_checked(model, folder / "adapter", steps)
    result.update(status="apps_supervised_warmup_completed", optimizer_steps=steps, nonzero_gradient_steps=steps,
                  examples_seen=steps * 4, changed_adapter_tensors=changed, adapter=adapter,
                  train_probe_nll_before=before, train_probe_nll_after=after, training_seconds=round(time.perf_counter() - started, 3),
                  probe_note="Same eight Train examples; measures fitting only, not generalization or RL improvement",
                  peak_vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3), update_verified=True)
    write(folder / "result.json", result)
    print("SFT COMPLETE; NOT RL", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", default=[])
    parser.add_argument("--reference-adapter", type=Path)
    from apps_lora import PROFILES
    parser.add_argument("--lora-profile", choices=tuple(PROFILES), default="attention-qv-v1")
    args = parser.parse_args()
    if not 1 <= args.steps <= 256 or any(not 1 <= step <= args.steps for step in args.checkpoint_steps):
        parser.error("SFT steps must be 1..256 and checkpoint steps must be within that budget")
    if args.reference_adapter and 64 not in args.checkpoint_steps:
        parser.error("Reference replay requires a checkpoint at step 64")
    import os
    from apps_core import ASSETS
    os.environ.update(HF_HUB_OFFLINE="1", HF_HOME=str(ASSETS / "cache/huggingface"), TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"), PYTHONUTF8="1")
    folder = run_folder("sft-v2")
    print("EVIDENCE:", folder, flush=True)
    try:
        train(folder, args.steps, 42, "apps-call-v2", args.checkpoint_steps, args.reference_adapter, args.lora_profile)
    except Exception as error:
        result = read(folder / "result.json") if (folder / "result.json").exists() else {}
        result.update(status="failed", error=repr(error))
        write(folder / "result.json", result)
        raise
