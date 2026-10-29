import inspect
import re

from apps_lora import DEFAULT_PROFILE, PROFILES, coverage
from apps_model_runner import load_model
from check_lora_update import TARGET_MODULES


def check():
    if PROFILES[DEFAULT_PROFILE] != TARGET_MODULES or inspect.signature(load_model).parameters["lora_profile"].default != DEFAULT_PROFILE:
        raise AssertionError("Historical default LoRA configuration changed")
    old, expanded = coverage(DEFAULT_PROFILE), coverage("hybrid-qkv-v1")
    if len(old["covered_blocks"]) != 6 or old["target_module_count"] != 12 or old["trainable_parameters"] != 208896:
        raise AssertionError("Unexpected baseline coverage")
    if len(expanded["covered_blocks"]) != 24 or expanded["target_module_count"] != 30 or expanded["trainable_parameters"] != 798720:
        raise AssertionError("Unexpected hybrid coverage")
    old_names = {row["module"] for row in old["targets"]}
    new_names = {row["module"] for row in expanded["targets"]}
    if not old_names < new_names or any("linear_attn.in_proj_qkv" not in name for name in new_names - old_names):
        raise AssertionError("Research profile changes more than the declared projection coverage")
    for name in ("model.visual.blocks.0.attn.q_proj", "model.language_model.layers.0.mlp.up_proj", "model.language_model.embed_tokens"):
        if re.fullmatch(PROFILES["hybrid-qkv-v1"], name):
            raise AssertionError("Profile unexpectedly selects non-target weights")
    return {"status": "passed", "default_unchanged": True, "baseline": old, "candidate": expanded, "model_loaded_on_gpu": False}


if __name__ == "__main__":
    result = check()
    print({key: value if key not in ("baseline", "candidate") else {field: value[field] for field in ("target_module_count", "trainable_parameters", "covered_blocks")} for key, value in result.items()})
