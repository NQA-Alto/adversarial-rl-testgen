import json
import re
from pathlib import Path

from apps_core import ASSETS


DEFAULT_PROFILE = "attention-qv-v1"
PROFILES = {
    DEFAULT_PROFILE: r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)",
    "hybrid-qkv-v1": r"model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|v_proj)|linear_attn\.in_proj_qkv)",
}


def coverage(profile):
    from safetensors import safe_open

    pattern = PROFILES[profile]
    model = ASSETS / "models/Qwen3.5-2B"
    index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    selected = []
    for key, filename in sorted(index.items()):
        name = key.removesuffix(".weight")
        if not key.endswith(".weight") or re.fullmatch(pattern, name) is None:
            continue
        target = (model / filename).resolve()
        if not target.is_relative_to(model.resolve()):
            raise ValueError("Checkpoint shard outside model directory")
        with safe_open(target, framework="pt", device="cpu") as stream:
            shape = stream.get_slice(key).get_shape()
        if len(shape) != 2:
            raise ValueError("LoRA target must be a matrix")
        selected.append({"module": name, "shape": shape, "lora_parameters_rank4": 4 * sum(shape)})
    if not selected:
        raise ValueError("LoRA pattern did not match checkpoint matrices")
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))["text_config"]
    blocks = sorted({int(row["module"].split(".layers.")[1].split(".")[0]) for row in selected})
    return {"profile": profile, "rank": 4, "alpha": 8, "target_modules": pattern,
            "transformer_blocks": config["num_hidden_layers"], "covered_blocks": blocks,
            "target_module_count": len(selected), "adapter_tensor_count": 2 * len(selected),
            "trainable_parameters": sum(row["lora_parameters_rank4"] for row in selected), "targets": selected}
