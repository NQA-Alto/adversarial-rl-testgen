import json
from pathlib import Path


def load_qwen_local(path, **kwargs):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, GenerationConfig, Qwen3_5ForConditionalGeneration

    path = Path(path)
    index = path / "model.safetensors.index.json"
    filenames = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())) if index.is_file() else ["model.safetensors"]
    weights = {}
    for filename in filenames:
        target = (path / filename).resolve()
        if not target.is_relative_to(path.resolve()):
            raise ValueError("Checkpoint path outside model directory")
        weights.update(load_file(target, device="cpu", backend="mmap"))
    values = dict(dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa")
    values.update(kwargs)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(None,
        config=AutoConfig.from_pretrained(path, local_files_only=True), state_dict=weights, **values)
    del weights
    base = model[0] if isinstance(model, tuple) else model
    if (path / "generation_config.json").is_file():
        base.generation_config = GenerationConfig.from_pretrained(path, local_files_only=True)
    return model
