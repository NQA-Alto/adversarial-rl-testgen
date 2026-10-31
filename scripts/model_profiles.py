from pathlib import Path


PROFILES = {
    "2b": {"directory": "Qwen3.5-2B", "name": "Qwen3.5-2B", "quantized": False},
    "4b": {"directory": "Qwen3.5-4B", "name": "Qwen3.5-4B NF4", "quantized": True},
    "coder3b": {"directory": "Qwen2.5-Coder-3B-Instruct", "name": "Qwen2.5-Coder-3B NF4", "quantized": True},
}


def model_path(assets, profile):
    return Path(assets) / "models" / PROFILES[profile]["directory"]


def load_base(assets, profile):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

    path = model_path(assets, profile)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side="left")
    if profile == "2b":
        from local_model_loader import load_qwen_local
        model, loading = load_qwen_local(path, output_loading_info=True)
    else:
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model_class = Qwen3_5ForConditionalGeneration if profile == "4b" else AutoModelForCausalLM
        model, loading = model_class.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16,
                            device_map={"": 0}, quantization_config=quantization, attn_implementation="sdpa",
                            output_loading_info=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        raise RuntimeError("Model checkpoint does not match its architecture")
    return model, tokenizer
