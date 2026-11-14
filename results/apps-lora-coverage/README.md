# LoRA target coverage

Comparison of self-attention q_proj/v_proj targets with a profile that also includes linear-attention projections.

The original profile has 208,896 trainable parameters across 6/24 blocks; the expanded profile has 798,720 across 24/24. Both SFT64 runs completed, but the trial stopped before Validation to investigate target ordering.

## Data files

[attempt.json](attempt.json) · [coverage.json](coverage.json)

[Evaluation guide](../../docs/EXPERIMENTS.md).
