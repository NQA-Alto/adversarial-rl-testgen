# Expected-first resampling

Four checkpoints evaluated with generation seeds 42/43/44. Each checkpoint comes from one training run.

| Configuration | Valid /60 | Correct expected /60 | Bug-detecting tests /60 | Functions detected /15 |
|---|---|---|---|---|
| Base Qwen | 35.33 ± 0.58 | 14.67 ± 2.52 | 3.67 ± 0.58 | 3.33 ± 0.58 |
| SFT64 | 43.00 ± 2.65 | 11.67 ± 1.53 | 3.67 ± 1.53 | 3.33 ± 1.53 |
| Fixed-code GRPO | 42.67 ± 3.06 | 10.67 ± 2.52 | 3.00 ± 1.00 | 2.67 ± 1.15 |
| Adversarial GRPO | 42.33 ± 3.21 | 11.00 ± 1.73 | 3.33 ± 1.53 | 3.00 ± 1.73 |

Mean ± sample SD across three generation seeds; one training run per configuration.

## Per-run counts

| Configuration | Generation seed | Valid /60 | Correct expected /60 | Bug-detecting tests /60 | Functions detected /15 |
|---|---|---|---|---|---|
| Base Qwen | 42 | 35 | 15 | 4 | 3 |
| SFT64 | 42 | 41 | 13 | 4 | 3 |
| Fixed-code GRPO | 42 | 42 | 13 | 3 | 2 |
| Adversarial GRPO | 42 | 41 | 12 | 3 | 2 |
| Base Qwen | 43 | 35 | 17 | 3 | 3 |
| SFT64 | 43 | 42 | 12 | 2 | 2 |
| Fixed-code GRPO | 43 | 40 | 11 | 2 | 2 |
| Adversarial GRPO | 43 | 40 | 12 | 2 | 2 |
| Base Qwen | 44 | 36 | 12 | 4 | 4 |
| SFT64 | 44 | 46 | 10 | 5 | 5 |
| Fixed-code GRPO | 44 | 46 | 8 | 4 | 4 |
| Adversarial GRPO | 44 | 46 | 9 | 5 | 5 |

The fixed-code branch detected 2.67/15 functions on average; the adversarial branch detected 3.00/15. Generation and filtering costs are recorded in costs.csv.

![Expected-first resampling](sampling.png)

Figure 1. Expected-first resampling.

## Data files

[costs.csv](costs.csv) · [costs.json](costs.json) · [metrics.csv](metrics.csv) · [metrics.json](metrics.json)

[Evaluation guide](../../docs/EXPERIMENTS.md).
