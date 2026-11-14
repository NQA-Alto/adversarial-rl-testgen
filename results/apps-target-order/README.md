# SFT target field order

The study keeps the same 319 input/expected pairs and training settings while changing the order of the two top-level target fields. SFT and GRPO checkpoints are compared separately.

| Configuration | Valid /60 | Correct expected /60 | Bug-detecting tests /60 | Functions detected /15 |
|---|---|---|---|---|
| sorted-v1 / sft | 41.00 ± 2.65 | 11.33 ± 2.52 | 3.67 ± 1.53 | 3.33 ± 1.15 |
| sorted-v1 / grpo | 43.33 ± 3.51 | 12.00 ± 2.65 | 3.67 ± 1.53 | 3.33 ± 1.53 |
| input-first-v1 / sft | 43.33 ± 3.21 | 19.00 ± 2.65 | 5.33 ± 0.58 | 4.00 ± 1.00 |
| input-first-v1 / grpo | 43.67 ± 3.51 | 20.67 ± 2.52 | 5.67 ± 2.08 | 4.33 ± 1.53 |

Mean ± sample SD across three generation seeds; one training run per configuration.

## Per-run counts

| Target order | Stage | Generation seed | Valid /60 | Correct expected /60 | Bug-detecting tests /60 | Functions detected /15 |
|---|---|---|---|---|---|---|
| sorted-v1 | SFT64 | 42 | 39 | 14 | 5 | 4 |
| sorted-v1 | GRPO | 42 | 43 | 14 | 4 | 3 |
| sorted-v1 | SFT64 | 43 | 40 | 11 | 2 | 2 |
| sorted-v1 | GRPO | 43 | 40 | 13 | 2 | 2 |
| sorted-v1 | SFT64 | 44 | 44 | 9 | 4 | 4 |
| sorted-v1 | GRPO | 44 | 47 | 9 | 5 | 5 |
| input-first-v1 | SFT64 | 42 | 41 | 18 | 6 | 3 |
| input-first-v1 | GRPO | 42 | 40 | 18 | 4 | 3 |
| input-first-v1 | SFT64 | 43 | 42 | 22 | 5 | 4 |
| input-first-v1 | GRPO | 43 | 44 | 23 | 5 | 4 |
| input-first-v1 | SFT64 | 44 | 47 | 17 | 5 | 5 |
| input-first-v1 | GRPO | 44 | 47 | 21 | 8 | 6 |

Input-first targets increased correct expected outputs at all three generation seeds. Most of the change appeared after SFT; the gain in bug-detecting tests was concentrated in rounding problem 2947.

![SFT target field order](comparison.png)

Figure 1. SFT target field order.

## Data files

[costs.csv](costs.csv) · [errors.csv](errors.csv) · [metrics.csv](metrics.csv) · [metrics.json](metrics.json) · [paired-deltas.csv](paired-deltas.csv) · [per-case.csv](per-case.csv) · [per-problem.csv](per-problem.csv)

[Evaluation guide](../../docs/EXPERIMENTS.md).
