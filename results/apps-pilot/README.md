# Initial APPS pilot

Base Qwen and the initial GRPO run, using prompt v1. Each Validation function receives four tests with generation seed 42.

## Per-run counts

| Configuration | Valid /60 | Correct expected /60 | Bug-detecting tests /60 | Functions detected /15 |
|---|---|---|---|---|
| Base Qwen | 14 | 10 | 2 | 2 |
| Initial GRPO | 17 | 11 | 1 | 1 |

This trial predates the revised prompt and SFT warm start.

## Data files

[metrics.csv](metrics.csv) · [metrics.json](metrics.json)

[Evaluation guide](../../docs/EXPERIMENTS.md).
