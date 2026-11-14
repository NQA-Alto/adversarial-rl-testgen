# Experiment results

Start with the expanded study. The other folders preserve earlier experiments and diagnostics with different datasets, prompts or training budgets.

## Main study

[Expanded APPS experiment](apps-expanded-adaptive/README.md): SFT840, fixed-code GRPO and adversarial GRPO, with Validation and held-out Test results. Includes the dataset ID, protocol and summary metrics.

## Training comparisons

| Study | Question |
|---|---|
| [Initial pilot](apps-pilot/README.md) | How does the first GRPO run compare with the base model? |
| [SFT and GRPO](apps-sft-grpo/README.md) | What changes after SFT64 and GRPO96 with expected-first labels? |
| [Target ordering](apps-target-order/README.md) | Does emitting input before expected change performance? |
| [Shared SFT trajectory](apps-shared-sft-budget/README.md) | How do checkpoints at 64 and 256 steps compare within one run? |
| [GRPO after SFT256](apps-sft256-grpo/README.md) | Does another 96-step GRPO stage help the stronger parent? |
| [Expected-first SFT budget](apps-sft-budget/README.md) | What changes between 64 and 256 steps with the earlier labels? |
| [Reward weights](apps-reward-ablation/README.md) | How do weighted and equal rewards compare? |

## Adversarial curricula

| Study | Evaluation |
|---|---|
| [Expected-first curriculum](apps-adversarial/README.md) | Seed 42 |
| [Expected-first resampling](apps-adversarial-review/README.md) | Generation seeds 42/43/44 |
| [Input-first curriculum](apps-input-first-adversarial/README.md) | Seed 42 |
| [Input-first resampling](apps-input-first-review/README.md) | Generation seeds 42/43/44 |

## Diagnostics and stopped trials

- [Earlier dataset profile](apps-data-profile/README.md)
- [Output error breakdown](apps-error-analysis/README.md)
- [Incorrect expected outputs](apps-expected-diagnosis/README.md)
- [Input difficulty](apps-input-difficulty/README.md)
- [Target serialization](apps-serialization-diagnosis/README.md)
- [LoRA coverage](apps-lora-coverage/README.md)
- [SFT replay gate](apps-expected-budget/README.md)
- [Thinking and output contracts](apps-tuning/README.md)

Each study links to its stored data. Metric definitions and comparison limits are in the [evaluation guide](../docs/EXPERIMENTS.md).

## Render the figures

With Matplotlib installed, run `python scripts/render_saved_results.py` from the repository root. It reads the committed CSV and JSON summaries without loading a model or running an evaluation. Error bars show sample standard deviation across the seeds stated in each figure.

Per-step training curves require the external training logs and are not included here. Recorded source and artifact hashes refer to the original runs.
