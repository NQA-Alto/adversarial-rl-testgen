# LLM Test Generation

A student project exploring how a local language model can generate tests that expose bugs in Python functions. Each test contains an input and an expected output. An execution-based verifier checks the expected output against a reference solution before counting a detected bug.

The training pipeline combines supervised fine-tuning (SFT), Group Relative Policy Optimization (GRPO) and an adversarial code curriculum, inspired by ATGen. It uses Qwen3.5-2B with LoRA on a bounded subset of APPS Introductory.

## Start here

- **Try the verifier:** run the [small example](#try-the-verifier) below with Python, without downloading a model.
- **Review the experiment:** see the [recorded results](results/apps-expanded-adaptive/README.md) and [evaluation protocol](docs/EXPERIMENTS.md).
- **Explore the code:** use the [source map](#source-map).
- **Run the full application:** see [setup and training](docs/APPS_TRAINING.md). The original Windows setup requires local assets that are not included in this repository.

## How it works

```text
Problem + faulty Python function + input contract
                        |
                  Test generator
                        |
                {input, expected}
                        |
               Execution-based verifier
                 /                 \
       Reference solution      Faulty function
                 \                 /
           Correct expected AND different output?
                        |
                 Bug detected
```

SFT teaches the output format using reference-checked examples. GRPO uses rewards for valid inputs, correct expected outputs and bug detection. In the adversarial branch, a frozen code model proposes new faulty functions; filtering retains candidates that pass the generator's current valid tests but still contain a known error within the declared input domain.

## Try the verifier

From the repository root, using Python 3.12:

```sh
python examples/verify_test.py
python scripts/check_apps_rewards.py
python scripts/check_runtime_limits.py
```

These commands use the Python standard library. The example scores three hand-written tests against a small boundary-condition bug. It demonstrates the verifier, not model inference or the reported APPS results.

## Recorded results

The expanded experiment uses 58 training problems, 5 validation problems and 7 held-out problems, all drawn from APPS Train. Each faulty function receives four generated tests. The table reports mean counts across generation seeds 42, 43 and 44, with one training seed per configuration.

| Checkpoint | Correct expected / 72 | Bug-detecting tests / 72 | Faulty functions detected / 18 |
|---|---:|---:|---:|
| Base Qwen | 14.00 | 7.00 | 5.00 |
| SFT, 840 steps | 24.67 | 14.33 | 8.67 |
| SFT + fixed-code GRPO | 24.67 | 13.33 | 7.67 |
| SFT + adversarial GRPO | 24.67 | 14.00 | 8.33 |

SFT was selected using validation results before the held-out evaluation. It improved on the base model; neither GRPO branch improved bug detection over SFT in this experiment.

[Metrics and variability](results/apps-expanded-adaptive/README.md) · [Machine-readable results](results/apps-expanded-adaptive/metrics.json) · [Frozen protocol](results/apps-expanded-adaptive/protocol.json)

## Source map

| Component | Entry points |
|---|---|
| Input contracts, prompts and scoring | [`apps_core.py`](scripts/apps_core.py), [`restricted_python.py`](scripts/restricted_python.py) |
| Data preparation | [`prepare_apps.py`](scripts/prepare_apps.py), [`prepare_apps_expanded.py`](scripts/prepare_apps_expanded.py) |
| SFT and GRPO | [`apps_sft.py`](scripts/apps_sft.py), [`apps_model_runner.py`](scripts/apps_model_runner.py), [`apps_rewards.py`](scripts/apps_rewards.py) |
| Adversarial code and curriculum | [`apps_adversary.py`](scripts/apps_adversary.py), [`apps_adversarial_round.py`](scripts/apps_adversarial_round.py) |
| Evaluation and audit | [`apps_final_evaluate.py`](scripts/apps_final_evaluate.py), [`audit_apps_run.py`](scripts/audit_apps_run.py) |
| Gradio application | [`apps_ui.py`](scripts/apps_ui.py), [`apps_presentation.py`](scripts/apps_presentation.py), [`ui_services.py`](scripts/ui_services.py) |

The repository also retains earlier experiments and internal-dataset scripts. Start with the APPS entry points above; the [results index](results/README.md) separates the expanded study from earlier trials.

## Scope

The verifier supports a restricted Python syntax and explicitly bounded input domains. The held-out set is small and separate from the official APPS Test split. Generation seeds measure sampling variation, not variation across independently trained models. The results therefore describe this experimental setup rather than a full APPS benchmark reproduction.

Source code, result summaries and plots are included. Model weights, adapters, prepared datasets and raw run artifacts remain external. The original full training/UI environment has not been reproduced on a clean machine.

## Documentation

- [Dataset construction](docs/APPS_DATASET.md)
- [Setup and training](docs/APPS_TRAINING.md)
- [Demo guide](docs/DEMO.md)
- [Evaluation and experiment history](docs/EXPERIMENTS.md)
- [Relationship to ATGen](docs/ATGEN_ALIGNMENT.md)

## References

- [ATGen paper](https://arxiv.org/abs/2510.14635)
- [APPS](https://github.com/hendrycks/apps) and the [codeparrot/apps dataset](https://huggingface.co/datasets/codeparrot/apps)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
