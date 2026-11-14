# Setup and training

There are two ways to explore this repository: run the standalone verifier on a normal Python installation, or restore the original local model environment for inference and training.

## Standalone verifier

Python 3.12 is the recorded runtime. No third-party packages or GPU are required for:

```sh
python examples/verify_test.py
python scripts/check_apps_rewards.py
python scripts/check_runtime_limits.py
```

These checks exercise scoring and interpreter limits. They do not train a model or reproduce the stored benchmark results.

## Original full environment

| Component | Recorded configuration |
|---|---|
| Operating system | Windows |
| CPU | AMD Ryzen 7 7840H |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB VRAM |
| RAM | 16 GB |
| Python | 3.12.15 |
| PyTorch | 2.14.1+cu132 |
| Transformers / PEFT / TRL | 5.20.2 / 0.21.2 / 1.16.0 |
| Base model | Qwen3.5-2B, BF16 |
| LoRA | Rank 4, alpha 8; 208,896 trainable parameters in the selected profile |

`requirements.txt` records the training environment; `requirements-ui.txt` lists the UI dependencies. `requirements-quantization.txt` adds bitsandbytes for quantization experiments. These are recorded dependency snapshots, not a verified clean-install recipe. In particular, the PyTorch CUDA build needs a compatible package source and driver setup.

The original resource layout is:

```text
D:/DA/TTNT_De19_Assets/
  env/                              training Python environment
  tools/gui/env/                    UI Python environment
  models/Qwen3.5-2B/                 base weights and tokenizer
  data/apps-source/<revision>/       pinned APPS source
  data/apps-introductory-v2/         prepared base subset
  data/apps-introductory-expanded-v4/ expanded training subset
  evidence/                         adapters, raw outputs and run records
```

Those assets are not included in Git. Launchers and several scripts also refer to the original project location, `D:/DA/DoAn_TTNT_De19/adversarial-rl-testgen`. Moving the checkout alone is insufficient to run the full application. A clean-machine installation has not been validated.

## Dataset preparation

[`fetch_apps.py`](../scripts/fetch_apps.py) downloads the pinned source. [`prepare_apps.py`](../scripts/prepare_apps.py) builds v2, and [`prepare_apps_expanded.py`](../scripts/prepare_apps_expanded.py) extends Train using the local tokenizer. The expansion copies Validation and Test unchanged and refuses to overwrite an existing target. See [dataset construction](APPS_DATASET.md) for the selection criteria.

The commands below assume that the original assets and environments have been restored. Run them from the repository root.

```powershell
$env:PYTHONUTF8 = '1'
$env:TTNT_APPS_DATA_DIR = 'D:/DA/TTNT_De19_Assets/data/apps-introductory-expanded-v4'
$trainPython = 'D:/DA/TTNT_De19_Assets/env/Scripts/python.exe'
& $trainPython -B scripts/verify_apps.py
& $trainPython -B scripts/apps_sft.py --steps 840 --target-order input-first-v1
```

The selected SFT run used 1,050 labels, four examples per step, learning rate `5e-5` and seed 42.

## GRPO

Each group contains four tests for the same faulty function. Reward profile `equal-v1` assigns equal contributions to valid format/input, correct expected output and bug detection. A wrong expected output cannot earn bug-detection reward. Training updates the LoRA adapter.

For an individual fixed-code run, substitute actual adapter and baseline directories:

```powershell
& $trainPython -B scripts/apps_model_runner.py train `
  --steps 64 --seed 42 --prompt-version apps-call-v2 `
  --reward-profile equal-v1 --learning-rate 0.000005 `
  --init-adapter '<SFT adapter directory>' `
  --baseline-run '<matching Validation run directory>' --no-publish

& $trainPython -B scripts/apps_model_runner.py evaluate `
  --adapter '<adapter directory>' --seed 42 `
  --prompt-version apps-call-v2 --no-publish
```

`--no-publish` prevents an update to local demo state; it has no connection to GitHub publication. Each run still writes its own output directory.

The recorded expanded comparison used two rounds of 32 steps per branch from the same SFT840 parent. The commands above are individual runs, not a complete replay of that comparison. Its settings and hashes are in the [frozen protocol](../results/apps-expanded-adaptive/protocol.json).

## Balanced-curriculum follow-up

`apps_balanced_study.py` implements a separate two-round, 64-step-per-round comparison. Half the sampled groups come from adversarial positions; the fixed-code branch uses matching case IDs. Before round two, unresolved faulty functions are retained and already-detected ones become replacement candidates.

```powershell
& $trainPython -B scripts/apps_balanced_study.py `
  --parent-study '<existing SFT840 comparison directory>'
```

The pilot uses training seed 42. A seed-43 confirmation is conditional on its validation gate. This follow-up uses Train/Validation and is separate from the published 32-step-per-round held-out result.

## Inspecting runs

`audit_apps_run.py` checks stored outputs, rewards, sampling schedules and adapter hashes. `check_apps_sampling.py` checks the actual TRL sampler. Both require the relevant run artifacts or training dependencies.

For the application controls, see the [demo guide](DEMO.md).
