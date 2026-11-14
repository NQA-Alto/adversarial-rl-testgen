# Dataset construction

The project uses call-based Python problems from the Introductory portion of APPS Train. The source is [codeparrot/apps](https://huggingface.co/datasets/codeparrot/apps), pinned to revision `21e74ddf8de1a21436da12e3e653065c5213e9d1`.

## Expanded dataset

Dataset ID: `apps-intro-expanded-v4-0b00cd05b256`.

| Split | Source problems | Faulty functions |
|---|---:|---:|
| Train | 58 | 145 |
| Validation | 5 | 15 |
| Held-out Test | 7 | 18 |

The expanded training set provides 1,050 SFT examples. Validation and Test retain the files from v2 byte for byte. Problem IDs and source URLs do not overlap across splits. All three splits come from APPS Train; the official APPS Test set was not used.

## Construction pipeline

1. Select problems with an explicit finite input contract and compatible Python solutions.
2. Normalize the entry point to `solve`.
3. Require two source solutions to pass the original tests and agree over every input in the declared domain.
4. Create faulty functions by mutating an AST operator, comparison, constant or index.
5. Reject unsupported code, behaviorally duplicate mutations and mutations that fail on every allowed input.
6. Save split files, source identifiers, reference solutions, witnesses and hashes for verification.

The v4 audit records 56,152 function/input checks. This is a dataset audit, separate from model evaluation.

During adversarial training, additional faulty functions are sampled from the frozen code model and filtered against the current policy. That step uses Train only.

## Model input and target

The test generator receives the problem statement, the faulty function and the input contract. It returns one JSON object with `input` and `expected`. Reference solutions and known failing inputs belong to the verifier and are not included in the test-generation prompt.

SFT targets contain valid inputs with expected outputs computed by a reference solution. Some detect a bug; others are correct tests that do not distinguish the faulty function.

Malformed, out-of-domain and truncated model outputs remain in evaluation denominators.

## Files and configuration

- [`fetch_apps.py`](../scripts/fetch_apps.py): download and verify the pinned source files.
- [`prepare_apps.py`](../scripts/prepare_apps.py): construct the bounded v2 dataset.
- [`prepare_apps_expanded.py`](../scripts/prepare_apps_expanded.py): extend Train while retaining Validation and Test.
- [`verify_apps.py`](../scripts/verify_apps.py): verify prepared data.
- [`protocol.json`](../results/apps-expanded-adaptive/protocol.json): split hashes for the expanded experiment.

The original expanded files were stored under `D:/DA/TTNT_De19_Assets/data/apps-introductory-expanded-v4`. Set `TTNT_APPS_DATA_DIR` explicitly when using that dataset: `apps_core.py` otherwise defaults to v2. The configuration file in `configs/` describes the earlier v2 setup and is not the expanded experiment's protocol.

## Limitations

Bounded input domains allow exhaustive reference checks but substantially reduce task diversity and difficulty. Multiple mutants from one source problem are correlated observations. Public APPS content may also have appeared in model pretraining; that exposure has not been measured.
