# Evaluation and experiment history

## Metrics

| Metric | Numerator | Denominator |
|---|---|---|
| Valid rate | Outputs with valid JSON and an allowed input | All generated outputs |
| IO accuracy | Tests with a correct expected output | All generated outputs |
| Attack rate | Correct tests that expose a bug | All generated outputs |
| Detection at 4 | Faulty functions detected by at least one correct test | All evaluated faulty functions |

Invalid and truncated outputs remain in the denominator. Detection at 4 uses a budget of four tests per faulty function. It measures test generation, rather than the accuracy of a code-writing model.

## Expanded study

The [expanded experiment](../results/apps-expanded-adaptive/README.md) uses 58 Train, 5 Validation and 7 held-out Test problems from APPS Train. SFT840 was selected using Validation before the final evaluation.

The fixed-code and adversarial GRPO branches share the SFT parent, prompt, reward profile, training positions and two rounds of 32 optimizer steps. The adversarial branch changes eligible faulty functions between rounds; the fixed branch retains the original code.

The base model, SFT and both GRPO branches were evaluated with generation seeds 42, 43 and 44. There is one training seed per configuration. Reported standard deviations describe sampling variability; the study does not estimate variation across independent training runs. Mutants from the same source problem are also correlated.

SFT improved on the base model. Neither GRPO branch improved held-out bug detection over SFT in this configuration.

## Subsequent experiments

The balanced-curriculum implementation uses two rounds of 64 steps, with half the sampled groups drawn from adversarial positions. The fixed and adversarial branches use matching case IDs, learning rates, parents and rewards. A seed-43 confirmation is conditional on the seed-42 validation gate.

These trials use Train/Validation. The stored Test result belongs to the earlier frozen protocol. A later checkpoint would require a fresh held-out evaluation to support a new final claim.

## Earlier trials

The [results index](../results/README.md) groups the earlier studies, including target ordering, SFT budget, reward weighting, adversarial curricula and error analysis. They use earlier datasets or prompts and should be compared within their own settings.

Stopped runs and negative results are retained alongside successful runs. In particular, the replay-gate failure and the LoRA trial stopped before Validation do not provide new model-quality estimates.

## Available evidence

Git contains summary CSV/JSON files, plots and the expanded protocol with source and split hashes. Raw generations, adapters and frozen run sources were stored in `D:/DA/TTNT_De19_Assets/evidence` and are not included in this checkout. Reading a summary or passing the standalone checks is not a rerun of the model experiments.
