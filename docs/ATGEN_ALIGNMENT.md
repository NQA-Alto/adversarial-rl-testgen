# Relationship to ATGen

The [ATGen paper](https://arxiv.org/abs/2510.14635) motivates the use of execution rewards and adversarial code to train a test generator. This project implements a small local experiment around those ideas.

## Implemented components

| Component | Implementation |
|---|---|
| Test generator | Qwen3.5-2B with a rank-4 LoRA adapter |
| Warm start | 840 SFT steps on 1,050 reference-checked examples |
| GRPO reward | Equal contributions for validity, correct expected output and bug detection |
| Code generator | Frozen base Qwen model; sampled candidates are filtered |
| Curriculum | Retain unresolved faulty functions; refresh detected ones between rounds |
| Comparison | Shared SFT parent, 145 Train positions, two rounds of 32 GRPO steps per branch |

Only the test generator's adapter is trained. The frozen code model changes the training examples through sampling and filtering.

## Adaptations

The model returns JSON directly, with no reasoning tags. The verifier supports bounded Python functions, and curriculum updates happen between rounds. Best-of-N selection, reinforcement learning for the code generator and full benchmark evaluation are outside this implementation.

These choices make the experiment feasible on the recorded laptop environment, but they change the model, task distribution and evaluation conditions. Results should be read within this setup, rather than compared numerically with the paper's benchmark scores.

## Outcome

SFT improved the recorded held-out bug-detection count over the base model. The fixed-code and adversarial GRPO branches did not improve on SFT. The [expanded experiment](../results/apps-expanded-adaptive/README.md) contains the counts, variability and protocol.

The [balanced-curriculum follow-up](APPS_TRAINING.md#balanced-curriculum-follow-up) is a separate implementation that increases sampling from adversarial positions and keeps case IDs matched across branches. Its settings do not replace the protocol of the recorded held-out result.
