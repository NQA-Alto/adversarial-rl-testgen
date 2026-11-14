import csv
import json
from pathlib import Path


FIELDS = ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")
HEADERS = ("Valid /60", "Correct expected /60", "Bug-detecting tests /60", "Functions detected /15")
LABELS = {'base': 'Base Qwen',
 'baseline_validation': 'Base Qwen',
 'fresh_grpo_pilot': 'Initial GRPO',
 'sft': 'SFT64',
 'sft256': 'SFT256',
 'sft256-grpo96': 'SFT256 + GRPO96',
 'grpo': 'GRPO',
 'control': 'Fixed-code GRPO',
 'adaptive': 'Adversarial GRPO',
 'sft_validation': 'SFT64',
 'pilot_validation': 'SFT64 + GRPO96',
 'SFT64_parent': 'SFT64',
 'legacy-v1': 'Weighted reward (0.1/0.2/0.7)',
 'equal-v1': 'Equal reward',
 'SFT64_historical': 'Earlier SFT64',
 'SFT64_same_trajectory': 'Same-run SFT64',
 'SFT256': 'SFT256'}
NOTES = {'apps-pilot': ('Initial APPS pilot',
                'Base Qwen and the initial GRPO run, using prompt v1. Each Validation function receives four '
                'tests with generation seed 42.',
                'This trial predates the revised prompt and SFT warm start.',
                []),
 'apps-sft-grpo': ('SFT and GRPO with expected-first labels',
                   'Base Qwen, SFT64 and GRPO96 evaluated on 15 Validation functions with generation seed '
                   '42. SFT targets place expected before input.',
                   'Training increased valid outputs, while detected functions remained at 3/15.',
                   ['validation-comparison.png']),
 'apps-adversarial': ('Expected-first adversarial curriculum',
                      'Both branches start from the same SFT64 adapter and run two rounds of 48 GRPO steps. '
                      'Results below use generation seed 42.',
                      'The fixed-code and adversarial branches each detected 2/15 functions.',
                      ['validation-comparison.png']),
 'apps-adversarial-review': ('Expected-first resampling',
                             'Four checkpoints evaluated with generation seeds 42/43/44. Each checkpoint '
                             'comes from one training run.',
                             'The fixed-code branch detected 2.67/15 functions on average, compared with '
                             '3.00/15 for the adversarial branch.',
                             ['sampling.png']),
 'apps-input-first-adversarial': ('Input-first adversarial curriculum',
                                  'Both branches start from SFT64 with input-first targets and run two '
                                  'rounds of 48 GRPO steps. Results below use generation seed 42.',
                                  'Both branches detected 3/15 functions at seed 42.',
                                  ['validation-comparison.png']),
 'apps-input-first-review': ('Input-first resampling',
                             'Base Qwen, SFT64 and both GRPO branches evaluated with generation seeds '
                             '42/43/44.',
                             'Both GRPO branches detected an average of 4/15 functions and produced 5/60 '
                             'bug-detecting tests.',
                             ['sampling.png']),
 'apps-target-order': ('SFT target field order',
                       'The study keeps the same 319 input/expected pairs and training settings while '
                       'changing the order of the two top-level target fields. SFT and GRPO checkpoints are '
                       'compared separately.',
                       'Input-first targets increased correct expected outputs at all three generation '
                       'seeds. Most of the change appeared after SFT; the gain in bug-detecting tests was '
                       'concentrated in rounding problem 2947.',
                       ['comparison.png']),
 'apps-shared-sft-budget': ('Input-first SFT budget',
                            'The 64-step and 256-step checkpoints come from one training trajectory, sharing '
                            'initialization and the first 64 steps. Validation uses three generation seeds.',
                            'The 256-step checkpoint improved correct outputs and bug detection over the '
                            '64-step checkpoint in this run.',
                            ['comparison.png']),
 'apps-sft256-grpo': ('GRPO after SFT256',
                      'A 96-step GRPO stage starts from the SFT256 adapter and uses fixed Train code. Parent '
                      'and child are evaluated with the same three generation seeds.',
                      'GRPO slightly increased valid outputs but reduced correct expected outputs and bug '
                      'detection.',
                      ['comparison.png']),
 'apps-sft-budget': ('Expected-first SFT budget',
                     'Comparison of 64 and 256 SFT steps with the same label pool and prompt. The new '
                     'training run retains its own 64-step checkpoint for comparison.',
                     'SFT256 increased correct expected outputs but detected fewer functions. This trial '
                     'uses the earlier expected-first labels.',
                     ['validation-comparison.png']),
 'apps-reward-ablation': ('GRPO reward weights',
                          'Both runs start from the same expected-first SFT64 adapter and use 96 GRPO steps. '
                          'Component weights 0.1/0.2/0.7 are compared with equal weights of 1/3.',
                          'Equal weights improved the seed-42 result slightly, but the gain was not '
                          'consistent when resampled with seeds 43/44.',
                          ['validation-comparison.png']),
 'apps-error-analysis': ('Output errors in the initial adversarial trial',
                         'Validation outputs from expected-first checkpoints are classified by failure type '
                         'and compared with inputs that could expose a bug.',
                         'Incorrect expected outputs prevent some otherwise useful inputs from counting as '
                         'bug-detecting tests.',
                         ['failure-breakdown.png']),
 'apps-expected-diagnosis': ('Incorrect expected outputs',
                             'Error counts across three generation seeds for base Qwen, input-first SFT64 '
                             'and the GRPO branches.',
                             'SFT64 + GRPO96 produced 69/180 valid samples with an incorrect expected '
                             'output. Of those, 36 inputs could expose a bug. Per-problem counts are in '
                             'per-problem.csv.',
                             ['expected-errors.png']),
 'apps-serialization-diagnosis': ('JSON field order after SFT',
                                  'Earlier SFT labels placed expected before input because canonical '
                                  'serialization sorted the keys.',
                                  'Most JSON outputs after SFT also began with expected. The controlled '
                                  'ordering comparison is recorded in [SFT target field '
                                  'order](../apps-target-order/README.md).',
                                  []),
 'apps-input-difficulty': ('Bug-exposing input density',
                           'Exhaustive enumeration of each Validation input domain measures the fraction of '
                           'inputs that distinguish the reference solution from a mutant.',
                           'With four independent uniform draws with replacement, detection probability is 1 '
                           '- (1 - p)^4, assuming every expected output is correct. This is a '
                           'reference-based calculation, not a model score.',
                           []),
 'apps-lora-coverage': ('LoRA target coverage',
                        'Comparison of self-attention q_proj/v_proj targets with a profile that also '
                        'includes linear-attention projections.',
                        'The original profile has 208,896 trainable parameters across 6/24 blocks; the '
                        'expanded profile has 798,720 across 24/24. Both SFT64 runs completed, but the trial '
                        'stopped before Validation to investigate target ordering.',
                        []),
 'apps-expected-budget': ('SFT replay gate',
                          'A replay of the 64-step checkpoint checks whether a longer training-budget '
                          'comparison can use the historical adapter as a matching baseline.',
                          'The run stopped after 116/256 steps. Relative L2 at step 64 was 0.002685916, '
                          'above the 0.001 threshold. Label, example-order and shared-source hashes matched; '
                          'the cause of the divergence remains unresolved. No new quality evaluation was '
                          'completed.',
                          []),
 'apps-tuning': ('Thinking and output contracts',
                 'Train-only inference trials with frozen adapters: enabling thinking and changing the '
                 'output-type instructions in the prompt.',
                 'All 20 saved thinking samples were truncated at the 768-token limit. Prompt variants v3/v4 '
                 'also failed the Train gate, so these candidates were not evaluated further on Validation.',
                 []),
 'apps-data-profile': ('Earlier APPS dataset profile',
                       'The earlier subset contains 29 problems and 74 mutants, split by source problem into '
                       'Train, Validation and Test. Feature summaries use Train; split sizes come from the '
                       'manifest.',
                       'See [dataset construction](../../docs/APPS_DATASET.md) for source and verifier '
                       'details. This profile describes the earlier subset, not the expanded 58-problem '
                       'Train set.',
                       ['dataset-profile.png', 'pipeline.png']),
 'apps-final-test': ('Held-out evaluation',
                     'Evaluation of the selected checkpoints and settings on the development holdout.',
                     'Results are summarized from saved evaluation runs.',
                     ['comparison.png'])}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def label(value):
    return LABELS.get(str(value), str(value))


def number(value):
    if isinstance(value, dict) and "mean" in value:
        return f'{value["mean"]:.2f} ± {value["sample_sd"]:.2f}'
    return str(value).replace("|", "\\|")


def table(headers, rows, highlight=False):
    best = {i: max(row[i]["mean"] for row in rows) for i in range(1, len(headers))} if highlight else {}
    def cell(value, index):
        text = number(value)
        return "**" + text + "**" if index in best and value["mean"] == best[index] else text
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(cell(v, i) for i, v in enumerate(row)) + " |" for row in rows])


def csv_table(path, metric_headers=HEADERS):
    with path.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        return ""
    headers = records[0]
    valid = "valid_tests" if "valid_tests" in headers else "valid"
    bug = "bug_detecting_tests" if "bug_detecting_tests" in headers else "bug_tests"
    if not {valid, "correct_expected", bug, "cases_detected"}.issubset(headers):
        return ""
    keys = [k for k in ("variant", "profile", "sft_steps", "order", "stage", "generation_seed") if k in headers]
    titles = {"variant": "Configuration", "profile": "LoRA", "sft_steps": "SFT steps", "order": "Target order", "stage": "Stage", "generation_seed": "Generation seed"}
    return table([titles[k] for k in keys] + list(metric_headers),
                 [[label(row[k]) for k in keys] + [row[f] for f in (valid, "correct_expected", bug, "cases_detected")] for row in records])


def summary_table(data, metric_headers=HEADERS):
    summary = data.get("summary", {})
    if not summary:
        return ""
    rows = []
    for name, fields in summary.items():
        if "correct_expected" in fields:
            rows.append([label(name), *[fields[f] for f in FIELDS]])
        else:
            for stage, values in fields.items():
                if isinstance(values, dict) and "correct_expected" in values:
                    rows.append([name + " / " + stage, *[values[f] for f in FIELDS]])
    return table(["Configuration", *metric_headers], rows, highlight=False) if rows else ""


def write_result_readme(folder):
    folder = Path(folder)
    title, intro, conclusion, figures = NOTES.get(folder.name, ("Experiment results", "Recorded settings and results.", "", []))
    metrics = folder / "metrics.json"
    data = read(metrics) if metrics.exists() else {}
    headers = HEADERS
    if folder.name == "apps-final-test" and data:
        slots, cases = data["test_slots_per_run"], data["test_cases"]
        headers = (f"Valid /{slots}", f"Correct expected /{slots}", f"Bug-detecting tests /{slots}", f"Functions detected /{cases}")
        selected = label(data["selected_variant_before_test"])
        intro = f'The development holdout contains {cases} faulty functions and {slots} generated tests per run. The checkpoint selected using Validation before evaluation was {selected}.'
        conclusion = 'The holdout is drawn from APPS Train; official APPS Test usage is recorded separately in protocol.json. Tables show generation-seed means and standard deviations, with one training run per configuration.'
    parts = ["# " + title, "", intro, ""]
    if metrics.exists():
        summary = data.get("summary", {})
        if folder.name == "apps-sft256-grpo":
            winner = label(data["selected_within_this_pair"])
            conclusion = f'{winner} was selected within this pair using the recorded checkpoint-selection criterion. The table retains counts before and after GRPO.'
        elif folder.name == "apps-shared-sft-budget" and summary:
            a, b = summary["64"], summary["256"]
            conclusion = f'Mean correct expected outputs were {a["correct_expected"]["mean"]:.2f}/60 at SFT64 and {b["correct_expected"]["mean"]:.2f}/60 at SFT256; mean detected functions were {a["cases_detected"]["mean"]:.2f}/15 and {b["cases_detected"]["mean"]:.2f}/15, respectively.'
        elif folder.name in ("apps-adversarial-review", "apps-input-first-review"):
            a, b = summary["control"], summary["adaptive"]
            conclusion = f'The fixed-code branch detected {a["cases_detected"]["mean"]:.2f}/15 functions on average; the adversarial branch detected {b["cases_detected"]["mean"]:.2f}/15. Generation and filtering costs are recorded in costs.csv.'
        elif folder.name == "apps-lora-coverage" and summary:
            conclusion = 'LoRA configurations are compared at the same budget and generation seeds. Parameters, costs and per-run results are available in the files below.'
        elif folder.name == "apps-expected-budget" and data.get("result", {}).get("status") != "stopped_at_replay_gate":
            title = "SFT budget"
            parts[0] = "# " + title
            conclusion = f'Status: {data["result"]["status"]}. Available evaluations are recorded separately by checkpoint and generation seed.'
        text = summary_table(data, headers)
        if text:
            parts += [text, "", "Mean ± sample SD across three generation seeds; one training run per configuration.", ""]
    if (folder / "metrics.csv").exists():
        details = csv_table(folder / "metrics.csv", headers)
        if details:
            parts += ["## Per-run counts", "", details, ""]
    if folder.name == "apps-expected-diagnosis":
        counts = folder / "counts.csv"
        with counts.open(encoding="utf-8", newline="") as stream:
            records = list(csv.DictReader(stream))
        parts += [table(["Checkpoint", "Correct expected /180", "Valid but incorrect expected /180", "Bug-exposing input with incorrect expected"],
                        [[label(r["variant"]), r["correct_expected"], r["wrong_expected"], r["input_exposes_bug_but_expected_wrong"]] for r in records]), ""]
    parts += [conclusion, ""]
    for index, image in enumerate(figures, 1):
        parts += [f'![{title}]({image})', "", f'Figure {index}. {title}.', ""]
    if folder.name == "apps-reward-ablation" and (folder / "sampling-metrics.csv").exists():
        parts += ["[Resampling across three seeds](SAMPLING.md).", ""]
    data_files = [p.name for p in sorted(folder.iterdir()) if p.suffix in (".csv", ".json") and p.name not in ("review.json", "ui-review.json")]
    if data_files:
        parts += ["## Data files", "", " · ".join(f'[{name}]({name})' for name in data_files), ""]
    parts += ["[Evaluation guide](../../docs/EXPERIMENTS.md).", ""]
    (folder / "README.md").write_text("\n".join(parts), encoding="utf-8", newline="\n")


def write_sampling_notes(folder):
    folder = Path(folder)
    text = "# Reward comparison by generation seed\n\nFixed checkpoints evaluated with generation seeds 42/43/44. Each run generates 60 tests for 15 Validation functions.\n\n"
    text += csv_table(folder / "sampling-metrics.csv")
    text += "\n\nOne training seed and three generation seeds. Per-seed counts show sampling variability.\n\n"
    text += "![Results by generation seed](sampling-variability.png)\n\nFigure 1. Resampled checkpoint results.\n\n[Data files](sampling-metrics.json) · [Reward experiment](README.md).\n"
    (folder / "SAMPLING.md").write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "results"
    for name in NOTES:
        folder = root / name
        if folder.is_dir():
            write_result_readme(folder)
    if (root / "apps-reward-ablation/sampling-metrics.csv").exists():
        write_sampling_notes(root / "apps-reward-ablation")
