import argparse
import csv
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

from round1_data import digest, read, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = read(args.run / "result.json")
    if result["status"] != "grpo_training_seed_replicates_completed":
        raise ValueError("Only review a completed experiment")
    plan = read(args.run / "plan.json")
    source = Path(plan["source_experiment"])
    assert digest(source / "result.json") == plan["source_result_sha256"]
    original_metrics = read(source / "post-warmup-comparison.json")["metrics"]
    assert result["all_metrics"]["42"] == original_metrics
    output = args.run / ("review-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    output.mkdir()
    table = []
    by_problem = []
    failures = []
    prediction_hashes = {}
    training_proofs = []
    from dataset_tasks import canonical, compile_trusted
    from probe_round1_learning import failure_reason, score
    from restricted_python import RestrictedFunction
    from safetensors.torch import load_file
    import torch
    for seed in (42, 43, 44):
        for arm in ("control", "adversarial"):
            folder = (source if seed == 42 else args.run / ("seed-" + str(seed))) / ("evaluate-" + arm)
            records = rows(folder / "predictions.jsonl")
            prediction_hashes[str(folder / "predictions.jsonl")] = digest(folder / "predictions.jsonl")
            metrics = read(folder / "result.json")["metrics"]
            assert metrics == result["all_metrics"][str(seed)][arm]
            table.append({"training_seed": seed, "decoding_seed": 42, "arm": arm, **metrics})
            validation_cases = {case["case_id"]: case for case in rows(Path(r"D:\DA\TTNT_De19_Assets\data\internal-v1\validation.jsonl"))}
            categories = Counter(failure_reason(sample, validation_cases[record["case_id"]]) for record in records for sample in record["samples"])
            failures.extend({"training_seed": seed, "decoding_seed": 42, "arm": arm, "category": category, "count": count} for category, count in sorted(categories.items()))
            if seed != 42:
                train_folder = folder.with_name("train-" + arm)
                trained = read(train_folder / "result.json")
                spec = read(folder.parent / "post-warmup-spec.json")
                assert trained["training_seed"] == seed and trained["configuration"]["seed"] == seed
                adapter = Path(trained["final_adapter"]["path"]) / "adapter_model.safetensors"
                assert digest(adapter) == trained["final_adapter"]["adapter_sha256"]
                parent_tensors = load_file(str(Path(spec["parent_adapter_path"]) / "adapter_model.safetensors"), device="cpu")
                tensors = load_file(str(adapter), device="cpu")
                assert set(tensors) == set(parent_tensors)
                changed = [name for name in tensors if not torch.equal(tensors[name], parent_tensors[name])]
                live_names = [name.replace(".default.", ".") for name in trained["changed_adapter_tensors"]]
                assert sorted(changed) == sorted(live_names) and changed
                cases = {case["case_id"]: case for case in rows(folder.parent / (arm + "-curriculum.jsonl"))}
                rollouts = rows(train_folder / "rollouts.jsonl")
                assert len(rollouts) == 96 and len({g["case_id"] for g in rollouts}) == 96
                checked = 0
                for group in rollouts:
                    case = cases[group["case_id"]]
                    reference, buggy = compile_trusted(case["reference"]), RestrictedFunction(case["buggy_code"], case["domain"])
                    assert len(group["samples"]) == 4
                    assert group["rewards"] == [sample["reward"] for sample in group["samples"]]
                    assert bool(group["policy_signal"]) == (len(set(group["rewards"])) > 1)
                    for sample in group["samples"]:
                        actual = score(sample["raw_output"], sample["terminated"], case, reference, buggy)
                        for key in ("valid_format", "correct_expected", "bug_detected", "reward"):
                            assert actual[key] == sample[key], (seed, arm, group["case_id"], key)
                        for key in ("gold_output", "buggy_output", "buggy_error"):
                            assert canonical(actual.get(key)) == canonical(sample.get(key))
                        checked += 1
                training_proofs.append({"training_seed": seed, "arm": arm, "train_samples_rescored": checked,
                    "changed_adapter_tensors": len(changed), "gradient_nonzero_steps": trained["nonzero_gradient_steps"], "adapter_sha256": digest(adapter)})
            for pid in sorted({r["problem_id"] for r in records}):
                selected = [r for r in records if r["problem_id"] == pid]
                samples = [s for r in selected for s in r["samples"]]
                by_problem.append({"training_seed": seed, "decoding_seed": 42, "arm": arm, "problem_id": pid,
                    "programs": len(selected), "programs_detected": sum(any(s["bug_detected"] for s in r["samples"]) for r in selected),
                    "tests": len(samples), "valid_tests": sum(s["valid_format"] for s in samples),
                    "correct_expected": sum(s["correct_expected"] for s in samples), "bug_detecting_tests": sum(s["bug_detected"] for s in samples)})
    for name, values in (("by-training-seed.csv", table), ("by-problem.csv", by_problem), ("failure-categories.csv", failures)):
        with (output / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    os.environ.setdefault("MPLCONFIGDIR", r"D:\DA\TTNT_De19_Assets\cache\matplotlib-ui")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="#fcfcfa")
    colors = ("#9ba995", "#486b52")
    for index, arm in enumerate(("control", "adversarial")):
        metric = result["aggregate"][arm]["empirical_case_detection_at_4"]
        axes[0].bar(index, 100 * metric["mean"], yerr=100 * metric["sample_std"], width=.55, color=colors[index], capsize=5)
        axes[0].text(index, min(97, 100 * (metric["mean"] + metric["sample_std"]) + 3), f'{100 * metric["mean"]:.2f}%', ha="center", fontsize=11)
    axes[0].set_xticks([0, 1], ["GRPO cố định", "GRPO đối kháng"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Code lỗi phát hiện với tối đa 4 test (%)")
    baseline = result["all_metrics"]["42"]["parent_warmup"]["empirical_case_detection_at_4"] * 100
    axes[0].axhline(baseline, color="#9b947f", linestyle="--", linewidth=1, label="Sau SFT (cố định)")
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    axes[0].set_title("Trung bình và độ lệch chuẩn giữa 3 seed train", loc="left", fontsize=11, pad=16)
    for index, arm in enumerate(("control", "adversarial")):
        values = [result["all_metrics"][str(seed)][arm]["cases_detected_with_up_to_4_tests"] for seed in (42, 43, 44)]
        positions = [position + (index - .5) * .3 for position in range(3)]
        label = "GRPO cố định" if arm == "control" else "GRPO đối kháng"
        axes[1].bar(positions, values, width=.28, color=colors[index], label=label)
        for position, value in zip(positions, values):
            axes[1].text(position, value + .4, str(value), ha="center", fontsize=10)
    axes[1].set_xticks([0, 1, 2], ["Seed train 42", "Seed train 43", "Seed train 44"])
    axes[1].set_ylim(0, 18.8)
    axes[1].set_ylabel("Số code lỗi phát hiện / 18")
    axes[1].set_title("Code lỗi phát hiện trong từng seed train", loc="left", fontsize=11, pad=16)
    axes[1].set_yticks(range(0, 19, 3))
    axes[1].legend(frameon=False, loc="upper right")
    for axis in axes:
        axis.set_facecolor("#fcfcfa")
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#d4dacd")
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    figure.text(.05, .02, "Seed sinh test cố định 42; cùng checkpoint SFT và cùng pool đối kháng. Thanh sai số là SD, không phải khoảng tin cậy.", fontsize=9, color="#6a7363")
    figure.tight_layout(rect=(0, .07, 1, 1))
    for suffix in ("png", "svg"):
        figure.savefig(output / ("training-seed-stability." + suffix), dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
    proof = {"status": "review_passed", "source_run": str(args.run), "rows": 6, "prediction_hashes": prediction_hashes,
             "training_seeds": [42, 43, 44], "decoding_seed": 42, "unique_validation_programs": 18,
             "new_validation_outputs": 288, "reused_validation_outputs": 144, "test_set_model_evaluation": False,
             "scope": plan["interpretation"], "training_proofs": training_proofs, "new_training_samples_rescored": sum(p["train_samples_rescored"] for p in training_proofs),
             "uncertainty": "Descriptive sample SD across three conditional GRPO training seeds; no significance claim"}
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    proof["review_source_sha256"] = digest(Path(__file__))
    (output / "verification.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
