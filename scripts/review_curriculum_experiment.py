import argparse
import collections
import csv
import hashlib
import importlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def figures(root, output, scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"control": "#49796a", "adversarial": "#a76b46"}
    labels = {"control": "Dữ liệu cũ", "adversarial": "Thêm lỗi đối kháng"}
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    for arm in colors:
        groups = rows(root / ("train-" + arm) / "rollouts.jsonl")
        x = [row["group"] for row in groups]
        means = [sum(row["rewards"]) / len(row["rewards"]) for row in groups]
        axes[0].plot(x, means, color=colors[arm], alpha=0.22, linewidth=0.8)
        smooth = [sum(means[max(0, index - 11):index + 1]) / len(means[max(0, index - 11):index + 1]) for index in range(len(means))]
        axes[0].plot(x, smooth, color=colors[arm], label=labels[arm], linewidth=1.6)
        cumulative = []
        for index in range(len(groups)):
            cumulative.append(sum(bool(row["policy_signal"]) for row in groups[:index + 1]) / (index + 1))
        axes[1].plot(x, cumulative, color=colors[arm], label=labels[arm], linewidth=1.5)
    axes[0].set_ylabel("Reward trung bình")
    axes[0].set_title("Diễn biến GRPO — đường đậm là trung bình trượt tối đa 12 nhóm")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("Tỷ lệ nhóm có tín hiệu học\n(tích lũy)")
    axes[1].set_xlabel("Nhóm GRPO / bước cập nhật")
    for ax in axes:
        ax.grid(axis="y", color="#e4e5e2", linewidth=0.6)
        ax.set_ylim(0, 1.02)
    fig.savefig(output / "training-curves.png", dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(output / "training-curves.svg", facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    names = ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected_with_up_to_4_tests")
    titles = ("Test hợp lệ / 72", "Kết quả kỳ vọng đúng / 72", "Test phát hiện lỗi / 72", "Chương trình lỗi phát hiện / 18")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for ax, key, title in zip(axes.flat, names, titles, strict=True):
        values = [scores[variant][key] for variant in ("parent_static_72", "control", "adversarial")]
        bars = ax.bar(["Trước vòng này", "Dữ liệu cũ", "Đối kháng"], values, color=["#999d9a", colors["control"], colors["adversarial"]], width=0.6)
        ax.bar_label(bars, padding=3)
        ax.set_ylim(0, 18 if key == names[-1] else 72)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e4e5e2", linewidth=0.6)
    fig.suptitle("Validation — một seed, cùng 4 test cho mỗi chương trình lỗi")
    fig.savefig(output / "validation-comparison.png", dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(output / "validation-comparison.svg", facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    args = parser.parse_args()
    plot_cache = args.assets / "cache/matplotlib"
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(plot_cache)
    root = args.root.resolve()
    if not root.is_relative_to((args.assets / "evidence").resolve()):
        raise ValueError("Expected experiment inside evidence")
    extended = (root / "extended-spec.json").exists()
    spec = read(root / ("extended-spec.json" if extended else "round1-spec.json"))
    overall = read(root / "result.json")
    expected = "extended_curriculum_experiment_completed" if extended else "round1_experiment_completed"
    if overall["status"] != expected:
        raise ValueError("Experiment is not complete")
    for name, sha in spec["source_hashes"].items():
        if digest(root / "source" / name) != sha:
            raise ValueError("Saved source changed: " + name)
    sys.path.insert(0, str(root / "source"))
    from dataset_tasks import compile_trusted, evaluate_output
    from restricted_python import RestrictedFunction
    from evaluate_model import summarize
    compare = importlib.import_module("run_extended_curriculum" if extended else "run_round1_experiment").compare
    import torch
    from safetensors.torch import load_file
    parent_path = Path(spec["parent_adapter_path"]) / "adapter_model.safetensors"
    if digest(parent_path) != spec["parent_adapter_sha256"]:
        raise ValueError("Parent adapter changed")
    parent = load_file(str(parent_path), device="cpu")
    validation_path = args.assets / "data/internal-v1/validation.jsonl"
    if digest(validation_path) != spec["validation_split_sha256"]:
        raise ValueError("Validation changed")
    val_cases = {case["case_id"]: case for case in rows(validation_path)}
    output = root / ("review-" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir()
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    audit = {"rescore_training_samples": 0, "rescore_validation_samples": 0, "adapter_checks": {},
             "saved_source_hashes_match": True, "test_set_model_evaluation": False, "auditor_sha256": digest(Path(__file__))}
    training, evaluation, problem_rows = {}, {}, []
    for arm in ("control", "adversarial"):
        train = read(root / ("train-" + arm) / "result.json")
        val = read(root / ("evaluate-" + arm) / "result.json")
        training[arm], evaluation[arm] = train, val
        adapter = Path(train["final_adapter"]["path"]) / "adapter_model.safetensors"
        if digest(adapter) != train["final_adapter"]["adapter_sha256"] or digest(adapter) != val["adapter_sha256"]:
            raise ValueError("Adapter provenance differs")
        weights = load_file(str(adapter), device="cpu")
        if set(weights) != set(parent) or not all(torch.isfinite(value).all() for value in weights.values()):
            raise ValueError("Invalid adapter tensors")
        changes = {key: float((value.float() - parent[key].float()).abs().max()) for key, value in weights.items()}
        if max(changes.values()) <= 0:
            raise ValueError("Adapter did not change from parent")
        gradients = rows(root / ("train-" + arm) / "gradient-audit.jsonl")
        if sum(item["gradient_norm"] > 0 for item in gradients) != train["nonzero_gradient_steps"]:
            raise ValueError("Gradient audit differs")
        audit["adapter_checks"][arm] = {"changed_tensors": sum(value > 0 for value in changes.values()), "max_absolute_delta_from_parent": max(changes.values()),
            "adapter_sha256": digest(adapter), "optimizer_steps": train["optimizer_steps"], "nonzero_gradient_steps": train["nonzero_gradient_steps"],
            "training_seconds": train["training_seconds"], "peak_torch_allocated_gib": train["peak_torch_allocated_gib"]}
        for phase in ("train", "evaluate"):
            cases = {case["case_id"]: case for case in rows(root / (arm + "-curriculum.jsonl"))} if phase == "train" else val_cases
            compiled = {key: (compile_trusted(case["reference"]), RestrictedFunction(case["buggy_code"], case["domain"])) for key, case in cases.items()}
            records = rows(root / (phase + "-" + arm) / ("rollouts.jsonl" if phase == "train" else "predictions.jsonl"))
            for record in records:
                for sample in record["samples"]:
                    rescored = evaluate_output(sample["raw_output"].strip(), cases[record["case_id"]], *compiled[record["case_id"]])
                    if rescored.get("buggy_error") == "RestrictedCodeError":
                        raise ValueError("Restricted resource limit was counted as a model result")
                    if not sample["terminated"]:
                        rescored.update(valid_format=False, correct_expected=False, bug_detected=False, reward=0.0)
                    if any(rescored[key] != sample[key] for key in ("valid_format", "correct_expected", "bug_detected", "reward")):
                        raise ValueError("Rescore differs: " + record["case_id"])
                    audit["rescore_training_samples" if phase == "train" else "rescore_validation_samples"] += 1
            if phase == "evaluate":
                if summarize(records, True, 18) != val["metrics"]:
                    raise ValueError("Validation metrics differ")
                for pid in sorted({record["problem_id"] for record in records}):
                    subset = [record for record in records if record["problem_id"] == pid]
                    metrics = summarize(subset, True, len(subset))
                    problem_rows.append({"arm": arm, "problem_id": pid, **{key: metrics[key] for key in ("generated_tests", "valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected_with_up_to_4_tests")}})
    parent_validation = read(args.assets / "evidence" / spec["round_config"]["parent_validation_run"] / "result.json")
    comparison = compare(root, spec, parent_validation, training, evaluation)
    if comparison != overall["extended_comparison" if extended else "round1_comparison"]:
        raise ValueError("Recomputed comparison differs")
    audit["fair_comparison_reverified"] = True
    write(output / "verification.json", audit)
    with (output / "validation-by-problem.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=problem_rows[0].keys())
        writer.writeheader()
        writer.writerows(problem_rows)
    figures(root, output, comparison["metrics"])
    write(output / "artifact-manifest.json", {path.name: digest(path) for path in output.iterdir() if path.is_file()})
    print(json.dumps(audit, indent=2))
    print("REVIEW SAVED:", output)


if __name__ == "__main__":
    main()
