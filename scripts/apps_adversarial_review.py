import argparse
import csv
import json
import os
import shutil
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from apps_core import ASSETS, digest, read, write
from apps_model_runner import PROJECT, run_folder
from apps_round import run
from apps_ui import gpu_busy
from audit_apps_run import audit
from result_docs import write_result_readme


LABELS = {"base": "Qwen gốc", "sft": "SFT64", "control": "GRPO cố định", "adaptive": "GRPO đối kháng"}
FIELDS = ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def execute(comparison=None, sft_sampling=None, output=None):
    if gpu_busy(ignore_pid=os.getpid()):
        raise RuntimeError("Another project GPU job is active")
    os.environ.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                      HF_HOME=str(ASSETS / "cache/huggingface"), TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"))
    state = read(PROJECT / "APPS_STATE.json")
    active_state = dict(state)
    if comparison:
        from apps_adversarial_round import record
        compared = read(comparison / "result.json")
        if compared["status"] != "completed" or not compared["comparison_prompt_and_seed_match"]:
            raise ValueError("Completed matched research comparison required")
        state["adversarial_round_path"] = str(comparison)
        state["sft_training"] = record(Path(compared["plan"]["parent_sft_training"]), True)
        state["sft_validation"] = record(Path(compared["plan"]["parent_sft_validation"]))
        for arm, prefix in (("control", "paired_control"), ("adaptive", "adversarial")):
            for stage in ("training", "validation"):
                state[prefix + "_" + stage] = compared["rounds"][-1]["arms"][arm][stage]
    root = run_folder("adversarial-review")
    print("EVIDENCE:", root, flush=True)
    variants = {}
    for name, prefix in (("base", "baseline"), ("sft", "sft"), ("control", "paired_control"), ("adaptive", "adversarial")):
        evaluated = state[prefix + "_validation"]
        if digest(Path(evaluated["path"]) / "result.json") != evaluated["result_sha256"]:
            raise ValueError("Original evaluation changed")
        adapter = state[prefix + "_training"] if name != "base" else None
        if adapter and digest(Path(adapter["adapter_path"]) / "adapter_model.safetensors") != adapter["adapter_sha256"]:
            raise ValueError("Original adapter changed")
        variants[name] = {"adapter": adapter["adapter_path"] if adapter else None,
                          "adapter_sha256": adapter["adapter_sha256"] if adapter else None,
                          "reuse": {"42": evaluated["path"]}}
    if comparison:
        previous = read(sft_sampling)
        if digest(Path(previous["root"]) / "result.json") != previous["trial_result_sha256"]:
            raise ValueError("SFT sampling trial changed")
        arm = previous["runs"]["input-first-v1"]
        if Path(arm["sft"]).resolve() != Path(state["sft_training"]["path"]).resolve():
            raise ValueError("SFT sampling does not match this research parent")
        reuse = [row for row in arm["evaluations"] if row["stage"] == "sft"]
        for row in reuse:
            folder = Path(row["path"])
            if digest(folder / "result.json") != row["result_sha256"] or read(folder / "result.json")["adapter_sha256"] != variants["sft"]["adapter_sha256"]:
                raise ValueError("Reused research SFT evaluation changed")
            variants["sft"]["reuse"][str(row["generation_seed"])] = row["path"]
        for row in read(PROJECT / "results/apps-adversarial-review/metrics.json")["runs"]:
            if row["variant"] == "base":
                if digest(Path(row["path"]) / "result.json") != row["result_sha256"]:
                    raise ValueError("Reused base evaluation changed")
                variants["base"]["reuse"][str(row["generation_seed"])] = row["path"]
    else:
        previous = read(PROJECT / "results/apps-reward-ablation/sampling-metrics.json")
        for row in previous["sampling_result"]["runs"]:
            if row["variant"] == "SFT64_parent" and row["generation_seed"] in (43, 44):
                if row["adapter_sha256"] != variants["sft"]["adapter_sha256"] or digest(Path(row["path"]) / "result.json") != row["result_sha256"]:
                    raise ValueError("Reused SFT sampling evidence changed")
                variants["sft"]["reuse"][str(row["generation_seed"])] = row["path"]
    source = root / "source"
    source.mkdir()
    hashes = {}
    for name in ("apps_adversarial_review.py", "apps_model_runner.py", "apps_core.py", "apps_profiles.py", "restricted_python.py", "audit_apps_run.py", "apps_round.py", "apps_adversarial_round.py", "apps_lora.py"):
        shutil.copy2(Path(__file__).with_name(name), source / name)
        hashes[name] = digest(source / name)
    plan = {"recorded_at": datetime.now(timezone.utc).isoformat(), "purpose": "Descriptive sampling robustness of the actual final adversarial and matched static checkpoints",
            "dataset_id": state["dataset_id"], "generation_seeds": [42, 43, 44], "variants": variants,
            "prompt_version": "apps-call-v2", "split": "validation", "tests_per_case": 4, "max_new_tokens": 96,
            "selection": "Report all 12 variant/seed combinations. No retraining, checkpoint replacement or favorable seed selection.",
            "timing": "Final seed42 results known; supplementary seeds planned before their new control/adaptive evaluations. Conditional research follow-up, when used, was documented before target-order Validation.",
            "reused_evaluations": sum(len(item["reuse"]) for item in variants.values()),
            "research_comparison": str(comparison) if comparison else None,
            "comparison_result_sha256": digest(comparison / "result.json") if comparison else None,
            "summary_output": str(output) if output else None,
            "limits": "One training trajectory per method; sampling seeds are not training replicates. Five Validation problems, 15 mutant programs. Descriptive SD, not significance or confidence intervals.",
            "source_hashes": hashes, "state_sha256": digest(PROJECT / "APPS_STATE.json"),
            "active_config_sha256": digest(PROJECT / "configs/active-dataset.json"),
            "adaptive_round": state["adversarial_round_path"], "model_test_evaluation": False}
    write(root / "plan.json", plan)
    write(root / "previous-state.json", state)
    write(root / "active-state-before.json", active_state)
    result = {"status": "running", "plan_sha256": digest(root / "plan.json"), "runs": [], "model_test_evaluation": False}
    write(root / "result.json", result)
    runner = [str(ASSETS / "env/Scripts/python.exe"), "-B", str(PROJECT / "scripts/apps_model_runner.py")]
    signatures = {}
    generation = None
    try:
        for seed in plan["generation_seeds"]:
            for name, item in variants.items():
                reused = str(seed) in item["reuse"]
                command = runner + ["evaluate", "--seed", str(seed), "--prompt-version", "apps-call-v2", "--no-publish"]
                if item["adapter"]:
                    command += ["--adapter", item["adapter"]]
                folder = Path(item["reuse"][str(seed)]) if reused else run(command, root, f'{name}-seed-{seed}')
                audit(folder)
                evaluated = read(folder / "result.json")
                if (evaluated["split"] != "validation" or evaluated["dataset_id"] != plan["dataset_id"] or evaluated["seed"] != seed
                        or evaluated["adapter_sha256"] != item["adapter_sha256"] or not evaluated["complete_split"]
                        or evaluated["metrics"]["generated_tests"] != 60 or evaluated["metrics"]["evaluated_cases"] != 15):
                    raise ValueError("Evaluation does not match fixed protocol")
                signature = {row["case_id"]: (row["prompt_sha256"], row["case_seed"]) for row in records(folder / "predictions.jsonl")}
                if seed in signatures and signatures[seed] != signature:
                    raise ValueError("Prompt or case seeds differ across methods")
                signatures[seed] = signature
                if generation is not None and generation != evaluated["generation"]:
                    raise ValueError("Decoding configuration differs across methods")
                generation = evaluated["generation"]
                result["runs"].append({"variant": name, "generation_seed": seed, "path": str(folder), "reused": reused,
                                       "result_sha256": digest(folder / "result.json"), "adapter_sha256": item["adapter_sha256"], "metrics": evaluated["metrics"]})
                write(root / "result.json", result)
        for name, value in hashes.items():
            if digest(Path(__file__).with_name(name)) != value:
                raise ValueError("Review source changed while running")
        if (digest(PROJECT / "APPS_STATE.json") != plan["state_sha256"]
                or digest(PROJECT / "configs/active-dataset.json") != plan["active_config_sha256"]):
            raise ValueError("Active experiment changed during review")
        result.update(status="completed", active_checkpoint_unchanged=True, training_seeds=1, sampling_seeds=3)
        write(root / "result.json", result)
        summarize(root, output)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        write(root / "result.json", result)
        raise


def cost_rows(root):
    plan = read(root / "plan.json")
    state = read(root / "previous-state.json")
    experiment = read(Path(plan["adaptive_round"]) / "result.json")
    output = []

    def append(name, arm, kind, folder):
        folder = Path(folder)
        result = read(folder / "result.json")
        if kind == "grpo":
            runtimes = [row["train_runtime"] for row in records(folder / "trainer-metrics.jsonl") if "train_runtime" in row]
            if len(runtimes) != 1:
                raise ValueError("Expected exactly one completed GRPO training runtime")
            seconds = runtimes[0]
        else:
            seconds = result["training_seconds"] if kind == "sft" else result["elapsed_seconds"]
        output.append({"stage": name, "arm": arm, "kind": kind, "seconds": seconds, "peak_pytorch_allocated_gib": result["peak_vram_gib"],
                       "path": str(folder), "result_sha256": digest(folder / "result.json"),
                       "measurement": "Trainer train_runtime" if kind == "grpo" else "training_seconds" if kind == "sft" else "elapsed_seconds"})

    append("SFT64 dùng chung", "shared", "sft", state["sft_training"]["path"])
    for stage in experiment["rounds"]:
        number = stage["round"]
        append(f'Probe Train vòng {number}', "adaptive", "probe", stage["probe"])
        append(f'Sinh và lọc code vòng {number}', "adaptive", "code_generation", stage["pool"])
        for arm in ("control", "adaptive"):
            append(f'GRPO vòng {number}', arm, "grpo", stage["arms"][arm]["training"]["path"])
            append(f'Validation vòng {number}', arm, "evaluation", stage["arms"][arm]["validation"]["path"])
    return output


def summarize(root, output=None):
    result, plan = read(root / "result.json"), read(root / "plan.json")
    if result["status"] != "completed" or digest(root / "plan.json") != result["plan_sha256"]:
        raise ValueError("Completed review with original plan required")
    expected = {(name, seed) for name in LABELS for seed in (42, 43, 44)}
    if len(result["runs"]) != 12 or {(row["variant"], row["generation_seed"]) for row in result["runs"]} != expected:
        raise ValueError("Every planned evaluation must be present exactly once")
    by_key = {}
    for row in result["runs"]:
        folder = Path(row["path"])
        if digest(folder / "result.json") != row["result_sha256"]:
            raise ValueError("Saved evidence changed")
        audit(folder)
        if read(folder / "result.json")["metrics"] != row["metrics"]:
            raise ValueError("Saved metrics changed")
        by_key[row["variant"], row["generation_seed"]] = row["metrics"]
    summary = {name: {field: {"mean": statistics.mean(by_key[name, seed][field] for seed in (42, 43, 44)),
                              "sample_sd": statistics.stdev(by_key[name, seed][field] for seed in (42, 43, 44))} for field in FIELDS} for name in LABELS}
    paired = [{"seed": seed, **{field: by_key["adaptive", seed][field] - by_key["control", seed][field] for field in FIELDS}} for seed in (42, 43, 44)]
    costs = cost_rows(root)
    totals = {arm: {"grpo_seconds": sum(row["seconds"] for row in costs if row["arm"] == arm and row["kind"] == "grpo"),
                    "adaptive_data_seconds": sum(row["seconds"] for row in costs if row["arm"] == arm and row["kind"] in ("probe", "code_generation")),
                    "validation_seconds": sum(row["seconds"] for row in costs if row["arm"] == arm and row["kind"] == "evaluation"),
                    "max_recorded_allocated_gib": max(row["peak_pytorch_allocated_gib"] for row in costs if row["arm"] == arm)} for arm in ("control", "adaptive")}
    output = Path(output or plan.get("summary_output") or PROJECT / "results/apps-adversarial-review")
    output.mkdir(parents=True, exist_ok=True)
    write(output / "metrics.json", {"folder": str(root), "plan": plan, "result_sha256": digest(root / "result.json"),
                                   "runs": result["runs"], "summary": summary, "adaptive_minus_control": paired,
                                   "training_seeds": 1, "generation_seeds": 3, "model_test_evaluation": False})
    write(output / "costs.json", {"rows": costs, "totals": totals, "shared_sft_seconds": costs[0]["seconds"],
                                 "limits": "Recorded phase runtimes, not total end-to-end or GPU kernel time; model loading, audits, orchestration and gaps are not fully included. Allocated PyTorch VRAM is not total device memory. Same RL budget, unequal total compute."})
    for filename, rows in (("metrics.csv", [{"variant": row["variant"], "generation_seed": row["generation_seed"], **{field: row["metrics"][field] for field in FIELDS}} for row in result["runs"]]), ("costs.csv", costs)):
        with (output / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_result_readme((output / 'README.md').parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="white")
    for name, color, marker in zip(LABELS, ("#aaa896", "#6d8493", "#9b7b52", "#38654e"), ("o", "s", "D", "^"), strict=True):
        for ax, field, title, maximum in zip(axes, ("correct_expected", "cases_detected"), ("Test có expected đúng", "Code phát hiện với tối đa 4 test"), (60, 15), strict=True):
            ax.plot((42, 43, 44), [by_key[name, seed][field] for seed in (42, 43, 44)], marker=marker, color=color, label=LABELS[name], alpha=.8)
            ax.set(xticks=(42, 43, 44), xlabel="Seed sinh test · cùng checkpoint", ylabel=f'Số lượng /{maximum}', title=title, ylim=(0, maximum))
            ax.grid(axis="y", alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "sampling.png", dpi=180, facecolor="white")
    plt.close(fig)
    print("ADVERSARIAL REVIEW COMPLETED:", output, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--sft-sampling", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.summarize and (args.comparison or args.sft_sampling) and not (args.comparison and args.sft_sampling and args.output):
        parser.error("Research comparison requires --comparison, --sft-sampling and a separate --output")
    summarize(args.summarize, args.output) if args.summarize else execute(args.comparison, args.sft_sampling, args.output)
