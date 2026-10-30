import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from apps_core import digest, read, write
from apps_model_runner import PROJECT
from audit_apps_run import audit
from apps_rewards import PROFILES
from result_docs import write_result_readme


FIELDS = ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")


def summarize(root):
    plan, result = read(root / "plan.json"), read(root / "result.json")
    if result["status"] != "completed" or digest(root / "plan.json") != result["plan_sha256"]:
        raise ValueError("Completed frozen RL follow-up required")
    training = Path(result["training"])
    audit(training)
    trained = read(training / "result.json")
    if trained["reward_profile"] != plan["reward_profile"] or trained["reward_weights"] != PROFILES[plan["reward_profile"]]:
        raise ValueError("Training reward differs from planned profile")
    if trained["parent_adapter_sha256"] != plan["parent_adapter_sha256"] or not trained["parent_adapter_exact_match"] or not trained["update_verified"] or trained["optimizer_steps"] != 96:
        raise ValueError("Exact SFT256 parent and verified96 RL updates required")
    reward_control = None
    if plan["reward_profile"] == "equal-v1":
        reference = read(PROJECT / "results/apps-sft256-grpo/metrics.json")
        reference_root = Path(reference["root"])
        if digest(reference_root / "result.json") != reference["result_sha256"]:
            raise ValueError("Default reward comparison changed")
        reference_training = Path(read(reference_root / "result.json")["training"])
        audit(reference_training)
        control = read(reference_training / "result.json")
        if control["parent_adapter_sha256"] != plan["parent_adapter_sha256"] or control["configuration"] != trained["configuration"] or control["versions"] != trained["versions"] or control["source_hashes"] != trained["source_hashes"]:
            raise ValueError("Reward arms differ beyond their reward profile")
        order = lambda folder: [json.loads(line)["case_id"] for line in (folder / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()]
        if order(reference_training) != order(training) or len(order(training)) != 96:
            raise ValueError("Reward arms used different case order")
        reward_control = {"training": str(reference_training), "result_sha256": digest(reference_training / "result.json"),
                          "parent_config_versions_sources_and_case_order_match": True}
    table, runs, signatures, errors = [], [], {}, []
    decoding = None
    for variant, items in (("sft256", plan["parent_validation"]), ("sft256-grpo96", result["evaluations"])):
        if len(items) != 3 or {r["generation_seed"] for r in items} != {42, 43, 44}:
            raise ValueError("All three predeclared sampling seeds required")
        for item in items:
            folder = Path(item["path"])
            audit(folder)
            info = read(folder / "result.json")
            if digest(folder / "result.json") != item["result_sha256"] or info["metrics"] != item["metrics"] or info["split"] != "validation" or info["model_test_evaluation"] or not info["complete_split"] or info["metrics"]["generated_tests"] != 60:
                raise ValueError("Validation evidence changed")
            records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            sig = {r["case_id"]: (r["prompt_sha256"], r["case_seed"]) for r in records}
            seed = item["generation_seed"]
            if seed in signatures and signatures[seed] != sig or decoding is not None and decoding != info["generation"]:
                raise ValueError("Comparison protocol mismatch")
            signatures[seed], decoding = sig, info["generation"]
            table.append({"variant": variant, "generation_seed": seed, **{f: info["metrics"][f] for f in FIELDS}})
            runs.append({"variant": variant, **item})
            for row in records:
                c = Counter()
                for sample in row["samples"]:
                    c["invalid" if not sample["valid_format"] else "wrong_expected" if not sample["correct_expected"] else "bug_detected" if sample["bug_detected"] else "correct_no_bug"] += 1
                errors.append({"variant": variant, "seed": seed, "case_id": row["case_id"], **{f: c[f] for f in ("invalid", "wrong_expected", "correct_no_bug", "bug_detected")}})
    summary = {v: {f: {"mean": statistics.mean(r[f] for r in table if r["variant"] == v), "sample_sd": statistics.stdev(r[f] for r in table if r["variant"] == v)} for f in FIELDS} for v in ("sft256", "sft256-grpo96")}
    best = max(summary, key=lambda v: tuple(summary[v][f]["mean"] for f in ("cases_detected", "bug_detecting_tests", "correct_expected", "valid_tests")) + (v == "sft256",))
    output = PROJECT / ("results/apps-sft256-grpo" if plan["reward_profile"] == "legacy-v1" else "results/apps-sft256-equal-reward")
    output.mkdir(parents=True, exist_ok=True)
    if result.get("evaluation_recovery"):
        recovery = Path(result["evaluation_recovery"])
        record = read(recovery / "plan.json")
        if record["original_plan_sha256"] != result["plan_sha256"] or digest(recovery / "failed-result.json") != record["failed_result_sha256"]:
            raise ValueError("Recovery evidence changed")
        write(output / "evaluation-recovery.json", {"record": record,
                                                   "recovery_plan_sha256": digest(recovery / "plan.json"),
                                                   "original_failure_preserved": True, "training_repeated": False})
    write(output / "metrics.json", {"root": str(root), "result_sha256": digest(root / "result.json"), "plan": plan,
                                   "summary": summary, "runs": runs, "selected_within_this_pair": best,
                                   "selection_rule": "Detected cases, bug tests, correct expected, validity, then SFT when tied; three generation seeds, one training seed",
                                   "model_test_evaluation": False})
    if reward_control:
        write(output / "reward-control.json", reward_control)
        costs = []
        for profile, folder in (("legacy-v1", reference_training), ("equal-v1", training)):
            events = [json.loads(line) for line in (folder / "trainer-metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            costs.append({"reward_profile": profile, "training_seconds": next(row["train_runtime"] for row in events if "train_runtime" in row),
                          "peak_pytorch_allocated_gib": read(folder / "result.json")["peak_vram_gib"]})
        write(output / "costs.json", {"rows": costs, "scope": "Trainer phase times, excluding model loading/audits. Separate sessions with uncontrolled background load; not a timing benchmark."})
    for filename, data in (("metrics.csv", table), ("errors.csv", errors)):
        with (output / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    write_result_readme((output / 'README.md').parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), facecolor="white")
    for ax, f, title, maximum in zip(axes, ("correct_expected", "bug_detecting_tests", "cases_detected"), ("Expected đúng", "Test bắt lỗi", "Code phát hiện @4"), (60, 60, 15), strict=True):
        values = [summary[v][f]["mean"] for v in summary]
        sd = [summary[v][f]["sample_sd"] for v in summary]
        bars = ax.bar(("SFT256", "+ GRPO96"), values, yerr=sd, capsize=4, color=("#8ba798", "#3e6750"))
        for bar, value, spread in zip(bars, values, sd, strict=True):
            ax.text(bar.get_x()+bar.get_width()/2, value+spread+.5, f'{value:.2f}', ha="center", fontweight="bold" if value == max(values) else "normal")
        ax.set(title=title, ylabel=f'Số lượng /{maximum}', xlabel="Checkpoint", ylim=(0, maximum))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180, facecolor="white")
    plt.close(fig)
    delta = {f: summary["sft256-grpo96"][f]["mean"] - summary["sft256"][f]["mean"] for f in FIELDS}
    print("SFT256 RL SUMMARY:", best, delta, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
