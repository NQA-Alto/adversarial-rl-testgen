import argparse
import csv
import json
import statistics
from pathlib import Path

from apps_core import digest, read, write
from apps_model_runner import PROJECT
from audit_apps_run import audit
from result_docs import write_result_readme


FIELDS = ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")


def summarize(root):
    result, plan = read(root / "result.json"), read(root / "plan.json")
    if result["status"] == "stopped_at_replay_gate":
        if digest(root / "plan.json") != result["plan_sha256"]:
            raise ValueError("Original plan changed")
        output = PROJECT / "results/apps-expected-budget"
        output.mkdir(parents=True, exist_ok=True)
        diagnosis = read(root / "replay-diagnosis.json")
        write(output / "metrics.json", {"root": str(root), "trial_result_sha256": digest(root / "result.json"),
                                       "plan": plan, "result": result, "replay_diagnosis": diagnosis,
                                       "quality_evaluated": False, "model_test_evaluation": False})
        write_result_readme((output / 'README.md').parent)
        print("EXPECTED SUMMARY: stopped_at_replay_gate", output, flush=True)
        return
    if result["status"] not in ("completed", "stopped_at_train_gate") or digest(root / "plan.json") != result["plan_sha256"]:
        raise ValueError("Finished original planned attempt required")
    for key in ("training", "train_probe"):
        audit(Path(result[key]))
    trained = read(Path(result["training"]) / "result.json")
    baseline = read(Path(plan["baseline_sft"]) / "result.json")
    table, summary, eligible = [], {}, False
    if result["train_gate_passed"]:
        audit(Path(result["grpo"]))
        old = read(PROJECT / "results/apps-target-order/metrics.json")
        if digest(Path(old["root"]) / "result.json") != old["trial_result_sha256"]:
            raise ValueError("Original comparison changed")
        signatures, decoding = {}, None
        for steps, rows in ((64, old["runs"]["input-first-v1"]["evaluations"]), (256, result["evaluations"])):
            if len(rows) != 6 or {(r["stage"], r["generation_seed"]) for r in rows} != {(s, k) for s in ("sft", "grpo") for k in (42, 43, 44)}:
                raise ValueError("Incomplete seed/stage combinations")
            for row in rows:
                folder = Path(row["path"])
                audit(folder)
                info = read(folder / "result.json")
                if digest(folder / "result.json") != row["result_sha256"] or info["metrics"] != row["metrics"] or info["split"] != "validation" or info["model_test_evaluation"]:
                    raise ValueError("Validation evidence changed")
                signature = {p["case_id"]: (p["prompt_sha256"], p["case_seed"]) for p in
                             [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]}
                seed = row["generation_seed"]
                if seed in signatures and signatures[seed] != signature:
                    raise ValueError("Prompt or seed mismatch")
                if decoding is not None and decoding != info["generation"]:
                    raise ValueError("Decoding mismatch")
                signatures[seed], decoding = signature, info["generation"]
                table.append({"sft_steps": steps, "stage": row["stage"], "generation_seed": seed,
                              **{f: info["metrics"][f] for f in FIELDS}, "path": str(folder)})
        for steps in (64, 256):
            for stage in ("sft", "grpo"):
                selected = [r for r in table if r["sft_steps"] == steps and r["stage"] == stage]
                summary[f'{steps}-{stage}'] = {f: {"mean": statistics.mean(r[f] for r in selected), "sample_sd": statistics.stdev(r[f] for r in selected)} for f in FIELDS}
        sums = {steps: {f: sum(r[f] for r in table if r["sft_steps"] == steps and r["stage"] == "grpo") for f in FIELDS} for steps in (64, 256)}
        eligible = sums[256]["correct_expected"] >= sums[64]["correct_expected"] + 9 and all(sums[256][f] >= sums[64][f] for f in ("valid_tests", "bug_detecting_tests", "cases_detected"))
    output = PROJECT / "results/apps-expected-budget"
    output.mkdir(parents=True, exist_ok=True)
    write(output / "metrics.json", {"root": str(root), "trial_result_sha256": digest(root / "result.json"), "plan": plan,
                                   "result": result, "validation_rows": table, "summary": summary,
                                   "validation_eligibility_passed": eligible, "model_test_evaluation": False})
    if table:
        with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    write_result_readme((output / 'README.md').parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    events = [json.loads(line) for line in (Path(result["training"]) / "sft-steps.jsonl").read_text(encoding="utf-8").splitlines()]
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
    ax.plot([e["step"] for e in events], [e["loss"] for e in events], color="#869d93", alpha=.55, label="Loss minibatch")
    smoothed = [statistics.mean(e["loss"] for e in events[max(0, i - 7):i + 1]) for i in range(len(events))]
    ax.plot([e["step"] for e in events], smoothed, color="#315f49", label="Trung bình trượt 8 bước")
    ax.axvline(64, linestyle="--", color="#a18b64", label="Ngân sách SFT đối chứng")
    ax.set(xlabel="Bước cập nhật SFT · 4 ví dụ/bước", ylabel="Loss cross-entropy", title="Học expected từ nhãn Train · input trước expected")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "learning-curve.png", dpi=180, facecolor="white")
    plt.close(fig)
    print("EXPECTED SUMMARY:", result["status"], eligible, output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
