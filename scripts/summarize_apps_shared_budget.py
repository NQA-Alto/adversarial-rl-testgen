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


def rows(folder):
    return [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]


def summarize(root):
    result, plan = read(root / "result.json"), read(root / "plan.json")
    if result["status"] not in ("completed", "stopped_at_train_gate") or digest(root / "plan.json") != result["plan_sha256"]:
        raise ValueError("Completed original protocol required")
    training = Path(result["training"])
    audit(training)
    trained = read(training / "result.json")
    signatures, decoding = {}, None
    for name, item in result["probes"].items():
        folder = Path(item["path"])
        audit(folder)
        if digest(folder / "result.json") != item["result_sha256"] or read(folder / "result.json")["split"] != "train":
            raise ValueError("Train probe evidence changed")
        sig = {r["case_id"]: (r["prompt_sha256"], r["case_seed"]) for r in rows(folder)}
        if signatures and signatures != sig:
            raise ValueError("Train prompt/seed mismatch")
        signatures = sig
    table, summary = [], {}
    signatures = {}
    for item in result["evaluations"]:
        folder = Path(item["path"])
        audit(folder)
        info = read(folder / "result.json")
        if digest(folder / "result.json") != item["result_sha256"] or info["metrics"] != item["metrics"] or info["split"] != "validation" or not info["complete_split"] or info["model_test_evaluation"]:
            raise ValueError("Validation evidence changed")
        sig = {r["case_id"]: (r["prompt_sha256"], r["case_seed"]) for r in rows(folder)}
        seed = item["generation_seed"]
        if seed in signatures and signatures[seed] != sig or decoding is not None and decoding != info["generation"]:
            raise ValueError("Validation protocol mismatch")
        signatures[seed], decoding = sig, info["generation"]
        table.append({"sft_steps": item["sft_steps"], "generation_seed": seed, **{f: info["metrics"][f] for f in FIELDS}, "path": str(folder)})
    eligible = False
    if result["train_gate_passed"]:
        if len(table) != 6 or {(r["sft_steps"], r["generation_seed"]) for r in table} != {(s, k) for s in (64, 256) for k in (42, 43, 44)}:
            raise ValueError("All six predeclared evaluations required")
        totals = {}
        for steps in (64, 256):
            selected = [r for r in table if r["sft_steps"] == steps]
            summary[str(steps)] = {f: {"mean": statistics.mean(r[f] for r in selected), "sample_sd": statistics.stdev(r[f] for r in selected)} for f in FIELDS}
            totals[steps] = {f: sum(r[f] for r in selected) for f in FIELDS}
        eligible = totals[256]["correct_expected"] >= totals[64]["correct_expected"] + 9 and all(totals[256][f] >= totals[64][f] for f in ("valid_tests", "bug_detecting_tests", "cases_detected"))
    elif table:
        raise ValueError("Train gate failed but new Validation was evaluated")
    output = PROJECT / "results/apps-shared-sft-budget"
    output.mkdir(parents=True, exist_ok=True)
    write(output / "metrics.json", {"root": str(root), "trial_result_sha256": digest(root / "result.json"), "plan": plan,
                                   "result": result, "summary": summary, "validation_rows": table, "eligible_for_rl_followup": eligible,
                                   "model_test_evaluation": False})
    if table:
        with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    write_result_readme((output / 'README.md').parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    events = [json.loads(line) for line in (training / "sft-steps.jsonl").read_text(encoding="utf-8").splitlines()]
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
    ax.plot([r["step"] for r in events], [r["loss"] for r in events], color="#a6b5a9", alpha=.65, label="Loss minibatch")
    ax.plot([r["step"] for r in events], [statistics.mean(x["loss"] for x in events[max(0, i-7):i+1]) for i in range(len(events))], color="#315f49", label="Trung bình trượt 8 bước")
    ax.axvline(64, color="#a18b64", linestyle="--", label="Checkpoint64")
    ax.set(xlabel="Bước cập nhật SFT · 4 ví dụ/bước", ylabel="Loss cross-entropy", title="Học nhãn input-first trên APPS Train")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "learning-curve.png", dpi=180, facecolor="white")
    plt.close(fig)
    print("SHARED SFT SUMMARY:", result["status"], "RL follow-up eligible:", eligible, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
