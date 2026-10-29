import argparse
import csv
import json
import statistics
from pathlib import Path

from apps_core import digest, load_cases, read, write
from apps_final_evaluate import LOCK, audit_saved
from apps_model_runner import PROJECT
from result_docs import write_result_readme


FIELDS = ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")
LABELS = {"base":"Qwen gốc", "sft":"SFT input-first", "grpo96":"GRPO 96 bước", "control":"GRPO cố định · 2 vòng", "adaptive":"GRPO đối kháng · 2 vòng"}


def summarize(root):
    plan, result, lock = read(root / "plan.json"), read(root / "result.json"), read(LOCK)
    if result["status"] != "completed" or not result["selected_variant_unchanged"] or not result["comparison_prompt_and_seed_match"]:
        raise ValueError("Complete locked final evaluation required")
    if digest(root / "plan.json") != result["plan_sha256"] or lock["plan_sha256"] != result["plan_sha256"] or Path(lock["root"]).resolve() != root.resolve():
        raise ValueError("Final lock does not match this protocol")
    expected = {(name,seed) for name in plan["variants"] for seed in plan["generation_seeds"]}
    if len(result["runs"]) != len(expected) or {(r["variant"],r["generation_seed"]) for r in result["runs"]} != expected:
        raise ValueError("Missing or duplicated final run")
    cases, manifest = load_cases("test", allow_test=True)
    if manifest["dataset_id"] != plan["dataset_id"]:
        raise ValueError("Holdout dataset changed")
    total, slots = len(cases), len(cases)*4
    table, per_case, errors = [], [], []
    indexed = {}
    for row in result["runs"]:
        folder = Path(row["path"])
        if digest(folder / "result.json") != row["result_sha256"]:
            raise ValueError("Final result changed")
        audit_saved(folder, cases)
        info = read(folder / "result.json")
        if info["metrics"] != row["metrics"] or info["metrics"]["generated_tests"] != slots or info["metrics"]["evaluated_cases"] != total:
            raise ValueError("Unexpected final denominator")
        indexed[row["variant"], row["generation_seed"]] = row["metrics"]
        table.append({"variant":row["variant"],"generation_seed":row["generation_seed"],**{f:row["metrics"][f] for f in FIELDS},
                      "test_slots":slots,"cases_total":total,"elapsed_seconds":info["elapsed_seconds"],
                      "elapsed_seconds_scope":info["elapsed_seconds_scope"],"peak_pytorch_allocated_gib":info["peak_vram_gib"]})
        counts = {k:0 for k in ("invalid_or_truncated","wrong_expected","correct_no_bug","bug_detected","prompt_budget_exceeded")}
        for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
            prediction = json.loads(line)
            for sample in prediction["samples"]:
                key = "bug_detected" if sample["bug_detected"] else "correct_no_bug" if sample["correct_expected"] else "wrong_expected" if sample["valid_format"] else "invalid_or_truncated"
                counts[key] += 1
                counts["prompt_budget_exceeded"] += int(sample.get("reason") == "prompt_budget_exceeded")
            per_case.append({"variant":row["variant"],"generation_seed":row["generation_seed"],"source_id":prediction["source_id"],"case_id":prediction["case_id"],
                             "valid_tests":sum(s["valid_format"] for s in prediction["samples"]),"correct_expected":sum(s["correct_expected"] for s in prediction["samples"]),
                             "bug_tests":sum(s["bug_detected"] for s in prediction["samples"]),"case_detected":int(any(s["bug_detected"] for s in prediction["samples"]))})
        errors.append({"variant":row["variant"],"generation_seed":row["generation_seed"],**counts})
    summary = {name:{f:{"mean":statistics.mean(indexed[name,k][f] for k in plan["generation_seeds"]),
                        "sample_sd":statistics.stdev(indexed[name,k][f] for k in plan["generation_seeds"])} for f in FIELDS} for name in plan["variants"]}
    output = PROJECT / "results/apps-final-test"
    output.mkdir(parents=True,exist_ok=True)
    write(output / "protocol.json", plan)
    write(output / "metrics.json", {"root":str(root),"result_sha256":digest(root / "result.json"),"protocol_sha256":result["plan_sha256"],
                                    "selected_variant_before_test":plan["selected_variant_before_test"],"summary":summary,"runs":result["runs"],
                                    "test_cases":total,"test_slots_per_run":slots,"training_seeds":1,"sampling_seeds":3,
                                    "development_test_evaluated":True,"official_apps_test_evaluated":False,"model_test_evaluation":True})
    for filename, rows in (("metrics.csv",table),("per-case.csv",per_case),("errors.csv",errors)):
        with (output / filename).open("w",encoding="utf-8",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    selected = plan["selected_variant_before_test"]
    write_result_readme((output / 'README.md').parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(12,7.5),facecolor="white")
    names = list(summary)
    for ax,f,title,maximum in zip(axes.flat,FIELDS,("Test hợp lệ","Test có expected đúng","Test bắt lỗi","Code phát hiện với tối đa 4 test"),(slots,slots,slots,total),strict=True):
        values = [summary[name][f] for name in names]
        bars = ax.bar(range(len(names)),[v["mean"] for v in values],yerr=[v["sample_sd"] for v in values],capsize=4,
                      color=["#b5b7a9","#91a39a","#788f9b","#a29a7b","#38654e"][:len(names)])
        for name,bar,value in zip(names,bars,values,strict=True):
            if name==selected:
                bar.set_edgecolor("#29372f")
                bar.set_linewidth(1.8)
            ax.annotate(f'{value["mean"]:.2f}',(bar.get_x()+bar.get_width()/2,value["mean"]+value["sample_sd"]),xytext=(0,5),textcoords="offset points",ha="center",fontsize=9)
        ax.set(xticks=range(len(names)),xticklabels=[LABELS.get(n,n).replace(" · ","\n") for n in names],title=title,ylabel=f'Số lượng /{maximum}',ylim=(0,maximum))
        ax.tick_params(axis="x",labelsize=8,rotation=12)
        ax.grid(axis="y",alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Test phát triển · model được chọn từ Validation có viền đậm",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(output / "comparison.png",dpi=180,facecolor="white")
    plt.close(fig)
    print(json.dumps({"output":str(output),"selected_before_test":selected,"summary":summary},ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    summarize(parser.parse_args().folder)
