import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from apps_core import digest, read, write
from apps_model_runner import PROJECT
from audit_apps_run import audit


FIELDS = ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")


def summarize(root):
    plan, result = read(root / "plan.json"), read(root / "result.json")
    if result["status"] != "completed" or digest(root / "plan.json") != result["plan_sha256"]:
        raise ValueError("Completed frozen RL follow-up required")
    training = Path(result["training"])
    audit(training)
    trained = read(training / "result.json")
    if trained["parent_adapter_sha256"] != plan["parent_adapter_sha256"] or not trained["parent_adapter_exact_match"] or not trained["update_verified"] or trained["optimizer_steps"] != 96:
        raise ValueError("Exact SFT256 parent and verified96 RL updates required")
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
    output = PROJECT / "results/apps-sft256-grpo"
    output.mkdir(parents=True, exist_ok=True)
    write(output / "metrics.json", {"root": str(root), "result_sha256": digest(root / "result.json"), "plan": plan,
                                   "summary": summary, "runs": runs, "selected_within_this_pair": best,
                                   "selection_rule": "Detected cases, bug tests, correct expected, validity, then SFT when tied; three generation seeds, one training seed",
                                   "model_test_evaluation": False})
    for filename, data in (("metrics.csv", table), ("errors.csv", errors)):
        with (output / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    lines = ["# Tác động thêm của GRPO từ SFT256", "", "Cùng adapter SFT256 input-first, thêm96 bước GRPO trên code Train cố định. Chỉ thay phần học RL; giữ prompt, model nền, bộ chấm và ngân sách sinh. Không phải phép thử đối kháng mới. Đối chứng hai vòng đối kháng trước vẫn dùng warmup SFT64 riêng.", "",
             "| Phiên bản | Seed sinh | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |", "|---|---:|---:|---:|---:|---:|"]
    for r in table:
        lines.append(f'| {r["variant"]} | {r["generation_seed"]} | ' + " | ".join(str(r[f]) for f in FIELDS) + " |")
    lines += ["", "| Phiên bản | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |", "|---|---:|---:|---:|---:|"]
    for variant, values in summary.items():
        cells = []
        for f in FIELDS:
            text = f'{values[f]["mean"]:.2f} ± {values[f]["sample_sd"]:.2f}'
            cells.append("**" + text + "**" if values[f]["mean"] == max(v[f]["mean"] for v in summary.values()) else text)
        lines.append("| " + variant + " | " + " | ".join(cells) + " |")
    delta = {f: summary["sft256-grpo96"][f]["mean"] - summary["sft256"][f]["mean"] for f in FIELDS}
    lines += ["", "Chênh lệch GRPO − SFT trung bình: " + "; ".join(f'{f}: {n:+.2f}' for f, n in delta.items()) + ".", "",
              f'Lựa chọn trong cặp này theo thứ tự code phát hiện, test bắt lỗi, expected, hợp lệ, rồi phương án đơn giản khi hòa: **{best}**. Chỉ là lựa chọn trên Validation, chưa phải thành tích Test.', "",
              "Đã xác nhận đúng parent, gradient, tham số LoRA thay đổi, bản lưu và chấm lại toàn bộ raw output. Các bằng chứng cập nhật không tự chứng minh chất lượng tăng; bảng trên mới là so sánh chất lượng.", "",
              "Một seed train, ba seed sinh. SD không phải khoảng tin cậy. Nhiều lượt điều chỉnh trên cùng năm bài Validation có nguy cơ chọn quá khớp; không chỉnh phương pháp theo Test. Phần tăng từ SFT64 lên SFT256 thuộc SFT, không cộng vào lợi ích RL.", "",
              "![So sánh trước và sau GRPO](comparison.png)", "", "Hình 1. Số đếm trung bình và SD lấy mẫu. Mẫu sai/bị cắt vẫn nằm trong mẫu số, trục ghi rõ đơn vị."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
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
    print("SFT256 RL SUMMARY:", best, delta, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
