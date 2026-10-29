import argparse
import csv
import json
import statistics
from pathlib import Path

from apps_core import digest, read, write
from apps_model_runner import PROJECT
from audit_apps_run import audit


FIELDS = ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")


def summarize(root):
    result, plan = read(root / "result.json"), read(root / "plan.json")
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
    lines = ["# Thử tăng thời lượng học expected", "", "Giữ input-first, 319 nhãn Train, LoRA và learning rate; tăng SFT64 lên SFT256 từ model gốc với seed42. SFT là học có giám sát, chưa phải RL. Nếu qua điều kiện Train, học thêm GRPO96 từ cả hai parent theo cùng giao thức; nhánh64 tái dùng kết quả đã kiểm chứng.", "",
             "## Kiểm tra Train trước Validation", "", "| Mốc SFT | Hợp lệ /164 | Expected đúng /164 | Test bắt lỗi /164 | Code phát hiện /41 |", "|---|---:|---:|---:|---:|"]
    for name, m in (("64 bước", result["baseline_train_metrics"]), ("256 bước", result["candidate_train_metrics"])):
        lines.append("| " + name + " | " + " | ".join(str(m[f]) for f in FIELDS) + " |")
    lines += ["", f'Qua điều kiện Train đã ghi trước: **{result["train_gate_passed"]}**. Replay64 relative L2 = {result["step64_replay"]["relative_l2_difference"]:.6g}. Train chỉ đo khả năng khớp dữ liệu đã học, không dùng để tuyên bố khái quát hóa.', ""]
    if not result["train_gate_passed"]:
        lines += ["Ứng viên dừng tại Train đúng kế hoạch. Không có GRPO hay Validation mới trong phép thử này. Không đổi checkpoint hiện hành và không mở Test."]
    else:
        lines += ["## Validation · ba seed sinh", "", "| SFT steps | Mốc | Seed sinh | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |", "|---:|---|---:|---:|---:|---:|---:|"]
        for r in table:
            lines.append(f'| {r["sft_steps"]} | {r["stage"]} | {r["generation_seed"]} | ' + " | ".join(str(r[f]) for f in FIELDS) + " |")
        lines += ["", "| SFT steps / mốc | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |", "|---|---:|---:|---:|---:|"]
        for label, data in summary.items():
            cells = []
            for f in FIELDS:
                value = f'{data[f]["mean"]:.2f} ± {data[f]["sample_sd"]:.2f}'
                cells.append("**" + value + "**" if data[f]["mean"] == max(x[f]["mean"] for x in summary.values()) else value)
            lines.append("| " + label + " | " + " | ".join(cells) + " |")
        lines += ["", f'Qua tiêu chí Validation đã ghi trước: **{eligible}**. Tiêu chí xét GRPO96: tổng expected ba seed tăng ít nhất9/180, các tổng hợp lệ/test bắt lỗi/code phát hiện không giảm. Không tự thay checkpoint chỉ vì loss thấp.', "",
                  "Chỉ một seed huấn luyện. Ba seed sinh mô tả biến động lấy mẫu, SD không phải khoảng tin cậy. Đây là kiểm chứng bổ sung sau nhiều lần nhìn Validation; nguy cơ chọn quá khớp Validation vẫn có. Test chưa được dùng để chỉnh thiết kế."]
    lines += ["", "## Chi phí SFT", "", f'SFT64: {baseline["training_seconds"]:.1f} giây; SFT256: {trained["training_seconds"]:.1f} giây. Đây là thời gian pha học/probe đã ghi, chưa bao gồm đầy đủ nạp model và điều phối. Đỉnh PyTorch cấp phát của ứng viên: {trained["peak_vram_gib"]:.3f}GiB.', "",
              "![Đường loss SFT của phép thử](learning-curve.png)", "", "Hình 1. Loss theo bước cập nhật trên minibatch Train. Các minibatch khác nhau; không phải đường lỗi Validation hay bằng chứng model tổng quát tốt hơn.", "",
              "Raw output, checkpoint, tham số, hash và kế hoạch nằm trong evidence ghi ở metrics.json. Không sửa expected sau sinh để làm đẹp kết quả."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
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
