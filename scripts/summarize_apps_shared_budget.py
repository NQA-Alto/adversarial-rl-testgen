import argparse
import csv
import json
import statistics
from pathlib import Path

from apps_core import digest, read, write
from apps_model_runner import PROJECT
from audit_apps_run import audit


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
    lines = ["# SFT64 và SFT256 từ cùng một lượt học", "", "Lấy checkpoint64 và cuối256 từ cùng một quỹ đạo optimizer mới. Chung model khởi đầu, 319 nhãn input-first, thứ tự ví dụ và 64 bước đầu. Không dùng adapter lịch sử làm đối chứng replay. Một seed train; đây là SFT, chưa phải RL.", "",
             "## Kiểm tra Train", "", "| Bước SFT | Hợp lệ /164 | Expected đúng /164 | Test bắt lỗi /164 | Code phát hiện /41 |", "|---:|---:|---:|---:|---:|"]
    for steps in (64, 256):
        m = result["probes"][str(steps)]["metrics"]
        lines.append(f'| {steps} | ' + " | ".join(str(m[f]) for f in FIELDS) + " |")
    lines += ["", f'Qua điều kiện Train: **{result["train_gate_passed"]}**. Các số Train chỉ đo mức khớp dữ liệu đã học.', ""]
    if table:
        lines += ["## Validation · đủ ba seed sinh", "", "| Bước SFT | Seed sinh | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |", "|---:|---:|---:|---:|---:|---:|"]
        for r in table:
            lines.append(f'| {r["sft_steps"]} | {r["generation_seed"]} | ' + " | ".join(str(r[f]) for f in FIELDS) + " |")
        lines += ["", "| Bước SFT | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |", "|---:|---:|---:|---:|---:|"]
        for steps, values in summary.items():
            cells = []
            for f in FIELDS:
                text = f'{values[f]["mean"]:.2f} ± {values[f]["sample_sd"]:.2f}'
                cells.append("**" + text + "**" if values[f]["mean"] == max(v[f]["mean"] for v in summary.values()) else text)
            lines.append("| " + steps + " | " + " | ".join(cells) + " |")
        lines += ["", f'Qua điều kiện đặt trước để cân nhắc thử RL tiếp: **{eligible}**. Expected tổng ba seed phải tăng ít nhất 9/180, tổng hợp lệ/test bắt lỗi/code phát hiện không giảm. Không tự thay checkpoint GRPO đang dùng bằng một mốc SFT chỉ vì expected cao.', ""]
    else:
        lines += ["Ứng viên dừng ở điều kiện Train; không có Validation mới. Không tự thử ngân sách lớn hơn.", ""]
    lines += ["Một seed train và ba seed sinh khác với ba lần huấn luyện độc lập. SD mô tả biến động lấy mẫu, không phải khoảng tin cậy. Phép thử bổ sung sau nhiều lần xem Validation có nguy cơ chọn quá khớp tập này. Test vẫn được giữ riêng.", "",
              f'Đã học {trained["optimizer_steps"]} bước, {trained["examples_seen"]} lượt ví dụ; thời gian pha học/probe đã ghi {trained["training_seconds"]:.1f} giây, đỉnh PyTorch cấp phát {trained["peak_vram_gib"]:.3f}GiB. Không phải tổng thời gian hệ thống hay tổng VRAM máy.', "",
              "![Đường loss Train](learning-curve.png)", "", "Hình 1. Loss minibatch Train và trung bình trượt 8 bước; không phải loss Validation. Mốc64 là checkpoint đối chứng cùng quỹ đạo.", "",
              "Mọi adapter, raw output, kế hoạch và mã nguồn snapshot nằm trong evidence ghi tại metrics.json. Không sửa đáp án model sau sinh."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
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
