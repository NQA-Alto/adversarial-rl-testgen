import argparse
import csv
import json
import statistics
from pathlib import Path

from apps_core import digest, load_cases, read, write
from apps_final_evaluate import LOCK, audit_saved
from apps_model_runner import PROJECT


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
    lines = ["# Đánh giá cuối trên Test phát triển APPS", "", f'Tập giữ lại: {len({c["source_id"] for c in cases})} bài nguồn, {total} code lỗi; mỗi lượt có {slots} slot test, tối đa bốn test/code. Đây là holdout chia theo bài bên trong APPS Train, **không phải Test chính thức APPS**.', "",
             f'Model được chọn trước khi mở Test: **{LABELS.get(selected,selected)}**. Quy tắc chọn dựa trên Validation đã khóa trong protocol.json. Không đổi model hoặc cấu hình theo kết quả Test.', "",
             "Các variant, model nền, adapter/config, nguồn bộ chấm, phiên bản thư viện, ngân sách và seed đã được khóa trước đợt đánh giá. Lưu đủ mọi lượt và chấm lại raw output; không huấn luyện trong đợt này.", "",
             f'| Phiên bản | Seed sinh | Hợp lệ /{slots} | Expected đúng /{slots} | Test bắt lỗi /{slots} | Code phát hiện /{total} |', "|---|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append("| " + LABELS.get(row["variant"],row["variant"]) + " | " + str(row["generation_seed"]) + " | " + " | ".join(str(row[f]) for f in FIELDS) + " |")
    lines += ["", "## Trung bình ± độ lệch chuẩn mẫu", "", f'| Phiên bản | Chọn từ Validation | Hợp lệ /{slots} | Expected đúng /{slots} | Test bắt lỗi /{slots} | Code phát hiện /{total} |', "|---|---|---:|---:|---:|---:|"]
    for name, item in summary.items():
        values = []
        for f in FIELDS:
            value = f'{item[f]["mean"]:.2f} ± {item[f]["sample_sd"]:.2f}'
            values.append("**"+value+"**" if name != "base" and item[f]["mean"] > summary["base"][f]["mean"] else value)
        lines.append("| " + LABELS.get(name,name) + " | " + ("Có" if name==selected else "") + " | " + " | ".join(values) + " |")
    lines += ["", "Chữ đậm là trung bình cao hơn Qwen gốc trên cùng Test, không phải bằng chứng ý nghĩa thống kê. Model được chọn vẫn là model đã chọn từ Validation, dù thứ hạng quan sát trên Test có thể khác.", "",
              "![Kết quả Test cuối](comparison.png)", "", "Hình 1. Đợt Test đã khóa; thanh sai số là SD qua ba seed sinh test. Một seed train, không phải ba lượt huấn luyện độc lập. Các mutant cùng bài có tương quan.", "",
              "## Cách đọc và giới hạn", "", "Expected đúng là cặp input/expected đúng theo reference; test bắt lỗi phải vừa đúng expected vừa làm code lỗi trả sai hoặc phát sinh lỗi được hỗ trợ. Code phát hiện @4 đếm code có ít nhất một trong bốn test bắt lỗi. Mẫu sai/bị cắt vẫn nằm trong mẫu số.", "",
              "Nếu prompt vượt ngân sách đã khóa, bốn slot thất bại vẫn được tính; số lượng ghi trong errors.csv. Đây là slot ngân sách, không khẳng định model thực sự sinh token cho prompt đã bị chặn.", "",
              "GRPO 96 bước liền và GRPO cố định hai vòng khác số lần khởi tạo optimizer; chỉ nhánh cố định hai vòng là đối chứng phù hợp cho nhánh đối kháng hai vòng. Không dùng thứ hạng Test để chọn checkpoint rồi coi Test còn độc lập.", "",
              "Đây là quy mô nhỏ với miền input hữu hạn và mutant AST giữ lại. Không so trực tiếp mức điểm với benchmark ATGen hoặc toàn bộ APPS. Không cập nhật thiết kế từ Test và chấm lại dưới tên kết quả cuối.", "",
              "metrics.csv ghi thời gian generation/chấm và đỉnh PyTorch cấp phát, chưa gồm nạp model/hash/audit/điều phối. Nếu có nối lại vì gián đoạn kỹ thuật, trường elapsed_seconds_scope ghi rõ phần thời gian còn lại; output đã hoàn tất được giữ nguyên.", "",
              "Protocol và bảng tổng hợp ở đây; raw output, model và khóa sự kiện nằm trên ổ D: tại root ghi trong metrics.json. Script dựng lại: scripts/summarize_apps_final.py. Chạy lại tổng hợp chỉ chấm dữ liệu đã lưu, không tạo lượt model mới."]
    (output / "README.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
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
