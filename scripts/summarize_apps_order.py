import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from apps_core import canonical, digest, read, write
from apps_model_runner import PROJECT
from audit_apps_run import audit


FIELDS = ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")
LABELS = {"sorted-v1": "Expected trước", "input-first-v1": "Input trước"}


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(root):
    plan, result = read(root / "plan.json"), read(root / "result.json")
    if result["status"] != "completed" or digest(root / "plan.json") != result["plan_sha256"]:
        raise ValueError("Complete trial and original plan required")
    indexed, table, failures, per_case, costs = {}, [], [], [], []
    for order in plan["arms"]:
        arm = result["arms"][order]
        if len(arm["evaluations"]) != 6 or {(r["stage"],r["generation_seed"]) for r in arm["evaluations"]} != {(s,k) for s in ("sft","grpo") for k in plan["generation_seeds"]}:
            raise ValueError("Missing or duplicated evaluation")
        for stage in ("sft", "grpo"):
            folder = Path(arm[stage])
            if digest(folder / "result.json") != arm[stage + "_result_sha256"]:
                raise ValueError("Training evidence changed")
            audit(folder)
            info = read(folder / "result.json")
            seconds = info["training_seconds"] if stage == "sft" else next(row["train_runtime"] for row in records(folder / "trainer-metrics.jsonl") if "train_runtime" in row)
            costs.append({"order":order,"stage":stage,"seconds":seconds,"peak_pytorch_allocated_gib":info["peak_vram_gib"],"optimizer_steps":info["optimizer_steps"]})
        for row in arm["evaluations"]:
            folder = Path(row["path"])
            if digest(folder / "result.json") != row["result_sha256"]:
                raise ValueError("Evaluation changed")
            audit(folder)
            info = read(folder / "result.json")
            if info["metrics"] != row["metrics"] or info["split"] != "validation" or info["metrics"]["generated_tests"] != 60:
                raise ValueError("Unexpected evaluation metrics")
            stage, seed = row["stage"], row["generation_seed"]
            indexed[order,stage,seed] = row["metrics"]
            table.append({"order":order,"stage":stage,"generation_seed":seed,**{f:row["metrics"][f] for f in FIELDS}})
            counts = Counter()
            for prediction in records(folder / "predictions.jsonl"):
                case_hits = 0
                for sample in prediction["samples"]:
                    counts["bug_detected" if sample["bug_detected"] else "correct_no_bug" if sample["correct_expected"] else "wrong_expected" if sample["valid_format"] else "invalid_or_truncated"] += 1
                    case_hits += int(sample["bug_detected"])
                    try:
                        obj = json.loads(sample["raw_output"])
                        first = next(iter(obj),"empty") if type(obj) is dict else "not_object"
                    except (ValueError,TypeError):
                        first = "parse_error"
                    counts["first_" + first] += 1
                    if sample["valid_format"]:
                        possible = "buggy_error" in sample or canonical(sample["gold_output"]) != canonical(sample.get("buggy_output"))
                        counts["input_only_opportunity"] += int(possible)
                        counts["lost_due_to_expected"] += int(possible and not sample["correct_expected"])
                per_case.append({"order":order,"stage":stage,"generation_seed":seed,"source_id":prediction["source_id"],"case_id":prediction["case_id"],
                                 "valid_tests":sum(s["valid_format"] for s in prediction["samples"]),
                                 "correct_expected":sum(s["correct_expected"] for s in prediction["samples"]),"bug_tests":case_hits,"case_detected":int(case_hits>0)})
            failures.append({"order":order,"stage":stage,"generation_seed":seed,**dict(counts)})
    summaries = {order:{stage:{field:{"mean":statistics.mean(indexed[order,stage,k][field] for k in plan["generation_seeds"]),
                                    "sample_sd":statistics.stdev(indexed[order,stage,k][field] for k in plan["generation_seeds"])} for field in FIELDS} for stage in ("sft","grpo")} for order in plan["arms"]}
    totals = {order:{f:sum(indexed[order,"grpo",k][f] for k in plan["generation_seeds"]) for f in FIELDS} for order in plan["arms"]}
    old, new = (totals[order] for order in plan["arms"])
    eligible = new["correct_expected"] >= old["correct_expected"] + 3 * len(plan["generation_seeds"]) and all(new[f] >= old[f] for f in ("valid_tests","bug_detecting_tests","cases_detected"))
    deltas = [{"stage":s,"generation_seed":k,**{f:indexed["input-first-v1",s,k][f]-indexed["sorted-v1",s,k][f] for f in FIELDS}} for s in ("sft","grpo") for k in plan["generation_seeds"]]
    rl_deltas = {order:{f:summaries[order]["grpo"][f]["mean"]-summaries[order]["sft"][f]["mean"] for f in FIELDS} for order in plan["arms"]}
    output = PROJECT / "results/apps-target-order"
    output.mkdir(parents=True,exist_ok=True)
    problem_groups = {}
    for row in per_case:
        key = row["order"], row["stage"], row["generation_seed"], row["source_id"]
        group = problem_groups.setdefault(key, {"order":key[0],"stage":key[1],"generation_seed":key[2],"source_id":key[3],
                                               "generated_tests":0,"cases":0,"valid_tests":0,"correct_expected":0,"bug_tests":0,"cases_detected":0})
        group["generated_tests"] += 4
        group["cases"] += 1
        for field in ("valid_tests","correct_expected","bug_tests"):
            group[field] += row[field]
        group["cases_detected"] += row["case_detected"]
    write(output / "metrics.json", {"root":str(root),"trial_result_sha256":digest(root / "result.json"),"plan":plan,"runs":result["arms"],"summary":summaries,
                                    "input_first_minus_sorted":deltas,"grpo_minus_sft_by_order":rl_deltas,"eligible_for_adaptive_followup":eligible,
                                    "training_seeds":1,"sampling_seeds":3,"model_test_evaluation":False})
    for filename, rows in (("metrics.csv",table),("errors.csv",failures),("per-case.csv",per_case),("per-problem.csv",list(problem_groups.values())),("costs.csv",costs),("paired-deltas.csv",deltas)):
        with (output / filename).open("w",encoding="utf-8",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Kiểm chứng thứ tự nhãn input/expected", "", "Giữ nguyên 319 cặp nhãn, profile LoRA 6 khối, model, dữ liệu, prompt, parser, reward, learning rate và ngân sách. Chỉ đổi thứ tự hai trường ngoài cùng của completion SFT. Mọi output model được chấm nguyên trạng.", "",
             "Đối chứng dùng SFT64 đã hoàn tất trước khi phát hiện vấn đề; kiểm tra byte target cũ và ngữ nghĩa target mới, giữ nguyên thứ tự ví dụ. Hai nhánh GRPO96 chạy mới đầy đủ; không dùng lượt GRPO dở của phép thử LoRA. Chỉ chấm Validation sau khi cả hai GRPO hoàn tất.", "",
             "| Thứ tự nhãn SFT | Mốc | Seed sinh | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append("| " + LABELS[row["order"]] + " | " + row["stage"].upper() + " | " + str(row["generation_seed"]) + " | " + " | ".join(str(row[f]) for f in FIELDS) + " |")
    lines += ["", "## Trung bình ± độ lệch chuẩn mẫu", "", "| Thứ tự nhãn | Mốc | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |", "|---|---|---:|---:|---:|---:|"]
    for order in plan["arms"]:
        for stage in ("sft","grpo"):
            values = []
            for f in FIELDS:
                item = summaries[order][stage][f]
                value = f'{item["mean"]:.2f} ± {item["sample_sd"]:.2f}'
                maximum = max(summaries[o][s][f]["mean"] for o in plan["arms"] for s in ("sft","grpo"))
                values.append("**" + value + "**" if item["mean"] == maximum else value)
            lines.append("| " + LABELS[order] + " | " + stage.upper() + " | " + " | ".join(values) + " |")
    lines += ["", "Giá trị trung bình cao nhất in đậm, kể cả khi hòa. Chỉ một seed train và ba seed sinh test; SD mô tả lấy mẫu, không phải khoảng tin cậy hay nhiều lượt train độc lập.", "",
              "![So sánh thứ tự nhãn](comparison.png)", "", "Hình 1. Số test/code trên Validation, giữ mẫu không hợp lệ trong mẫu số. Thanh sai số là SD của ba seed sinh test.", "",
              "## Tách lợi ích SFT và RL", "", "So cùng mốc giữa hai thứ tự nhãn để xem tác động biểu diễn target. So GRPO trừ SFT trong mỗi nhánh để xem tác động thêm của RL. Đây là GRPO cố định, chưa phải đánh giá curriculum đối kháng mới.", "",
              "| Nhánh | GRPO − SFT: hợp lệ | Expected đúng | Test bắt lỗi | Code phát hiện |", "|---|---:|---:|---:|---:|"]
    for order in plan["arms"]:
        lines.append("| " + LABELS[order] + " | " + " | ".join(f'{rl_deltas[order][f]:+.2f}' for f in FIELDS) + " |")
    lines += ["", f'Đủ tiêu chí đặt trước để cân nhắc thử đối kháng tiếp: **{eligible}**. Tiêu chí ở mốc GRPO là expected đúng trung bình tăng ít nhất 3/60, các trung bình hợp lệ/test bắt lỗi/code phát hiện không giảm. Ngưỡng sàng lọc không phải kiểm định ý nghĩa thống kê; không tự thay checkpoint chính.', "",
              "## Phân tích và chi phí", "", "errors.csv giữ đủ phân loại lỗi và thứ tự trường model thực sự sinh. Cột input-only chỉ chẩn đoán các cơ hội bị mất do expected sai, không dùng oracle để sửa điểm model. per-case.csv giữ ID từng code phát hiện; per-problem.csv gom theo bài nguồn để xem mức tăng có chỉ tập trung vào một bài hay không. Ba mutant cùng đề không phải ba bài độc lập.", "",
              "costs.csv ghi thời gian pha và đỉnh PyTorch cấp phát. SFT bao gồm probe sau học/lưu adapter; GRPO dùng Trainer train_runtime. Chưa tính đầy đủ thời gian nạp model, audit và điều phối, không gọi là tổng chi phí toàn quy trình.", "",
              "## Giới hạn và nguồn", "", "Năm bài Validation, miền hữu hạn và một seed huấn luyện. Nondeterminism CUDA chưa được loại trừ. Kết quả tốt nếu có vẫn cần phép so sánh đối kháng mới và đánh giá Test cuối sau khi đóng băng thiết kế.", "",
              "Thứ tự nhãn là điều chỉnh triển khai của đồ án. Không coi nó là thành phần nguyên bản hoặc kết quả tái lập [ATGen](https://arxiv.org/html/2510.14635v1). Test phát triển và Test chính thức chưa được model đánh giá.", "",
              "Kế hoạch, mã nguồn lúc chạy, adapter và raw output nằm tại thư mục evidence ghi trong metrics.json. Dựng lại bảng/hình: `python scripts/summarize_apps_order.py <folder>`. Không cần nạp model để tổng hợp."]
    (output / "README.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(11,7),facecolor="white")
    for ax,field,title,maximum in zip(axes.flat,FIELDS,("Test hợp lệ","Test có expected đúng","Test bắt lỗi","Code phát hiện với tối đa 4 test"),(60,60,60,15),strict=True):
        items = [summaries[o][s][field] for o in plan["arms"] for s in ("sft","grpo")]
        bars = ax.bar(range(4),[m["mean"] for m in items],yerr=[m["sample_sd"] for m in items],capsize=4,color=("#b5b7a9","#869481","#7c9b9a","#38654e"))
        for bar,item in zip(bars,items,strict=True):
            ax.annotate(f'{item["mean"]:.2f}',(bar.get_x()+bar.get_width()/2,item["mean"]+item["sample_sd"]),xytext=(0,6),textcoords="offset points",ha="center",fontsize=9,
                        fontweight="bold" if item["mean"] == max(m["mean"] for m in items) else "normal")
        ax.set(xticks=range(4),xticklabels=[LABELS[o]+"\n"+s.upper() for o in plan["arms"] for s in ("sft","grpo")],title=title,ylabel=f'Số lượng /{maximum}',ylim=(0,maximum))
        ax.tick_params(axis="x",labelsize=8)
        ax.grid(axis="y",alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "comparison.png",dpi=180,facecolor="white")
    plt.close(fig)
    print(json.dumps({"output":str(output),"eligible_for_adaptive_followup":eligible,"summary":summaries},ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    summarize(parser.parse_args().folder)
