import argparse
import csv
import statistics
from pathlib import Path

from apps_core import digest, read, write
from apps_model_runner import PROJECT
from audit_apps_run import audit


def summarize(folder):
    result = read(folder / "sampling-result.json")
    plan = read(folder / "sampling-plan.json")
    if result["status"] != "completed" or digest(folder / "sampling-plan.json") != result["sampling_plan_sha256"]:
        raise ValueError("Completed sampling probe with original plan required")
    keys = {(row["variant"], row["generation_seed"]) for row in result["runs"]}
    if len(result["runs"]) != 9 or keys != {(variant, seed) for variant in plan["variants"] for seed in plan["generation_seeds"]}:
        raise ValueError("All nine prespecified sampling evaluations are required")
    for row in result["runs"]:
        path = Path(row["path"])
        if digest(path / "result.json") != row["result_sha256"]:
            raise ValueError("Saved sampling evidence changed")
        audit(path)
        if read(path / "result.json")["metrics"] != row["metrics"]:
            raise ValueError("Sampling metrics do not match saved evaluation")
    labels = {"SFT64_parent": "SFT64 trước RL", "legacy-v1": "GRPO reward hiện tại", "equal-v1": "GRPO reward chia đều"}
    fields = ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")
    summaries = {}
    for variant in plan["variants"]:
        values = [row for row in result["runs"] if row["variant"] == variant]
        summaries[variant] = {field: {"mean": statistics.mean(row["metrics"][field] for row in values),
                                     "sample_sd": statistics.stdev(row["metrics"][field] for row in values),
                                     "values_by_seed": {str(row["generation_seed"]): row["metrics"][field] for row in values}} for field in fields}
    paired = []
    indexed = {(row["variant"], row["generation_seed"]): row["metrics"] for row in result["runs"]}
    for seed in plan["generation_seeds"]:
        paired.append({"generation_seed": seed, **{field: indexed[("equal-v1", seed)][field] - indexed[("legacy-v1", seed)][field] for field in fields}})
    summary = {"sampling_result": result, "sampling_plan": plan, "descriptive_summary": summaries,
               "equal_minus_legacy_by_sampling_seed": paired, "training_seeds": 1, "sampling_seeds": 3,
               "model_test_evaluation": False, "statistical_significance_claim": False}
    output = PROJECT / "results/apps-reward-ablation"
    output.mkdir(parents=True, exist_ok=True)
    write(output / "sampling-metrics.json", summary)
    with (output / "sampling-metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "generation_seed", *fields, "generated_tests", "cases_total"])
        for row in result["runs"]:
            m = row["metrics"]
            writer.writerow([row["variant"], row["generation_seed"], *(m[field] for field in fields), m["generated_tests"], m["evaluated_cases"]])
    lines = ["# Độ biến động theo seed sinh test", "",
             "Đây là kiểm tra bổ sung đã lên kế hoạch trước khi xem Validation của hai nhánh reward mới. Cùng ba checkpoint cố định, ba seed sinh test 42/43/44; không train thêm và không chọn seed thuận lợi. Mỗi lượt gồm 60 test trên 15 code của năm bài Validation.", "",
             "**Một seed huấn luyện, ba seed lấy mẫu.** Không được gọi đây là ba seed train hay độ ổn định giữa ba lần học. Các code cùng bài và các lần lấy mẫu có tương quan; SD bên dưới chỉ mô tả biến động, không phải khoảng tin cậy.", "",
             "| Phiên bản | Seed sinh | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |", "|---|---:|---:|---:|---:|---:|"]
    for row in result["runs"]:
        m = row["metrics"]
        lines.append("| " + labels[row["variant"]] + " | " + str(row["generation_seed"]) + " | " + " | ".join(str(m[field]) for field in fields) + " |")
    maximum = {field: max(summaries[variant][field]["mean"] for variant in summaries) for field in fields}
    lines += ["", "## Trung bình và độ lệch chuẩn mẫu", "",
              "| Phiên bản | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |", "|---|---:|---:|---:|---:|"]
    for variant, item in summaries.items():
        values = []
        for field in fields:
            text = f'{item[field]["mean"]:.2f} ± {item[field]["sample_sd"]:.2f}'
            values.append("**" + text + "**" if item[field]["mean"] == maximum[field] else text)
        lines.append("| " + labels[variant] + " | " + " | ".join(values) + " |")
    lines += ["", "Giá trị trung bình cao nhất mỗi cột được in đậm, kể cả khi hòa. Đây là số lượng test/code, không phải tỷ lệ phần trăm và không phải kết quả trên Test.", "",
              "## Chênh lệch chia đều trừ hiện tại theo cùng seed", "", "| Seed sinh | Hợp lệ | Expected đúng | Test bắt lỗi | Code phát hiện |", "|---:|---:|---:|---:|---:|"]
    for row in paired:
        lines.append("| " + str(row["generation_seed"]) + " | " + " | ".join(f'{row[field]:+d}' for field in fields) + " |")
    lines += ["", "![Kết quả từng seed sinh test](sampling-variability.png)", "",
              "Tiêu chí ứng viên ban đầu ở seed 42 được giữ nguyên trong README và metrics.json. Bảng bổ sung không được dùng để lặng lẽ đổi tiêu chí hoặc chọn lại checkpoint. Nếu các seed cho hướng khác nhau, phải ghi kết quả chưa ổn định; nếu cùng hướng vẫn chưa đủ kết luận trên dataset rộng hơn.", "",
              "Script dựng lại: scripts/summarize_apps_reward_sampling.py. Raw output và adapter vẫn ở thư mục evidence trên D:; đường dẫn, hash và toàn bộ số liệu nằm trong sampling-metrics.json. Test phát triển và Test chính thức không được sử dụng."]
    (output / "SAMPLING.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
    colors = ("#aaa896", "#8da38d", "#38654e")
    for variant, color, marker in zip(plan["variants"], colors, ("o", "s", "^"), strict=True):
        for ax, field, title, denominator in zip(axes, ("correct_expected", "cases_detected"),
                                               ("Test có expected đúng", "Code phát hiện với tối đa 4 test"), (60, 15), strict=True):
            values = [indexed[(variant, seed)][field] for seed in plan["generation_seeds"]]
            ax.plot(plan["generation_seeds"], values, color=color, marker=marker, linewidth=1.5, markersize=6, label=labels[variant])
            ax.set_xticks(plan["generation_seeds"])
            ax.set_xlabel("Seed sinh test · không phải seed huấn luyện")
            ax.set_ylabel(f'Số lượng /{denominator}')
            ax.set_title(title, loc="left", fontsize=11)
            ax.set_ylim(0, denominator)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis="y", alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Biến động lấy mẫu trên cùng checkpoint · Validation", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(output / "sampling-variability.png", dpi=180, facecolor="white")
    plt.close(fig)
    print("Saved sampling analysis:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
