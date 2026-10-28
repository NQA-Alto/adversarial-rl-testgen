import json
from collections import Counter

from apps_core import all_inputs, load_cases, read, write
from apps_model_runner import PROJECT


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    plt.rcParams["font.family"] = "DejaVu Sans"
    output = PROJECT / "results/apps-error-analysis"
    analysis = read(output / "analysis.json")
    variants = ("base", "sft", "control", "adaptive")
    labels = ["Qwen gốc", "SFT", "GRPO cố định", "GRPO đối kháng"]
    categories = [("invalid_input_or_json", "Input/JSON sai hoặc bị cắt", "#c9c8c0"),
                  ("wrong_expected", "Expected sai", "#c48e6d"),
                  ("correct_expected_no_bug", "Đúng, chưa bắt lỗi", "#93adbb"),
                  ("bug_detected", "Test bắt lỗi", "#47735b")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor="white")
    left = [0] * len(variants)
    for category, label, color in categories:
        values = [analysis["runs"][key]["counts"].get(category, 0) + (analysis["runs"][key]["counts"].get("truncated", 0) if category == "invalid_input_or_json" else 0) for key in variants]
        bars = axes[0].barh(labels, values, left=left, color=color, label=label)
        for bar, value in zip(bars, values, strict=True):
            if value:
                axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(value), ha="center", va="center", fontsize=10)
        left = [a + b for a, b in zip(left, values, strict=True)]
    axes[0].set_xlim(0, 60)
    axes[0].set_xlabel("Số test sinh trên Validation (60 / phiên bản)")
    axes[0].set_ylabel("Phiên bản model")
    axes[0].set_title("Expected sai là nhóm thất bại lớn nhất", loc="left", fontsize=12)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, -.48), frameon=False, ncol=2, fontsize=9)
    potential = [analysis["runs"][key]["counts"].get("input_only_attack", 0) for key in variants]
    usable = [analysis["runs"][key]["metrics"]["bug_detecting_tests"] for key in variants]
    x = list(range(len(variants)))
    axes[1].bar([i - .18 for i in x], potential, width=.36, color="#a4b2b6", label="Input có thể bắt lỗi (oracle)*")
    axes[1].bar([i + .18 for i in x], usable, width=.36, color="#47735b", label="Test model sinh bắt lỗi")
    for i, (a, b) in enumerate(zip(potential, usable, strict=True)):
        axes[1].text(i - .18, a + .25, str(a), ha="center", fontsize=10)
        axes[1].text(i + .18, b + .25, str(b), ha="center", fontsize=10)
    axes[1].set_xticks(x, labels=["Qwen gốc", "SFT", "GRPO\ncố định", "GRPO\nđối kháng"])
    axes[1].set_ylim(0, 16)
    axes[1].set_xlabel("Phiên bản model")
    axes[1].set_ylabel("Số test / 60 mẫu sinh")
    axes[1].set_title("Cơ hội bị mất khi expected sai", loc="left", fontsize=12)
    axes[1].legend(frameon=False, loc="upper right", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=.3, wspace=.3)
    fig.text(.08, .015, "*Oracle chỉ dùng chẩn đoán input; không sửa output hay cộng thành tích cho model. Cùng Validation, prompt v2 và seed sinh 42.", fontsize=9, color="#616a62")
    fig.savefig(output / "failure-breakdown.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    train, manifest = load_cases("train")
    sources = {case["source_id"]: case for case in train}
    types, volumes = Counter(), {}
    for case in sources.values():
        fields = list(case["domain"].values())
        if any(field["type"] == "list" and field.get("alphabet") and all(type(value) is str for value in field["alphabet"]) for field in fields):
            category = "Danh sách chuỗi"
        elif any(field["type"] == "list" for field in fields):
            category = "Danh sách số"
        elif any(field["type"] == "str" for field in fields):
            category = "Chuỗi ký tự"
        else:
            category = "Số nguyên"
        types[category] += 1
        volumes[str(case["source_id"])] = sum(1 for _ in all_inputs(case))
    data_output = PROJECT / "results/apps-data-profile"
    data_output.mkdir(parents=True, exist_ok=True)
    write(data_output / "profile.json", {"dataset_id": manifest["dataset_id"], "split_counts_from_manifest": manifest["splits"],
                                         "train_source_input_types": dict(types), "train_domain_sizes": volumes,
                                         "test_content_inspected": False, "model_test_evaluation": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor="white")
    splits = ("train", "validation", "test")
    counts = [manifest["splits"][key]["problems"] for key in splits]
    bars = axes[0].bar(["Train", "Validation", "Test giữ lại"], counts, color=["#47735b", "#93adbb", "#c9c8c0"])
    for bar, value in zip(bars, counts, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + .3, str(value), ha="center")
    axes[0].set_ylim(0, 20)
    axes[0].set_xlabel("Tập phát triển (trong APPS Train)")
    axes[0].set_ylabel("Số bài nguồn")
    axes[0].set_title("Tách theo bài, không trùng giữa tập", fontsize=11)
    axes[1].barh(list(types), list(types.values()), color="#93adbb")
    axes[1].set_xlabel("Số bài Train")
    axes[1].set_ylabel("Nhóm input")
    axes[1].set_title("Dữ liệu Train gồm những dạng nào?", fontsize=11)
    axes[2].bar(list(volumes), list(volumes.values()), color="#a5b395")
    axes[2].set_xlabel("ID bài APPS Train")
    axes[2].set_ylabel("Số input hợp lệ trong miền hữu hạn")
    axes[2].set_title("Quy mô miền kiểm tra của từng bài Train", fontsize=11)
    axes[2].tick_params(axis="x", labelrotation=90, labelsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(data_output / "dataset-profile.png", dpi=180, facecolor="white")
    plt.close(fig)
    (data_output / "README.md").write_text("# Hồ sơ dữ liệu APPS\n\nThống kê từ manifest và nội dung Train. Số lượng Test lấy từ manifest đã đóng băng; không đọc nội dung Test, không chạy model trên Test. Đây là tập con 29 bài dạng gọi hàm, không đại diện toàn bộ APPS.\n\n![Hồ sơ dữ liệu](dataset-profile.png)\n\nprofile.json chứa số liệu gốc; chạy scripts/plot_apps_analysis.py để dựng lại.\n", encoding="utf-8", newline="\n")

    fig, ax = plt.subplots(figsize=(13, 6.2), facecolor="white")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    def box(x, y, w, h, text, color="#f2f5ee"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", facecolor=color, edgecolor="#98a69b", linewidth=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    def arrow(start, end, text=None):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=13, color="#58685d"))
        if text:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + .13, text, ha="center", fontsize=9)
    box(.25, 4.9, 2.1, .8, "APPS Introductory\n29 bài gọi hàm")
    box(3.1, 4.9, 2.15, .8, "Chia theo bài\nTrain / Val / Test")
    box(6, 4.9, 2.1, .8, "SFT trên Train\nkhởi động adapter")
    box(9, 4.9, 3.2, .8, "GRPO cố định\n2 vòng × 48 bước")
    box(9, 3.1, 3.2, .8, "GRPO đối kháng\n2 vòng × 48 bước", "#e3eee3")
    box(.25, 2.5, 2.5, 1, "Policy hiện tại\nsinh test trên Train")
    box(3.4, 2.5, 2.4, 1, "Qwen local gốc\nsinh code, giữ trọng số")
    box(6.4, 2.5, 1.8, 1, "Lọc code sai\nqua test hiện tại")
    box(9, 1.25, 3.2, .8, "Chấm cùng Validation\nđúng expected + bắt lỗi")
    box(4.1, .35, 3.7, .8, "Chốt cấu hình → Test cuối\nchưa thực hiện", "#f5f0e6")
    arrow((2.5, 5.3), (2.95, 5.3))
    arrow((5.4, 5.3), (5.85, 5.3), "Train")
    arrow((8.25, 5.3), (8.85, 5.3))
    arrow((8.1, 4.95), (8.9, 3.8))
    arrow((2.9, 3), (3.25, 3))
    arrow((5.95, 3), (6.25, 3))
    arrow((8.35, 3), (8.85, 3.45))
    ax.plot([12.35, 12.65, 12.65], [5.3, 5.3, 1.65], color="#58685d", linewidth=1.2)
    arrow((12.65, 1.65), (12.35, 1.65))
    arrow((10.6, 2.95), (10.6, 2.2))
    arrow((8.85, 1.65), (7.6, .95))
    ax.plot([8.85, 8.65, 8.65, 1.5], [3.5, 3.5, 4.2, 4.2], color="#58685d", linewidth=1.2)
    arrow((1.5, 4.2), (1.5, 3.65))
    ax.text(4.7, 4.32, "Làm mới curriculum giữa hai vòng", ha="center", fontsize=9)
    ax.text(.25, 1.1, "Reference chỉ phục vụ bộ chấm.\nInput ngoài miền, code không được hỗ trợ\nvà output bị cắt đều được ghi nhận.", fontsize=10, color="#59685d")
    ax.text(.25, 5.95, "Pipeline hiện đã triển khai trên APPS", fontsize=15, fontweight="bold", color="#29372f")
    fig.savefig(data_output / "pipeline.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved error analysis, Train EDA and pipeline figures")


if __name__ == "__main__":
    main()
