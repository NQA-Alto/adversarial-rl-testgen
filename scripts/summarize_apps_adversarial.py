import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from apps_adversary import audit_pool, rows
from apps_core import digest, read, write
from apps_model_runner import PROJECT
from audit_apps_run import audit


def summarize(folder):
    result = read(folder / "result.json")
    if result["status"] != "completed" or not result["comparison_prompt_and_seed_match"]:
        raise ValueError("A completed matched comparison is required")
    previous = read(folder / "previous-state.json")
    runs = [("base", "Qwen gốc", Path(previous["baseline_validation"]["path"])),
            ("sft", "SFT", Path(previous["sft_validation"]["path"]))]
    final = result["rounds"][-1]["arms"]
    runs += [(arm, label, Path(final[arm]["validation"]["path"])) for arm, label in
             (("control", "SFT + GRPO cố định"), ("adaptive", "SFT + GRPO đối kháng"))]
    summary = {"round_folder": str(folder), "plan": result["plan"], "runs": {}, "rounds": [],
               "model_test_evaluation": False, "adversarial_generated_test_exposures": result["adversarial_generated_test_exposures"],
               "limits": "Five Validation source problems, 15 mutated programs, one seed; finite call-based input domains. No official APPS Test evaluation. Matched optimizer-step and generation budgets, not matched total wall-clock or code-sampling compute."}
    metric_rows, problem_rows = [], []
    for key, label, path in runs:
        audit(path)
        record = read(path / "result.json")
        breakdown, problems = Counter(), {}
        for prediction in rows(path / "predictions.jsonl"):
            problems.setdefault(prediction["source_id"], []).append(prediction)
            for sample in prediction["samples"]:
                category = "bug_detected" if sample["bug_detected"] else "correct_expected_no_bug" if sample["correct_expected"] else "wrong_expected" if sample["valid_format"] else "invalid_or_truncated"
                breakdown[category] += 1
        if sum(breakdown.values()) != 60:
            raise ValueError("Incomplete Validation")
        summary["runs"][key] = {"label": label, "result": record, "result_sha256": digest(path / "result.json"), "failure_breakdown": dict(breakdown)}
        m = record["metrics"]
        metric_rows.append([key, label, m["valid_tests"], m["correct_expected"], m["bug_detecting_tests"], m["cases_detected"], 60, 15])
        for source_id, predictions in problems.items():
            samples = [sample for prediction in predictions for sample in prediction["samples"]]
            problem_rows.append([key, source_id, len(samples), sum(sample["valid_format"] for sample in samples),
                                 sum(sample["correct_expected"] for sample in samples), sum(sample["bug_detected"] for sample in samples),
                                 sum(any(sample["bug_detected"] for sample in prediction["samples"]) for prediction in predictions)])
    seen_behaviors = set()
    for stage in result["rounds"]:
        pool = Path(stage["pool"])
        audit_pool(pool)
        reviewed = read(pool / "result.json")
        for arm in stage["arms"].values():
            audit(Path(arm["training"]["path"]))
            audit(Path(arm["validation"]["path"]))
            for key in ("training", "validation"):
                entry = arm[key]
                if digest(Path(entry["path"]) / "result.json") != entry["result_sha256"]:
                    raise ValueError("Recorded run hash changed")
        train = {key: read(Path(arm["training"]["path"]) / "result.json") for key, arm in stage["arms"].items()}
        orders = {key: [event["case_id"] for event in rows(Path(arm["training"]["path"]) / "rollouts.jsonl")]
                  for key, arm in stage["arms"].items()}
        if train["control"]["configuration"] != train["adaptive"]["configuration"] or orders["control"] != orders["adaptive"]:
            raise ValueError("Training configurations or case order differ between arms")
        for key, arm in stage["arms"].items():
            expected_parent = result["plan"]["parent_adapter_sha256"] if stage["round"] == 1 else result["rounds"][stage["round"] - 2]["arms"][key]["training"]["adapter_sha256"]
            if train[key]["parent_adapter_sha256"] != expected_parent or not train[key]["parent_adapter_exact_match"]:
                raise ValueError("Parent lineage mismatch")
        stage["matched_training_configuration_and_case_order"] = True
        accepted = rows(pool / "adversarial-cases.jsonl")
        behaviors = {(row["source_id"], row["gate"]["behavior_sha256"]) for row in accepted}
        summary["rounds"].append({"round": stage["round"], "pool": reviewed, "curricula": stage["curricula"],
                                  "arms": stage["arms"], "matched_training_configuration_and_case_order": True,
                                  "novel_behaviors_vs_prior_rounds": len(behaviors - seen_behaviors),
                                  "accepted_code_examples": accepted[:2]})
        seen_behaviors.update(behaviors)
    summary["distinct_accepted_behaviors_across_rounds"] = len(seen_behaviors)
    output = PROJECT / "results/apps-adversarial"
    output.mkdir(exist_ok=True, parents=True)
    write(output / "metrics.json", summary)
    for filename, header, data in (
        ("metrics.csv", ["variant", "label", "valid", "correct_expected", "bug_detecting_tests", "cases_detected", "generated_tests", "cases_total"], metric_rows),
        ("per-problem.csv", ["variant", "apps_source_id", "tests", "valid", "correct_expected", "bug_detecting_tests", "cases_detected"], problem_rows),
    ):
        with (output / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(data)
    maximum = {field: max(entry["result"]["metrics"][field] for entry in summary["runs"].values())
               for field in ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")}
    lines = ["# APPS: GRPO cố định và curriculum đối kháng", "",
             "Cùng checkpoint SFT, hai vòng × 48 bước/nhánh; optimizer được tạo lại ở đầu mỗi vòng ở cả hai nhánh. Validation: 15 code từ năm bài nguồn, bốn test/code, seed 42. Không đánh giá Test.", "",
             "| Phiên bản | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện @4 /15 |", "|---|---:|---:|---:|---:|"]
    for _, entry in summary["runs"].items():
        m = entry["result"]["metrics"]
        values = [f'**{m[field]}**' if m[field] == maximum[field] else str(m[field]) for field in maximum]
        lines.append("| " + entry["label"] + " | " + " | ".join(values) + " |")
    lines += ["", "Giá trị lớn nhất từng cột được in đậm, kể cả khi hòa. Đây là kết quả Validation của một seed, chưa chứng minh cải thiện ổn định.", "",
              "| Vòng | Code được sinh | Code qua bộ lọc | Vị trí Train dùng code đối kháng /41 | Test RL sinh khi gặp code đối kháng |", "|---|---:|---:|---:|---:|"]
    for stage in summary["rounds"]:
        pool = stage["pool"]
        lines.append(f'| {stage["round"]} | {pool["sampled_candidates"]} | {pool["accepted_candidates"]} | {stage["curricula"]["adaptive"]["counts"].get("local_adversary", 0)} | {stage["arms"]["adaptive"]["adversarial_generated_test_exposures"]} |')
    control = summary["runs"]["control"]["result"]["metrics"]
    adaptive = summary["runs"]["adaptive"]["result"]["metrics"]
    lines += ["", f'Nhánh đối kháng phát hiện {adaptive["cases_detected"]}/15 code, đối chứng cố định phát hiện {control["cases_detected"]}/15. Test bắt lỗi lần lượt {adaptive["bug_detecting_tests"]}/60 và {control["bug_detecting_tests"]}/60.', "",
              f'Tổng cộng {summary["distinct_accepted_behaviors_across_rounds"]} hành vi code đối kháng khác nhau giữa các vòng; số code qua bộ lọc trong bảng có thể gồm hành vi lặp lại từ vòng trước.', "",
              "Đối kháng ở đây có nghĩa là code vượt qua những test đúng mà policy hiện tại sinh được, đồng thời còn nhân chứng sai trong miền input đã khai báo. Điều đó không chứng minh code khó hơn với mọi model hoặc sai ở ít input hơn. Code không được hỗ trợ, code đúng toàn miền, code sai toàn miền, code trùng hành vi và code bị cắt đều bị loại.", "",
              "Hai nhánh khớp số bước học và ngân sách sinh test; nhánh đối kháng tốn thêm chi phí probe và sinh/lọc code. Không tuyên bố tổng chi phí tính toán bằng nhau.", "",
              "Các lượt GRPO cố định 96 bước một mạch trước đây vẫn được giữ trong results/apps-sft-grpo; không dùng chúng thay cho đối chứng hai vòng trong bảng này.", "",
              "![Kết quả Validation](validation-comparison.png)", "", "metrics.json giữ kế hoạch, hash, gradient/update, phân loại thất bại và thống kê từng vòng. per-problem.csv tách kết quả theo bài nguồn. Log đầy đủ và adapter lưu trên ổ D, không đưa vào Git."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor="white")
    keys = list(summary["runs"])
    labels = ["Qwen gốc", "SFT", "GRPO\ncố định", "GRPO\nđối kháng"]
    for ax, field, title in zip(axes, ("valid_rate", "io_accuracy", "case_detection_at_4"),
                               ("Test đúng định dạng và miền input", "Test có expected đúng", "Code lỗi phát hiện với 4 test"), strict=True):
        values = [summary["runs"][key]["result"]["metrics"][field] * 100 for key in keys]
        bars = ax.bar(labels, values, color=["#9ca49c", "#b5bea4", "#7d9baf", "#47735b"], width=.65)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f'{value:.1f}%', ha="center", fontsize=9,
                    fontweight="bold" if value == max(values) else "normal")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Phiên bản model")
        ax.set_ylabel("Tỷ lệ (%)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("APPS Validation · cùng prompt, seed và ngân sách sinh test", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "validation-comparison.png", dpi=180, facecolor="white")
    fig.savefig(output / "validation-comparison.svg", facecolor="white")
    svg = output / "validation-comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    print("Saved:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
