import argparse
from pathlib import Path

from apps_core import digest, read, write
from apps_model_runner import PROJECT
from audit_apps_run import audit


def summarize(thinking, trials):
    result = read(thinking / "result.json")
    review = read(thinking / "futility-review.json")
    if result["status"] != "stopped_early_infeasible" or review["status"] != "passed":
        raise ValueError("Thinking feasibility record is incomplete")
    from apps_reasoning_probe import audit as audit_reasoning
    audit_reasoning(thinking)
    records = []
    for path in trials:
        trial = read(path / "result.json")
        if trial["status"] != "completed":
            raise ValueError("Prompt trial is not complete")
        audit(Path(trial["train_run"]))
        if trial["validation_evaluated"]:
            audit(Path(trial["validation_run"]))
        records.append({"path": str(path), "result_sha256": digest(path / "result.json"), "result": trial})
    output = PROJECT / "results/apps-tuning"
    output.mkdir(parents=True, exist_ok=True)
    write(output / "experiments.json", {"thinking_probe": {"path": str(thinking), "result": result, "review": review},
                                       "prompt_trials": records, "training": False, "model_test_evaluation": False,
                                       "active_configuration_changed": False})
    lines = ["# APPS: kiểm tra suy luận và hợp đồng đầu ra", "", "Đây là các thử nghiệm suy luận với adapter SFT đóng băng, không phải các lượt RL mới. Checkpoint và cấu hình demo hiện tại giữ nguyên. Test chưa được model đánh giá.", "",
             "## Thinking có thẻ đáp án", "", "Thử cùng prompt, checkpoint và trần 768 token trên tám bài Train định trước; chỉ đổi native enable_thinking. Parser yêu cầu định dạng thẻ đã khai báo, không sửa JSON hay đáp án của model.", ""]
    partial = result["modes"]["thinking"]
    lines += [f'Đã lưu {partial["completed_samples"]}/32 mẫu thinking, trong đó {partial["terminated_samples"]} mẫu kết thúc. Ngay cả khi toàn bộ mẫu còn lại kết thúc, tỷ lệ tối đa chỉ còn {partial["best_possible_terminated_fraction"] * 100:.1f}%, dưới ngưỡng 80% đặt trước. Lượt được dừng sớm vì không còn khả năng đạt tiêu chí.', "",
              "Việc dừng vì vô vọng được bổ sung sau kế hoạch ban đầu và đã ghi trong result.json; không mô tả đây là một nhánh benchmark hoàn chỉnh. Các mẫu đang sinh dở lúc dừng không nằm trong bản ghi. Trực tiếp dùng định dạng thẻ mới cũng gặp lỗi tuân thủ định dạng, vì vậy kết quả này chưa cô lập hoàn toàn chất lượng suy luận. Chỉ kết luận cấu hình thử hiện tại chưa đủ điều kiện dùng tiếp.", "",
              "Model card local của Qwen3.5-2B có cảnh báo thinking loops; không suy rộng thất bại dưới trần token này thành việc model không có khả năng suy luận.", "",
              "## Hai lượt thử prompt JSON", "", "Cùng SFT, cùng toàn bộ 41 code Train, bốn test/code, seed 42, tối đa 96 token. Bộ chấm và trọng số không đổi. Các kiểu expected là annotation giao diện cho bài APPS đã biết, được kiểm tra bằng reference trên miền hữu hạn; không đưa reference hay đáp án cụ thể vào prompt. Đây là thông tin giao diện bổ sung, không phải thành phần nguyên bản của ATGen.", "",
              "| Prompt | Hợp lệ /164 | Expected đúng /164 | Test bắt lỗi /164 | Qua tiêu chí Train | Chấm Validation |", "|---|---:|---:|---:|---|---|"]
    baseline = records[0]["result"]["baseline_train_metrics"]
    lines.append(f'| v2, đối chứng SFT | {baseline["valid_tests"]} | {baseline["correct_expected"]} | {baseline["bug_detecting_tests"]} | Mốc đối chứng | Đã có từ lượt trước |')
    for entry in records:
        trial = entry["result"]
        m = trial["train_metrics"]
        lines.append(f'| {trial["plan"]["candidate"]} | {m["valid_tests"]} | {m["correct_expected"]} | {m["bug_detecting_tests"]} | {"Có" if trial["train_gate_passed"] else "Không"} | {"Có" if trial["validation_evaluated"] else "Không"} |')
    lines += ["", "v3 bỏ ví dụ JSON và viết lại prompt; nhiều mẫu chỉ trả trường expected, thiếu input. v4 giữ nguyên toàn bộ prompt v2 và chỉ thêm kiểu expected vào system instruction để kiểm tra một thay đổi hẹp hơn.", "",
              "Tiêu chí Train được ghi trước từng lượt: expected đúng tăng ít nhất 5 điểm phần trăm, số test hợp lệ và test bắt lỗi không giảm. Chỉ khi qua Train mới chấm Validation một lần. Có thể xem kết quả Train chưa qua tiêu chí để điều chỉnh thiết kế; không dùng Test cho việc này. Không chọn lại checkpoint hay seed để làm đẹp bảng.", ""]
    for entry in records:
        trial = entry["result"]
        if trial["validation_evaluated"]:
            m = trial["validation_metrics"]
            lines += [f'Validation của {trial["plan"]["candidate"]}: {m["correct_expected"]}/60 expected đúng, {m["bug_detecting_tests"]}/60 test bắt lỗi, {m["cases_detected"]}/15 code phát hiện. Đủ tiêu chí ứng viên: {trial["eligible_prompt_candidate"]}.', ""]
    lines += ["## Liên hệ với phân tích lỗi", "", "[Phân tích hiện tại](../apps-error-analysis/README.md) cho thấy expected sai làm mất nhiều cơ hội bắt lỗi. Cách tính input-only attack chỉ dùng oracle để chẩn đoán; không biến expected do reference tạo thành thành tích của model. Ý tưởng phân tích này tham khảo ATGen §5.3.", "",
              "Nếu một prompt cải thiện, phải ghi đây là tác động prompt/giao diện. Chỉ sau một thí nghiệm cập nhật trọng số riêng có đối chứng mới được bàn về phần cải thiện do RL. Một seed và tập bài nhỏ vẫn là giới hạn.", "",
              "Log thô, bản chụp mã nguồn và kế hoạch trước chạy được lưu trong các thư mục evidence ghi tại experiments.json; không đưa model weights hay raw dataset lên Git."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("Saved:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thinking", type=Path, required=True)
    parser.add_argument("--trial", type=Path, action="append", required=True)
    args = parser.parse_args()
    summarize(args.thinking, args.trial)
