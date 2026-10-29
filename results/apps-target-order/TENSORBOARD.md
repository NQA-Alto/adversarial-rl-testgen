# Đọc TensorBoard cho phép thử thứ tự nhãn

Mở http://127.0.0.1:6006, chọn SCALARS. Danh sách dưới đây chỉ gồm các lượt đầy đủ của phép thử, không gồm GRPO bị ngắt của thử nghiệm LoRA.

| Nhánh | Mốc | Run |
|---|---|---|
| Expected trước | SFT | apps-sft-v2-20261029-141301-018558 |
| Expected trước | GRPO | apps-grpo-pilot-20261029-143736-431372 |
| Input trước | SFT | apps-sft-v2-20261029-143238-169569 |
| Input trước | GRPO | apps-grpo-pilot-20261029-144757-263819 |

Ở bộ lọc Runs, nhập:

`20261029-141301-018558|20261029-143736-431372|20261029-143238-169569|20261029-144757-263819`

Ở Filter tags, dùng `verified_io_accuracy|verified_attack_rate|sft/loss|gradient/norm_observed`.

- Chọn trục Step. Đối với log được chuyển đổi, WALL/RELATIVE có thể phản ánh lúc export, không phải thời gian train thực.
- rollout/verified_io_accuracy và rollout/verified_attack_rate là bốn mẫu trong nhóm Train; không phải điểm Validation.
- Không so trực tiếp loss SFT với loss GRPO; chúng tối ưu các mục tiêu khác nhau.
- Smoothing chỉ làm mượt đường hiển thị. Kết luận chất lượng lấy từ bảng Validation đã chấm lại.
- Có gradient và adapter thay đổi là bằng chứng học, không thay cho bằng chứng test tốt hơn.
