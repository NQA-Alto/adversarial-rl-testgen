# Kiểm tra phạm vi LoRA — lượt dừng sớm

Đã xác nhận profile cũ có LoRA tại 6/24 khối (208.896 tham số), profile mới phủ 24/24 khối (798.720 tham số). Cả hai lượt SFT64 hoàn tất và lưu adapter; các projection được chọn có tensor B thay đổi.

Phép so sánh được chủ động dừng trong nhánh GRPO đầu tiên, trước mọi lượt Validation mới, sau khi phát hiện thứ tự nhãn SFT expected trước input. Phần GRPO chưa hoàn chỉnh không dùng làm bằng chứng so sánh chất lượng. Chưa có kết luận profile rộng tốt hơn hay kém hơn.

Kế hoạch gốc, mã lúc chạy, log, adapter SFT, STOP_REQUEST.json và trạng thái ngắt gốc đều được giữ tại evidence trên ổ D:. attempt.json ghi đường dẫn và hash. Trạng thái này không phải lỗi GPU hay lỗi bộ chấm.

Ưu tiên tiếp theo là phép thử riêng thứ tự input/expected, giữ profile LoRA cũ để không đổi đồng thời hai yếu tố. Xem [chẩn đoán nhãn](../apps-serialization-diagnosis/README.md) và [kế hoạch phạm vi LoRA](../../docs/APPS_LORA_EXPERIMENT.md).
