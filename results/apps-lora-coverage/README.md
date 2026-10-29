# Phạm vi adapter LoRA

So profile q_proj/v_proj của self-attention với profile thêm projection của linear attention.

Profile cũ có 208.896 tham số trên 6/24 khối; profile mở rộng có 798.720 tham số trên 24/24 khối. Hai lượt SFT64 hoàn tất; phép thử dừng trước Validation để kiểm tra thứ tự nhãn.

## Dữ liệu

[attempt.json](attempt.json) · [coverage.json](coverage.json)

[Cấu hình và chỉ số đánh giá](../../docs/EXPERIMENTS.md).
