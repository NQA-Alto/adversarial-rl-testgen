# Thử nghiệm APPS ban đầu

Prompt v1, Qwen gốc và lượt GRPO đầu tiên. Mỗi code Validation sinh bốn test, seed 42.

## Kết quả từng lượt

| Cấu hình | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---|---|---|---|---|
| Qwen gốc | 14 | 10 | 2 | 2 |
| GRPO ban đầu | 17 | 11 | 1 | 1 |

Đây là mốc đầu trước khi đổi prompt và thêm SFT.

## Dữ liệu

[metrics.csv](metrics.csv) · [metrics.json](metrics.json)

[Cấu hình và chỉ số đánh giá](../../docs/EXPERIMENTS.md).
