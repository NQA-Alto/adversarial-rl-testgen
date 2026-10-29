# Curriculum đối kháng với nhãn input-first

Hai nhánh bắt đầu từ SFT64 input-first, cùng hai vòng × 48 bước GRPO. Bảng dùng seed sinh 42.

## Kết quả từng lượt

| Cấu hình | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---|---|---|---|---|
| Qwen gốc | 35 | 15 | 4 | 3 |
| SFT64 | 41 | 18 | 6 | 3 |
| GRPO cố định | 41 | 19 | 4 | 3 |
| GRPO đối kháng | 41 | 16 | 5 | 3 |

Hai nhánh cùng phát hiện 3/15 code ở seed 42.

![Curriculum đối kháng với nhãn input-first](validation-comparison.png)

Hình 1. Curriculum đối kháng với nhãn input-first.

## Dữ liệu

[metrics.csv](metrics.csv) · [metrics.json](metrics.json) · [per-problem.csv](per-problem.csv)

[Cấu hình và chỉ số đánh giá](../../docs/EXPERIMENTS.md).
