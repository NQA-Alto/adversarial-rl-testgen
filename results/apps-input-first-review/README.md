# Lấy mẫu lại hai nhánh input-first

So sánh Qwen gốc, SFT64 và hai nhánh GRPO trên ba seed sinh 42/43/44.

| Cấu hình | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---|---|---|---|---|
| Qwen gốc | 35,33 ± 0,58 | 14,67 ± 2,52 | 3,67 ± 0,58 | 3,33 ± 0,58 |
| SFT64 | 43,33 ± 3,21 | 19,00 ± 2,65 | **5,33 ± 0,58** | **4,00 ± 1,00** |
| GRPO cố định | 43,33 ± 3,21 | **20,33 ± 1,53** | 5,00 ± 1,73 | **4,00 ± 1,73** |
| GRPO đối kháng | **43,67 ± 3,06** | 19,00 ± 3,00 | 5,00 ± 1,00 | **4,00 ± 1,73** |

Trung bình ± SD trên ba seed sinh; mỗi cấu hình có một lượt train.

## Kết quả từng lượt

| Cấu hình | Seed sinh | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---|---|---|---|---|---|
| Qwen gốc | 42 | 35 | 15 | 4 | 3 |
| SFT64 | 42 | 41 | 18 | 6 | 3 |
| GRPO cố định | 42 | 41 | 19 | 4 | 3 |
| GRPO đối kháng | 42 | 41 | 16 | 5 | 3 |
| Qwen gốc | 43 | 35 | 17 | 3 | 3 |
| SFT64 | 43 | 42 | 22 | 5 | 4 |
| GRPO cố định | 43 | 42 | 22 | 4 | 3 |
| GRPO đối kháng | 43 | 43 | 22 | 4 | 3 |
| Qwen gốc | 44 | 36 | 12 | 4 | 4 |
| SFT64 | 44 | 47 | 17 | 5 | 5 |
| GRPO cố định | 44 | 47 | 20 | 7 | 6 |
| GRPO đối kháng | 44 | 47 | 19 | 6 | 6 |

Nhánh cố định phát hiện trung bình 4.00/15 code; nhánh đối kháng 4.00/15. Chi phí sinh và lọc code được lưu trong costs.csv.

![Lấy mẫu lại hai nhánh input-first](sampling.png)

Hình 1. Lấy mẫu lại hai nhánh input-first.

## Dữ liệu

[costs.csv](costs.csv) · [costs.json](costs.json) · [metrics.csv](metrics.csv) · [metrics.json](metrics.json)

[Cấu hình và chỉ số đánh giá](../../docs/EXPERIMENTS.md).
