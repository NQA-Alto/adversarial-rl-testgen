# Lấy mẫu lại hai nhánh expected-first

Đánh giá bốn checkpoint trên các seed sinh 42/43/44. Mỗi checkpoint được huấn luyện một lần.

| Cấu hình | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---|---|---|---|---|
| Qwen gốc | 35,33 ± 0,58 | **14,67 ± 2,52** | **3,67 ± 0,58** | **3,33 ± 0,58** |
| SFT64 | **43,00 ± 2,65** | 11,67 ± 1,53 | **3,67 ± 1,53** | **3,33 ± 1,53** |
| GRPO cố định | 42,67 ± 3,06 | 10,67 ± 2,52 | 3,00 ± 1,00 | 2,67 ± 1,15 |
| GRPO đối kháng | 42,33 ± 3,21 | 11,00 ± 1,73 | 3,33 ± 1,53 | 3,00 ± 1,73 |

Trung bình ± SD trên ba seed sinh; mỗi cấu hình có một lượt train.

## Kết quả từng lượt

| Cấu hình | Seed sinh | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---|---|---|---|---|---|
| Qwen gốc | 42 | 35 | 15 | 4 | 3 |
| SFT64 | 42 | 41 | 13 | 4 | 3 |
| GRPO cố định | 42 | 42 | 13 | 3 | 2 |
| GRPO đối kháng | 42 | 41 | 12 | 3 | 2 |
| Qwen gốc | 43 | 35 | 17 | 3 | 3 |
| SFT64 | 43 | 42 | 12 | 2 | 2 |
| GRPO cố định | 43 | 40 | 11 | 2 | 2 |
| GRPO đối kháng | 43 | 40 | 12 | 2 | 2 |
| Qwen gốc | 44 | 36 | 12 | 4 | 4 |
| SFT64 | 44 | 46 | 10 | 5 | 5 |
| GRPO cố định | 44 | 46 | 8 | 4 | 4 |
| GRPO đối kháng | 44 | 46 | 9 | 5 | 5 |

Nhánh cố định phát hiện trung bình 2.67/15 code; nhánh đối kháng 3.00/15. Chi phí sinh và lọc code được lưu trong costs.csv.

![Lấy mẫu lại hai nhánh expected-first](sampling.png)

Hình 1. Lấy mẫu lại hai nhánh expected-first.

## Dữ liệu

[costs.csv](costs.csv) · [costs.json](costs.json) · [metrics.csv](metrics.csv) · [metrics.json](metrics.json)

[Cấu hình và chỉ số đánh giá](../../docs/EXPERIMENTS.md).
