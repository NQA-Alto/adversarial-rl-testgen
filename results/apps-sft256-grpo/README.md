# GRPO sau SFT256

Thêm 96 bước GRPO từ đúng adapter SFT256, trên code Train cố định. So với parent trên cùng ba seed sinh.

| Cấu hình | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---|---|---|---|---|
| SFT256 | 54,33 ± 1,53 | **26,00 ± 1,73** | **10,33 ± 0,58** | **7,00 ± 0,00** |
| SFT256 + GRPO96 | **55,33 ± 2,08** | 24,67 ± 0,58 | 8,67 ± 1,53 | 6,00 ± 0,00 |

Trung bình ± SD trên ba seed sinh; mỗi cấu hình có một lượt train.

## Kết quả từng lượt

| Cấu hình | Seed sinh | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---|---|---|---|---|---|
| SFT256 | 42 | 53 | 27 | 10 | 7 |
| SFT256 | 43 | 56 | 27 | 10 | 7 |
| SFT256 | 44 | 54 | 24 | 11 | 7 |
| SFT256 + GRPO96 | 42 | 53 | 24 | 9 | 6 |
| SFT256 + GRPO96 | 43 | 56 | 25 | 7 | 6 |
| SFT256 + GRPO96 | 44 | 57 | 25 | 10 | 6 |

Trong cặp so sánh này, SFT256 có kết quả tốt hơn theo tiêu chí chọn checkpoint. Các số đếm trước và sau GRPO được giữ riêng trong bảng.

![GRPO sau SFT256](comparison.png)

Hình 1. GRPO sau SFT256.

## Dữ liệu

[errors.csv](errors.csv) · [metrics.csv](metrics.csv) · [metrics.json](metrics.json)

[Cấu hình và chỉ số đánh giá](../../docs/EXPERIMENTS.md).
