# Tỷ trọng reward trong GRPO

Hai nhánh cùng SFT64 expected-first, cùng 96 bước GRPO; so trọng số 0,1/0,2/0,7 với chia đều 1/3.

## Kết quả từng lượt

| Cấu hình | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---|---|---|---|---|
| SFT64 | 41 | 13 | 4 | 3 |
| Reward 0,1/0,2/0,7 | 41 | 13 | 4 | 3 |
| Reward chia đều | 45 | 15 | 5 | 4 |

Reward chia đều tăng nhẹ ở seed 42, nhưng mức tăng chưa ổn định khi sinh lại với seed 43/44.

![Tỷ trọng reward trong GRPO](validation-comparison.png)

Hình 1. Tỷ trọng reward trong GRPO.

![Tỷ trọng reward trong GRPO](training-components.png)

Hình 2. Tỷ trọng reward trong GRPO.

[Kết quả sinh lại với ba seed](SAMPLING.md).

## Dữ liệu

[error-examples.json](error-examples.json) · [metrics.csv](metrics.csv) · [metrics.json](metrics.json) · [per-case.csv](per-case.csv) · [sampling-metrics.csv](sampling-metrics.csv) · [sampling-metrics.json](sampling-metrics.json)

[Cấu hình và chỉ số đánh giá](../../docs/EXPERIMENTS.md).
