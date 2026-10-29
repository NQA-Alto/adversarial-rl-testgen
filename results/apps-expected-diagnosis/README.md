# Phân tích expected sai

Đếm lỗi trên ba seed sinh của Qwen gốc, SFT64 input-first và các nhánh GRPO.

| Checkpoint | Expected đúng /180 | Hợp lệ, expected sai /180 | Input lộ lỗi nhưng expected sai |
|---|---|---|---|
| grpo96 | 62 | 69 | 36 |
| Qwen gốc | 44 | 62 | 29 |
| SFT64 | 57 | 73 | 35 |
| GRPO cố định | 61 | 69 | 36 |
| GRPO đối kháng | 57 | 74 | 38 |

Ở SFT64 + GRPO96, có 69/180 mẫu hợp lệ nhưng expected sai; 36 input trong số đó có thể làm lộ lỗi. Dữ liệu từng bài nằm trong per-problem.csv.

![Phân tích expected sai](expected-errors.png)

Hình 1. Phân tích expected sai.

## Dữ liệu

[analysis.json](analysis.json) · [counts.csv](counts.csv) · [per-problem.csv](per-problem.csv) · [wrong-expected-examples.json](wrong-expected-examples.json)

[Cấu hình và chỉ số đánh giá](../../docs/EXPERIMENTS.md).
