# Thực nghiệm

Các phép so sánh dùng Qwen3.5-2B, BF16 và LoRA rank 4. Chỉ Train tham gia học; Validation gồm năm bài nguồn với 15 code lỗi. Model chưa được đánh giá trên Test.

## Chỉ số

Mỗi code sinh bốn test. Một lượt đánh giá tạo 60 test, kể cả đầu ra hỏng JSON, ngoài miền hoặc bị cắt.

| Chỉ số | Cách tính |
|---|---|
| Tỷ lệ hợp lệ | Số test đúng JSON và miền input / tổng test sinh |
| IO Accuracy | Số test có expected đúng / tổng test sinh |
| Attack Rate | Số test có expected đúng và phát hiện lỗi / tổng test sinh |
| Code phát hiện @4 | Số mutant có ít nhất một test phát hiện lỗi / tổng mutant |

Test chỉ bắt lỗi khi expected khớp reference và mutant trả khác đáp án hoặc phát sinh lỗi thực thi được hỗ trợ. Lỗi của bộ thông dịch được báo riêng. Phân tích input-only dùng reference để xác định cơ hội bắt lỗi, không thay expected model trong các chỉ số chính.

Các bảng nhiều seed dùng trung bình và độ lệch chuẩn mẫu của ba lần sinh 42/43/44. Mỗi phương án chỉ có một lượt train, nên độ lệch chuẩn này chưa phản ánh biến động giữa các lần huấn luyện độc lập.

## SFT64 và SFT256

Hai checkpoint được lưu trong cùng một quỹ đạo học, với chung model khởi đầu, 319 nhãn input-first và 64 bước đầu. Mỗi bước xử lý bốn ví dụ; learning rate 5e-5.

| Ngân sách SFT | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---|---:|---:|---:|---:|
| 64 bước | 43,33 | 19,00 | 5,33 | 4,00 |
| 256 bước | **54,33** | **26,00** | **10,33** | **7,00** |

SFT256 cải thiện ở cả ba seed sinh. [Kết quả từng lượt và đường loss](../results/apps-shared-sft-budget/README.md).

## GRPO sau SFT256

Thêm 96 bước GRPO từ đúng adapter SFT256, giữ code Train cố định và reward mặc định.

| Checkpoint | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---|---:|---:|---:|---:|
| SFT256 | 54,33 | **26,00** | **10,33** | **7,00** |
| SFT256 + GRPO96 | **55,33** | 24,67 | 8,67 | 6,00 |

Số test hợp lệ tăng nhẹ nhưng các chỉ số còn lại giảm. SFT256 được giữ làm checkpoint tốt hơn trong cặp này. [Bảng trước/sau RL](../results/apps-sft256-grpo/README.md).

## Curriculum đối kháng

Hai nhánh bắt đầu từ SFT64 input-first, mỗi nhánh hai vòng × 48 bước GRPO. Cả hai cùng khởi tạo lại optimizer ở đầu mỗi vòng. Nhánh đối kháng làm mới một phần code Train giữa các vòng; nhánh cố định giữ nguyên code.

| Nhánh | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---|---:|---:|---:|
| Cố định | 20,33 | 5,00 | 4,00 |
| Đối kháng | 19,00 | 5,00 | 4,00 |

Nhánh đối kháng có 120/384 test RL tiếp xúc code bổ sung. Hai nhánh bằng nhau về số code phát hiện và số test bắt lỗi; chi phí sinh/lọc code khiến tổng thời gian nhánh đối kháng cao hơn. [Kết quả và chi phí](../results/apps-input-first-review/README.md).

## Các phép thử khác

| Thử nghiệm | Nội dung | Kết quả |
|---|---|---|
| [Thứ tự nhãn SFT](../results/apps-target-order/README.md) | Expected trước so với input trước | Sau GRPO, expected trung bình 12 lên 20,67/60; phần lớn mức tăng đã xuất hiện ở SFT |
| [Tỷ trọng reward](../results/apps-reward-ablation/README.md) | Trọng số 0,1/0,2/0,7 so với chia đều | Tăng nhẹ ở một seed, chưa ổn định qua các seed sinh |
| [SFT với nhãn cũ](../results/apps-sft-budget/README.md) | Tăng 64 lên 256 bước | Expected tăng nhưng số code phát hiện giảm |
| [Phạm vi LoRA](../results/apps-lora-coverage/README.md) | Thêm projection của linear attention | Dừng trước đánh giá Validation |
| [Thinking và prompt](../results/apps-tuning/README.md) | Kiểm tra trần token và hợp đồng đầu ra | Các ứng viên không qua điều kiện trên Train |
| [Replay SFT](../results/apps-expected-budget/README.md) | Chạy lại mốc 64 lịch sử | Dừng do độ lệch tham số vượt ngưỡng; chuyển sang đối chứng cùng quỹ đạo |

[Toàn bộ bảng và biểu đồ](../results/README.md).

## Phân tích expected sai

Ở SFT64 + GRPO96 input-first, có 69/180 đầu ra hợp lệ nhưng expected sai. Trong đó, 36 input phân biệt được reference với mutant. Expected sai làm mất các cơ hội phát hiện lỗi này. [Phân loại theo bài](../results/apps-expected-diagnosis/README.md).

## Phạm vi kết luận

Các phép thử hiện cho thấy SFT có ích với cấu hình và dữ liệu đã chọn. GRPO và curriculum đối kháng chưa cải thiện các chỉ số chính. Số bài Validation còn ít và đã được dùng trong nhiều phép điều chỉnh, nên cần đánh giá trên Test sau khi chốt cấu hình. Các kết quả này không đại diện cho toàn bộ APPS hoặc mức cải thiện trong bài báo ATGen.

`metrics.csv` và `metrics.json` giữ các số đếm, cấu hình, seed và đường dẫn lượt chạy. Raw output, adapter và mã nguồn theo từng lượt được lưu ngoài Git trong `D:/DA/TTNT_De19_Assets/evidence`. Các script `summarize_apps_*.py` dựng lại bảng và biểu đồ từ những bản ghi đó.
