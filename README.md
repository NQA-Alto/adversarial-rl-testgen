# Adversarial Reinforcement Learning for Test Case Generation

Sinh ca kiểm thử cho hàm Python bằng Qwen3.5-2B chạy trên máy cá nhân. Mỗi ca kiểm thử gồm dữ liệu đầu vào và kết quả mong đợi (`input`, `expected`). Chương trình đối chiếu với lời giải tham chiếu để kiểm tra đáp án và xác định ca kiểm thử có phát hiện được lỗi hay không.

Đồ án thử nghiệm SFT, GRPO và curriculum đối kháng trên một tập con APPS Introductory. Model sinh test được tinh chỉnh bằng LoRA; model sinh code bổ sung giữ nguyên trọng số. Quá trình suy luận và huấn luyện đều chạy local.

## Dữ liệu

Tập con APPS Introductory dạng gọi hàm Python. Miền input được giới hạn để kiểm tra bằng hai lời giải tham chiếu. Code lỗi ban đầu tạo bằng đột biến AST; code bổ sung được model local sinh và lọc trong các vòng đối kháng.

| Tập | Bài toán | Code lỗi |
|---|---:|---:|
| Train | 58 | 145 |
| Validation | 5 | 15 |
| Test | 7 | 18 |

Các tập tách theo bài và URL nguồn trong APPS Train. Tập Test ở đây là holdout của đồ án, khác APPS Test chính thức.

## Kết quả

Đánh giá cuối trên Test, trung bình ba seed sinh 42/43/44; bốn test cho mỗi code.

| Phiên bản | Expected đúng /72 | Test bắt lỗi /72 | Code phát hiện /18 |
|---|---:|---:|---:|
| Qwen gốc | 14.00 | 7.00 | 5.00 |
| Sau SFT | **24.67** | **14.33** | **8.67** |
| GRPO cố định | **24.67** | **13.33** | **7.67** |
| GRPO đối kháng | **24.67** | **14.00** | **8.33** |

Phiên bản sử dụng được chọn trên Validation trước khi mở Test: **Sau SFT**.

Phép so sánh GRPO dùng cùng SFT 840 cha, cùng reward và hai vòng × 32 bước mỗi nhánh. Ba seed sinh chỉ phản ánh biến động lấy mẫu; mỗi cấu hình có một seed huấn luyện.

[Thiết lập, số liệu Validation và phân tích giới hạn](results/apps-expanded-adaptive/README.md)

![Kết quả Test](results/apps-expanded-adaptive/test.png)

Hình 1. Expected đúng và code lỗi phát hiện trên Test. Thanh sai số là độ lệch chuẩn qua ba seed sinh.

Một phép thử tiếp theo tăng tỷ trọng học trên code đối kháng lên 50% và sửa vòng thích nghi để kiểm tra đúng curriculum hiện tại. Cả hai nhánh vẫn phát hiện trung bình 8,33/15 code trên Validation; nhánh đối kháng chưa vượt SFT. Vì pilot chưa đạt điều kiện cải thiện đã chốt, không chạy seed xác nhận và giữ nguyên bản sử dụng. [Cấu hình và kết quả](results/apps-balanced-adversarial/README.md).

## Chạy demo

Bản chạy hiện tại dùng Windows; model, dữ liệu và adapter lưu ở `D:/DA/TTNT_De19_Assets`.

1. Chạy `MO_GIAO_DIEN.cmd`.
2. Mở [demo](http://127.0.0.1:7860/?__theme=light).
3. Chọn bài và phiên bản, xem kết quả đã lưu hoặc bấm **Sinh 4 test mới**.

Tab Kết quả hiển thị đối chứng và đánh giá cuối. Tab Thực nghiệm cho phép chạy kiểm tra dữ liệu, GRPO trên Train và xem đường học. [TensorBoard](http://127.0.0.1:6006) chạy cùng bộ công cụ. Dùng `DONG_GIAO_DIEN.cmd` để dừng dịch vụ.

## Môi trường

| Thành phần | Cấu hình đã dùng |
|---|---|
| CPU | AMD Ryzen 7 7840H |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB VRAM |
| RAM | 16 GB |
| Python | 3.12.15 |
| PyTorch | 2.14.1+cu132 |
| Transformers / PEFT / TRL | 5.20.2 / 0.21.2 / 1.16.0 |
| Model | Qwen3.5-2B, BF16 |
| LoRA | Rank 4, alpha 8, 208.896 tham số huấn luyện |

Các lệnh khởi chạy còn dùng đường dẫn của máy thực nghiệm. Model nền và checkpoint không được đưa vào Git; việc cài lại trên máy khác chưa được kiểm tra.

## Mã nguồn và tài liệu

- [Dữ liệu](docs/APPS_DATASET.md)
- [Cách dùng demo](docs/DEMO.md)
- [Phương pháp tham khảo](docs/ATGEN_ALIGNMENT.md)
- [Thực nghiệm cuối](results/apps-expanded-adaptive/README.md)

`apps_core.py` chứa prompt và bộ chấm; `restricted_python.py` thực thi tập con Python; `apps_model_runner.py` chạy GRPO và đánh giá; `apps_adversary.py` sinh và lọc code; `apps_ui.py` cùng `apps_presentation.py` tạo giao diện Gradio.

## Giới hạn

Hỗ trợ hàm Python trong cú pháp và miền input đã khai báo. Chưa phải hệ thống chấm mọi chương trình hoặc benchmark APPS đầy đủ. Bộ dữ liệu và holdout nhỏ; expected sai vẫn là một nguồn thất bại. Việc chạy lại trên máy khác chưa được kiểm tra.

## Tham khảo

- [ATGen](https://arxiv.org/abs/2510.14635)
- [APPS](https://github.com/hendrycks/apps) và [codeparrot/apps](https://huggingface.co/datasets/codeparrot/apps)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
