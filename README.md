# Adversarial Reinforcement Learning for Test Case Generation

Sinh ca kiểm thử cho hàm Python bằng Qwen3.5-2B chạy trên máy cá nhân. Mỗi ca kiểm thử gồm dữ liệu đầu vào và kết quả mong đợi (`input`, `expected`). Chương trình đối chiếu với lời giải tham chiếu để kiểm tra đáp án và xác định ca kiểm thử có phát hiện được lỗi hay không.

Đồ án thử nghiệm SFT, GRPO và curriculum đối kháng trên một tập con APPS Introductory. Model sinh test được tinh chỉnh bằng LoRA; model sinh code bổ sung giữ nguyên trọng số. Quá trình suy luận và huấn luyện đều chạy local.

## Dữ liệu

Từ APPS Train, chọn 29 bài dạng gọi hàm và tạo 74 phiên bản code lỗi bằng đột biến AST. Miền input được giới hạn để kiểm tra toàn bộ bằng hai lời giải tham chiếu.

| Tập | Số bài | Số code lỗi |
|---|---:|---:|
| Train | 17 | 41 |
| Validation | 5 | 15 |
| Test | 7 | 18 |

Các tập được chia theo bài và URL nguồn. Tập Test này là phần giữ lại từ APPS Train; APPS Test chính thức là tập khác và chưa được sử dụng.

## Kết quả

Kết quả Validation, trung bình trên ba seed sinh test 42, 43 và 44. Mỗi code sinh bốn test, tương ứng 60 test cho mỗi lượt đánh giá.

| Cấu hình | Expected đúng /60 | Test phát hiện lỗi /60 | Code phát hiện /15 |
|---|---:|---:|---:|
| SFT 64 bước | 19,00 | 5,33 | 4,00 |
| SFT 256 bước | 26,00 | 10,33 | **7,00** |
| SFT 256, LoRA 24 khối | **27,33** | 9,67 | **7,00** |
| SFT 256 + GRPO 96 bước | 24,67 | 8,67 | 6,00 |
| SFT 256 + GRPO 96, reward chia đều | 25,00 | **10,67** | **7,00** |

Hai mốc SFT 64 và 256 được lấy từ cùng một lượt học. Tăng thời lượng SFT cải thiện cả expected lẫn khả năng phát hiện lỗi. Mở rộng LoRA từ 6 lên 24 khối giúp expected tăng nhẹ, nhưng số test bắt lỗi giảm.

Reward chia đều giúp nhánh RL trở lại mức 7/15 code phát hiện, so với 6/15 của reward mặc định. So với SFT256, nhánh này chỉ thêm một test bắt lỗi trên tổng 180 test, trong khi expected giảm và kết quả giữa các seed biến động hơn. Demo vẫn dùng SFT256 6 khối làm mặc định.

Trong thí nghiệm đối kháng riêng, hai nhánh cùng bắt đầu từ SFT64 và học hai vòng, mỗi vòng 48 bước. Cả hai phát hiện trung bình 4/15 code. Kết quả hiện tại chưa cho thấy lợi ích của curriculum đối kháng ở cấu hình đã thử.

Mỗi cấu hình mới được huấn luyện với một seed. Ba seed sinh test chỉ phản ánh biến động khi lấy mẫu. Toàn bộ kết quả trên đây thuộc Validation; model chưa được đánh giá trên Test.

[Bảng thực nghiệm và cách tính chỉ số](docs/EXPERIMENTS.md)

![So sánh trước và sau GRPO](results/apps-sft256-equal-reward/comparison.png)

Hình 1. Kết quả SFT256 và SFT256 + GRPO96 với reward chia đều trên Validation. Thanh sai số là độ lệch chuẩn qua ba seed sinh test.

## Chạy demo

Bản chạy hiện tại dùng Windows và lưu model, dữ liệu, adapter tại `D:/DA/TTNT_De19_Assets`.

1. Chạy `MO_GIAO_DIEN.cmd`.
2. Mở [Gradio](http://127.0.0.1:7860/?__theme=light).
3. Chọn bài toán để xem code, input, expected và kết quả kiểm tra.

Giao diện có phần xem kết quả đã lưu, sinh test trực tiếp, chạy thực nghiệm và theo dõi huấn luyện. TensorBoard ở [cổng 6006](http://127.0.0.1:6006). Chạy `DONG_GIAO_DIEN.cmd` để dừng các dịch vụ.

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

## Tài liệu và mã nguồn

- [Chuẩn bị dữ liệu](docs/APPS_DATASET.md)
- [Chạy và huấn luyện](docs/APPS_TRAINING.md)
- [Các thí nghiệm](docs/EXPERIMENTS.md)
- [Phần triển khai so với ATGen](docs/ATGEN_ALIGNMENT.md)
- [Ví dụ demo](docs/DEMO.md)

Các script chính nằm trong `scripts/`: `apps_core.py` chứa bộ chấm, `apps_sft.py` thực hiện SFT, `apps_model_runner.py` chạy GRPO và đánh giá, `apps_adversarial_round.py` điều phối curriculum đối kháng, `apps_ui.py` tạo giao diện Gradio. Bảng số liệu và biểu đồ nằm trong `results/`.

## Phạm vi

Chương trình xử lý hàm Python trong tập cú pháp và miền input đã khai báo. Đây chưa phải hệ thống chấm mọi chương trình được nhập vào hoặc benchmark đầy đủ APPS. Kết quả trên năm bài Validation còn nhỏ; expected sai vẫn là nguồn thất bại chính.

## Tham khảo

- [ATGen: Adversarial Reinforcement Learning for Test Case Generation](https://arxiv.org/abs/2510.14635)
- [APPS](https://github.com/hendrycks/apps) và [codeparrot/apps](https://huggingface.co/datasets/codeparrot/apps)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
