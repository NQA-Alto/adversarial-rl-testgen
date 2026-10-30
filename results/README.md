# Kết quả thực nghiệm

Các thư mục dưới đây lưu bảng số liệu, cấu hình và biểu đồ của từng phép thử. Cách tính chỉ số nằm trong [tài liệu thực nghiệm](../docs/EXPERIMENTS.md).

## Các phép so sánh chính

| Thư mục | Nội dung |
|---|---|
| [apps-shared-sft-budget](apps-shared-sft-budget/README.md) | SFT64 và SFT256 trên cùng một lượt học, nhãn input-first |
| [apps-sft256-grpo](apps-sft256-grpo/README.md) | SFT256 trước và sau 96 bước GRPO |
| [apps-sft256-equal-reward](apps-sft256-equal-reward/README.md) | GRPO từ SFT256 với reward chia đều; đối chiếu với lượt reward mặc định |
| [apps-input-first-adversarial](apps-input-first-adversarial/README.md) | Hai vòng GRPO cố định và đối kháng, bắt đầu từ SFT64 input-first |
| [apps-input-first-review](apps-input-first-review/README.md) | Lấy mẫu lại hai nhánh trên ba seed sinh và tổng hợp chi phí |

Các bảng APPS hiện dùng Validation gồm 15 code lỗi. Chưa có kết quả đánh giá model trên Test. Mỗi cấu hình được train một lần; ba seed sinh không phải ba seed train.

## Phân tích và các phép thử khác

| Thư mục | Nội dung |
|---|---|
| [apps-data-profile](apps-data-profile/README.md) | Thống kê tập dữ liệu và quy trình xử lý |
| [apps-target-order](apps-target-order/README.md) | So sánh thứ tự input/expected trong nhãn SFT |
| [apps-expected-diagnosis](apps-expected-diagnosis/README.md) | Expected sai theo checkpoint và bài toán |
| [apps-input-difficulty](apps-input-difficulty/README.md) | Tỷ lệ input làm lộ lỗi trong miền hữu hạn |
| [apps-serialization-diagnosis](apps-serialization-diagnosis/README.md) | Thứ tự khóa JSON trong nhãn và đầu ra |
| [apps-reward-ablation](apps-reward-ablation/README.md) | Thay tỷ trọng các thành phần reward |
| [apps-lora-coverage](apps-lora-coverage/README.md) | Thử mở rộng các lớp gắn LoRA; dừng trước Validation |
| [apps-lora-input-first](apps-lora-input-first/README.md) | LoRA 6 và 24 khối với cùng 256 bước, nhãn input-first |
| [apps-expected-budget](apps-expected-budget/README.md) | Lượt SFT dừng ở kiểm tra replay |
| [apps-tuning](apps-tuning/README.md) | Thinking và cách diễn đạt hợp đồng đầu ra |

## Các mốc APPS trước đó

Các lượt này dùng prompt hoặc thứ tự nhãn cũ, được lưu để đối chiếu quá trình thử nghiệm.

- [GRPO ban đầu](apps-pilot/README.md)
- [SFT và GRPO với nhãn expected-first](apps-sft-grpo/README.md)
- [Ngân sách SFT với nhãn expected-first](apps-sft-budget/README.md)
- [Curriculum đối kháng với nhãn expected-first](apps-adversarial/README.md)
- [Lấy mẫu lại hai nhánh expected-first](apps-adversarial-review/README.md)
- [Phân tích lỗi của các mốc đầu](apps-error-analysis/README.md)

`grpo-seed-check/` thuộc tập bài toán nội bộ trước khi chuyển sang APPS. Số liệu trong thư mục đó không được gộp vào các bảng APPS. `apps-active-selection/selection.json` ghi checkpoint đang được chọn cho giao diện.
