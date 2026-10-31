# Kết quả thực nghiệm

| Kết quả | Phạm vi |
|---|---|
| [Curriculum cân bằng](apps-balanced-adversarial/README.md) | Hai vòng GRPO, kiểm tra code hiện tại và lấy mẫu đối kháng 50%; Validation |
| [APPS mở rộng](apps-expanded-adaptive/README.md) | SFT 840, hai nhánh GRPO, Validation và Test đã khóa |
| [Ngân sách SFT](apps-shared-sft-budget/README.md) | Mốc SFT 64 và 256 trên cùng quỹ đạo, tập Train cũ |
| [GRPO từ SFT 256](apps-sft256-grpo/README.md) | Đối chứng trước và sau RL trên Validation cũ |
| [Curriculum đối kháng từ SFT 64](apps-input-first-review/README.md) | Hai vòng đối kháng và cố định trên Validation cũ |
| [Phân tích expected](apps-expected-diagnosis/README.md) | Các dạng expected sai ở mốc SFT 64 + GRPO |

Số liệu giữa các phiên bản dữ liệu và prompt được giữ riêng. Các thư mục còn lại lưu những phép thử trước đó; chúng không thay kết quả của checkpoint đã chốt. Cách tính chỉ số và giới hạn so sánh nằm trong [tài liệu thực nghiệm](../docs/EXPERIMENTS.md).
