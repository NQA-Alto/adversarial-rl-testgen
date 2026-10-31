# Thực nghiệm

## Cách tính chỉ số

| Chỉ số | Tử số | Mẫu số |
|---|---|---|
| Valid rate | Output có JSON và input hợp lệ | Toàn bộ output sinh ra |
| IO accuracy | Test có expected đúng theo reference | Toàn bộ output sinh ra |
| Attack rate | Test đúng và làm lộ lỗi của code | Toàn bộ output sinh ra |
| Code detection at 4 | Code có ít nhất một test đúng bắt lỗi | Toàn bộ code được đánh giá |

Output không hợp lệ hoặc bị cắt vẫn nằm trong mẫu số. Code detection dùng ngân sách tối đa bốn test mỗi code, khác pass@1 của model viết code. Các chỉ số không được thay thế cho nhau khi so với ATGen.

## Mốc đã đánh giá trên Test

[Tập Train mở rộng và SFT 840](../results/apps-expanded-adaptive/README.md) gồm 58 bài Train, 5 bài Validation và 7 bài Test giữ riêng từ APPS Train. Checkpoint được chọn bằng Validation trước khi chạy Test. SFT tốt hơn Qwen gốc; hai nhánh GRPO ở ngân sách hai vòng 32 bước chưa vượt SFT.

Ba seed sinh 42, 43, 44 phản ánh biến động lấy mẫu. Các mutant cùng bài có tương quan; số mutant không phải số bài toán độc lập.

## Thử nghiệm tiếp theo

Lịch cân bằng tăng số nhóm được học trên code đối kháng. Nhánh cố định và đối kháng có cùng parent SFT, case ID, số bước, learning rate và reward. Mỗi nhánh được học hai vòng 64 bước trên hai seed train. Chỉ code tại các vị trí đối kháng được thay; sau vòng đầu, bộ sinh code được lọc lại theo policy mới.

Các phép thử mới chỉ dùng Train/Validation. Test cũ giữ nguyên để mô tả phiên bản đã chốt, không được gán cho checkpoint mới. Muốn đánh giá cuối một phiên bản tiếp theo cần một holdout mới chưa tham gia điều chỉnh.

## Các mốc trước

- [SFT 64 và 256 trên cùng quỹ đạo](../results/apps-shared-sft-budget/README.md)
- [SFT 256 trước và sau GRPO](../results/apps-sft256-grpo/README.md)
- [Hai nhánh đối kháng từ SFT 64](../results/apps-input-first-review/README.md)

Đây là kết quả lịch sử trên tập Train nhỏ hơn, không gộp với tập mở rộng. Các lượt không cải thiện được giữ lại trong nhật ký và phân tích riêng.

Output gốc, adapter, mã nguồn đóng băng và các phép chấm lại được lưu trong `D:/DA/TTNT_De19_Assets/evidence`. Git giữ mã nguồn, cấu hình, bảng tổng hợp và biểu đồ.
