# Phương pháp tham khảo: ATGen

Tài liệu tham khảo chính là [ATGen: Adversarial Reinforcement Learning for Test Case Generation](https://arxiv.org/abs/2510.14635), bản v1. Đồ án triển khai vòng sinh test, học từ reward thực thi và làm mới code huấn luyện theo khả năng của policy.

## Phần triển khai

| Thành phần | Cách thực hiện |
|---|---|
| Test generator | Qwen3.5-2B local, trả JSON gồm input và expected |
| Bộ chấm | Đối chiếu expected với reference, rồi kiểm tra kết quả code cần thử |
| GRPO | TRL, bốn completion mỗi nhóm, cập nhật adapter LoRA |
| Reward | Kết hợp tính hợp lệ, độ đúng expected và khả năng bắt lỗi |
| Code generator | Qwen gốc, giữ nguyên trọng số, lấy mẫu lời giải cho bài Train |
| Lọc code | Giữ code còn sai trong miền thực nghiệm nhưng qua được test đúng của policy |
| Curriculum | Thay một phần code Train và làm mới giữa các vòng |
| Đối chứng | Hai nhánh cùng parent SFT64, cùng hai vòng × 48 bước và cách đánh giá |

Chỉ model sinh test được cập nhật bằng RL. Model sinh code không được huấn luyện cùng lúc. Code bổ sung được lấy từ các lời giải model sinh còn lỗi, thay vì yêu cầu model cố tình chèn một lỗi định trước.

## Những điều chỉnh

Đồ án dùng Qwen3.5-2B, LoRA rank 4 và GPU 8 GB; đầu ra là JSON trực tiếp, tắt chế độ thinking. SFT được thêm làm bước khởi động trước GRPO. Vì vậy, ảnh hưởng của SFT được đo riêng trước khi xét phần RL thêm vào.

Dữ liệu gồm 29 bài APPS dạng gọi hàm, với miền input hữu hạn và bộ thông dịch Python giới hạn. Code ban đầu là mutant AST; code bổ sung trong curriculum được model local sinh rồi lọc. Phạm vi này khác bộ benchmark APPS và Codeforces của bài báo.

Reward mặc định dùng trọng số định dạng/expected/attack là 0,1/0,2/0,7. Đã có phép so sánh với trọng số chia đều. Điều kiện định dạng JSON của đồ án khác định dạng có thẻ suy luận trong bài báo.

## Kết quả

Ở phép so sánh input-first, nhánh cố định và đối kháng đều phát hiện trung bình 4/15 code qua ba seed sinh. Nhánh đối kháng có 120/384 test RL được sinh trên các vị trí code bổ sung.

Tăng SFT64 lên SFT256 giúp số code phát hiện tăng từ 4 lên 7/15. Thêm GRPO96 với reward mặc định từ SFT256 còn 6/15.

Khi đổi reward sang chia đều, nhánh GRPO từ SFT256 đạt lại 7/15 code; test bắt lỗi trung bình 10,67/60 so với 10,33 của SFT, nhưng expected giảm từ 26 xuống 25/60. Đây là một phép điều chỉnh trọng số reward; chưa đủ để kết luận RL vượt SFT hoặc tái hiện mức tăng của bài báo.

Đồ án chưa triển khai toàn bộ benchmark của ATGen, lọc Best-of-N hay RL cho model sinh code. Số liệu ở đây dùng để đánh giá các thành phần của bản triển khai, không so trực tiếp với điểm benchmark trong bài báo.

[Kết quả đối kháng](../results/apps-input-first-review/README.md) · [Phép so sánh SFT và GRPO](../results/apps-sft256-grpo/README.md)
