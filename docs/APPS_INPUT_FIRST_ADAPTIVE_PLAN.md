# Kế hoạch đối kháng có điều kiện sau kiểm chứng thứ tự nhãn

Ghi ngày 29/10/2026, khi phép thử thứ tự nhãn đang học và chưa chấm Validation mới.

Chỉ chạy nếu ứng viên input-first qua tiêu chí đã ghi trong APPS_TARGET_ORDER_EXPERIMENT.md. Không nới tiêu chí sau khi nhìn kết quả. Nếu không qua, ghi chưa đủ điều kiện và giữ nguyên trạng thái hiện tại.

## Phép so sánh tiếp theo khi đủ điều kiện

- Cùng bắt đầu từ đúng adapter input-first SFT64 đã chấm, không bắt đầu từ hai parent khác nhau.
- Hai nhánh: code Train cố định và curriculum đối kháng theo vòng. Mỗi nhánh 2 vòng × 48 bước GRPO, cùng seed42, thứ tự bài, reward legacy-v1, prompt v2 và ngân sách sinh.
- Model sinh code vẫn là Qwen local gốc, đóng băng; lấy 16 code/bài đủ điều kiện rồi lọc. Dùng Train probe của đúng policy hiện tại, làm mới giữa hai vòng. Không dùng witness làm input prompt.
- Nhánh cố định cũng khởi tạo lại optimizer mỗi vòng, không dùng lượt 96 bước liền làm đối chứng tương đương cho hai vòng.
- Giữ checkpoint cuối theo ngân sách. Nếu chấm Validation giữa vòng để vẽ đường tiến triển thì không chọn checkpoint vòng một vì điểm tốt hơn.
- Sau hai vòng, kiểm tra lấy mẫu cuối với seed sinh 42/43/44 cho cả hai nhánh. Giữ tất cả kết quả; so thêm SFT cha và Qwen gốc trong cùng điều kiện.

Kết luận chính là chênh lệch code phát hiện và test bắt lỗi giữa hai nhánh cùng seed. Expected đúng và tỷ lệ hợp lệ là chỉ số bổ sung cần báo cáo. Một seed train chưa đủ kết luận độ ổn định của việc huấn luyện; ba seed sinh không khắc phục giới hạn này.

## Ranh giới kết luận

Lợi ích đổi thứ tự nhãn thuộc biểu diễn target SFT; lợi ích GRPO thuộc so sánh trước/sau RL từ cùng parent; lợi ích đối kháng thuộc so sánh curriculum cố định và làm mới. Không gộp ba thay đổi thành một tác động đối kháng.

Chạy nghiên cứu bằng parent chỉ định và --no-publish để tránh tự thay APPS_STATE hoặc demo. Kết quả mới phải lưu ở thư mục riêng, không ghi đè results/apps-adversarial của lượt trước.

## Trước đánh giá cuối

Sau khi kết thúc các phép so sánh đã chốt, khóa model, prompt, reward, checkpoint, seed/ngân sách và metric. Khi chọn một checkpoint phục vụ demo, dùng Validation, ưu tiên số code phát hiện, tiếp theo số test bắt lỗi, expected đúng, rồi tính đơn giản khi hòa. Mọi nhánh vẫn được trình bày dù không được chọn.

Chỉ sau bước khóa mới mở Test phát triển cho đợt đánh giá cuối. Không dùng kết quả Test để sửa thiết kế và chấm lại dưới nhãn “cuối”. Video dự phòng, đóng gói và báo cáo vẫn là các hạng mục riêng theo ưu tiên của người dùng.
