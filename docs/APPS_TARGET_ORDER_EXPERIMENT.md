# Kiểm chứng thứ tự nhãn SFT

Kế hoạch lập ngày 29/10/2026, trước khi học ứng viên input-first. Mục tiêu là kiểm tra riêng một thay đổi biểu diễn nhãn, không đổi bài toán hoặc cách chấm.

## Chẩn đoán

319 target SFT cũ đều đặt expected trước input do canonical JSON sắp xếp khóa. Trên 180 output Validation đã lưu mỗi phiên bản, Qwen gốc có 155 JSON bắt đầu bằng input, còn sau SFT có 177 JSON bắt đầu bằng expected. Số còn lại không đọc được JSON. Đây là quan sát về thứ tự sinh, chưa đủ chứng minh nguyên nhân gây accuracy thấp.

Giả thuyết: với model sinh từ trái sang phải, học chọn input trước rồi dự đoán expected có thể phù hợp hơn việc trả expected trước khi chốt input. JSON cả hai dạng vẫn hợp lệ; kết quả cũ không bị hủy hay chấm lại thành số khác.

## Đối chứng và ứng viên

- sorted-v1: thứ tự expected rồi input, dùng adapter SFT64 đối chứng vừa hoàn tất và có hash. Train probe đúng parent cũng được giữ lại.
- input-first-v1: bắt đầu lại từ cùng model gốc, cùng LoRA q/v tại sáu khối, SFT64 với input rồi expected.
- Giữ ngữ nghĩa mọi cặp input/expected và thứ tự khóa bên trong input. Không đổi thứ tự bài, learning rate, model, profile/rank LoRA, prompt, reward hay parser.
- Hai nhánh chạy GRPO96 mới đầy đủ từ SFT tương ứng. Không dùng phần GRPO bị ngắt của thử nghiệm phạm vi LoRA.
- Sau khi cả hai GRPO hoàn tất mới chấm bốn checkpoint SFT/GRPO trên Validation với seed sinh 42/43/44. Giữ cả 12 lượt, không chọn seed đẹp; Test chưa dùng.

Tiêu chí sàng lọc cho phép cân nhắc thử đối kháng tiếp: ở mốc GRPO, expected đúng trung bình tăng ít nhất 3/60, trong khi trung bình hợp lệ, test bắt lỗi và code phát hiện không giảm so với đối chứng. Không tự thay checkpoint chính khi chưa rà kết quả.

## Kiểm tra kỹ thuật

check_apps_target_order.py so đủ 319 target: byte của định dạng mặc định cũ không thay đổi; phiên bản mới giữ đúng các giá trị và nhãn bắt lỗi; chỉ thứ tự hai khóa ngoài cùng thay đổi. Parser đánh giá vẫn không phụ thuộc thứ tự khóa. Không sửa hay sắp xếp lại raw output của model khi chấm.

SFT và GRPO tiếp tục kiểm tra gradient, adapter thay đổi, số bước, hash và bản lưu. Phải tách ảnh hưởng đổi target SFT khỏi ảnh hưởng thêm của RL; thử nghiệm này chưa đo lợi ích curriculum đối kháng.

## Nơi lưu và giới hạn

Evidence gốc: apps-target-order-trial-20261029-143221-872463 trên ổ D:. Cấu hình demo và APPS_STATE chưa bị thay. Kết quả đầy đủ sẽ dựng bằng scripts/summarize_apps_order.py vào results/apps-target-order.

Vẫn chỉ một seed huấn luyện và năm bài Validation; ba seed sinh test không thay nhiều lần học độc lập. Đối chứng SFT được tái sử dụng và nondeterminism CUDA chưa loại trừ. Nếu không qua tiêu chí, giữ nguyên kết quả và kết luận không xác nhận cải thiện.
