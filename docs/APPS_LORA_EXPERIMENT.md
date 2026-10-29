# Kiểm chứng phạm vi adapter LoRA

Ngày lập kế hoạch: 29/10/2026. Đây là thí nghiệm cải thiện cấu hình huấn luyện; chưa thay cấu hình đang dùng trong demo.

## Vấn đề quan sát được

Expected sai làm mất nhiều test có input đủ khả năng phát hiện lỗi. Rà soát checkpoint và mã Transformers đang cài cho thấy Qwen3.5-2B có 24 khối ngôn ngữ: 18 khối linear attention và sáu khối full attention.

Profile LoRA ban đầu chỉ chọn q_proj/v_proj trong self_attn. Nó khớp 12 projection của sáu khối full attention, tương ứng 208.896 tham số được học. Những lượt trước vẫn là huấn luyện hợp lệ; giới hạn là phạm vi thích nghi hẹp. Chưa thể từ quan sát này kết luận đây là nguyên nhân duy nhất gây kết quả thấp.

## Thay đổi thử nghiệm

Giữ profile cũ attention-qv-v1 làm mặc định. Thêm profile hybrid-qkv-v1: giữ các q/v cũ và thêm in_proj_qkv ở 18 khối linear attention. Có 30 projection ở 24 khối, 798.720 tham số được học, cùng rank4/alpha8/dropout0. Không thêm LoRA cho vision, embedding hoặc MLP.

Phạm vi khớp được kiểm tra từ tên và shape của checkpoint trước khi nạp GPU; số tham số live và tensor B thay đổi được kiểm tra trong lượt học. Không chỉ suy đoán từ tên model.

## Phép so sánh đã chốt trước kết quả mới

Hai nhánh cùng bắt đầu từ model nền gốc; mỗi nhánh SFT64 rồi GRPO cố định96. Cùng Train, target pool 319 mẫu, thứ tự ví dụ, seed học42, learning rate, prompt v2 và reward legacy-v1. Khác profile LoRA và số tham số tương ứng.

Sau cả hai SFT, dùng đánh giá Train làm baseline đúng parent cho GRPO. Chỉ sau khi học xong cả hai nhánh GRPO mới chấm bốn checkpoint SFT/GRPO trên Validation, với ba seed sinh test 42/43/44. Giữ đủ 12 lượt, không chọn seed thuận lợi hoặc checkpoint trung gian. Tập Test vẫn giữ lại.

Tiêu chí sàng lọc để cân nhắc thử curriculum đối kháng tiếp: ở mốc GRPO, expected đúng trung bình tăng ít nhất 3/60 so với đối chứng mới, đồng thời trung bình test hợp lệ, test bắt lỗi và code phát hiện không giảm. Qua tiêu chí chưa có nghĩa tự thay cấu hình chính hoặc chứng minh hiệu quả thống kê.

## Cách kết luận

- So cùng mốc giữa hai profile: tác động thay phạm vi LoRA.
- So SFT và GRPO trong một profile: tác động thêm của RL.
- Phép thử này dùng code Train cố định, chưa đo lợi ích curriculum đối kháng. Nếu thử đối kháng tiếp, cần nhánh cố định cùng checkpoint cha và ngân sách tương ứng.
- Vẫn chỉ một seed huấn luyện; ba seed sinh test không thay thế nhiều lượt train độc lập.

Đây là điều chỉnh kỹ thuật của đồ án cho kiến trúc model local đang dùng. Không gọi nó là thí nghiệm nguyên bản của [ATGen](https://arxiv.org/html/2510.14635v1), cũng không thay thế các kết quả đối kháng đã lưu.

## Vận hành

Nguồn, kế hoạch, profile, hash và kết quả được lưu trong apps-lora-coverage-trial-20261029-141204-932567 dưới evidence trên ổ D:. GPU phải có ít nhất 6.000 MiB trống trước khi chạy; không tự đóng ứng dụng khác để lấy bộ nhớ. Model nền và tập dữ liệu không bị thay đổi.

Lượt này đã chủ động dừng trong nhánh GRPO đầu tiên, trước mọi Validation mới, khi phát hiện vấn đề thứ tự nhãn SFT. Hai adapter SFT64 được giữ; chưa có so sánh chất lượng LoRA hoàn chỉnh. Không tính phần GRPO dở vào kết quả. Xem [bản ghi dừng sớm](../results/apps-lora-coverage/README.md) và [chẩn đoán thứ tự nhãn](../results/apps-serialization-diagnosis/README.md).

scripts/summarize_apps_lora.py chỉ dùng cho một lượt đủ toàn bộ kế hoạch; không chạy để biến lượt thiếu này thành bảng so sánh hoàn chỉnh.
