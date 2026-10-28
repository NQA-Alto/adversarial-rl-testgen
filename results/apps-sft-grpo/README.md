# APPS: prompt v2, SFT và GRPO

Cùng 15 code lỗi từ năm bài Validation, bốn test/code, seed sinh 42. Tập Test chưa được model đánh giá.

| Phiên bản | Test hợp lệ | Expected đúng | Test bắt lỗi | Code phát hiện @4 |
|---|---:|---:|---:|---:|
| Qwen gốc | 35/60 | 15/60 | 4/60 | 3/15 |
| Sau SFT | 41/60 | 13/60 | 4/60 | 3/15 |
| Sau SFT + GRPO | 44/60 | 15/60 | 4/60 | 3/15 |

Qwen gốc ở bảng này dùng prompt v2. Tác động đổi prompt so với v1 được ghi riêng trong metrics.json; không gộp thành cải thiện do huấn luyện.

SFT dùng nhãn input/expected từ APPS Train; GRPO dùng reward và bắt đầu từ đúng adapter SFT. Đây là GRPO với code lỗi cố định, chưa có curriculum adversary trên APPS.

![So sánh Validation](validation-comparison.png)

Log gốc, target SFT, adapter, gradient, bản chụp mã nguồn và các biên bản kiểm chứng nằm trong thư mục round ghi tại metrics.json. Các đầu ra sai/bị cắt vẫn ở trong mẫu số.

Một seed và năm bài Validation chưa đủ kết luận hiệu quả ổn định. per-problem.csv cho phép xem khác biệt theo bài; metrics.json giữ phân loại lỗi và thông tin từng lượt.
