# Độ biến động theo seed sinh test

Đây là kiểm tra bổ sung đã lên kế hoạch trước khi xem Validation của hai nhánh reward mới. Cùng ba checkpoint cố định, ba seed sinh test 42/43/44; không train thêm và không chọn seed thuận lợi. Mỗi lượt gồm 60 test trên 15 code của năm bài Validation.

**Một seed huấn luyện, ba seed lấy mẫu.** Không được gọi đây là ba seed train hay độ ổn định giữa ba lần học. Các code cùng bài và các lần lấy mẫu có tương quan; SD bên dưới chỉ mô tả biến động, không phải khoảng tin cậy.

| Phiên bản | Seed sinh | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---|---:|---:|---:|---:|---:|
| SFT64 trước RL | 42 | 41 | 13 | 4 | 3 |
| GRPO reward hiện tại | 42 | 41 | 13 | 4 | 3 |
| GRPO reward chia đều | 42 | 45 | 15 | 5 | 4 |
| SFT64 trước RL | 43 | 42 | 12 | 2 | 2 |
| GRPO reward hiện tại | 43 | 40 | 13 | 2 | 2 |
| GRPO reward chia đều | 43 | 42 | 14 | 2 | 2 |
| SFT64 trước RL | 44 | 46 | 10 | 5 | 5 |
| GRPO reward hiện tại | 44 | 45 | 10 | 5 | 5 |
| GRPO reward chia đều | 44 | 46 | 10 | 5 | 5 |

## Trung bình và độ lệch chuẩn mẫu

| Phiên bản | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---|---:|---:|---:|---:|
| SFT64 trước RL | 43.00 ± 2.65 | 11.67 ± 1.53 | 3.67 ± 1.53 | 3.33 ± 1.53 |
| GRPO reward hiện tại | 42.00 ± 2.65 | 12.00 ± 1.73 | 3.67 ± 1.53 | 3.33 ± 1.53 |
| GRPO reward chia đều | **44.33 ± 2.08** | **13.00 ± 2.65** | **4.00 ± 1.73** | **3.67 ± 1.53** |

Giá trị trung bình cao nhất mỗi cột được in đậm, kể cả khi hòa. Đây là số lượng test/code, không phải tỷ lệ phần trăm và không phải kết quả trên Test.

## Chênh lệch chia đều trừ hiện tại theo cùng seed

| Seed sinh | Hợp lệ | Expected đúng | Test bắt lỗi | Code phát hiện |
|---:|---:|---:|---:|---:|
| 42 | +4 | +2 | +1 | +1 |
| 43 | +2 | +1 | +0 | +0 |
| 44 | +1 | +0 | +0 | +0 |

![Kết quả từng seed sinh test](sampling-variability.png)

Tiêu chí ứng viên ban đầu ở seed 42 được giữ nguyên trong README và metrics.json. Bảng bổ sung không được dùng để lặng lẽ đổi tiêu chí hoặc chọn lại checkpoint. Nếu các seed cho hướng khác nhau, phải ghi kết quả chưa ổn định; nếu cùng hướng vẫn chưa đủ kết luận trên dataset rộng hơn.

Script dựng lại: scripts/summarize_apps_reward_sampling.py. Raw output và adapter vẫn ở thư mục evidence trên D:; đường dẫn, hash và toàn bộ số liệu nằm trong sampling-metrics.json. Test phát triển và Test chính thức không được sử dụng.
