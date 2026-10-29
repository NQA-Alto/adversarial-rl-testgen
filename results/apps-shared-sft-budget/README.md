# SFT64 và SFT256 từ cùng một lượt học

Lấy checkpoint64 và cuối256 từ cùng một quỹ đạo optimizer mới. Chung model khởi đầu, 319 nhãn input-first, thứ tự ví dụ và 64 bước đầu. Không dùng adapter lịch sử làm đối chứng replay. Một seed train; đây là SFT, chưa phải RL.

## Kiểm tra Train

| Bước SFT | Hợp lệ /164 | Expected đúng /164 | Test bắt lỗi /164 | Code phát hiện /41 |
|---:|---:|---:|---:|---:|
| 64 | 119 | 59 | 27 | 21 |
| 256 | 151 | 97 | 54 | 31 |

Qua điều kiện Train: **True**. Các số Train chỉ đo mức khớp dữ liệu đã học.

## Validation · đủ ba seed sinh

| Bước SFT | Seed sinh | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---:|---:|---:|---:|---:|---:|
| 64 | 42 | 40 | 17 | 6 | 3 |
| 256 | 42 | 53 | 27 | 10 | 7 |
| 64 | 43 | 42 | 22 | 5 | 4 |
| 256 | 43 | 56 | 27 | 10 | 7 |
| 64 | 44 | 48 | 18 | 5 | 5 |
| 256 | 44 | 54 | 24 | 11 | 7 |

| Bước SFT | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---:|---:|---:|---:|---:|
| 64 | 43.33 ± 4.16 | 19.00 ± 2.65 | 5.33 ± 0.58 | 4.00 ± 1.00 |
| 256 | **54.33 ± 1.53** | **26.00 ± 1.73** | **10.33 ± 0.58** | **7.00 ± 0.00** |

Qua điều kiện đặt trước để cân nhắc thử RL tiếp: **True**. Expected tổng ba seed phải tăng ít nhất 9/180, tổng hợp lệ/test bắt lỗi/code phát hiện không giảm. Không tự thay checkpoint GRPO đang dùng bằng một mốc SFT chỉ vì expected cao.

Một seed train và ba seed sinh khác với ba lần huấn luyện độc lập. SD mô tả biến động lấy mẫu, không phải khoảng tin cậy. Phép thử bổ sung sau nhiều lần xem Validation có nguy cơ chọn quá khớp tập này. Test vẫn được giữ riêng.

Đã học 256 bước, 1024 lượt ví dụ; thời gian pha học/probe đã ghi 779.4 giây, đỉnh PyTorch cấp phát 4.362GiB. Không phải tổng thời gian hệ thống hay tổng VRAM máy.

![Đường loss Train](learning-curve.png)

Hình 1. Loss minibatch Train và trung bình trượt 8 bước; không phải loss Validation. Mốc64 là checkpoint đối chứng cùng quỹ đạo.

Mọi adapter, raw output, kế hoạch và mã nguồn snapshot nằm trong evidence ghi tại metrics.json. Không sửa đáp án model sau sinh.
