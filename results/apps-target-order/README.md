# Kiểm chứng thứ tự nhãn input/expected

Giữ nguyên 319 cặp nhãn, profile LoRA 6 khối, model, dữ liệu, prompt, parser, reward, learning rate và ngân sách. Chỉ đổi thứ tự hai trường ngoài cùng của completion SFT. Mọi output model được chấm nguyên trạng.

Đối chứng dùng SFT64 đã hoàn tất trước khi phát hiện vấn đề; kiểm tra byte target cũ và ngữ nghĩa target mới, giữ nguyên thứ tự ví dụ. Hai nhánh GRPO96 chạy mới đầy đủ; không dùng lượt GRPO dở của phép thử LoRA. Chỉ chấm Validation sau khi cả hai GRPO hoàn tất.

| Thứ tự nhãn SFT | Mốc | Seed sinh | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---|---|---:|---:|---:|---:|---:|
| Expected trước | SFT | 42 | 39 | 14 | 5 | 4 |
| Expected trước | GRPO | 42 | 43 | 14 | 4 | 3 |
| Expected trước | SFT | 43 | 40 | 11 | 2 | 2 |
| Expected trước | GRPO | 43 | 40 | 13 | 2 | 2 |
| Expected trước | SFT | 44 | 44 | 9 | 4 | 4 |
| Expected trước | GRPO | 44 | 47 | 9 | 5 | 5 |
| Input trước | SFT | 42 | 41 | 18 | 6 | 3 |
| Input trước | GRPO | 42 | 40 | 18 | 4 | 3 |
| Input trước | SFT | 43 | 42 | 22 | 5 | 4 |
| Input trước | GRPO | 43 | 44 | 23 | 5 | 4 |
| Input trước | SFT | 44 | 47 | 17 | 5 | 5 |
| Input trước | GRPO | 44 | 47 | 21 | 8 | 6 |

## Trung bình ± độ lệch chuẩn mẫu

| Thứ tự nhãn | Mốc | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---|---|---:|---:|---:|---:|
| Expected trước | SFT | 41.00 ± 2.65 | 11.33 ± 2.52 | 3.67 ± 1.53 | 3.33 ± 1.15 |
| Expected trước | GRPO | 43.33 ± 3.51 | 12.00 ± 2.65 | 3.67 ± 1.53 | 3.33 ± 1.53 |
| Input trước | SFT | 43.33 ± 3.21 | 19.00 ± 2.65 | 5.33 ± 0.58 | 4.00 ± 1.00 |
| Input trước | GRPO | **43.67 ± 3.51** | **20.67 ± 2.52** | **5.67 ± 2.08** | **4.33 ± 1.53** |

Giá trị trung bình cao nhất in đậm, kể cả khi hòa. Chỉ một seed train và ba seed sinh test; SD mô tả lấy mẫu, không phải khoảng tin cậy hay nhiều lượt train độc lập.

![So sánh thứ tự nhãn](comparison.png)

Hình 1. Số test/code trên Validation, giữ mẫu không hợp lệ trong mẫu số. Thanh sai số là SD của ba seed sinh test.

## Tách lợi ích SFT và RL

So cùng mốc giữa hai thứ tự nhãn để xem tác động biểu diễn target. So GRPO trừ SFT trong mỗi nhánh để xem tác động thêm của RL. Đây là GRPO cố định, chưa phải đánh giá curriculum đối kháng mới.

| Nhánh | GRPO − SFT: hợp lệ | Expected đúng | Test bắt lỗi | Code phát hiện |
|---|---:|---:|---:|---:|
| Expected trước | +2.33 | +0.67 | +0.00 | +0.00 |
| Input trước | +0.33 | +1.67 | +0.33 | +0.33 |

Đủ tiêu chí đặt trước để cân nhắc thử đối kháng tiếp: **True**. Tiêu chí ở mốc GRPO là expected đúng trung bình tăng ít nhất 3/60, các trung bình hợp lệ/test bắt lỗi/code phát hiện không giảm. Ngưỡng sàng lọc không phải kiểm định ý nghĩa thống kê; không tự thay checkpoint chính.

## Phân tích và chi phí

errors.csv giữ đủ phân loại lỗi và thứ tự trường model thực sự sinh. Cột input-only chỉ chẩn đoán các cơ hội bị mất do expected sai, không dùng oracle để sửa điểm model. per-case.csv giữ ID từng code phát hiện; per-problem.csv gom theo bài nguồn để xem mức tăng có chỉ tập trung vào một bài hay không. Ba mutant cùng đề không phải ba bài độc lập.

costs.csv ghi thời gian pha và đỉnh PyTorch cấp phát. SFT bao gồm probe sau học/lưu adapter; GRPO dùng Trainer train_runtime. Chưa tính đầy đủ thời gian nạp model, audit và điều phối, không gọi là tổng chi phí toàn quy trình.

## Giới hạn và nguồn

Năm bài Validation, miền hữu hạn và một seed huấn luyện. Nondeterminism CUDA chưa được loại trừ. Kết quả tốt nếu có vẫn cần phép so sánh đối kháng mới và đánh giá Test cuối sau khi đóng băng thiết kế.

Thứ tự nhãn là điều chỉnh triển khai của đồ án. Không coi nó là thành phần nguyên bản hoặc kết quả tái lập [ATGen](https://arxiv.org/html/2510.14635v1). Test phát triển và Test chính thức chưa được model đánh giá.

Kế hoạch, mã nguồn lúc chạy, adapter và raw output nằm tại thư mục evidence ghi trong metrics.json. Dựng lại bảng/hình: `python scripts/summarize_apps_order.py <folder>`. Không cần nạp model để tổng hợp.

## Diễn giải kết quả lượt này

Expected đúng tăng ở cả ba seed sinh test khi so cùng mốc. Ở mốc GRPO, test bắt lỗi tăng ở seed43/44 và hòa seed42; số code phát hiện cũng không tăng tại seed42. Tỷ lệ input hợp lệ không tăng ở mọi seed.

Theo bài nguồn, expected trung bình tăng ở cả năm bài. Phần tăng test bắt lỗi ở mốc GRPO tập trung tại bài làm tròn 2947, còn tăng code phát hiện nằm tại 2947 và 2868. Vì vậy chưa có bằng chứng về lợi ích đồng đều mọi bài hoặc mọi loại bug.

Tại nhánh input-first, GRPO tăng thêm trung bình 1,67 expected đúng, 0,33 test bắt lỗi và 0,33 code so với SFT. Phải giữ mức tăng này riêng với thay đổi biểu diễn target. Đủ điều kiện để thử đối kháng tiếp không có nghĩa đã chọn phương pháp cuối. [Kế hoạch có điều kiện](../../docs/APPS_INPUT_FIRST_ADAPTIVE_PLAN.md).
