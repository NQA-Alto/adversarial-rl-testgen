# Dữ liệu

Nguồn: [codeparrot/apps](https://huggingface.co/datasets/codeparrot/apps), revision `21e74ddf8de1a21436da12e3e653065c5213e9d1`. Dùng bài Introductory dạng gọi hàm trong APPS Train.

| Tập | Bài nguồn | Code lỗi |
|---|---:|---:|
| Train | 58 | 145 |
| Validation | 5 | 15 |
| Test | 7 | 18 |

Bản dữ liệu: `apps-intro-expanded-v4-0b00cd05b256`. Train có1050 nhãn SFT. Validation và Test giữ nguyên so với bản v2; các bài và URL nguồn không giao giữa các tập.

Hai lời giải nguồn phải qua test gốc và đồng ý trên toàn bộ miền input đã khai báo. Entry point được đổi thành `solve`. Các lời giải và mutant được chạy trong bộ thông dịch AST giới hạn; có56.152 cặp code/input trong audit bản v4. Đây là kiểm tra dữ liệu, khác đánh giá model trên Test.

Code lỗi ban đầu tạo bằng thay một toán tử, phép so sánh, hằng số hoặc chỉ số trong AST. Loại code trùng hành vi, sai mọi input hoặc vượt cú pháp hỗ trợ. Code đối kháng bổ sung được lấy mẫu từ model gốc và lọc theo policy hiện tại; chỉ làm trên Train.

## Dữ liệu model nhìn thấy

Đề bài, code cần kiểm tra và miền input. Model trả JSON gồm `input` và `expected`. Reference và witness chỉ thuộc bộ chấm, không đưa vào prompt sinh test. Đầu ra model được giữ nguyên; mẫu JSON lỗi, sai miền và bị cắt vẫn nằm trong mẫu số.

## Giới hạn

Không nạp toàn bộ APPS vì bộ chạy hiện chỉ hỗ trợ hàm Python, cú pháp hạn chế và miền input hữu hạn. Tập Test chính thức của APPS không được sử dụng. Dữ liệu công khai có thể từng xuất hiện trong tiền huấn luyện của model; không xác định được mức độ đó.

Tệp dữ liệu và hash từng tập nằm tại `D:/DA/TTNT_De19_Assets/data/apps-introductory-expanded-v4`. Các lượt thí nghiệm nhỏ trước đó vẫn được giữ riêng.
