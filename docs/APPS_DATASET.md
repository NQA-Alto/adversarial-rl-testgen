# Dữ liệu

Nguồn là [codeparrot/apps](https://huggingface.co/datasets/codeparrot/apps), revision `21e74ddf8de1a21436da12e3e653065c5213e9d1`. Đồ án dùng các bài Introductory dạng gọi hàm trong APPS Train.

## Chọn bài và tạo code lỗi

Danh sách ban đầu gồm 35 bài, với miền input khai báo trong `scripts/apps_profiles.py`. Miền này gồm số nguyên, chuỗi hoặc danh sách ngắn; một số bài có thêm ràng buộc giữa các tham số.

Với mỗi bài, entry point được đổi tên thành `solve`. Hai lời giải nguồn phải chạy đúng các test gốc và cho cùng kết quả trên toàn bộ miền thực nghiệm. Code được chạy bằng bộ thông dịch AST giới hạn trong `restricted_python.py`.

Code lỗi ban đầu được tạo bằng một thay đổi AST, chẳng hạn đổi toán tử, phép so sánh, hằng số hoặc chỉ số. Các bản trùng hành vi, sai với mọi input hoặc dùng cú pháp ngoài phạm vi hỗ trợ bị loại. Mỗi mutant còn lại có ít nhất một input làm lộ lỗi và một input chưa làm lộ lỗi.

Sau lọc, còn 29 bài và 74 mutant. Có 26.833 cặp code/input được kiểm tra. Đây là các mutant do đồ án tạo từ lời giải nguồn, không phải code sai có sẵn trong APPS.

## Chia tập

| Tập | Bài nguồn | Mutant |
|---|---:|---:|
| Train | 17 | 41 |
| Validation | 5 | 15 |
| Test | 7 | 18 |

Mutant của cùng một bài luôn thuộc cùng tập. Các URL nguồn cũng được tách giữa các tập. Dữ liệu và hash từng tập nằm trong `manifest.json` của bản dữ liệu `apps-intro-call-v2-eb69b8869857`.

Test phát triển được chia từ APPS Train, khác APPS Test chính thức. Cả hai chưa được dùng để đánh giá model.

## Code bổ sung cho curriculum đối kháng

Trong các vòng đối kháng, Qwen gốc sinh lời giải cho bài Train. Bộ lọc giữ những lời giải còn sai trong miền thực nghiệm nhưng vượt qua test đúng do policy hiện tại tạo ra. Code này được dùng để thay một phần mutant ở vòng học tiếp theo; dữ liệu Validation và Test giữ nguyên.

## Đầu vào và đầu ra

Model nhận đề bài, code cần kiểm tra và hợp đồng input. Đầu ra là một đối tượng JSON có hai khóa `input` và `expected`.

```json
{"input":{"text":"ba"},"expected":"ba"}
```

Expected được so chính xác với kết quả reference, kể cả kiểu, thứ tự phần tử và khoảng trắng. Reference và nhân chứng lỗi nằm ở bộ chấm, không được đưa vào prompt sinh test. Bộ chấm giữ nguyên đầu ra model; JSON hỏng, input ngoài miền hoặc đầu ra bị cắt đều được tính là thất bại.

## Giới hạn dữ liệu

Đây là tập con nhỏ, được chọn theo khả năng chạy và kiểm tra trong miền hữu hạn. Hai lời giải cùng cho một kết quả chưa bảo đảm đúng ở mọi input ngoài miền. Các bài nguồn là dữ liệu công khai, nên không xác định được mức độ model nền từng gặp chúng trong tiền huấn luyện.

Các tệp dữ liệu nằm tại `D:/DA/TTNT_De19_Assets/data/apps-introductory-v2`; model và adapter được lưu ngoài repository.
