# Chẩn đoán thứ tự trường trong nhãn SFT

319 nhãn SFT lịch sử đều bắt đầu bằng expected. Nguyên nhân là hàm canonical sắp xếp tên khóa khi tuần tự hóa. JSON vẫn hợp lệ và bộ chấm không phụ thuộc thứ tự khóa.

Ví dụ nhãn gốc:

```json
{"expected":true,"input":{"distance_to_pump":4,"fuel_left":4,"mpg":1}}
```

## Đầu ra đã lưu trên ba seed sinh test

Mỗi phiên bản có 180 completion trên Validation. Bảng phân loại theo khóa đầu tiên trong JSON đọc được; parse_error là không đọc được JSON.

| Phiên bản | Trường đầu | Số mẫu | Input/JSON hợp lệ | Expected đúng | Test bắt lỗi |
|---|---|---:|---:|---:|---:|
| base | input | 155 | 106 | 44 | 11 |
| base | parse_error | 25 | 0 | 0 | 0 |
| sft | expected | 177 | 129 | 35 | 11 |
| sft | parse_error | 3 | 0 | 0 | 0 |
| control | expected | 176 | 128 | 32 | 9 |
| control | parse_error | 4 | 0 | 0 | 0 |
| adaptive | expected | 176 | 127 | 33 | 10 |
| adaptive | parse_error | 4 | 0 | 0 | 0 |

## Ý nghĩa và kiểm chứng cần làm

Model tự hồi quy sinh token từ trái sang phải. Nếu nhãn đặt expected trước input, nó được dạy trả đáp án trước khi hoàn tất input cụ thể trong văn bản. Dữ liệu cho thấy thứ tự đầu ra sau SFT thay đổi rõ rệt, nhưng chưa chứng minh đổi thứ tự sẽ cải thiện accuracy: còn nhiều yếu tố học khác cùng thay đổi.

Phép thử tiếp theo cần giữ nguyên profile LoRA, target input/expected, thứ tự bài, learning rate và ngân sách; chỉ đổi thứ tự hai khóa ngoài cùng sang input rồi expected. So riêng SFT và phần GRPO thêm vào; giữ parser và metric cũ. Không sắp xếp lại đầu ra model khi chấm hoặc dùng oracle sửa expected.

Đây là vấn đề biểu diễn nhãn được phát hiện khi rà mã, chưa phải bằng chứng toàn bộ kết quả cũ bị chấm sai. Test chưa được model đánh giá. Kết quả ba seed có tương quan và không phải nhiều seed train.
