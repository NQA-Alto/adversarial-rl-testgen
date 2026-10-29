# Độ khó của việc chọn input trong miền hữu hạn

Đây là chẩn đoán bằng chương trình tham chiếu trên Validation, không gọi hay huấn luyện model. Vét cạn tất cả input đã khai báo cho từng mutant để đếm bao nhiêu input có thể làm lộ lỗi.

Với p là tỷ lệ input làm lộ lỗi, bốn lần chọn đều ngẫu nhiên độc lập, có hoàn lại, sẽ có xác suất phát hiện code là 1 − (1 − p)^4. Phép tính giả sử đáp án expected luôn do oracle hoàn hảo cung cấp.

| Code | Input trong miền | Input làm lộ lỗi | Tỷ lệ input làm lộ lỗi | Xác suất phát hiện với 4 input + oracle |
|---|---:|---:|---:|---:|
| apps-train-2834/m1 | 81 | 54 | 66.67% | 98.77% |
| apps-train-2834/m2 | 81 | 54 | 66.67% | 98.77% |
| apps-train-2834/m3 | 81 | 54 | 66.67% | 98.77% |
| apps-train-2868/m1 | 341 | 1 | 0.29% | 1.17% |
| apps-train-2868/m2 | 341 | 22 | 6.45% | 23.41% |
| apps-train-2868/m3 | 341 | 232 | 68.04% | 98.96% |
| apps-train-2947/m1 | 72 | 7 | 9.72% | 33.58% |
| apps-train-2947/m2 | 72 | 51 | 70.83% | 99.28% |
| apps-train-2947/m3 | 72 | 49 | 68.06% | 98.96% |
| apps-train-2968/m1 | 363 | 360 | 99.17% | 100.00% |
| apps-train-2968/m2 | 363 | 270 | 74.38% | 99.57% |
| apps-train-2968/m3 | 363 | 351 | 96.69% | 100.00% |
| apps-train-3014/m1 | 364 | 360 | 98.90% | 100.00% |
| apps-train-3014/m2 | 364 | 324 | 89.01% | 99.99% |
| apps-train-3014/m3 | 364 | 351 | 96.43% | 100.00% |

Tổng số code kỳ vọng được phát hiện theo giả định trên: **12.51/15**. Đây là kỳ vọng tính bằng công thức, không phải số code của một lượt sinh test thực tế.

## Cách dùng chẩn đoán

Nếu nhiều input trong miền đã làm lộ lỗi mà model vẫn có ít test dùng được, cần xem expected sai và input không hợp lệ trước khi chỉ tăng mức đối kháng. Kết quả này không chứng minh riêng nguyên nhân nào, vì model có thể lấy mẫu input khác phân bố đều và bốn completion có thể trùng nhau.

Oracle biết lời giải đúng, còn model trong đồ án phải tự dự đoán expected từ đề. Vì vậy không dùng con số này để tuyên bố thuật toán ngẫu nhiên tốt hơn model trong cùng điều kiện, không thay expected do model sinh bằng đáp án oracle để tăng điểm. Đây cũng không phải trần tối ưu của mọi phương pháp chọn input.

Miền input nhỏ và mutant AST là giới hạn của dataset đồ án; kết quả không suy rộng sang toàn bộ APPS. Tập Test phát triển chưa được model đánh giá. Chạy lại bằng scripts/analyze_apps_input_difficulty.py.
