# Demo

Chạy `MO_GIAO_DIEN.cmd` và mở [giao diện](http://127.0.0.1:7860/?__theme=light).

## Demo

Chọn một bài và phiên bản model. Bảng hiển thị bốn test đã lưu trên Validation với seed 42, đối chiếu Qwen gốc với phiên bản đã chọn. **Sinh 4 test mới** chạy suy luận trên GPU; kết quả có thể khác do lấy mẫu.

Mỗi dòng có input, expected model sinh và kết quả code. Nếu expected sai, bảng ghi thêm đáp án tham chiếu. Test đúng nhưng chưa làm lộ lỗi được phân biệt với test không hợp lệ. **JSON gốc** giữ đầu ra để kiểm tra.

Bài mặc định lấy một hoặc hai ký tự giữa: độ dài lẻ trả một ký tự, độ dài chẵn trả hai ký tự. Chẳng hạn `bAA` có kết quả `A`, còn `bb` có kết quả `bb`. Miền input của từng thí nghiệm hiển thị cạnh code.

## Kết quả và dữ liệu

Kết quả gồm Qwen gốc, SFT840, GRPO cố định và GRPO đối kháng. Chỉ số tổng hợp dùng toàn bộ code trong tập tương ứng. Test giữ lại được chấm sau khi chọn checkpoint bằng Validation.

Tab Dữ liệu cho biết nguồn, số bài, số code lỗi và miền hỗ trợ. Tất cả là tập con APPS Train, không phải APPS Test chính thức.

## Thực nghiệm

Có thể kiểm tra dữ liệu, chấm Qwen gốc, chạy1–96 bước GRPO hoặc chấm adapter mới. Nút GRPO dùng code Train cố định và khởi tạo lại từ SFT840 với optimizer mới. Lượt bấm tay không thay checkpoint của phép đánh giá cuối.

Nhật ký có thanh cuộn, nút về đầu/cuối và nơi lưu kết quả. Đường học GRPO nằm trong phần thu gọn; TensorBoard mở ở cổng 6006. Dùng `DONG_GIAO_DIEN.cmd` để dừng các dịch vụ.
