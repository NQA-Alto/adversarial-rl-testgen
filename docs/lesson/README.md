# Nguồn bài giảng đồ án

File dành để đọc và chia sẻ: `../BAI_GIANG_DO_AN_TTNT.html`. Đây là một file HTML tự chứa: CSS, JavaScript, sơ đồ SVG và ảnh đều được nhúng sẵn. Mở bằng trình duyệt; không cần chạy Python, model hoặc Gradio để đọc tài liệu.

Năm file `01-…` đến `05-…` chứa nội dung biên soạn. `tensorboard.png` là ảnh giao diện thật chụp ngày 29/10/2026. Mục24 cập nhật ngày30/10 giải thích expected sai, hình điểm đối xứng và kết quả input-first. Các bảng/hình kết quả lấy từ thư mục results; chỉ đọc Train/Validation đã có, không mở nội dung Test.

Để dựng lại trên bản cài hiện tại, chạy Python với `scripts/build_project_lesson.py`. Script kiểm tra dataset và số liệu của ví dụ trước khi dựng, rồi ghi build-manifest.json. Nếu kết quả hoặc cấu hình thay đổi, phải rà lại cả phần diễn giải bằng lời, không chỉ cập nhật bảng.

Kiểm tra đã thực hiện: hiển thị qua trang xem trước localhost, ảnh được nạp từ dữ liệu nhúng, mục lục, phép tính hai mô phỏng và hai profile reward. review.json phân biệt kiểm tra cấu trúc tự chứa với việc mở file trực tiếp trong trình duyệt; khung trình duyệt nhúng chỉ cho HTTP/HTTPS nên không có kiểm tra trực tiếp giao thức file.

Tài liệu phục vụ tự học, chưa phải báo cáo nộp cuối. Mốc số liệu tham chiếu được ghi trong chính bài giảng.
