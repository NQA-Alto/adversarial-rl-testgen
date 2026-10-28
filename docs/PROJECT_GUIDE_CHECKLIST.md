# Đối chiếu Project Guide

Nguồn: CMP169-Project-Guide.pdf, REV.10-26, 38 trang. Rà soát ngày 28/10/2026. Các mục tổ chức công việc được áp dụng phù hợp với đồ án cá nhân.

| Mục trong guide | Tình trạng hiện tại | Việc còn cần làm |
|---|---|---|
| Tái hiện và hiểu phương pháp ở quy mô nhỏ (tr.11) | Có baseline, SFT, GRPO cố định và hai vòng curriculum sinh code local; có bảng đối chiếu ATGen | Mở rộng độ ổn định, ghi rõ giản lược; chưa gọi là tái hiện đầy đủ benchmark ATGen |
| Chia tập trước, chỉ học trên Train, giữ Test đến cuối (tr.16) | Train/Validation/Test tách theo bài; SFT và GRPO chỉ dùng Train; model chưa đánh giá Test | Chốt phương pháp và cấu hình trước lần đánh giá cuối trên Test |
| Nhật ký cả lần thất bại, lưu cấu hình và thời gian (tr.17–18) | Có log, seed, hash, adapter, gradient, thời gian và kết quả từng mốc | Khi viết báo cáo dùng số liệu từ artifact, không sửa đầu ra rồi tính lại thành đúng |
| Baseline, so với bài báo, phân tích lỗi (tr.22) | Có mốc gốc/SFT và so sánh GRPO cố định với GRPO đối kháng cùng ngân sách; có ATGEN_ALIGNMENT.md | Tiếp tục phân tích expected sai và kiểm tra độ ổn định; không so trực tiếp điểm với benchmark khác dữ liệu |
| EDA và pipeline (tr.18,21) | Có thống kê nguồn, tập con và kiểm chứng dữ liệu | Hoàn thiện hình EDA cho APPS và hình pipeline tổng thể |
| Git thường xuyên, commit nhỏ, README và cách tải (tr.14,35–36) | Các thay đổi APPS được chia theo dữ liệu, huấn luyện/giao diện, kết quả và tài liệu | Duy trì commit nhỏ theo các lần cập nhật tiếp theo |
| Chạy lại trên máy sạch, tránh đường dẫn cứng (tr.35–36) | Chạy được trên máy hiện tại; còn phụ thuộc D: và môi trường đã cài | Tạm hoãn theo ưu tiên hiện tại; trước bản nộp cần tách cấu hình và kiểm tra trên môi trường mới |
| Demo với model đã lưu, không train lại tại buổi demo (tr.30) | Có checkpoint local và giao diện suy luận | Chuẩn bị 2–3 ví dụ gồm một ví dụ sai, video dự phòng 2 phút; nút train dùng cho thí nghiệm, không làm nội dung chính của demo bảo vệ |
| Báo cáo, nguồn, hình bảng và trích dẫn (tr.19,23–25,36) | Đang lưu dữ liệu phục vụ biên soạn; chưa làm báo cáo cuối | Theo mẫu đồ án cơ sở, năm chương, nguồn IEEE đầy đủ kể cả công cụ hỗ trợ; bảng phải là bảng, trục và đơn vị rõ |

Kết quả thấp hoặc không tăng sau RL không tự động làm sai quy trình: guide yêu cầu lưu cả lần thất bại và phân tích nguyên nhân. Tuy nhiên, không dùng việc có gradient hoặc adapter thay đổi để tuyên bố khả năng tìm lỗi đã cải thiện.

Trước khi chốt bản nộp, cần kiểm tra khả năng chạy lại và phạm vi phương pháp đã triển khai. Các thay đổi dữ liệu, prompt, model, reward hay cách chấm cần có phiên bản và mốc so sánh tương ứng.
