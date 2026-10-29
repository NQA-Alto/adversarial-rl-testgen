# Đối chiếu Project Guide

Nguồn: CMP169-Project-Guide.pdf, REV.10-26, 38 trang. Rà soát ngày 28/10/2026. Các mục tổ chức công việc được áp dụng phù hợp với đồ án cá nhân.

| Mục trong guide | Tình trạng hiện tại | Việc còn cần làm |
|---|---|---|
| Tái hiện và hiểu phương pháp ở quy mô nhỏ (tr.11) | Có baseline, SFT, GRPO cố định và hai vòng curriculum sinh code local; có bảng đối chiếu ATGen | Mở rộng độ ổn định, ghi rõ giản lược; chưa gọi là tái hiện đầy đủ benchmark ATGen |
| Chia tập trước, chỉ học trên Train, giữ Test đến cuối (tr.16) | Train/Validation/Test tách theo bài; SFT và GRPO chỉ dùng Train; model chưa đánh giá Test | Chốt phương pháp và cấu hình trước lần đánh giá cuối trên Test |
| Nhật ký cả lần thất bại, lưu cấu hình và thời gian (tr.17–18) | Có log, seed, hash, adapter, gradient, thời gian và kết quả từng mốc | Khi viết báo cáo dùng số liệu từ artifact, không sửa đầu ra rồi tính lại thành đúng |
| Baseline, so với bài báo, phân tích lỗi (tr.22) | Có mốc gốc/SFT và so sánh GRPO cố định với GRPO đối kháng cùng ngân sách; có ATGEN_ALIGNMENT.md | Đã thêm phân tích expected sai và kiểm tra hai nhánh phát hiện cùng code; còn kiểm tra độ ổn định APPS, không so trực tiếp điểm với benchmark khác dữ liệu |
| EDA và pipeline (tr.18,21) | Có thống kê nguồn, hình EDA Train, pipeline và biểu đồ phân loại thất bại | Bổ sung chú thích, số hình và dẫn chiếu khi biên soạn báo cáo cuối |
| Git thường xuyên, commit nhỏ, README và cách tải (tr.14,35–36) | Các thay đổi APPS được chia theo dữ liệu, huấn luyện/giao diện, kết quả và tài liệu | Duy trì commit nhỏ theo các lần cập nhật tiếp theo |
| Chạy lại trên máy sạch, tránh đường dẫn cứng (tr.35–36) | Chạy được trên máy hiện tại; còn phụ thuộc D: và môi trường đã cài | Tạm hoãn theo ưu tiên hiện tại; trước bản nộp cần tách cấu hình và kiểm tra trên môi trường mới |
| Demo với model đã lưu, không train lại tại buổi demo (tr.30) | Có checkpoint local và giao diện suy luận | Chuẩn bị 2–3 ví dụ gồm một ví dụ sai, video dự phòng 2 phút; nút train dùng cho thí nghiệm, không làm nội dung chính của demo bảo vệ |
| Báo cáo, nguồn, hình bảng và trích dẫn (tr.19,23–25,36) | Đang lưu dữ liệu phục vụ biên soạn; chưa làm báo cáo cuối | Theo mẫu đồ án cơ sở, năm chương, nguồn IEEE đầy đủ kể cả công cụ hỗ trợ; bảng phải là bảng, trục và đơn vị rõ |

Kết quả thấp hoặc không tăng sau RL không tự động làm sai quy trình: guide yêu cầu lưu cả lần thất bại và phân tích nguyên nhân. Tuy nhiên, không dùng việc có gradient hoặc adapter thay đổi để tuyên bố khả năng tìm lỗi đã cải thiện.

Trước khi chốt bản nộp, cần kiểm tra khả năng chạy lại và phạm vi phương pháp đã triển khai. Các thay đổi dữ liệu, prompt, model, reward hay cách chấm cần có phiên bản và mốc so sánh tương ứng.

## Kiểm chứng reward và biến động lấy mẫu

Đã bổ sung ablation tỷ trọng reward GRPO theo hướng tr.22: hai nhánh cùng parent, dữ liệu, prompt và ngân sách; kế hoạch ghi trước đánh giá. Bộ chấm/metric chính giữ nguyên, reward học có phiên bản riêng. Từng rollout và adapter được kiểm tra lại; trường hợp không đạt tiêu chí tiếp tục được lưu.

Kiểm tra thêm ba seed sinh test được lên kế hoạch trước Validation mới và giữ toàn bộ kết quả. Đây không thay thế nhiều seed huấn luyện. Test vẫn chưa dùng; đóng gói, video và báo cáo cuối tiếp tục theo ưu tiên trước đó của người dùng.

## Kiểm tra lấy mẫu và thống kê chi phí ngày 29/10/2026

Đã giữ đủ 12 kết quả của bốn checkpoint trên ba seed sinh test; chấm lại mẫu thô, đối chiếu prompt/seed/cấu hình và giữ Test. Đã thống kê riêng thời gian học, tạo dữ liệu đối kháng và chấm Validation. Các số thời gian chỉ bao quát pha đã ghi, chưa phải thời gian toàn quy trình.

Phần nhiều seed huấn luyện vẫn là giới hạn, không đánh dấu hoàn tất. Đã lập [ước lượng tiến độ toàn đề tài](PROJECT_STATUS.md) khoảng 70%, với các hạng mục đánh giá cuối, demo, cài lại và báo cáo/slide còn riêng. Đây không phải tỷ lệ điểm chấm.

## Rà biểu diễn nhãn ngày 29/10/2026

Giữ đủ 12 lượt Validation và phân tích theo năm bài nguồn, tách hiệu ứng thứ tự nhãn SFT khỏi RL. Kế hoạch có tiêu chí trước kết quả; target mới giữ nguyên ngữ nghĩa, parser/metric không đổi. Lượt LoRA dừng sớm có ghi nguyên nhân và bằng chứng thiếu; không trình bày như so sánh hoàn chỉnh. Test vẫn giữ lại.

## Cập nhật expected và RL · 30/10

Đã chấm đủ sáu lượt SFT64/256 từ cùng quỹ đạo và sáu lượt trước/sau GRPO từ SFT256; raw output được chấm lại, prompt/seed/generation khớp. SFT tăng expected và số code phát hiện, nhưng thêm GRPO làm giảm các chỉ số chính. Đã giữ checkpoint SFT tốt hơn, lưu cả kết quả RL không tăng và biên bản lượt dừng do replay. Phù hợp yêu cầu tr.22 về ablation và phân tích lỗi, không dùng bằng chứng gradient thay thành tích chất lượng.

Có ba ví dụ demo Validation kèm cách tính tay, gồm trường hợp expected sai; chưa quay video dự phòng hoặc đánh giá Test cuối. Báo cáo chính thức và kiểm chứng cài lại vẫn để sau theo ưu tiên đã thống nhất.
