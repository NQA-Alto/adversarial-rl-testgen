# Ba ví dụ để giải thích demo

Snapshot ngày 30/10/2026: input-first SFT64 + GRPO96, seed sinh42, prompt apps-call-v2. Đầu ra lấy từ `apps-adapter-validation-20261029-150214-438503`; khi đổi checkpoint phải chọn và kiểm tra lại ví dụ. Đây là ví dụ minh họa Validation, không phải mẫu đại diện dùng để tính điểm chung.

## 1. Làm tròn — model bắt được lỗi

Chọn `apps-train-2947/m2`, bài làm tròn đến bội gần nhất; nếu ở chính giữa hai bội thì giữ nguyên số ban đầu.

Test đầu tiên model sinh: `n = 1, m = 5, expected = 0`. Hai bội kề là0 và5; khoảng cách lần lượt1 và4 nên đáp án0. Code lỗi trả5. Input hợp lệ, expected đúng, code trả khác: tính là phát hiện lỗi.

Test thứ hai: `n = 8, m = 5, expected = 10`. Bội5 cách3, bội10 cách2; code lỗi lại trả8. Đây là một test bắt lỗi nữa, nhưng cùng một code nên số code phát hiện vẫn chỉ tăng1.

Lời nói gợi ý: “AI đề xuất input và đáp án kỳ vọng. Bộ chấm kiểm tra đáp án bằng reference, rồi so với code cần kiểm tra. Model không được xem reference khi sinh test.”

## 2. Chuyển vị hai hàng — test chuỗi

Chọn `apps-train-3014/m1`. Lấy các ký tự ở chỉ số0,2,4… thành hàng đầu; các ký tự1,3,5… thành hàng sau, rồi nối hai hàng.

Test thứ hai của model: `text = "ba", expected = "ba"`. Hàng đầu là`b`, hàng sau là`a`; đáp án`ba`. Code mutant lấy cả chuỗi cho hàng đầu, rồi cộng tiếp ký tự chỉ số1, nên trả`baa`.

Lời nói gợi ý: “Cùng kiến trúc input/expected nhưng bài toán có thể là số, chuỗi hoặc danh sách. Miền input được quy định riêng cho từng bài.”

## 3. Điểm đối xứng — model vẫn sai expected

Chọn `apps-train-2834/m1`. Điểm đối xứng P qua tâm Q là2Q−P.

Test thứ hai model sinh: `p = [2, 0], q = [0, 0], expected = [4, 0]`. Đáp án đúng phải là`[-2, 0]`; code mutant trả`[2, 0]`. Input đã làm lộ lỗi nhưng model tính expected sai, nên bộ chấm không công nhận test bắt lỗi.

Lời nói gợi ý: “Đây là giới hạn thực tế của hệ thống. Tạo được JSON và chọn được input nhạy với lỗi chưa đủ; model còn phải tính đúng đáp án.”

## Trình tự video khoảng hai phút

| Thời lượng dự kiến | Nội dung |
|---|---|
| 0:00–0:15 | Nêu mục tiêu sinh test bằng model local; chỉ vào đề, code và miền input |
| 0:15–0:45 | Ví dụ làm tròn: input, expected, đáp án reference và code lỗi |
| 0:45–1:10 | Ví dụ chuỗi; phân biệt test bắt lỗi với số code phát hiện |
| 1:10–1:35 | Ví dụ expected sai; nói rõ giới hạn |
| 1:35–2:00 | Mở bảng kết quả đủ tập và giải thích so sánh trước/sau, đối kháng/cố định |

Dùng model và kết quả đã lưu khi bảo vệ theo Project Guide tr.30. Không train lại để làm nội dung chính của video. Nếu chạy trực tiếp, đầu ra lấy mẫu có thể khác; dùng kết quả đã lưu để đối chiếu chứ không hứa đúng nguyên văn ví dụ.

Khi trình bày thành tích, dùng bảng đủ15 code ×4 test và cả ba seed sinh, không lấy riêng ba ví dụ này để tính accuracy. Phân biệt hiệu ứng SFT, GRPO và curriculum đối kháng. Lượt hai vòng hiện chưa chứng minh đối kháng hơn cố định.
