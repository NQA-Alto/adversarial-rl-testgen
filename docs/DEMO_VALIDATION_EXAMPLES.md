# Ba ví dụ để giải thích demo

Snapshot 30/10/2026: SFT256 input-first, seed sinh 42, prompt apps-call-v2. Đầu ra lấy từ `apps-adapter-validation-20261030-002837-023045`; đây là checkpoint đang tốt hơn lượt thêm GRPO96. Ví dụ chỉ minh họa, không dùng riêng chúng để tính thành tích chung.

## 1. Chuyển vị hai hàng

Chọn `apps-train-3014/m1`. Lấy ký tự ở chỉ số 0,2,4… thành hàng đầu, các ký tự 1,3,5… thành hàng sau rồi nối hai hàng.

Test thứ hai: `text = "ba", expected = "ba"`. Hàng đầu là `b`, hàng sau là `a`; đáp án `ba`. Code mutant lấy cả chuỗi cho hàng đầu, rồi cộng tiếp ký tự ở chỉ số 1, nên trả `baa`. Test phát hiện lỗi.

Test thứ ba dùng chuỗi rỗng và expected rỗng. Reference và mutant đều trả chuỗi rỗng: đây là test đúng nhưng chưa bắt lỗi.

Lời nói gợi ý: “Model local sinh cả input lẫn expected. Bộ chấm dùng reference để xác nhận expected rồi so với code cần kiểm tra. Reference không được đưa vào prompt sinh test.”

## 2. Lấy ký tự giữa

Chọn `apps-train-2968/m1`. Chuỗi dài lẻ lấy một ký tự giữa; dài chẵn lấy hai ký tự giữa.

Test thứ hai: `s = "bAA", expected = "A"`. Chuỗi có ba ký tự, ký tự giữa ở chỉ số 1 là `A`. Code lỗi bắt đầu lát cắt tại chỉ số 2 nhưng kết thúc cũng tại 2, nên trả chuỗi rỗng. Test bắt được lỗi.

Test thứ tư: `s = "bb", expected = "b"`. Đáp án phải là `bb`, nên model vẫn tính sai expected ở trường hợp này. Không lấy output `b` của code lỗi làm chuẩn.

Lời nói gợi ý: “Cùng một bài, model có thể tạo được test bắt lỗi và cũng có thể sai đáp án. Bảng kết quả tính đủ cả hai loại.”

## 3. Điểm đối xứng — expected sai

Chọn `apps-train-2834/m1`. Điểm đối xứng P qua tâm Q là 2Q−P.

Test thứ hai: `p = [2, 0], q = [0, 0], expected = [4, 0]`. Đáp án đúng phải là `[-2, 0]`; code mutant trả `[2, 0]`. Input đã làm lộ lỗi nhưng model tính expected sai, nên không được tính test bắt lỗi.

Lời nói gợi ý: “Tạo được JSON và chọn được input nhạy với lỗi chưa đủ; model còn phải tính đúng đáp án. Hệ thống giữ nguyên đầu ra sai để phân tích.”

## Video khoảng hai phút

| Thời lượng dự kiến | Nội dung |
|---|---|
| 0:00–0:15 | Mục tiêu sinh test bằng model local; đề, code và miền input |
| 0:15–0:45 | Ví dụ chuyển vị: input, expected, reference và output mutant |
| 0:45–1:10 | Ví dụ ký tự giữa; test đúng chưa chắc bắt lỗi |
| 1:10–1:35 | Ví dụ điểm đối xứng và giới hạn expected sai |
| 1:35–2:00 | Bảng đủ tập: lợi ích SFT riêng, kết quả GRPO và đối kháng riêng |

Theo Project Guide tr.30, dùng model/kết quả đã lưu, không train lại làm nội dung chính khi bảo vệ. Nếu sinh trực tiếp, đầu ra có thể khác; không hứa sẽ trùng nguyên văn các test này.

SFT256 phát hiện trung bình 7/15 code; thêm GRPO96 còn 6/15 trên ba seed sinh. Đây là một seed train, không phải ba lần huấn luyện. Đối chứng curriculum hai vòng dùng parent SFT64 riêng và chưa chứng minh lợi ích đối kháng. Không gộp các tác động thành một mức tăng nhờ RL.

Demo mở đầu bằng ví dụ chuyển vị để dễ theo dõi. Tất cả 15 code Validation, gồm các bài model làm kém, vẫn có trong danh sách; số liệu đánh giá không bị lọc theo ví dụ trình bày.
