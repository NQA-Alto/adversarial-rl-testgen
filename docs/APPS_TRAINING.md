# Huấn luyện trên APPS

Đề tài trong danh sách giảng viên: **Adversarial Reinforcement Learning for Test Case Generation**, số 19. Trang demo dùng tên này; trạng thái thực nghiệm được mô tả riêng để không đồng nhất bản thử với phương pháp đối kháng hoàn chỉnh.

## Vì sao test ban đầu bị loại?

Model phải trả một đối tượng JSON có `input` và `expected`. Input còn phải đúng tên tham số, kiểu và miền con được ghi trong đề thí nghiệm. Ví dụ tọa độ chỉ cho phép các phần tử `[-1, 0, 2]`; số 1 hợp lệ trong nhiều bài hình học nhưng nằm ngoài hợp đồng thực nghiệm này. Miền nhỏ giúp đối chiếu tất cả input bằng hai lời giải nguồn, đồng thời làm bài toán tuân thủ yêu cầu khó hơn.

Prompt v2 đưa miền và định dạng xuống gần yêu cầu trả lời, nhắc rõ chúng thay thế miền rộng hơn trong ví dụ gốc, và thêm một ví dụ JSON của bài cộng hai số không liên quan. Vì vậy đây là prompting có ví dụ định dạng, không gọi là zero-shot tuyệt đối.

Trước khi train, chương trình chấm lại toàn bộ 60 đầu ra baseline, thử 15 test biết trước bắt được lỗi và 15 đối chứng có expected sai. Kết quả được lưu trong `diagnosis.json`. Những phép thử này có thể phát hiện một số lỗi bộ chấm; không phải chứng minh toàn bộ phần mềm không còn lỗi.

## Ba mốc cần phân biệt

1. **Qwen gốc với prompt v2:** đo ảnh hưởng của prompt trước khi cập nhật tham số.
2. **SFT:** tạo cặp input/expected đúng từ các bài APPS Train. Mỗi code chọn tối đa bốn test bắt lỗi và bốn test chưa bắt lỗi, tất cả qua bộ chấm. Pool có 319 mẫu; ngân sách cố định 64 bước, bốn mẫu/bước. Giai đoạn này dùng nhãn tham chiếu nên là học có giám sát, không phải RL.
3. **GRPO:** bắt đầu từ đúng adapter SFT đã chấm, optimizer mới, sinh bốn test/nhóm và học từ reward. Lượt chính đặt 96 bước; so sánh adapter cuối theo kế hoạch đã lưu, không chọn checkpoint theo điểm Test.

Các mốc cùng dùng dữ liệu APPS v2, prompt `apps-call-v2`, seed sinh 42, bốn test/code Validation và tối đa 96 token/completion. Lợi ích của SFT được đo bằng mốc SFT so với Qwen gốc; lợi ích thêm của GRPO được đo bằng mốc cuối so với SFT. Không gộp tác động đổi prompt hoặc SFT thành cải thiện do RL.

Chỉ LoRA được cập nhật, model gốc giữ cố định. SFT và GRPO đều lưu gradient, thay đổi tham số và hash adapter. Các test sai định dạng, ngoài miền và bị cắt vẫn ở trong mẫu số khi đánh giá.

## Điều khiển trên giao diện

Tác vụ **Huấn luyện SFT + GRPO trên APPS** chạy từ đầu toàn bộ quy trình, gồm kiểm chứng dữ liệu và chấm từng mốc. Sau khi một lượt đầy đủ hoàn tất, tác vụ **Huấn luyện GRPO trên Train** bắt đầu từ adapter SFT của lượt đó; nó không tự cộng thêm bước vào adapter GRPO cuối. Mỗi lượt có optimizer và thư mục evidence mới.

Tab Kết quả đọc các mốc cùng lượt đã hoàn tất. Trong lúc lượt mới đang chạy, kết quả đang công bố trước đó vẫn được giữ. Nút Sinh 4 test ở Demo chỉ suy luận/chấm một code; không huấn luyện.

Trong log, tắt Tự cập nhật log để đọc, hoặc dùng Đầu log/Cuối log. Khung giao diện giữ phần cuối log để hiển thị nhanh; `console.txt` của tác vụ lưu đầy đủ trên D:.

## Giới hạn

Đây là tập con APPS dạng gọi hàm với code đột biến, không phải benchmark đầy đủ. Validation gồm năm bài nguồn và 15 code lỗi; một seed chưa đủ kết luận ổn định. Test phát triển và Test chính thức chưa được model đánh giá. Vòng sinh curriculum bằng adversary trên APPS chưa nằm trong lượt SFT + GRPO cố định này.
