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

## Curriculum đối kháng theo vòng

Đã bổ sung `apps_adversary.py` và `apps_adversarial_round.py`. Model sinh code là Qwen local gốc, không cập nhật trọng số. Model sinh test bắt đầu từ SFT và học adapter bằng GRPO.

Mỗi vòng probe policy trên Train; chỉ những code đã bị bắt lỗi mới được xét thay. Code sinh thêm phải sai ở một số input trong miền khai báo, đúng ở một số input khác, vượt qua các test đúng hiện tại và khác hành vi các mutant ban đầu. Code ngoài phần Python được hỗ trợ bị loại. Nhân chứng lỗi chỉ phục vụ audit.

Hai vòng, mỗi vòng 48 bước/nhánh. Nhánh đối chứng dùng code cố định, cũng khởi tạo lại optimizer hai lần. Nhánh đối kháng dùng curriculum đóng băng ở đầu từng vòng. Validation và ngân sách sinh test giữ nguyên; Test chưa được model đánh giá. Đây là đối chứng khớp ngân sách học, không khớp tổng chi phí vì nhánh đối kháng cần sinh và lọc code.

Xem [kết quả hai vòng](../results/apps-adversarial) và [phạm vi so với ATGen](ATGEN_ALIGNMENT.md). Mục kết quả cố định phía trên vẫn mô tả lượt chạy trước đó.

## Một bước GRPO diễn ra như thế nào?

Model nhận đề bài và một code có thể sai, rồi sinh bốn cặp input/expected. Bộ chấm chạy reference để xác nhận expected, sau đó chạy code cần kiểm tra để xem test có làm lộ lỗi không. Model không được xem reference hoặc nhân chứng lỗi trong prompt; chúng chỉ nằm ở phía chấm.

Ví dụ nhóm đầu của lượt đối chứng reward mới có bốn điểm `[0; 0,1; 0,3; 0,1]`. Điểm trung bình là 0,125. GRPO so mỗi điểm với trung bình nhóm rồi chia cho độ lệch chuẩn; mẫu 0,3 có tín hiệu tương đối tốt hơn, còn ba mẫu kia có tín hiệu thấp hơn. Đây là so sánh trong nhóm, không phải nhãn đáp án mà người dùng nhập sẵn.

Với cấu hình hiện tại, một nhóm có bốn completion và một bước cập nhật optimizer sau phần gradient accumulation. Lượt 96 bước sinh 384 test trong lúc học. “96 bước” không có nghĩa 96 bài khác nhau hoặc 96 epoch; có thể gặp lại cùng code Train. Script kiểm tra số bước và số rollout sau chạy.

Một nhóm có bốn reward bằng nhau không cung cấp tín hiệu phân biệt cho GRPO. Vì vậy số bước được gọi và số bước có gradient khác 0 có thể khác nhau. Ngược lại, loss hiển thị gần 0 không tự chứng minh model không học: ở cấu hình policy cập nhật một lần và advantage được chuẩn hóa, các số hạng có thể triệt tiêu ở giá trị loss dù gradient vẫn khác 0. Phải xem gradient, thay đổi tensor, kiểm tra adapter đã lưu và chất lượng trên Validation.

SFT khác ở chỗ được cung cấp đáp án mục tiêu đúng và học để khớp đáp án đó. GRPO lấy đầu ra do chính model sinh, chấm bằng môi trường và dùng reward để cập nhật. Bấm “Sinh 4 test” ở Demo chỉ thực hiện phần sinh/chấm, không có bước gradient và cập nhật optimizer.

## Thử tỷ trọng reward

Hai profile được tách riêng khỏi bộ chấm đánh giá: legacy-v1 dùng trọng số định dạng/expected/attack là 0,1/0,2/0,7; equal-v1 dùng 1/3 cho mỗi thành phần. Trần reward đều bằng 1, nhưng các mức giữa khác nhau. Mọi profile vẫn chỉ thưởng attack nếu expected đúng.

Phép so sánh reward giữ cùng checkpoint SFT64, dữ liệu và 96 bước/nhánh, không đồng thời làm mới code đối kháng. Xem [kế hoạch kiểm chứng](APPS_REWARD_EXPERIMENT.md). Đây là thử riêng một thành phần của phương pháp; kết quả phải tách khỏi so sánh curriculum adaptive.

## Thứ tự trường trong completion SFT

Định dạng sorted-v1 lịch sử sắp xếp khóa nên dạy expected rồi input. Cấu hình thử input-first-v1 chỉ đảo hai trường ngoài cùng, giữ nguyên giá trị và thứ tự khóa bên trong input. Tham số CLI: --target-order input-first-v1. Mặc định lịch sử vẫn giữ để các lượt cũ có thể đối chiếu; chưa tự thay cấu hình demo.

Tất cả 319 cặp nhãn và byte định dạng cũ đã được kiểm tra. [Phép so sánh mới](../results/apps-target-order/README.md) cho thấy expected trung bình tăng, nhưng phần lớn thay đổi thuộc SFT. Không gọi cả mức tăng đó là lợi ích thêm của GRPO hay adversary.
