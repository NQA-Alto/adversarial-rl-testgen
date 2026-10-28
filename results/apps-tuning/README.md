# APPS: kiểm tra suy luận và hợp đồng đầu ra

Đây là các thử nghiệm suy luận với adapter SFT đóng băng, không phải các lượt RL mới. Checkpoint và cấu hình demo hiện tại giữ nguyên. Test chưa được model đánh giá.

## Thinking có thẻ đáp án

Thử cùng prompt, checkpoint và trần 768 token trên tám bài Train định trước; chỉ đổi native enable_thinking. Parser yêu cầu định dạng thẻ đã khai báo, không sửa JSON hay đáp án của model.

Đã lưu 20/32 mẫu thinking, trong đó 0 mẫu kết thúc. Ngay cả khi toàn bộ mẫu còn lại kết thúc, tỷ lệ tối đa chỉ còn 37.5%, dưới ngưỡng 80% đặt trước. Lượt được dừng sớm vì không còn khả năng đạt tiêu chí.

Việc dừng vì vô vọng được bổ sung sau kế hoạch ban đầu và đã ghi trong result.json; không mô tả đây là một nhánh benchmark hoàn chỉnh. Các mẫu đang sinh dở lúc dừng không nằm trong bản ghi. Trực tiếp dùng định dạng thẻ mới cũng gặp lỗi tuân thủ định dạng, vì vậy kết quả này chưa cô lập hoàn toàn chất lượng suy luận. Chỉ kết luận cấu hình thử hiện tại chưa đủ điều kiện dùng tiếp.

Model card local của Qwen3.5-2B có cảnh báo thinking loops; không suy rộng thất bại dưới trần token này thành việc model không có khả năng suy luận.

## Hai lượt thử prompt JSON

Cùng SFT, cùng toàn bộ 41 code Train, bốn test/code, seed 42, tối đa 96 token. Bộ chấm và trọng số không đổi. Các kiểu expected là annotation giao diện cho bài APPS đã biết, được kiểm tra bằng reference trên miền hữu hạn; không đưa reference hay đáp án cụ thể vào prompt. Đây là thông tin giao diện bổ sung, không phải thành phần nguyên bản của ATGen.

| Prompt | Hợp lệ /164 | Expected đúng /164 | Test bắt lỗi /164 | Qua tiêu chí Train | Chấm Validation |
|---|---:|---:|---:|---|---|
| v2, đối chứng SFT | 111 | 33 | 15 | Mốc đối chứng | Đã có từ lượt trước |
| apps-call-v3 | 2 | 1 | 1 | Không | Không |
| apps-call-v4 | 109 | 34 | 13 | Không | Không |

v3 bỏ ví dụ JSON và viết lại prompt; nhiều mẫu chỉ trả trường expected, thiếu input. v4 giữ nguyên toàn bộ prompt v2 và chỉ thêm kiểu expected vào system instruction để kiểm tra một thay đổi hẹp hơn.

Tiêu chí Train được ghi trước từng lượt: expected đúng tăng ít nhất 5 điểm phần trăm, số test hợp lệ và test bắt lỗi không giảm. Chỉ khi qua Train mới chấm Validation một lần. Có thể xem kết quả Train chưa qua tiêu chí để điều chỉnh thiết kế; không dùng Test cho việc này. Không chọn lại checkpoint hay seed để làm đẹp bảng.

## Liên hệ với phân tích lỗi

[Phân tích hiện tại](../apps-error-analysis/README.md) cho thấy expected sai làm mất nhiều cơ hội bắt lỗi. Cách tính input-only attack chỉ dùng oracle để chẩn đoán; không biến expected do reference tạo thành thành tích của model. Ý tưởng phân tích này tham khảo ATGen §5.3.

Nếu một prompt cải thiện, phải ghi đây là tác động prompt/giao diện. Chỉ sau một thí nghiệm cập nhật trọng số riêng có đối chứng mới được bàn về phần cải thiện do RL. Một seed và tập bài nhỏ vẫn là giới hạn.

Log thô, bản chụp mã nguồn và kế hoạch trước chạy được lưu trong các thư mục evidence ghi tại experiments.json; không đưa model weights hay raw dataset lên Git.
