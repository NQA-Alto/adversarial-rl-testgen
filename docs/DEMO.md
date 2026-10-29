# Demo

Chạy `MO_GIAO_DIEN.cmd` rồi mở [Gradio](http://127.0.0.1:7860/?__theme=light). Tab Demo hiển thị Qwen gốc, SFT256 và SFT256 + GRPO96. Đây là các đầu ra đã lưu với seed 42; có thể mở phần chạy trực tiếp để sinh bốn test mới.

## Chuyển vị hai hàng

Case `apps-train-3014/m1` lấy các ký tự chỉ số chẵn rồi nối với các ký tự chỉ số lẻ. Code lỗi dùng `text[::1]` thay cho `text[::2]`.

Với `text = "ba"`, SFT256 sinh `expected = "ba"`; code lỗi trả `"baa"`. Test này phát hiện lỗi. Với chuỗi rỗng, reference và mutant cùng trả chuỗi rỗng, nên test đúng nhưng chưa phân biệt được hai chương trình.

## Lấy ký tự giữa

Case `apps-train-2968/m1` trả một ký tự giữa nếu độ dài lẻ, hai ký tự giữa nếu độ dài chẵn.

Với `s = "bAA"`, SFT256 sinh `expected = "A"`; code lỗi trả chuỗi rỗng. Cũng trong lượt đó, model sinh `s = "bb", expected = "b"`, trong khi đáp án đúng là `"bb"`. Đây là trường hợp expected sai.

## Điểm đối xứng

Case `apps-train-2834/m1` tính điểm đối xứng P qua Q theo công thức 2Q−P. Model sinh `p = [2,0], q = [0,0], expected = [4,0]`. Reference trả `[-2,0]`, còn mutant trả `[2,0]`. Input có thể làm lộ lỗi nhưng expected sai, nên test không được tính thành công.

Các ví dụ trên thuộc checkpoint SFT256, kết quả Validation tại `apps-adapter-validation-20261030-002837-023045`. Khi sinh trực tiếp, đầu ra có thể khác do lấy mẫu. Số liệu đánh giá trong tab Kết quả vẫn tính đủ 15 code, không chỉ ba ví dụ này.
