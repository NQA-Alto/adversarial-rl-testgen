# Kiểm tra expected sai

Expected là đáp án đúng của input theo đề bài. Một test chỉ được tính bắt lỗi khi input hợp lệ, expected đúng và code cần kiểm tra cho kết quả khác expected hoặc phát sinh lỗi. Có input tốt nhưng expected sai vẫn là một test sai.

## Kết quả đã kiểm tra

Sau phép thử thứ tự nhãn, input-first SFT64 + GRPO96 có 62/180 expected đúng trên ba seed sinh Validation. Trong 69 đầu ra hợp lệ nhưng expected sai, 36 input thực sự phân biệt reference và mutant. Chỉ 6 expected sai trùng kết quả mutant; không đủ để nói nguyên nhân chính là model sao chép code lỗi. Không có sai khác kiểu Python ở mức ngoài cùng trong 69 trường hợp này.

Phân loại và ví dụ tính tay: [phân tích expected](../results/apps-expected-diagnosis/README.md). Reference chỉ được dùng để chấm và tạo nhãn Train, không dùng để sửa expected khi tính thành tích model.

Lượt đối kháng input-first đã hoàn tất hai vòng, mỗi vòng 48 bước cho cả hai nhánh. Trên ba seed sinh, cố định và đối kháng đều phát hiện trung bình 4/15 code; expected trung bình lần lượt 20,33 và 19/60. Chưa có bằng chứng nhánh đối kháng tốt hơn. [Kết quả đủ từng seed và chi phí](../results/apps-input-first-review/README.md).

## Phép thử có giới hạn ngày 30/10/2026

SFT64 chỉ xử lý 256 lượt ví dụ trên 319 nhãn Train. Thử SFT256 từ cùng model gốc, cùng LoRA, thứ tự input-first, seed, learning rate và pool nhãn. Đây là kiểm tra thời lượng học có giám sát; không gọi phần này là RL.

Kế hoạch được ghi trước khi ứng viên mới học/chấm. Giữ checkpoint 64 để đối chiếu replay với adapter SFT64 đã có, không chọn checkpoint giữa chừng theo Validation.

Sau SFT256, chỉ probe Train. Điều kiện đi tiếp: expected tăng ít nhất 8/164; số hợp lệ, test bắt lỗi và code phát hiện không giảm; replay64 relative L2 không quá 0,001. Nếu trượt thì dừng, không chấm Validation mới và không tự tăng ngân sách tiếp.

Nếu qua, học GRPO96 từ SFT256 với reward legacy-v1. Chỉ khi hoàn tất RL mới chấm hai checkpoint SFT256 và SFT256+GRPO96 trên ba seed sinh 42/43/44. Đối chứng là input-first SFT64 và SFT64+GRPO96 đã kiểm chứng, không phải nhánh GRPO hai lần khởi tạo optimizer.

Điều kiện Validation: tổng expected của GRPO ứng viên trên ba seed tăng ít nhất 9/180 so với GRPO đối chứng; tổng hợp lệ, test bắt lỗi và code phát hiện không giảm. Vẫn công bố mọi seed và cả kết quả SFT. Không thay model chính chỉ vì loss Train giảm.

## Giới hạn

Chỉ có một seed huấn luyện và năm bài nguồn Validation. Đây là thử nghiệm bổ sung sau khi đã nhìn nhiều kết quả Validation; nguy cơ chọn quá khớp Validation tăng theo số lần thử. Giới hạn một lần thử ngân sách này và giữ Test đến sau khi chốt lựa chọn. Không dùng Test để tìm cấu hình tốt hơn rồi đánh giá lại dưới nhãn cuối.

Project Guide tr.16 yêu cầu chia tập trước và chỉ dùng Test cuối; tr.22 yêu cầu phân tích mẫu sai, so baseline và giải thích chênh lệch với bài báo. Bài toán oracle expected vẫn là khó khăn của sinh test bằng model nhỏ; tăng thời lượng học có thể giúp nhưng không bảo đảm suy luận đúng ở bài chưa học.
