# APPS: kiểm chứng ngân sách khởi động SFT

So sánh 64 và 256 bước, cùng model gốc, LoRA rank 4, 319 target Train, thứ tự target, seed 42, prompt v2 và optimizer. Bốn ví dụ/bước, tương ứng 256 và 1.024 lượt ví dụ. Đây là SFT, không phải RL.

| Ngân sách | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện @4 /15 |
|---|---:|---:|---:|---:|
| 64 bước, lịch sử | 41 | 13 | 4 | 3 |
| 64 bước, cùng lượt mới | 41 | 15 | **5** | **4** |
| 256 bước | **54** | **21** | 3 | 2 |

Giá trị lớn nhất mỗi cột được in đậm, kể cả khi hòa. Cùng 15 code từ năm bài Validation; mỗi code sinh bốn test, cùng prompt, seed và giới hạn 96 token. Mẫu sai/bị cắt vẫn nằm trong mẫu số.

## Kiểm chứng phép so sánh

- Checkpoint tại bước 64 khớp từng tensor với lượt cũ: False. Sai khác L2 tương đối: 0.0019802373.
- Hash target pool và thứ tự target khớp lượt 64 bước cũ. Không đổi prompt, cách chấm hoặc kiểu expected.
- Checkpoint 128 chỉ được lưu để phục hồi; không chấm Validation để chọn mốc thuận lợi.
- Trọng số nền giữ nguyên; gradient chỉ đi vào LoRA. Adapter cuối được kiểm tra khớp bản lưu.
- Test phát triển và Test chính thức không được model đánh giá.

Ứng viên qua tiêu chí đã đặt trước để cân nhắc thí nghiệm RL tiếp theo: **False**. Cấu hình và checkpoint đang dùng không tự động bị thay thế.

![So sánh Validation](validation-comparison.png)

![Loss SFT trên Train](learning-curve.png)

Loss Train giảm chỉ chứng minh model khớp dữ liệu học hơn, không tự chứng minh tổng quát hóa. Kết quả hiện chỉ có một seed và năm bài Validation. Nếu warmup tốt hơn, phần thay đổi này thuộc SFT; cần phép so sánh RL riêng để kết luận về GRPO hoặc đối kháng.

Log, target, checkpoint 64/128/256 và bản chụp nguồn ở thư mục trial ghi trong metrics.json. Các bảng/hình được dựng lại bằng scripts/summarize_apps_sft_budget.py.

## Đối chứng bổ sung trong cùng quỹ đạo học

Kế hoạch gốc chỉ chấm checkpoint 256 và yêu cầu tái hiện bước 64 với sai khác L2 không quá 0,1%. Thực tế checkpoint 64 khác bản lịch sử khoảng 0,198%; ngưỡng tái hiện ban đầu không đạt và kết quả đó được giữ nguyên.

Trước khi xem Validation mới, đã ghi paired64-plan.json để bổ sung checkpoint 64 của chính lượt hiện tại làm đối chứng. Hai checkpoint 64/256 chia sẻ chính xác cùng 64 bước đầu. Cả hai chỉ được chấm sau khi đã học xong 256 bước; checkpoint 128 không được đánh giá. Đây là phép đối chiếu bổ sung, không sửa lại kế hoạch gốc hay chọn checkpoint theo điểm thuận lợi.

Ứng viên qua tiêu chí bổ sung so với cả hai mốc 64: **False**. Trạng thái đủ điều kiện theo kế hoạch gốc vẫn là **False**.

Sai khác khi tái hiện chưa được quy cho một nguyên nhân cụ thể. [Tài liệu PyTorch về reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html) mô tả các nguồn không xác định và đặc tính của CUDA attention; đây là những khả năng cần kiểm tra, không phải kết luận đã xác định kernel gây ra sai khác trong lượt này.
