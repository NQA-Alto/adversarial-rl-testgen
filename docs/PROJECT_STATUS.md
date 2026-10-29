# Tiến độ toàn đề tài

Cập nhật ngày 29/10/2026. Đây là ước lượng khối lượng công việc để điều phối đồ án, không phải điểm chấm hoặc phép đo chất lượng model.

**Ước lượng hiện tại: khoảng 70% toàn bộ đồ án; khoảng 90% phần triển khai kỹ thuật cốt lõi.** Chạy được chương trình và huấn luyện được adapter không có nghĩa là đã xong phần đánh giá, bàn giao và bảo vệ.

| Nhóm công việc | Tỷ trọng ước lượng | Đã xử lý | Phần còn lại |
|---|---:|---:|---|
| Phạm vi, môi trường, nguồn và kiểm tra dữ liệu | 15 | 15 | Đã có cấu hình, nguồn, chia tập và kiểm chứng miền input; phải giữ nguyên phiên bản trong đánh giá cuối. |
| Model local, bộ chấm, SFT, GRPO, curriculum đối kháng | 25 | 23 | Đã chạy và có bằng chứng cập nhật LoRA; cần chốt cấu hình dùng trong bản nộp và rà soát tích hợp cuối. |
| Thực nghiệm và phân tích | 25 | 18 | Đã có baseline, đối chứng, ablation và phân tích lỗi; còn quyết định mức kiểm tra nhiều seed train, chốt phương pháp và chấm Test giữ lại. |
| Giao diện và demo bảo vệ | 15 | 10 | Có giao diện và kết quả đã lưu; còn kịch bản 2–3 ví dụ, kiểm thử luồng trình diễn và video dự phòng. |
| Chạy lại, báo cáo, slide và chuẩn bị bảo vệ | 20 | 4 | Có README, nguồn, bằng chứng và bài giảng. Chưa hoàn thiện báo cáo/slide/video, chưa kiểm chứng cài lại trên môi trường mới. |
| Tổng | **100** | **70** | Ước lượng được làm tròn; không suy ra từ số lượt train. |

Các hạng mục báo cáo và đóng gói chưa làm theo ưu tiên trước đó của người dùng. Bảng này không tự thay đổi ưu tiên ấy. Mức độ hiểu và khả năng trả lời vấn đáp của người học cần được ôn riêng, không tự suy ra từ mức hoàn thiện mã nguồn.

## Những gì đã làm được

- Chạy Qwen3.5-2B local trên RTX 4060 Laptop 8 GB; kiểm chứng suy luận, gradient, LoRA, lưu và nạp adapter.
- Chuyển từ dữ liệu nội bộ sang tập con APPS Introductory có nguồn và phiên bản; 29 bài, 74 code lỗi, chia theo bài trước khi học.
- Có bộ chấm input/expected và khả năng phát hiện lỗi trong miền hữu hạn; lưu output thô và mẫu không hợp lệ.
- Chạy SFT, GRPO cố định, hai vòng curriculum đối kháng có nhánh đối chứng cùng ngân sách RL; model sinh code giữ nguyên trọng số.
- Kiểm tra riêng ảnh hưởng ngân sách SFT, cách diễn đạt prompt, thinking và tỷ trọng reward. Các thử nghiệm không đạt vẫn được giữ lại.
- Có EDA, phân tích lỗi, biểu đồ, TensorBoard và bài giảng HTML tự chứa để học cách giải thích đồ án.
- Duy trì lịch sử Git, lưu cấu hình/hash, nhật ký và adapter trên ổ D:.

## Giới hạn cần nói đúng

Hệ thống đã thực sự học bằng RL, nhưng kết quả Validation hiện có chưa chứng minh curriculum đối kháng cải thiện khả năng tìm lỗi. Đây là hai kết luận khác nhau. Không cần sửa biểu đồ hoặc chọn seed thuận lợi để tạo chênh lệch.

Phép kiểm tra seed sinh test bổ sung giữ nguyên checkpoint. Nó đo biến động đầu ra khi lấy mẫu, không thay thế nhiều lượt huấn luyện độc lập. Nếu chưa làm nhiều seed train, phải ghi rõ giới hạn đó.

Tập con APPS và bộ thông dịch giới hạn không phải benchmark APPS đầy đủ. Tập Test phát triển giữ lại nằm trong APPS Train; Test chính thức APPS chưa tải. Không so trực tiếp điểm của đồ án với con số trong ATGen nếu khác dữ liệu và cách chấm.

## Thứ tự tiếp tục

1. Hoàn tất và đọc phép kiểm tra lấy mẫu của nhánh đối kháng, đối chứng và các baseline; tổng hợp chi phí từ log gốc. Xem [bảng kiểm chứng bổ sung](../results/apps-adversarial-review/README.md).
2. Chốt kế hoạch thực nghiệm cuối: nếu cần nhiều seed train, lên kế hoạch đủ các nhánh trước khi chạy; báo cáo tất cả kết quả. Không tiếp tục train vô hạn chỉ để đạt cột cao hơn.
3. Đóng băng model, prompt, reward, ngân sách, phương pháp chọn checkpoint và thước đo; sau đó mới đánh giá Test giữ lại. Không dùng kết quả Test để chỉnh thiết kế rồi gọi là đánh giá cuối.
4. Chọn 2–3 ví dụ demo theo tiêu chí giải thích được: một test đúng bắt lỗi, một test đúng chưa bắt lỗi và một thất bại. Giữ output model nguyên trạng; khi cần dùng kết quả đã lưu phải nói rõ.
5. Quay video dự phòng khoảng hai phút, kiểm thử luồng trình diễn với checkpoint đã lưu. Nút train phục vụ thực nghiệm, không phải bước bắt buộc tại buổi bảo vệ.
6. Khi chuyển sang giai đoạn bàn giao: tách đường dẫn/cấu hình, kiểm chứng môi trường mới; hoàn thiện báo cáo và slide theo Project Guide; ôn vấn đáp dựa trên [bài giảng](BAI_GIANG_DO_AN_TTNT.html).

Đối chiếu yêu cầu cụ thể tại [checklist Project Guide](PROJECT_GUIDE_CHECKLIST.md) và [phạm vi so với ATGen](ATGEN_ALIGNMENT.md). Bản hiện tại chưa được coi là sản phẩm nộp hoàn chỉnh.
