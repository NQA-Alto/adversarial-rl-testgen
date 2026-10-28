# Tiến độ đồ án trên APPS Introductory

Mốc cập nhật: 28/10/2026. Đây là nhật ký triển khai và thực nghiệm, chưa phải báo cáo chính thức.

| Công việc | Trạng thái APPS | Bằng chứng và phần còn lại |
|---|---|---|
| Chốt bài toán và phạm vi | Đã chốt | Local Qwen3.5-2B, sinh input/expected cho code Python; LoRA + GRPO; model sinh code giữ nguyên trọng số. |
| Môi trường | Đã kiểm chứng | Ryzen 7 7840H, RTX 4060 Laptop 8188 MiB, RAM khoảng 16 GB; CUDA, suy luận, gradient và lưu/nạp adapter đã kiểm tra. Phiên bản thư viện/model và hash trọng số được lưu. |
| Nguồn và chia dữ liệu | Đã có phiên bản đóng băng | 29 bài APPS Introductory, 74 code đột biến: Train 17/41, Validation 5/15, Test phát triển 7/18; chia theo bài/URL trong APPS Train. Test chính thức APPS chưa tải. |
| Tiền xử lý và bộ chấm | Đã kiểm chứng | Hai reference qua test nguồn và đồng ý trên miền hữu hạn; 26.833 cặp code/input đã kiểm tra. Chỉ hỗ trợ phần Python và miền input được khai báo. |
| EDA và minh họa pipeline | Đã bổ sung hình tái tạo được | Có phân bố bài theo tập, kiểu input và kích thước miền input trên Train; có pipeline và hình phân tích lỗi. Không đọc nội dung Test. |
| Baseline và SFT | Đã kiểm chứng 64 và 256 bước | Cùng prompt v2, 319 target Train. SFT256 có 21/60 expected đúng và phát hiện 2/15 code trong Validation; cần giữ riêng với tác động RL. |
| GRPO cố định | Đã chạy | Lượt 96 bước liền được giữ làm lịch sử; lượt đối chứng mới gồm 2 vòng × 48 bước từ SFT. |
| Curriculum đối kháng | Đã chạy hai vòng | Sinh 128 và 192 code; lọc 7 và 14 code, tổng 20 hành vi khác nhau. Thay 6 và 9 vị trí Train; model sinh 72 test khi gặp code đối kháng. |
| Đánh giá Validation | Đã có so sánh có kiểm soát | Cùng checkpoint SFT ban đầu, seed, thứ tự bài và ngân sách. Kết quả cuối: cố định 2/15 code, đối kháng 2/15; SFT 3/15. Chưa có bằng chứng cải thiện. |
| Phân tích lỗi | Đã bổ sung | Hai nhánh phát hiện cùng hai code; 48/60 đầu ra trùng. Nhánh đối kháng có 11 input tiềm năng bắt lỗi nhưng mất 8 cơ hội vì expected sai. Đây là chẩn đoán bằng oracle, không cộng vào thành tích của model. |
| Thử nghiệm tinh chỉnh | Đã kiểm chứng, chưa chọn ứng viên mới | Thinking bị dừng sớm: 20 mẫu đã lưu đều hết trần 768 token. Prompt v3 chỉ 2/164 test Train hợp lệ; v4 đạt 109/164 hợp lệ, 34 expected đúng và 13 test bắt lỗi, chưa qua tiêu chí so với v2. Không chấm thêm Validation/Test; giữ checkpoint và prompt hiện tại. |
| Độ ổn định APPS | Chưa đủ | Hiện một seed huấn luyện. Ba seed và biểu đồ 10/18 trước đây thuộc dữ liệu nội bộ, không chuyển kết quả đó sang APPS. |
| Đánh giá Test cuối | Chưa làm | Chỉ mở sau khi chốt phương pháp, cấu hình và quy tắc chọn checkpoint; không dùng Test để chỉnh thiết kế. |
| Demo | Đã có, chưa chốt bản bảo vệ | Gradio, kết quả lưu và suy luận local; TensorBoard. Cần chọn ví dụ giải thích được, gồm thất bại, và quay video dự phòng khoảng hai phút. |
| GitHub | Đang duy trì | Nhánh apps-introductory có commit theo dữ liệu, thuật toán, thực nghiệm và kết quả. Không đưa weights, dữ liệu thô hay log lớn lên Git. |
| Chạy lại trên máy khác | Tạm hoãn | Theo ưu tiên hiện tại của người dùng; phải quay lại xử lý trước bản nộp. |
| Báo cáo và diễn tập bảo vệ | Để sau | Chưa biên soạn báo cáo cuối; tiếp tục giữ nguồn, cấu hình, lỗi tiêu biểu và bằng chứng để dùng sau. |

## Vị trí hiện tại trong quy trình Project Guide

Đã qua định nghĩa bài toán, chuẩn bị dữ liệu, tiền xử lý, huấn luyện và đánh giá Validation đầu tiên. Đang ở bước tinh chỉnh có kiểm chứng và phân tích lỗi. Bản hiện tại tái hiện các cơ chế chính ở quy mô nhỏ; chưa thể gọi là hoàn tất thực nghiệm hoặc tái lập benchmark ATGen.

Guide yêu cầu giữ Test đến cuối, lưu cả lần thất bại, có baseline, đối chiếu bài báo và phân tích lỗi. Các mốc không cải thiện vẫn được giữ; không chọn lại checkpoint hoặc seed chỉ để tạo cột cao hơn. Khi chốt thiết kế có thể dùng Validation theo quy tắc chọn đã ghi trước, rồi đánh giá Test một lần.

## Công việc tiếp theo

1. Đã kiểm chứng ngân sách SFT. Ứng viên 256 bước chưa đủ tiêu chí bổ sung. Ưu tiên kiểm chứng cách cân bằng các thành phần reward theo ATGen, giữ mốc đối chứng và phân biệt tác động SFT với RL.
2. Chấm SFT mới tách biệt với RL, rồi mới cân nhắc lặp lại hai nhánh GRPO từ checkpoint phù hợp; không sửa output khi chấm.
3. Các hướng suy luận/định dạng vừa thử chưa qua tiêu chí. Nếu thử tiếp phải lưu phiên bản và mốc đối chứng; không đổi số liệu cũ hay dùng Test để lựa chọn.
4. Kiểm tra độ ổn định của cấu hình APPS đã chọn; hoàn thiện EDA, pipeline, phân tích chi phí và phần đối chiếu ATGen.
5. Chốt cấu hình, đánh giá Test cuối; hoàn thiện video demo và nội dung bảo vệ. Đóng gói và báo cáo thực hiện sau theo yêu cầu người dùng.

## Tài liệu và kết quả

- docs/ATGEN_ALIGNMENT.md: phần tái hiện, điều chỉnh và giới hạn so với bài báo.
- docs/PROJECT_GUIDE_CHECKLIST.md: đối chiếu các yêu cầu của giảng viên.
- results/apps-sft-grpo: lượt SFT và GRPO cố định đầu tiên.
- results/apps-adversarial: hai vòng đối kháng và đối chứng cùng ngân sách.
- results/apps-error-analysis: nguyên nhân thất bại và so sánh theo code.
- Evidence, adapter và nguồn dữ liệu nằm dưới D:/DA/TTNT_De19_Assets.

Bản roadmap dữ liệu nội bộ trước khi chuyển đổi đã được giữ riêng trong evidence. Không xóa kết quả hay tài liệu cũ.

Hình và dữ liệu bổ sung: results/apps-data-profile, results/apps-error-analysis và results/apps-tuning. Các thử nghiệm tinh chỉnh lần này không cập nhật trọng số, không được gọi là lượt train RL mới.

## Kết quả kiểm chứng ngân sách SFT

Đã chạy SFT 256 bước với pool 319 target Train, 1.024 lượt ví dụ, giữ prompt v2 và các yếu tố học khác. Cả 64 và 256 ở đối chứng bổ sung nằm trong cùng một quỹ đạo học; checkpoint 128 chỉ lưu để phục hồi và không được chấm Validation.

| Mốc cùng lượt học | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện @4 /15 |
|---|---:|---:|---:|---:|
| SFT 64 bước | 41 | 15 | 5 | 4 |
| SFT 256 bước | 54 | 21 | 3 | 2 |

Ứng viên qua tiêu chí bổ sung so với cả mốc lịch sử và đối chứng cùng lượt: False. Tiêu chí ban đầu có ngưỡng tái hiện không đạt vì checkpoint 64 lệch L2 khoảng 0,198% so với lượt lịch sử; kết quả và kế hoạch gốc được giữ nguyên. Kế hoạch bổ sung được ghi trước khi xem Validation mới. Chưa xác định chắc nguyên nhân sai khác khi tái hiện.

Mốc 256 được lưu riêng trong giao diện để kiểm chứng. Các nhánh GRPO/đối kháng đã có vẫn bắt đầu từ SFT 64 bước lịch sử; không gộp tác động của SFT mới thành cải thiện của RL. Test chưa được model đánh giá. Xem results/apps-sft-budget để lấy bảng, đường loss và nhật ký.
