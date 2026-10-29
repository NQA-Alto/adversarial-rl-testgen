# Tiến độ đồ án trên APPS Introductory

Mốc cập nhật: 29/10/2026. Đây là nhật ký triển khai và thực nghiệm, chưa phải báo cáo chính thức.

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
| Thử nghiệm tinh chỉnh | Có ứng viên thứ tự nhãn input-first | Thinking bị dừng sớm: 20 mẫu đã lưu đều hết trần 768 token. Prompt v3 chỉ 2/164 test Train hợp lệ; v4 đạt 109/164 hợp lệ, 34 expected đúng và 13 test bắt lỗi, chưa qua tiêu chí so với v2. Không chấm thêm Validation/Test; giữ checkpoint và prompt hiện tại. |
| Độ ổn định APPS | Đã kiểm tra lấy mẫu; còn giới hạn seed train | Bốn checkpoint chính đã được kiểm tra với seed sinh 42/43/44. Vẫn chỉ có một seed huấn luyện mỗi phương pháp; không chuyển kết quả dữ liệu nội bộ cũ sang APPS. |
| Đánh giá Test cuối | Chưa làm | Chỉ mở sau khi chốt phương pháp, cấu hình và quy tắc chọn checkpoint; không dùng Test để chỉnh thiết kế. |
| Demo | Đã có, chưa chốt bản bảo vệ | Gradio, kết quả lưu và suy luận local; TensorBoard. Cần chọn ví dụ giải thích được, gồm thất bại, và quay video dự phòng khoảng hai phút. |
| GitHub | Đang duy trì | Nhánh apps-introductory có commit theo dữ liệu, thuật toán, thực nghiệm và kết quả. Không đưa weights, dữ liệu thô hay log lớn lên Git. |
| Chạy lại trên máy khác | Tạm hoãn | Theo ưu tiên hiện tại của người dùng; phải quay lại xử lý trước bản nộp. |
| Báo cáo và diễn tập bảo vệ | Để sau | Chưa biên soạn báo cáo cuối; tiếp tục giữ nguồn, cấu hình, lỗi tiêu biểu và bằng chứng để dùng sau. |

## Vị trí hiện tại trong quy trình Project Guide

Đã qua định nghĩa bài toán, chuẩn bị dữ liệu, tiền xử lý, huấn luyện và đánh giá Validation đầu tiên. Đang ở bước tinh chỉnh có kiểm chứng và phân tích lỗi. Bản hiện tại tái hiện các cơ chế chính ở quy mô nhỏ; chưa thể gọi là hoàn tất thực nghiệm hoặc tái lập benchmark ATGen.

Guide yêu cầu giữ Test đến cuối, lưu cả lần thất bại, có baseline, đối chiếu bài báo và phân tích lỗi. Các mốc không cải thiện vẫn được giữ; không chọn lại checkpoint hoặc seed chỉ để tạo cột cao hơn. Khi chốt thiết kế có thể dùng Validation theo quy tắc chọn đã ghi trước, rồi đánh giá Test một lần.

## Công việc tiếp theo

1. Đọc toàn bộ kiểm chứng SFT, reward và biến động lấy mẫu của chính hai nhánh đối kháng/cố định. Giữ kết luận chưa có lợi ích đối kháng được chứng minh; không tự nâng ứng viên chưa đạt tiêu chí.
2. Chốt nhu cầu và ngân sách kiểm tra nhiều seed huấn luyện. Nếu làm, lên kế hoạch cùng các nhánh và báo cáo mọi seed; không dùng seed lấy mẫu để thay thế.
3. Đóng băng cấu hình, thước đo và quy tắc chọn checkpoint; đánh giá Test giữ lại sau khi chốt.
4. Hoàn thiện ví dụ demo gồm thành công và thất bại, kiểm thử luồng trình diễn và quay video dự phòng.
5. Khi đến giai đoạn bàn giao mới xử lý môi trường mới, báo cáo và slide theo ưu tiên người dùng. Xem docs/PROJECT_STATUS.md để biết phần hoàn thành và phần còn lại.

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

## Kiểm chứng tỷ trọng reward GRPO

Hai lượt mới cùng xuất phát từ SFT64 lịch sử, cùng 41 code Train, prompt v2, seed 42 và 96 bước/nhánh. Thử trọng số định dạng/expected/attack 0,1/0,2/0,7 so với 1/3 mỗi thành phần theo tỷ trọng ATGen. Điều kiện định dạng vẫn là JSON của đồ án, khác thinking tags của bài báo. Chỉ thay tỷ trọng, không thay curriculum trong phép thử này.

| Phiên bản | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện @4 /15 |
|---|---:|---:|---:|---:|
| SFT64 trước RL | 41 | 13 | 4 | 3 |
| GRPO reward hiện tại · lượt mới | 41 | 13 | 4 | 3 |
| GRPO reward chia đều | **45** | **15** | **5** | **4** |

Chữ đậm là giá trị cao nhất từng cột, kể cả khi hòa. Mẫu sai/bị cắt vẫn nằm trong mẫu số. Đây là lượt đối chứng mới; không dùng lại số liệu của lượt 96 bước trước để thay thế.

Ứng viên chia đều qua tiêu chí đặt trước để cân nhắc thử đối kháng tiếp: **False**. Chưa tự thay cấu hình hoặc checkpoint chính. Gradient, thay đổi LoRA, bản lưu và toàn bộ rollout đã được kiểm chứng độc lập; không dùng reward tăng để thay cho đánh giá chất lượng.

[Bảng, hình và phân tích](../results/apps-reward-ablation/README.md) · [Kế hoạch thử nghiệm](APPS_REWARD_EXPERIMENT.md). Test phát triển và Test chính thức vẫn được giữ lại.

Đã kiểm tra thêm seed sinh 42/43/44 trên ba checkpoint cố định. Code phát hiện trung bình của nhánh hiện tại: 3.33 ± 1.53/15; chia đều: 3.67 ± 1.53/15. Đây là một seed huấn luyện và ba seed lấy mẫu; SD mô tả biến động, không phải khoảng tin cậy. [Toàn bộ chín lượt chấm](../results/apps-reward-ablation/SAMPLING.md).

## Kiểm chứng bổ sung hai nhánh đối kháng và chi phí

Đã kiểm tra Qwen gốc, SFT64 và hai checkpoint GRPO cuối với seed sinh test 42/43/44; mỗi lượt 60 test trên 15 code Validation. Sáu lượt mới và sáu kết quả cũ được chấm lại/đối chiếu; cùng prompt, cách lấy mẫu và bộ chấm. Đây là **một seed huấn luyện, ba seed sinh test**, không phải ba lượt train độc lập.

Code phát hiện trung bình ± SD: Qwen gốc 3.33 ± 0.58/15; SFT64 3.33 ± 1.53/15; cố định 2.67 ± 1.15/15; đối kháng 3.00 ± 1.73/15. Không có bằng chứng từ phép kiểm tra nhỏ này để tuyên bố đối kháng tốt hơn; SD chỉ mô tả biến động lấy mẫu.

[Bảng từng seed, biểu đồ và chi phí](../results/apps-adversarial-review/README.md). Checkpoint đang dùng và tập Test không thay đổi.

## Kiểm chứng thứ tự nhãn SFT

Đã kiểm chứng riêng thứ tự hai trường trong target SFT. Cấu hình cũ dạy expected trước input; cấu hình mới dạy input trước expected. Giữ nguyên 319 cặp nhãn, profile LoRA, thứ tự ví dụ, prompt, parser, reward và ngân sách. Đối chứng SFT64 được giữ từ lượt đã hoàn tất; hai GRPO96 chạy mới đầy đủ.

| Nhãn SFT | Mốc | Hợp lệ /60 | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |
|---|---|---:|---:|---:|---:|
| Expected trước | SFT | 41.00 | 11.33 | 3.67 | 3.33 |
| Expected trước | GRPO | 43.33 | 12.00 | 3.67 | 3.33 |
| Input trước | SFT | 43.33 | 19.00 | 5.33 | 4.00 |
| Input trước | GRPO | 43.67 | 20.67 | 5.67 | 4.33 |

Trung bình trên ba seed sinh test 42/43/44, một seed huấn luyện. Qua tiêu chí đặt trước để cân nhắc thử đối kháng tiếp: **True**. Phần lớn mức tăng thuộc thay đổi target SFT; tác động thêm của GRPO còn nhỏ. Đây không phải phép so sánh đối kháng mới.

Expected tăng ở cả năm bài nguồn, nhưng số test bắt lỗi tăng tập trung ở bài làm tròn APPS 2947. Không kết luận cải thiện đồng đều mọi loại lỗi. Các output sai/bị cắt vẫn ở mẫu số.

Phép thử mở rộng LoRA đã dừng trước Validation mới để ưu tiên vấn đề nhãn; không tính GRPO dở thành kết quả hoàn chỉnh. Đang thực hiện kế hoạch đối kháng có điều kiện từ adapter input-first mới; chưa thay checkpoint demo hoặc dùng Test.

[Bảng, hình và phân tích](../results/apps-target-order/README.md).

## Đối kháng input-first và phân tích expected · 30/10

Đã hoàn tất hai vòng × 48 bước từ SFT64 input-first cho cả nhánh cố định và đối kháng, sau đó kiểm tra đủ ba seed sinh 42/43/44. Cố định và đối kháng đều phát hiện trung bình 4/15 code; số test bắt lỗi đều 5/60. Expected trung bình tương ứng 20,33 và 19/60. Chưa có bằng chứng tăng nhờ curriculum đối kháng trong lượt này.

Nhánh đối kháng có 120 test RL thực sự tiếp xúc các vị trí code adversary trong tổng 384 test RL. Hai vòng dùng 8 rồi 16/41 vị trí Train đối kháng. Không nhầm số vị trí này với số code ứng viên đã sinh hoặc số lỗi phát hiện trên Validation.

Phân tích checkpoint SFT64 + GRPO96 liền: 62/180 expected đúng, 69 mẫu hợp lệ nhưng expected sai; 36 input làm lộ lỗi nhưng expected sai. Có bảng theo bài và ví dụ điểm đối xứng tính tay. [Phân tích lỗi](../results/apps-expected-diagnosis/README.md) · [Đối kháng và chi phí](../results/apps-input-first-review/README.md).

Đã hoàn tất kiểm chứng thời lượng SFT bằng checkpoint 64/256 từ cùng một quỹ đạo, qua điều kiện Train rồi chấm đủ ba seed Validation. Expected trung bình tăng 19 lên 26/60; test bắt lỗi 5,33 lên 10,33/60; code phát hiện 4 lên 7/15. Lượt dùng replay lịch sử đã dừng theo chốt kỹ thuật được giữ riêng, không tính là kiểm chứng chất lượng hoàn chỉnh. [Kết quả SFT](../results/apps-shared-sft-budget/README.md).

GRPO96 từ đúng SFT256 mới đã chạy và kiểm chứng đầy đủ. Expected trung bình 24,67/60; test bắt lỗi 8,67/60; code phát hiện 6/15, thấp hơn parent SFT256. Giữ SFT256 làm checkpoint đang tốt hơn cho demo; không gán lợi ích SFT thành thành tích RL. Demo lưu đủ Qwen gốc, SFT256 và SFT256+GRPO96; các đối chứng đối kháng SFT64 vẫn xem được riêng. [Phần thêm của RL](../results/apps-sft256-grpo/README.md).

Các lần train GRPO thủ công mới khởi tạo từ SFT256 đã chọn; nút chấm adapter hiện tại vẫn chấm lượt pilot mới nhất. Chấm lại và chọn model trước khi thay checkpoint demo, không tự dùng bản mới nhất chỉ vì vừa train xong. Test cuối, video dự phòng, báo cáo/slide và kiểm chứng cài lại vẫn là các hạng mục còn lại.
