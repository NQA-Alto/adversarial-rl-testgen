# Đối chiếu đồ án với ATGen

Tài liệu gốc: [ATGen: Adversarial Reinforcement Learning for Test Case Generation](https://arxiv.org/abs/2510.14635), bản v1, mục 3–5.

Đồ án triển khai có điều chỉnh ở quy mô nhỏ. Mục tiêu là kiểm tra khả năng sinh test đúng đáp án và phát hiện lỗi của code Python bằng mô hình local, sau đó đo tác động của curriculum đối kháng. Chưa thể xem kết quả trên tập con này là tái lập kết quả benchmark của bài báo.

| Thành phần tham chiếu | Cách triển khai trong đồ án | Bằng chứng cần kiểm tra |
|---|---|---|
| Đầu vào: đề bài và code cần kiểm tra (§3) | Đề APPS Introductory, hàm `solve`, miền input hữu hạn. Model sinh một cặp input/expected bằng JSON. | Prompt, mã bài nguồn và hash dữ liệu được lưu theo lượt chạy. |
| IO Accuracy và Attack Rate (§3, §5.1) | Expected phải khớp lời giải tham chiếu; chỉ tính bắt lỗi khi expected đúng và code cần kiểm tra trả khác hoặc phát sinh lỗi thực thi được hỗ trợ. | Từng mẫu được chấm lại từ đầu; mẫu sai định dạng/bị cắt vẫn nằm trong mẫu số. |
| GRPO (§4.1) | Dùng TRL, nhóm bốn test; chỉ cập nhật LoRA của model sinh test. | Reward từng nhóm, gradient, tham số thay đổi, hash adapter và kiểm tra lưu/nạp. |
| Reward nhiều thành phần (§4.1) | Mặc định dùng các mức 0 / 0,1 / 0,3 / 1; đã thử riêng trọng số chia đều 1/3 theo tỷ trọng bài báo. Format vẫn là JSON/miền input, khác thinking tags. Không gộp hai profile hay sửa cách chấm cũ. | Hàm `score` và log rollout; không nói đây là cấu hình reward nguyên bản. |
| Sinh code rồi lọc (§4.2) | Model local gốc được yêu cầu giải đúng đề. Code sinh ra phải qua bộ thông dịch giới hạn. Chỉ giữ code sai ở một số input, đúng ở một số input, vượt qua test đúng của policy hiện tại và khác hành vi các mutant ban đầu. | Toàn bộ code được lấy mẫu, lý do loại, nhân chứng lỗi và kết quả kiểm tra lại. Nhân chứng chỉ phục vụ bộ chấm, không đưa vào prompt. |
| Adaptive curriculum (§4.2) | Chỉ thay những code Train mà policy hiện tại đã bắt được lỗi. Làm mới code giữa các vòng, giữ nguyên các vị trí bài Train còn lại. | Probe trên Train, curriculum đóng băng theo vòng và số mẫu RL thật sự tiếp xúc code đối kháng. |
| So sánh không có đối kháng (§5.2) | Hai nhánh cùng xuất phát từ SFT, cùng số bước GRPO, cùng số lần khởi tạo optimizer, seed và cách đánh giá. Khác biệt chủ đích là code huấn luyện. | Nhánh GRPO cố định và nhánh adaptive; không dùng lượt 96 bước một mạch trước đây làm đối chứng tương đương cho hai vòng 48 bước. |

## Những điều chỉnh phải nêu khi bảo vệ

- Dùng Qwen3.5-2B chạy trên RTX 4060 Laptop 8 GB và LoRA rank 4. Thư viện huấn luyện là Transformers/PEFT/TRL.
- Hai vai trò dùng hai lần nạp model local tuần tự. Model sinh code giữ nguyên trọng số; model sinh test có adapter được học bằng RL. Không phải hai model cùng được cập nhật bằng RL.
- JSON trực tiếp, tắt chế độ thinking; chưa triển khai định dạng suy luận có thẻ của bài báo.
- Chỉ 29 bài APPS, chia theo bài thành Train/Validation/Test phát triển. Code ban đầu là đột biến AST; code bổ sung được model local sinh rồi lọc. Không phải bộ dữ liệu APPS + Codeforces của tác giả.
- Bộ chấm vét cạn miền input đã khai báo và chỉ hỗ trợ một phần Python. “Sai” hoặc “đúng” khi lọc code chỉ được xác nhận trong miền đó. Đây không phải chứng minh cho mọi input có thể có của đề gốc.
- Làm mới curriculum theo vòng là bản xấp xỉ theo lô. Không làm mới theo từng minibatch; không khẳng định độ khó tăng đơn điệu hay đã vượt giới hạn dữ liệu cố định.
- SFT là bước khởi động bổ sung của đồ án, không được cộng lợi ích của SFT hoặc đổi prompt vào lợi ích của RL.
- Chưa làm ứng dụng lọc Best-of-N và RL để huấn luyện model sinh code ở phần ứng dụng của bài báo.

## Cách giải thích phạm vi

Đồ án giải quyết phần sinh và kiểm chứng ca kiểm thử cho code Python trong một tập bài APPS giới hạn. Hệ thống vừa kiểm tra test có đúng đáp án không, vừa kiểm tra test có làm lộ lỗi không; sau đó dùng reward để học adapter sinh test. Phần đối kháng bổ sung những code mà test hiện tại bỏ sót để tạo dữ liệu học mới. Đồ án đo phần này bằng một nhánh đối chứng có cùng ngân sách học.

Đã chạy được thuật toán không đồng nghĩa với đã chứng minh hiệu quả. Chỉ được kết luận cải thiện khi có số liệu đánh giá tương ứng; kết quả thấp hoặc không cải thiện phải được giữ lại. Test phát triển dành cho lần đánh giá cuối; Test chính thức APPS chưa tải và chưa đánh giá. Các mốc hiện tại dùng Validation, không gọi là kết quả Test chính thức.

## Một ví dụ đối kháng thực sự đã sinh được

Ở vòng đầu, bài APPS 3019 yêu cầu đếm số lần một ký tự xuất hiện, có phân biệt chữ hoa và chữ thường. Candidate `r1-3019-08` lại chuyển ký tự cần tìm sang chữ thường trước khi đếm.

- Test hiện tại có `strng="bba", letter="a", expected=1`: code mới vẫn trả 1, nên test này bỏ sót lỗi.
- Bộ chấm tìm được `strng="a", letter="A", expected=0`: code mới trả 1. Như vậy code mới vẫn sai, không phải lời giải đúng.
- Candidate này vượt qua các test đúng hiện tại của ba vị trí bài 3019 và sai ở 90/363 input đã khai báo. Nó được đưa vào pool để thay code ở những vị trí đủ điều kiện.

Đây là “đối kháng” theo khả năng của policy tại thời điểm lọc. Không đồng nghĩa với code khó hơn cho mọi người hay lỗi chỉ xuất hiện ở rất ít input. Input nhân chứng trên được lưu để kiểm chứng, không đưa vào prompt cho model sinh test. Bằng chứng gốc nằm trong `apps-adversary-20261028-160208-467069`, các file `policy-suites.json` và `adversarial-cases.jsonl`.

## Nơi lưu bằng chứng

Mốc 28/10/2026 đã hoàn tất hai vòng × 48 bước cho mỗi nhánh. Model sinh test thực sự gặp code đối kháng khi sinh 72/384 test trong quá trình GRPO; hai pool có 20 hành vi code khác nhau. Nhánh đối kháng có 86/96 bước gradient khác 0, adapter sau từng vòng được kiểm tra khớp bản lưu. Đây là bằng chứng chạy và học, không phải bằng chứng hiệu quả.

Đánh giá cuối trên cùng Validation: GRPO cố định và GRPO đối kháng đều phát hiện 2/15 code, trong khi SFT là 3/15. Kết quả này chưa xác nhận lợi ích đối kháng. Bảng đầy đủ, phân loại thất bại và số liệu từng vòng nằm ở [results/apps-adversarial](../results/apps-adversarial/README.md). Không chọn lại checkpoint vòng một chỉ vì điểm Validation cao hơn; mốc cuối giữ đúng ngân sách đã định trước.

- `scripts/apps_model_runner.py`: GRPO, đánh giá và lưu adapter.
- `scripts/apps_adversary.py`: sinh code local, lọc và kiểm tra lại pool.
- `scripts/apps_curriculum.py`: kiểm tra curriculum giữ nguyên bài, reference, miền input và tập chia.
- `scripts/apps_adversarial_round.py`: so sánh hai nhánh qua hai vòng.
- `scripts/audit_apps_run.py`: chấm lại mẫu sinh, target SFT và reward RL.
- Log, code lấy mẫu và adapter nằm ở `D:/DA/TTNT_De19_Assets/evidence`; mã nguồn và bảng kết quả tổng hợp được đưa vào Git.

## Phân tích và kiểm tra bổ sung sau hai vòng

Đã bổ sung chẩn đoán input-only attack theo hướng ATGen §5.3. Với nhánh đối kháng, 11 input hợp lệ có thể làm lộ lỗi, nhưng chỉ 3 test có expected đúng để tính là bắt lỗi; 8 cơ hội mất do expected sai. Oracle chỉ phục vụ chẩn đoán, không sửa đáp án để tăng chỉ số chính. Hai nhánh cuối phát hiện cùng hai code và có 48/60 đầu ra giống hệt.

Đã thử native thinking với định dạng thẻ và trần 768 token trên Train. Lượt dừng sớm khi 20 mẫu đã lưu đều bị cắt; tiêu chí hoàn tất 80% không còn khả năng đạt. Đây là kiểm tra khả thi chưa đủ toàn bộ nhánh, không phải benchmark hoàn chỉnh hay bằng chứng model không thể suy luận. Định dạng thẻ chưa được SFT nên cũng là yếu tố cần nêu.

Hai phép thử kiểu đầu ra v3/v4 không đạt tiêu chí Train đã đặt, nên không chấm tiếp Validation và không thay cấu hình đang dùng. Các thí nghiệm này là suy luận/prompt, không cập nhật trọng số và không gọi là RL mới. Xem results/apps-error-analysis và results/apps-tuning để lấy số liệu chi tiết.

## Kiểm chứng riêng giai đoạn SFT

Đã so sánh ngân sách 64/256 bước với cùng target pool và prompt v2, có đối chứng cùng quỹ đạo học được lên kế hoạch trước Validation sau khi kiểm tra tái hiện lịch sử không khớp tuyệt đối. SFT256 đạt 21/60 expected đúng, 3/60 test bắt lỗi và 2/15 code được phát hiện. Đủ tiêu chí bổ sung để cân nhắc so sánh RL tiếp theo: False.

Phần kết quả này thuộc SFT khởi động bổ sung của đồ án. Không được dùng nó để nói ATGen đối kháng đã cải thiện. Các phép so sánh GRPO từ SFT64 giữ nguyên; nếu chuyển warmup, cả hai nhánh phải được thử lại từ cùng checkpoint. Xem results/apps-sft-budget để đối chiếu kế hoạch ban đầu, đối chứng bổ sung và số liệu.

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

## Kiểm tra bổ sung nhánh đối kháng đã triển khai

Đã kiểm tra Qwen gốc, SFT64 và hai checkpoint GRPO cuối với seed sinh test 42/43/44; mỗi lượt 60 test trên 15 code Validation. Sáu lượt mới và sáu kết quả cũ được chấm lại/đối chiếu; cùng prompt, cách lấy mẫu và bộ chấm. Đây là **một seed huấn luyện, ba seed sinh test**, không phải ba lượt train độc lập.

Code phát hiện trung bình ± SD: Qwen gốc 3.33 ± 0.58/15; SFT64 3.33 ± 1.53/15; cố định 2.67 ± 1.15/15; đối kháng 3.00 ± 1.73/15. Không có bằng chứng từ phép kiểm tra nhỏ này để tuyên bố đối kháng tốt hơn; SD chỉ mô tả biến động lấy mẫu.

Đã tách chi phí probe/sinh/lọc code khỏi thời gian GRPO, phù hợp với việc nêu giới hạn ngân sách so sánh. Đây là kiểm tra triển khai theo lô của đồ án, chưa phải tái lập nhiều seed train hoặc benchmark của ATGen. [Bằng chứng](../results/apps-adversarial-review/README.md).

## Điều chỉnh biểu diễn target SFT

Đã phát hiện canonical JSON đặt expected trước input trong nhãn SFT; chẩn đoán cho thấy thứ tự đầu ra model thay đổi theo nhãn. Phép so sánh chỉ đổi thứ tự hai trường, không đổi semantic label hoặc cách chấm. Cấu hình input-first qua tiêu chí sàng lọc trên ba seed lấy mẫu.

Đây là điều chỉnh triển khai SFT của đồ án, không phải thành phần nguyên bản của ATGen hoặc bằng chứng lợi ích đối kháng. Phải chạy lại đối chứng/curriculum từ cùng warmup mới; không nối số liệu hai warmup thành một chuỗi cải thiện. [Kết quả](../results/apps-target-order/README.md) và [kế hoạch tiếp theo](APPS_INPUT_FIRST_ADAPTIVE_PLAN.md).
