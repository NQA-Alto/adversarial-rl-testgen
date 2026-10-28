# Kiểm chứng tỷ trọng phần thưởng

Mục đích là tách ảnh hưởng của reward khỏi số bước SFT, prompt và curriculum. Thử nghiệm SFT 64/256 trước đó cho thấy expected đúng có thể tăng trong khi số code bắt lỗi giảm. Vì vậy, lần này giữ checkpoint SFT64 lịch sử và chỉ thay tỷ trọng phần thưởng trong GRPO.

## Hai nhánh

| Thành phần được bộ chấm xác nhận | legacy-v1 | equal-v1 |
|---|---:|---:|
| JSON và input hợp lệ | 0,1 | 1/3 |
| Expected đúng | 0,2 | 1/3 |
| Expected đúng và bắt được lỗi | 0,7 | 1/3 |

Các thành phần cộng dồn, tổng tối đa bằng 1. Expected sai không được nhận phần thưởng attack; test bị cắt hoặc không hợp lệ nhận 0. `apps_core.score` và chỉ số đánh giá giữ nguyên, còn `apps_rewards.training_reward` là mục tiêu học có phiên bản riêng. Raw output, cờ chấm và cả hai giá trị reward được giữ lại để có thể kiểm tra độc lập.

[ATGen §4.1 và §5.1](https://arxiv.org/html/2510.14635v1) sử dụng trọng số bằng nhau cho ba thành phần. Đồ án thử tỷ trọng này với chuẩn hóa tổng tối đa về 1; điều kiện định dạng vẫn là JSON và miền input hữu hạn, khác định dạng thinking của tác giả. Không gọi đây là tái lập nguyên vẹn cấu hình bài báo.

## Các yếu tố giữ cố định

- Cùng Qwen3.5-2B BF16, adapter SFT64 lịch sử đã lưu và kiểm tra hash; LoRA rank 4.
- Cùng 41 code Train, prompt apps-call-v2 và seed 42; optimizer GRPO mới ở mỗi nhánh.
- 96 bước/nhánh; bốn mẫu/nhóm, tối đa 96 token/mẫu, learning rate 1e-5. Kiểm tra thứ tự case và cấu hình hai nhánh khớp sau học.
- Huấn luyện xong cả hai nhánh trước khi chấm. Chỉ chấm checkpoint cuối trên Train và Validation với cùng prompt/hash, seed theo case và ngân sách sinh. Không lựa mốc trung gian theo điểm Validation.
- Đây là dữ liệu cố định để cô lập tỷ trọng reward. Không đồng thời thay code đối kháng, nên chưa đo lợi ích của adaptive curriculum.

## Quy tắc đọc kết quả đặt trước

Chỉ số chính: số test thực sự bắt lỗi và số code phát hiện với tối đa bốn test. Expected đúng và tỷ lệ hợp lệ là chỉ số bổ sung. Mẫu sai vẫn nằm trong mẫu số; không dùng oracle sửa expected rồi tính là đầu ra đúng của model.

Ứng viên chia đều chỉ đủ điều kiện cân nhắc thử đối kháng tiếp nếu expected đúng tăng ít nhất 3/60 so với đối chứng GRPO mới, đồng thời số test hợp lệ, test bắt lỗi và code phát hiện không giảm so với cả đối chứng và checkpoint SFT ban đầu. Đủ điều kiện không đồng nghĩa tự động thay checkpoint chính.

Reward trung bình giữa hai profile không trực tiếp so được vì tỷ trọng khác nhau. Hình quá trình học phải dùng các thành phần được chấm theo cùng định nghĩa. Một seed và năm bài Validation chưa đủ kết luận thống kê; CUDA nondeterminism chưa loại trừ.

## Bằng chứng và Project Guide

`apps_reward_trial.py` ghi kế hoạch và timestamp trước khi gọi model, đóng băng nguồn và parent, giữ toàn bộ rollout/gradient/adapter trên D:. `audit_apps_run.py` chấm lại từng mẫu theo đúng profile và kiểm tra hash adapter. `check_apps_rewards.py` kiểm tra tích hợp với bộ chấm, gồm expected sai, sai miền, trùng khóa JSON và đầu ra bị cắt.

Quy trình bám hướng dẫn Project Guide tr.16 về Train/Validation/Test và tr.22 về baseline, phân tích lỗi và ablation. Cả Test phát triển và Test chính thức được giữ lại. Các lần không cải thiện tiếp tục được lưu theo tr.17–18.

Kết quả cuối được tổng hợp riêng tại `results/apps-reward-ablation` bằng `summarize_apps_rewards.py`; không ghi đè các mốc RL hoặc SFT trước đó.
