# Phương pháp tham khảo

[ATGen](https://arxiv.org/html/2510.14635v1) là tài liệu chính. Đồ án tái hiện GRPO từ reward thực thi và curriculum code vượt qua test hiện tại.

| Thành phần | Bản triển khai |
|---|---|
| Protagonist | Qwen3.5-2B, LoRA rank 4 |
| Khởi động | SFT 840, 1050 nhãn Train |
| Reward | Định dạng, expected, bắt lỗi; mỗi phần 1/3 |
| Adversary | Qwen gốc giữ trọng số; lấy mẫu lời giải rồi lọc |
| Curriculum | Hai vòng, làm mới sau vòng đầu |
| Đối chứng | Cùng parent, 145 vị trí Train và 2 × 32 bước |

Dùng JSON trực tiếp, không có thẻ suy luận; Python và input hữu hạn; refresh theo vòng. Chưa triển khai Best-of-N, RL cho model sinh code hoặc benchmark đầy đủ. Số liệu không so ngang điều kiện với bài báo.

[Thực nghiệm cuối](../results/apps-expanded-adaptive/README.md) tách hiệu quả SFT, GRPO và đối kháng. Chỉ protagonist cập nhật tham số; adversary thay dữ liệu theo policy, không đồng thời huấn luyện trọng số.

## Cập nhật curriculum

Vòng thích nghi kiểm tra policy trên code đang có trong pool. Chỉ vị trí đã bị test hiện tại bắt lỗi mới tìm code thay thế; các code còn vượt qua test được giữ lại. Bộ lọc loại cả hành vi trùng với AST ban đầu và code đang học. Lịch cân bằng lấy một nửa số nhóm từ các vị trí đối kháng, dùng cùng case ID cho nhánh cố định.
