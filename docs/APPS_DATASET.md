# Dữ liệu APPS Introductory

Nguồn: [codeparrot/apps](https://huggingface.co/datasets/codeparrot/apps), phiên bản `21e74ddf8de1a21436da12e3e653065c5213e9d1`. Bản phân phối này được kho [APPS của tác giả](https://github.com/hendrycks/apps) dẫn tới. Nguồn ghi giấy phép MIT.

Chỉ tải `train.jsonl`, README và script mô tả dữ liệu; không thực thi script tải về. SHA-256 của Train gốc là `45e82ef22ed8e7c0c04d881a21b923e9dd233157896b0b8d5b3493e887499cae`. Tập Test chính thức chưa tải.

## Phạm vi

Bản Train chứa 5.000 bài, trong đó 2.639 bài Introductory: 2.504 bài dạng gọi hàm, 134 bài nhập/xuất chuẩn và một bài lỗi JSON. Đợt tích hợp đầu chỉ hỗ trợ dạng hàm. Khảo sát tìm được 528 bài có ít nhất một lời giải trong 12 lời giải nguồn đầu tiên vượt qua ít nhất ba test gốc bằng trình thông dịch hiện tại. Đây là kiểm tra dữ liệu, không phải điểm model.

Danh sách 35 bài và miền input được khai báo trong `scripts/apps_profiles.py` trước khi đánh giá model. Miền input nhỏ gồm số nguyên, chuỗi và danh sách; một số bài có điều kiện liên kết như hai chuỗi cùng độ dài. Chỉ đánh giá trên các miền con hợp lệ này, không tuyên bố bao phủ toàn bộ miền của đề gốc.

## Kiểm chứng

1. Giữ nguyên câu hỏi, ID, URL và độ khó của nguồn.
2. Đổi tên entry point của lời giải thành `solve`, bỏ annotation và `self` khi có. Chỉ giữ hàm tự chứa có thể chạy bằng trình thông dịch AST giới hạn. Không dùng `exec`/`eval` để thực thi code tải về.
3. Hai lời giải nguồn phải qua toàn bộ test gốc và đồng ý trên mọi input trong miền đã khai báo. Hai lời giải nguồn không có nghĩa là hai chứng minh độc lập về tính đúng.
4. Test gốc dùng quy ước trực tiếp hoặc bỏ một lớp singleton của APPS call-based. Expected mới do model sinh được so chính xác theo JSON với kết quả tham chiếu; không sửa JSON/kiểu dữ liệu hoặc lấy đáp án từ code lỗi.
5. Tạo code lỗi bằng một thay đổi AST: toán tử, phép so sánh, hằng số hoặc chỉ số. Loại cú pháp không hỗ trợ, lỗi giới hạn, hành vi trùng trên miền và mutant sai với mọi input. Mỗi code giữ lại phải có input bắt lỗi và input chưa bắt lỗi.
6. Kiểm tra lại dữ liệu trên đĩa: thứ tự tham số, hash, reward, tập chia và việc từ chối cú pháp ngoài phạm vi.

29/35 bài được giữ, tạo 74 code lỗi. Năm bài không có mutant phù hợp với bộ lọc. Bài 2704 bị loại vì hai lời giải nguồn bất đồng trên miền thực nghiệm. Danh sách loại đầy đủ nằm trong manifest.

Đây là **tập con APPS Introductory bổ sung các phiên bản đột biến**. APPS không vốn chứa sẵn những code lỗi này. Đợt này chưa có code lỗi do adversary local sinh.

## Chia dữ liệu

| Tập phát triển | Bài nguồn | Code lỗi |
|---|---:|---:|
| Train | 17 | 41 |
| Validation | 5 | 15 |
| Test giữ lại | 7 | 18 |

Các tập tách theo bài và URL trong **APPS Train**. Mutant của cùng bài luôn ở cùng tập. Test phát triển không phải Test chính thức của APPS. Kiểm tra bộ chấm không phải đánh giá model trên Test; model chưa được gọi cho tập giữ lại.

Nếu công bố điểm benchmark APPS, phải có kế hoạch và bộ chạy tương thích cho Test chính thức; không gọi điểm Validation này là điểm benchmark đó.

## Thí nghiệm

Pilot đầu dùng prompt `apps-call-v1`, chứa đề gốc, miền con thực nghiệm và code lỗi; không chứa lời giải tham chiếu hay test chứng minh lỗi. Prompt đó không thêm bài giải mẫu; ví dụ vốn thuộc đề gốc được giữ nguyên.

Reward: 0 nếu sai JSON/ngoài miền/bị cắt; 0,1 nếu input hợp lệ nhưng expected sai; 0,3 nếu expected đúng nhưng chưa bắt lỗi; 1 nếu expected đúng và bắt lỗi. Giới hạn bộ chấm không tính thành bug và chặn lượt train/evaluate.

Baseline đầu dùng Qwen3.5-2B gốc. Pilot tạo LoRA mới hạng 4 trên q_proj/v_proj self-attention, không kế thừa adapter dữ liệu nội bộ. Mỗi nhóm GRPO sinh 4 test, 48 bước, learning rate 1e-5, beta 0. Validation dùng 4 test/code, seed 42, tối đa 96 token/completion. Giữ cả đầu ra sai và bị cắt trong mẫu số.

Lượt hiện hành dùng prompt `apps-call-v2`, thêm một ví dụ định dạng JSON của bài cộng hai số không liên quan và nhấn mạnh miền input. Qwen gốc được chấm lại với prompt này. Sau đó tạo LoRA mới, SFT 64 bước bằng target từ Train, rồi GRPO 96 bước từ adapter SFT, optimizer mới. Ba mốc cùng dữ liệu, prompt và cấu hình đánh giá. Tác động đổi prompt, SFT và GRPO được ghi tách biệt; không dùng lời giải hoặc witness của Validation trong prompt hay SFT. Xem [quy trình huấn luyện](APPS_TRAINING.md) và [kết quả](../results/apps-sft-grpo/README.md).

## Giới hạn

- Tập nhỏ, chọn theo bộ chạy và hợp đồng input; không đại diện toàn bộ APPS Introductory.
- Miền hữu hạn thuận lợi cho kiểm chứng nhưng giới hạn khái quát.
- Hai lời giải và test nguồn vẫn có thể cùng sai ngoài miền đã kiểm tra.
- Mutation có kiểm soát chưa thay thế code sai tự nhiên do model sinh.
- Không xác nhận được model nền có từng gặp những bài công khai này khi tiền huấn luyện hay không.
- Pilot một seed chưa đủ kết luận cải thiện ổn định hoặc lợi ích đối kháng.

Tài nguyên ở D:. Bản dữ liệu nội bộ và kết quả cũ giữ trong bản sao lưu trước APPS và lịch sử Git.
