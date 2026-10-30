# Chạy và huấn luyện

Các lệnh dưới đây dùng môi trường đã cài trên máy thực nghiệm. Python huấn luyện nằm ở `D:/DA/TTNT_De19_Assets/env`; model nền ở `D:/DA/TTNT_De19_Assets/models/Qwen3.5-2B`. `requirements.txt` lưu phiên bản thư viện của môi trường này, còn `requirements-ui.txt` dành cho giao diện.

## Giao diện

Chạy `MO_GIAO_DIEN.cmd`, rồi mở [Gradio](http://127.0.0.1:7860/?__theme=light). `DONG_GIAO_DIEN.cmd` dừng Gradio, TensorBoard và bộ xuất log do launcher khởi chạy.

| Tab | Chức năng |
|---|---|
| Demo | Xem đề, code và test đã sinh; chạy suy luận trên một code |
| Kết quả | Xem bảng Validation và các phép so sánh |
| Dữ liệu | Xem nguồn, tập chia và miền thực nghiệm |
| Thực nghiệm | Kiểm tra dữ liệu, huấn luyện GRPO hoặc chấm checkpoint |
| Huấn luyện | Xem loss, reward, gradient và mở TensorBoard |

Nút **Sinh 4 test** chỉ suy luận và chấm. Tác vụ **Huấn luyện GRPO trên Train** tạo một lượt học mới từ SFT256 đã lưu, với optimizer mới. **Chấm adapter hiện tại trên Validation** đánh giá adapter pilot mới nhất. Tác vụ **Huấn luyện SFT + GRPO trên APPS** chạy lại pipeline từ model gốc, SFT64 rồi GRPO.

Mỗi lượt tạo một thư mục riêng trong `evidence/`. Dừng tác vụ có thể mất cập nhật chưa lưu; checkpoint đã ghi vẫn còn.

## Kiểm tra dữ liệu

```powershell
$env:PYTHONUTF8 = '1'
$python = 'D:/DA/TTNT_De19_Assets/env/Scripts/python.exe'
$project = 'D:/DA/DoAn_TTNT_De19/adversarial-rl-testgen'
& $python -B "$project/scripts/verify_apps.py"
```

## SFT

SFT dùng 319 cặp input/expected từ Train. Mỗi code lấy tối đa bốn test bắt lỗi và bốn test đúng nhưng chưa bắt lỗi. Mỗi bước xử lý bốn ví dụ, learning rate `5e-5`.

```powershell
& $python -B "$project/scripts/apps_sft.py" --steps 256 --target-order input-first-v1
```

Adapter được ghi vào thư mục `apps-sft-v2-.../adapter`. Bản hiện hành đặt `input` trước `expected`; cấu hình `sorted-v1` được giữ để chạy lại các thí nghiệm trước.

LoRA mặc định gắn vào `q_proj` và `v_proj` của sáu khối full attention. Cấu hình `hybrid-qkv-v1` thêm `in_proj_qkv` của 18 khối linear attention, tăng số tham số học từ 208.896 lên 798.720. Có thể chọn bằng `--lora-profile hybrid-qkv-v1`; rank và alpha vẫn là 4 và 8.

Phép so sánh với SFT256 hiện có được chạy riêng:

```powershell
& $python -B "$project/scripts/apps_lora_input_trial.py" --prepare
& $python -B "$project/scripts/apps_lora_input_trial.py" --run '<thư mục vừa chuẩn bị>'
& $python -B "$project/scripts/summarize_apps_lora_input.py" '<thư mục vừa chuẩn bị>'
```

Chương trình kiểm tra hash nhãn, thứ tự học và mã huấn luyện của đối chứng. Ứng viên chỉ được chấm Validation khi qua điều kiện trên Train; các lượt này không tự thay checkpoint của demo. Đường dẫn đối chứng nằm trong bảng `results/apps-shared-sft-budget/metrics.json`, nên cần có dữ liệu và kết quả tương ứng trên máy.

## GRPO và đánh giá

GRPO sinh bốn test cho mỗi nhóm. Reward lần lượt là 0 nếu đầu ra không hợp lệ, 0,1 nếu input hợp lệ nhưng expected sai, 0,3 nếu expected đúng và 1 nếu test phát hiện lỗi. Chỉ tham số LoRA được cập nhật; model nền giữ nguyên.

```powershell
& $python -B "$project/scripts/apps_model_runner.py" train `
  --steps 96 --seed 42 --prompt-version apps-call-v2 `
  --init-adapter '<thư mục adapter SFT>' `
  --baseline-run '<kết quả chấm đúng adapter SFT đó>' --no-publish

& $python -B "$project/scripts/apps_model_runner.py" evaluate `
  --adapter '<thư mục adapter cần chấm>' `
  --seed 42 --prompt-version apps-call-v2 --no-publish
```

`--no-publish` lưu kết quả của lượt chạy mà không thay trạng thái đang hiển thị trên giao diện. Validation dùng bốn test/code, tối đa 96 token/test, temperature 1,0, top-p 0,95 và top-k 20.

Pipeline cố định đầy đủ:

```powershell
& $python -B "$project/scripts/apps_round.py" --grpo-steps 96
```

Pipeline này cập nhật các mốc đang dùng sau khi hoàn tất. Kết quả của các lượt trước vẫn nằm trong thư mục evidence riêng.

Để thử reward chia đều từ SFT256, dùng cùng kết quả đối chứng của phép so sánh ngân sách SFT:

```powershell
& $python -B "$project/scripts/apps_sft256_rl_followup.py" '<thư mục shared-sft-budget>' --reward-profile equal-v1
& $python -B "$project/scripts/summarize_apps_sft256_rl.py" '<thư mục sft256-equal-reward vừa tạo>'
```

Lượt này giữ 96 bước, học trên code Train cố định và chấm ba seed sinh. Nó không tự thay checkpoint của demo.

Nếu tiến trình đánh giá dừng trước khi sinh đầu ra, `resume_sft256_evaluation.py '<thư mục lượt thử>'` có thể chạy nốt các seed thiếu. Chương trình giữ nguyên adapter và các lượt đã xong, lưu bản trạng thái lỗi, rồi kiểm tra lại cấu hình. Trường hợp đã có đầu ra một phần sẽ được dừng để xử lý riêng.

## Curriculum đối kháng

`apps_adversarial_round.py` điều phối hai nhánh cố định và đối kháng. Mỗi nhánh học hai vòng, mỗi vòng 48 bước; optimizer được tạo lại ở đầu vòng. Model sinh code giữ nguyên trọng số, sinh các lời giải rồi lọc theo bộ test của policy. Curriculum được làm mới giữa hai vòng.

Thí nghiệm này dùng SFT64 làm parent. Lượt SFT256 + GRPO96 là thí nghiệm trên code cố định, được ghi riêng trong [bảng thực nghiệm](EXPERIMENTS.md).

## Log và checkpoint

| Tệp | Nội dung |
|---|---|
| `plan.json` | Cấu hình và ngân sách |
| `result.json` | Trạng thái, số liệu, hash và đường dẫn adapter |
| `predictions.jsonl` | Test sinh khi đánh giá |
| `rollouts.jsonl` | Nhóm test và reward trong GRPO |
| `sft-steps.jsonl` | Loss từng bước SFT |
| `adapter_model.safetensors` | Tham số LoRA đã lưu |
| `source/` | Mã nguồn tại thời điểm chạy |

`audit_apps_run.py <thư mục lượt chạy>` chấm lại các bản ghi đã lưu. TensorBoard đọc event log trong `evidence/tensorboard-apps`, không khởi chạy huấn luyện.
