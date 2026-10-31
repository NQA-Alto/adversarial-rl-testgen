# Chạy và huấn luyện

Môi trường huấn luyện nằm ở `D:/DA/TTNT_De19_Assets/env`, model nền ở `models/Qwen3.5-2B` trong cùng thư mục tài nguyên. `requirements.txt` dành cho huấn luyện, `requirements-ui.txt` dành cho giao diện. Các lệnh hiện còn dùng đường dẫn của máy thực nghiệm.

## Giao diện

Chạy `MO_GIAO_DIEN.cmd`, mở [demo](http://127.0.0.1:7860/?__theme=light). Dùng `DONG_GIAO_DIEN.cmd` để dừng các dịch vụ.

| Tab | Chức năng |
|---|---|
| Demo | Xem đề, code và test đã lưu; sinh bốn test mới |
| Kết quả | Bảng Validation, Test và biểu đồ so sánh |
| Dữ liệu | Nguồn APPS, tập chia và miền input |
| Thực nghiệm | Kiểm tra dữ liệu, chạy GRPO, chấm adapter, xem log và TensorBoard |

Nút sinh test chỉ chạy suy luận. Tác vụ GRPO tạo adapter mới từ SFT đã chọn, với optimizer mới; không ghi đè checkpoint gốc. Kết quả, log và adapter được lưu riêng trong `evidence/`. Nút dừng giữ lại những tệp đã ghi xuống đĩa.

## Dữ liệu và SFT

Tập mở rộng dùng 58 bài Train và 1.050 nhãn SFT. Nhãn gồm input hợp lệ và expected tính từ lời giải tham chiếu; có cả test bắt lỗi và test đúng chưa bắt lỗi. SFT dùng bốn ví dụ mỗi bước, learning rate `5e-5`, seed 42, input trước expected. Bản được chọn đã học 840 bước.

```powershell
$env:PYTHONUTF8 = '1'
$env:TTNT_APPS_DATA_DIR = 'D:/DA/TTNT_De19_Assets/data/apps-introductory-expanded-v4'
$python = 'D:/DA/TTNT_De19_Assets/env/Scripts/python.exe'
$project = 'D:/DA/DoAn_TTNT_De19/adversarial-rl-testgen'
& $python -B "$project/scripts/verify_apps.py"
& $python -B "$project/scripts/apps_sft.py" --steps 840 --target-order input-first-v1
```

`prepare_apps_expanded.py` bổ sung các profile Train vào tập v2 và từ chối ghi đè tập đã có. Validation và Test được sao chép nguyên byte từ v2. Cần tải nguồn APPS và chuẩn bị v2 trước khi chạy bước mở rộng.

## GRPO

Mỗi nhóm sinh bốn test cho cùng một code. Cấu hình `equal-v1` cộng ba thành phần reward bằng nhau: định dạng/input hợp lệ, expected đúng và bắt được lỗi. Test có expected sai không được tính là bắt lỗi. Chỉ LoRA được cập nhật.

```powershell
& $python -B "$project/scripts/apps_model_runner.py" train `
  --steps 64 --seed 42 --prompt-version apps-call-v2 `
  --reward-profile equal-v1 --learning-rate 0.000005 `
  --init-adapter '<adapter SFT>' --baseline-run '<lượt Validation của adapter đó>' --no-publish

& $python -B "$project/scripts/apps_model_runner.py" evaluate `
  --adapter '<adapter cần chấm>' --seed 42 --prompt-version apps-call-v2 --no-publish
```

`--no-publish` chỉ ngăn thay trạng thái demo, không liên quan đến GitHub. Mỗi lần chạy vẫn lưu kết quả riêng.

## Lấy mẫu đối kháng cân bằng

`apps_balanced_study.py` điều phối hai vòng, mỗi vòng 64 bước cho từng nhánh. Pilot dùng seed train 42. Seed 43 chỉ chạy xác nhận khi pilot đạt điều kiện cải thiện đã chốt trước. Một nửa số nhóm lấy từ các vị trí có code đối kháng, một nửa từ phần Train còn lại. Nhánh cố định dùng đúng lịch case ID đó nhưng giữ code AST ban đầu. Trước vòng hai, policy kiểm tra đúng code đang học. Vị trí chưa bắt được lỗi giữ nguyên; chỉ vị trí đã bắt được mới tìm code thay thế.

```powershell
& $python -B "$project/scripts/apps_balanced_study.py" `
  --parent-study '<thư mục thực nghiệm SFT 840 và hai nhánh đã lưu>'
```

Lịch học có hash và được đối chiếu lại với từng nhóm thực tế sau huấn luyện. Bản này chỉ đánh giá Train/Validation, giữ nguyên sự kiện Test đã công bố và chưa tự thay checkpoint của demo.

`check_apps_sampling.py` kiểm tra lịch lấy mẫu với sampler thật của TRL. `audit_apps_run.py` chấm lại output đã lưu, kiểm tra reward, lịch học và hash adapter.
