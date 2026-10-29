# Lượt tăng SFT dừng ở kiểm tra replay

Đã dừng có chủ đích sau 116 bước, không hoàn tất ngân sách256. Checkpoint64 lệch relative L2 = 0.002685916 so với SFT64 lịch sử, vượt ngưỡng0,001 ghi trước. Không nới ngưỡng để gọi lượt này thành công.

Nhãn Train, thứ tự ví dụ và các mã nguồn dùng chung đều trùng hash. Loss bước đầu giống nhau, bắt đầu lệch từ bước2; chênh lệch loss lớn nhất64 bước đầu là 0.007842. Nguyên nhân số học cụ thể chưa được xác định. Không coi đây là bằng chứng model kém đi hoặc bộ chấm sai.

Không chạy Train probe, GRPO mới, Validation hay Test trong lượt này. Phần sau checkpoint64 chỉ có log cập nhật, không có adapter cuối256. Giữ checkpoint64, log, kế hoạch, STOP_REQUEST và bản trạng thái lỗi ban đầu trong evidence.

Phép thử kế tiếp dùng checkpoint64 và256 của cùng một quỹ đạo huấn luyện mới làm cặp đối chứng, không phụ thuộc việc replay chính xác adapter lịch sử. Đây là kế hoạch mới được ghi riêng trước khi chạy; không tái sử dụng bước116 như ứng viên256 và không sửa trạng thái lượt dừng này thành thành công.
