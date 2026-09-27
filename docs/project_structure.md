# Project Structure

Tài liệu này mô tả vị trí và mục đích của các thư mục/file trong repository. Thành viên phải đọc trước khi tạo hoặc chỉnh sửa file.

## Nguyên tắc chính

- Không tự ý tạo thêm thư mục cấp cao hoặc file `.py` mới nếu chưa thống nhất với nhóm trưởng.
- Khi được giao task, chỉ sửa các file được chỉ định trong task.
- Các config chung do nhóm trưởng quản lý.
- Logic chính nằm trong `src/`.
- Notebook Colab chỉ dùng để thực thi, gọi code từ `src/`, quan sát và lưu kết quả.
- Notebook không được tạo sẵn trong repo; khi task yêu cầu, thành viên tạo trên Colab và lưu về đúng tên/path được quy định.
- Báo cáo công việc cá nhân chỉ lưu trong thư mục tương ứng dưới `docs/task_reports/`.

## Cấu trúc

### `configs/`
Cấu hình chung và cấu hình E0-E9. Không tự tạo config mới nếu chưa được thống nhất.

### `data/`
- `raw/`: dữ liệu gốc, không commit dataset lớn.
- `processed/`: dữ liệu trung gian nếu cần.
- `splits/`: Train/Validation/Calibration và thông tin Official Test đã khóa.
- `metadata/`: metadata dùng chung, đặc biệt `class_mapping.json`.

### `notebooks/`
- `exploration/`: notebook audit/khám phá dữ liệu.
- `training/`: notebook chạy training trên Colab.
- `evaluation/`: notebook chạy các bước đánh giá.
- `demo/`: notebook thử nghiệm demo.

Tên notebook cụ thể được quy định trong từng task.

### `src/`
Source code chính, được chia theo chức năng: data, models, training, evaluation, calibration, selective prediction, robustness, explainability, inference và utils.

### `scripts/`
Entry point để chạy pipeline. Không đặt logic lớn trong scripts; logic thật nằm trong `src/`.

### `outputs/`
Kết quả chạy gồm checkpoints, logs, metrics, predictions và figures.

### `app/`
Ứng dụng Gradio cuối kỳ.

### `tests/`
Test cho các module dùng chung và model.

### `docs/`
- `proposal/`: đề cương chính thức.
- `protocol/`: quy định kỹ thuật chung.
- `pipeline/`: pipeline tổng thể E0-E9.
- `team_tasks/`: file giao việc theo ngày, ví dụ `27_09_task.md`.
- `task_reports/`: báo cáo của từng thành viên sau khi hoàn thành công việc.
