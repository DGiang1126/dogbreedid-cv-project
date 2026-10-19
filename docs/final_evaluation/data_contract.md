# DOGBREEDID — HỢP ĐỒNG DỮ LIỆU FINAL EVALUATION

**Trạng thái:** ĐÃ CHỐT QUY ƯỚC TRIỂN KHAI TASK 04 — CHỜ KIỂM THỬ TÍCH HỢP

**Ngày chuẩn bị:** 19/10/2026

**Tài liệu liên quan:** `docs/final_evaluation/configuration_lock.md`

## 1. Mục đích

Thống nhất giao diện dữ liệu giữa các thành phần E4, E6, E7 và E8 trong giai đoạn Final Evaluation.

Mỗi thành viên phải sử dụng đúng các tham số và artifacts đã nghiệm thu, cùng quy ước định danh ảnh và class mapping. Configuration Lock tổng thể vẫn đang ở trạng thái bản nháp, chưa được phê duyệt.

Trong giai đoạn chuẩn bị, chỉ được sử dụng dữ liệu giả lập hoặc Validation. **Chưa được truy cập Official Test.**

## 2. Quy ước Image ID

Mỗi ảnh được định danh bằng giá trị `relative_path` trong file split CSV đã được phê duyệt.

Yêu cầu:

- `image_id` phải là chuỗi không rỗng và duy nhất trong tập đánh giá.
- Không sử dụng vị trí trong batch hoặc chỉ số DataLoader làm Image ID.
- Mỗi `image_id` phải gắn với đúng một nhãn và một hàng logits.
- Cần đối chiếu Image ID và nhãn với split CSV tương ứng.
- Giữ nguyên biểu diễn của Image ID giữa các thành phần, không tự đổi dấu phân cách hoặc sửa đường dẫn.

## 3. Định dạng dữ liệu dự đoán

Thống nhất sử dụng định dạng NPZ tương tự E6.

Tên file dự kiến cho lần đánh giá chính thức:

`outputs/predictions/final/clean_predictions.npz`

| Trường      | Kích thước | Kiểu dữ liệu         |
| ----------- | ---------- | -------------------- |
| `logits`    | `[N, 30]`  | float32 hoặc float64 |
| `labels`    | `[N]`      | int64                |
| `image_ids` | `[N]`      | Chuỗi Unicode        |

Quy định:

1. Các hàng dữ liệu phải tương ứng theo cùng một ảnh.
2. Logits là đầu ra gốc, chưa áp dụng Temperature Scaling.
3. Labels thuộc khoảng 0–29.
4. Logits phải hữu hạn; Image IDs phải duy nhất.
5. Đọc NPZ bằng `allow_pickle=False`.
6. Không ghi đè kết quả chính thức đã được nghiệm thu.

Trong giai đoạn chuẩn bị, chỉ dùng NPZ giả lập hoặc Validation và phải đặt tên, lưu vị trí riêng để tránh nhầm với Official Test.

## 4. Bàn giao giữa các thành viên

### 4.1. Thuận — Final Classification & Calibration

Thuận chịu trách nhiệm xây dựng pipeline tạo raw predictions từ checkpoint E4 seed 42 đã được lựa chọn và xác minh SHA-256, để chuẩn bị cho Final Evaluation sau khi Configuration Lock được phê duyệt.

Đầu ra:

- Logits, labels và image_ids đúng định dạng.
- Classification metrics: Top-1 Accuracy, Top-5 Accuracy, Macro-F1.
- Calibration metrics: NLL, ECE và Brier Score trước/sau Temperature Scaling.
- Metadata: Git SHA, checkpoint SHA-256, split, runtime và preprocessing.

### 4.2. Kiệt — Final Selective Prediction

Kiệt sử dụng raw predictions do Thuận bàn giao.

Các bước:

1. Đọc logits, labels và image_ids.
2. Áp dụng Temperature E6 đã khóa.
3. Tính calibrated softmax probabilities.
4. Áp dụng Threshold E7 đã khóa với quy tắc ACCEPT `>=`.
5. Tính Coverage, Selective Accuracy, Selective Risk, Risk–Coverage Curve và AURC.

Không được chọn lại threshold dựa trên Official Test.

### 4.3. Đức Anh — Final Robustness

Đức Anh sử dụng cùng checkpoint E4, quy trình tiền xử lý và danh sách ảnh/nhãn.

Đánh giá bốn điều kiện:

- Clean.
- Gaussian Blur.
- Gaussian Noise.
- Occlusion.

Noise và Occlusion phải dùng seed xác định theo Image ID. Corruption thực hiện trước ImageNet Normalization.

Kết quả gồm Clean Accuracy, Corrupted Accuracy và Accuracy Drop.

Trong giai đoạn chuẩn bị, chỉ tích hợp và kiểm thử runner; chưa chạy Official Test.

## 5. Dữ liệu đầu vào và đầu ra cần kiểm thử

Các thành viên phải kiểm tra:

- Shape, dtype, giá trị NaN/Inf.
- Số lượng Image ID và tính duy nhất.
- Sự tương ứng giữa Image ID, label và logits.
- Khớp class mapping và split được phê duyệt.
- Tính tái lập khi thay đổi batch size hoặc thứ tự xử lý.
- Công thức tính metrics trên dữ liệu giả lập.
- Chặn Official Test khi Configuration Lock chưa được phê duyệt.

## 6. Quy định lưu kết quả

Mỗi lần thực nghiệm phải ghi nhận:

- Experiment ID và split.
- Git commit SHA.
- Checkpoint SHA-256.
- Phiên bản phần mềm và thiết bị.
- Cấu hình và lệnh chạy.
- Thời điểm thực hiện.
- Số lượng mẫu và class mapping.
- Kết quả metrics và raw predictions cần thiết.

Không đưa checkpoint hoặc dataset ảnh gốc lên GitHub.

## 7. Điều kiện nghiệm thu Data Contract

**Trạng thái quy ước:** ĐÃ ĐƯỢC TRƯỞNG NHÓM CHỐT ĐỂ TRIỂN KHAI TASK 04.

Các thành viên bắt buộc tuân thủ định dạng dữ liệu và quy tắc Image ID được quy định trong tài liệu này.

### 7.1. Phê duyệt quy ước triển khai

- [x] Trưởng nhóm thống nhất định dạng NPZ gồm `logits`, `labels`, `image_ids`.
- [x] Trưởng nhóm thống nhất sử dụng `relative_path` từ split CSV làm `image_id`.
- [x] Trưởng nhóm phê duyệt Data Contract làm quy ước triển khai Task 04.

### 7.2. Điều kiện nghiệm thu sau Task 04

- [ ] Thuận chứng minh có thể xuất đúng định dạng raw predictions.
- [ ] Kiệt chứng minh có thể đọc và tính E7 từ định dạng này.
- [ ] Đức Anh chứng minh sử dụng đúng Image IDs từ split CSV.
- [ ] Integration tests trên dữ liệu giả lập hoặc Validation đạt yêu cầu.
- [ ] Trưởng nhóm kiểm tra và nghiệm thu việc tuân thủ Data Contract.

**Trạng thái nghiệm thu: CHƯA HOÀN THÀNH — CHỜ KẾT QUẢ TASK 04.**

**Official Test chưa được phép sử dụng.**
