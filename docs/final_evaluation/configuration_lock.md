# DogBreedID — BIÊN BẢN KHÓA CẤU HÌNH ĐÁNH GIÁ CUỐI CÙNG

**Trạng thái:** BẢN NHÁP — CHƯA KHÓA CẤU HÌNH

**Ngày chuẩn bị:** 19/10/2026

**Dự án:** DogBreedID — Nhận diện 30 giống chó bằng thị giác máy tính

**Repository:** `DGiang1126/dogbreedid-cv-project`


---

## 1. Mục đích

Tài liệu này ghi nhận các thành phần và tham số đã được lựa chọn cho hệ thống DogBreedID trước khi thực hiện đánh giá cuối cùng trên tập Official Test.

Các nội dung bao gồm:

- Mô hình phân loại và checkpoint được lựa chọn.
- Cấu hình dữ liệu và tiền xử lý ảnh.
- Tham số hiệu chỉnh xác suất dự đoán E6.
- Ngưỡng chấp nhận hoặc từ chối dự đoán E7.
- Các tham số đánh giá độ bền vững E8.
- Quy định về khả năng tái lập thực nghiệm và kiểm soát dữ liệu.

**Nghiêm cấm sử dụng Official Test cho đến khi tài liệu được kiểm tra, phê duyệt và chuyển sang trạng thái ĐÃ KHÓA CẤU HÌNH.**

## 2. Mô hình phân loại được lựa chọn

| Thành phần                    | Giá trị                                                            |
| ----------------------------- | ------------------------------------------------------------------ |
| Experiment                    | E4 — MobileNetV2 Frozen + CutMix                                   |
| Kiến trúc mô hình             | MobileNetV2                                                        |
| Random seed                   | 42                                                                 |
| Số lớp phân loại              | 30                                                                 |
| Kích thước ảnh đầu vào        | 224 × 224                                                          |
| Epoch lưu checkpoint tốt nhất | 25                                                                 |
| Đường dẫn checkpoint          | `outputs/checkpoints/E4/seed_42/best.pt`                           |
| SHA-256 checkpoint            | `73df0cc1c5b16866d4facc910f3e4d3ab481e3f12f1681a59fa7d61e3845c47b` |

### 2.1. Kết quả xác minh checkpoint

Đã thực hiện kiểm tra checkpoint trên máy của trưởng nhóm:

- Nạp checkpoint bằng `torch.load(..., weights_only=True)`: **ĐẠT**.
- Nạp trọng số bằng `load_state_dict(..., strict=True)`: **ĐẠT**.
- Số khóa trọng số bị thiếu: **0**.
- Số khóa trọng số không mong đợi: **0**.
- Số lớp đầu ra của tầng phân loại: **30**.
- Metadata ghi nhận: E4, seed 42, sử dụng CutMix.
- Đối chiếu SHA-256 với checkpoint E4/seed_42/best.pt tải lại từ Google Drive chung: **ĐẠT — HASH TRÙNG KHỚP**.

Các kiểm tra trên xác nhận checkpoint có cấu trúc và trọng số tương thích với MobileNetV2 30 lớp. Chưa xem đây là bằng chứng xác minh độc lập toàn bộ quá trình huấn luyện trước đó.

### 2.2. Kết quả trên tập Validation được lưu trong checkpoint

| Chỉ số          |            Giá trị |
| --------------- | -----------------: |
| Macro-F1        |  0.815099216821925 |
| Top-1 Accuracy  | 0.8166666626930237 |
| Top-5 Accuracy  | 0.9866666793823242 |
| Validation Loss | 0.6620433596769969 |

Đây là các kết quả trên tập **Validation**, không phải kết quả đánh giá cuối cùng trên Official Test.

## 3. Dữ liệu và quy trình tiền xử lý

### 3.1. Bộ dữ liệu

- Bộ dữ liệu: **Stanford Dogs**.
- Phạm vi: **30 giống chó cố định**.
- Ánh xạ nhãn: `data/metadata/class_mapping.json`.
- Các tập dữ liệu sử dụng: Train, Validation, Calibration và Official Test.
- Số lượng ảnh Official Test dự kiến: **1.934 ảnh**.
- Các tập dữ liệu được chia theo giao thức đã thống nhất và không được thay đổi.

### 3.2. Quy trình tiền xử lý khi đánh giá

```text
Ảnh đầu vào
    ↓
Chuyển sang RGB
    ↓
Resize(256)
    ↓
CenterCrop(224)
    ↓
ToTensor
    ↓
ImageNet Normalization
    ↓
Mô hình E4
```

Tham số ImageNet Normalization:

- Mean: `[0.485, 0.456, 0.406]`
- Standard Deviation: `[0.229, 0.224, 0.225]`

Không được tự ý thay đổi cách chia dữ liệu, ánh xạ nhãn hoặc quy trình tiền xử lý đã được phê duyệt.

## 4. E6 — Hiệu chỉnh độ tin cậy bằng Temperature Scaling

| Thành phần                          | Giá trị                               |
| ----------------------------------- | ------------------------------------- |
| Phương pháp                         | Scalar Temperature Scaling            |
| Temperature tối ưu                  | `0.80246283961825`                    |
| Tập dữ liệu dùng để tìm Temperature | Calibration                           |
| File kết quả                        | `outputs/metrics/E6/temperature.json` |

Temperature Scaling điều chỉnh xác suất dự đoán của mô hình mà không thay đổi trọng số hoặc thứ tự các lớp được dự đoán.

Giá trị Temperature đã được lựa chọn từ tập Calibration và **không được phép tính lại hoặc điều chỉnh dựa trên Official Test**.

## 5. E7 — Cơ chế chấp nhận và từ chối dự đoán

| Thành phần                       | Giá trị                                            |
| -------------------------------- | -------------------------------------------------- |
| Confidence sử dụng               | Xác suất Softmax lớn nhất sau Temperature Scaling  |
| Target Coverage trên Calibration | 0.90                                               |
| Ngưỡng Confidence                | `0.5011672918550012`                               |
| Điều kiện ACCEPT                 | `confidence >= threshold`                          |
| Quy tắc xử lý bằng ngưỡng        | Chấp nhận toàn bộ mẫu có confidence bằng threshold |
| File kết quả                     | `outputs/metrics/E7/threshold.json`                |

Cơ chế Selective Prediction hoạt động như sau:

- **ACCEPT:** Mô hình đủ tự tin để đưa ra kết quả phân loại.
- **REJECT:** Mô hình chưa đủ tự tin để đưa ra kết luận.

REJECT không có nghĩa ảnh thuộc một giống chó chưa từng được học.

Threshold đã được lựa chọn trên tập Calibration. Coverage thực tế trên Official Test có thể khác kết quả Calibration.

**Không được thay đổi threshold dựa trên kết quả Official Test nhằm đạt một mức Coverage mong muốn.**

## 6. E8 — Đánh giá độ bền vững của mô hình

### 6.1. Các phép biến đổi ảnh

Các tham số thực nghiệm được sử dụng theo cấu hình E8 đã phê duyệt.

| Phép biến đổi  | Tham số cố định                                           |
| -------------- | --------------------------------------------------------- |
| Gaussian Blur  | Kernel 7 × 7; sigma = 1.5                                 |
| Gaussian Noise | Mean = 0.0; std = 0.05; clamp [0,1]                       |
| Occlusion      | Hình vuông; cạnh bằng 30% cạnh ảnh; màu RGB [0.5,0.5,0.5] |

### 6.2. Quy định thực hiện

- Global seed: **42**.
- Gaussian Noise và Occlusion sử dụng seed xác định theo ID cố định của từng ảnh.
- Corruption được áp dụng trên tensor RGB có pixel thuộc `[0,1]`.
- Corruption được thực hiện trước ImageNet Normalization.
- Các điều kiện đánh giá phải sử dụng cùng checkpoint, tập ảnh và nhãn.
- Không cập nhật trọng số mô hình khi đánh giá.

### 6.3. Chỉ số đánh giá

Công thức:

**Accuracy Drop = Clean Accuracy − Corrupted Accuracy**

Trong đó:

- Clean Accuracy: Độ chính xác trên ảnh gốc.
- Corrupted Accuracy: Độ chính xác trên ảnh đã được làm mờ, thêm nhiễu hoặc che khuất.
- Accuracy Drop: Mức thay đổi độ chính xác giữa hai điều kiện.

Accuracy Drop là hiệu có dấu và có thể âm nếu độ chính xác trên ảnh bị biến đổi cao hơn ảnh sạch.

**Trạng thái E8 Official Evaluation: CHƯA THỰC HIỆN.**

## 7. Quản lý mã nguồn và khả năng tái lập thực nghiệm

| Hạng mục                                                | Trạng thái      |
| ------------------------------------------------------- | --------------- |
| Git commit nền sau Task 03                              | `77612d7`       |
| SHA-256 checkpoint                                      | Đã ghi nhận     |
| Git commit dùng cho Final Evaluation                    | CHƯA XÁC ĐỊNH   |
| Môi trường Python và thư viện dùng cho Final Evaluation | CHƯA XÁC NHẬN   |
| Kiểm tra tính toàn vẹn của dữ liệu và artifacts         | CHƯA HOÀN THÀNH |
| Đối chiếu checkpoint với Google Drive                   | ĐÃ HOÀN THÀNH — SHA-256 TRÙNG KHỚP |
| Kiểm thử tích hợp các runner Final Evaluation           | CHƯA HOÀN THÀNH |

Git commit `77612d7` xác định phiên bản repository sau khi tích hợp Task 03. Đây không mặc nhiên là commit sẽ được sử dụng để chạy Final Evaluation.

Khi đánh giá chính thức, phải ghi nhận đầy đủ:

- Git commit thực sự được sử dụng.
- SHA-256 của checkpoint.
- Phiên bản Python, PyTorch và các thư viện quan trọng.
- CPU/GPU và môi trường thực nghiệm.
- Cấu hình và lệnh thực thi.
- Các chỉ số, dữ liệu dự đoán và thời gian chạy.

## 8. Quy định bắt buộc trước và trong Final Evaluation

1. Không huấn luyện lại các mô hình E0–E5.
2. Không thay đổi mô hình E4 hoặc checkpoint đã được lựa chọn.
3. Không thay đổi dataset splits, class mapping và preprocessing.
4. Không tính lại hoặc điều chỉnh Temperature E6.
5. Không chọn lại threshold E7.
6. Không thay đổi các mức độ corruption E8.
7. Không dùng Official Test để lựa chọn mô hình hoặc tối ưu hyperparameters.
8. Ghi nhận đầy đủ Git SHA, môi trường chạy, checkpoint SHA-256, lệnh chạy và kết quả.
9. Lưu raw predictions để có thể tính lại các metrics mà không cần chạy inference lại.
10. Không đưa dataset gốc hoặc checkpoint `.pt` lên GitHub.

Mọi thay đổi có ảnh hưởng đến giao thức thực nghiệm phải được xem xét và phê duyệt trước khi thực hiện Final Evaluation.

## 9. Danh sách kiểm tra trước khi phê duyệt

### 9.1. Hạng mục đã hoàn thành

- [x] Checkpoint E4 tồn tại và nạp được với `weights_only=True`.
- [x] Trọng số checkpoint E4 nạp thành công vào MobileNetV2 30 lớp bằng `strict=True`.
- [x] Temperature E6 khớp với artifact đã nghiệm thu.
- [x] Threshold E7 khớp với artifact đã nghiệm thu.
- [x] Các tham số corruption E8 khớp với YAML đã phê duyệt.
- [x] SHA-256 checkpoint được đối chiếu với bản tải lại từ Google Drive chung.
- [x] Hợp đồng dữ liệu được trưởng nhóm chốt làm quy ước triển khai Task 04.

### 9.2. Hạng mục chưa hoàn thành
- [ ] Hợp đồng dữ liệu được kiểm chứng bằng integration tests và nghiệm thu sau Task 04.
- [ ] Mã nguồn Final Evaluation và các thư viện phụ thuộc được kiểm tra.
- [ ] Các kiểm thử tích hợp Final Evaluation đạt yêu cầu.
- [ ] Trưởng nhóm phê duyệt Configuration Lock.

**Quyết định hiện tại: CHƯA KHÓA CẤU HÌNH.**

**Official Test tiếp tục bị khóa và chưa được phép sử dụng.**

## 10. Phê duyệt khóa cấu hình

| Hạng mục                                   | Trạng thái    |
| ------------------------------------------ | ------------- |
| Trạng thái phê duyệt                       | CHỜ PHÊ DUYỆT |
| Ngày phê duyệt                             | CHƯA XÁC ĐỊNH |
| Git commit Final Evaluation được phê duyệt | CHƯA XÁC ĐỊNH |
| Người phê duyệt                            | Trưởng nhóm   |
| Cho phép sử dụng Official Test             | CHƯA CHO PHÉP |

Tài liệu chỉ được chuyển sang trạng thái **ĐÃ KHÓA CẤU HÌNH (LOCKED)** khi toàn bộ các điều kiện cần thiết đã hoàn thành và được trưởng nhóm xác nhận.
