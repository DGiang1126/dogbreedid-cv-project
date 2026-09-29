# Dataset Protocol — DogBreedID

Tài liệu này quy định dataset, danh sách lớp, class mapping và split dùng chung cho toàn bộ project **DogBreedID**.

## 1. Dataset nguồn

- Dataset: **Stanford Dogs**
- Core scope: **30 classes**
- Bài toán: **30-class single-label fine-grained image classification**
- Official Train và Official Test của Stanford Dogs được giữ theo 30 lớp đã chọn.
- Official Train được chia tiếp thành Train / Validation / Calibration.
- Official Test bị khóa trong giai đoạn phát triển.

## 2. Quy tắc chọn 30 giống

30 giống được chọn **trước khi huấn luyện** theo thiết kế:

- 10 nhóm hình thái × 3 giống.
- Trong mỗi nhóm ưu tiên các giống có mức tương đồng thị giác tương đối cao.
- Giữa các nhóm vẫn giữ sự đa dạng về hình thái.
- Các nhóm này là nhóm tương đồng thị giác do nhóm xây dựng để thiết kế subset, không phải taxonomy chính thức của Stanford Dogs.
- Không thêm, xóa hoặc thay lớp dựa trên kết quả model, Validation hoặc Official Test.

Mục tiêu là tạo một subset có độ khó fine-grained hợp lý thay vì chọn 30 lớp ngẫu nhiên hoặc chỉ chọn các giống dễ phân biệt.

## 3. Danh sách 30 giống chính thức

| Class ID | Breed | Visual Group |
|---:|---|---|
| 0 | Eskimo Dog | Northern Spitz |
| 1 | Malamute | Northern Spitz |
| 2 | Siberian Husky | Northern Spitz |
| 3 | Greater Swiss Mountain Dog | Swiss Mountain Dogs |
| 4 | Appenzeller | Swiss Mountain Dogs |
| 5 | EntleBucher | Swiss Mountain Dogs |
| 6 | Toy Poodle | Poodle |
| 7 | Miniature Poodle | Poodle |
| 8 | Standard Poodle | Poodle |
| 9 | Miniature Schnauzer | Schnauzer |
| 10 | Standard Schnauzer | Schnauzer |
| 11 | Giant Schnauzer | Schnauzer |
| 12 | English Setter | Setter |
| 13 | Irish Setter | Setter |
| 14 | Gordon Setter | Setter |
| 15 | English Springer | Spaniel |
| 16 | Welsh Springer Spaniel | Spaniel |
| 17 | Cocker Spaniel | Spaniel |
| 18 | Norfolk Terrier | Small Terrier |
| 19 | Norwich Terrier | Small Terrier |
| 20 | Border Terrier | Small Terrier |
| 21 | Collie | Collie-type |
| 22 | Shetland Sheepdog | Collie-type |
| 23 | Border Collie | Collie-type |
| 24 | Golden Retriever | Retriever |
| 25 | Labrador Retriever | Retriever |
| 26 | Chesapeake Bay Retriever | Retriever |
| 27 | Japanese Spaniel | Small Companion |
| 28 | Pekinese | Small Companion |
| 29 | Shih-Tzu | Small Companion |

Class ID 0–29 là mapping cố định của project. Tên class khi triển khai phải được đối chiếu với tên class/folder thực tế của Stanford Dogs.

## 4. Class Mapping

Mapping chính thức được lưu tại:

```text
data/metadata/class_mapping.json
```

Dataset, DataLoader, training, evaluation, calibration, robustness, inference và Gradio phải dùng cùng mapping này.

Không hard-code một thứ tự class khác trong notebook hoặc source code.

## 5. Data Split

Official Train được chia bằng **stratified split theo class**:

```text
Train       = 80%
Validation  = 10%
Calibration = 10%
split_seed  = 42
```

Vai trò:

- **Train:** cập nhật weights.
- **Validation:** chọn checkpoint, model và hyperparameter.
- **Calibration:** Temperature Scaling và confidence threshold.
- **Official Test:** final evaluation sau khi khóa configuration.

Split chỉ được sinh một lần và lưu trong:

```text
data/splits/
```

Sau khi split được kiểm tra và commit, mọi experiment phải dùng lại đúng split đó.

Official Test giữ nguyên split chính thức của Stanford Dogs đối với 30 lớp đã chọn và không được dùng để chọn epoch/checkpoint, learning rate, optimizer, scheduler, backbone, augmentation, CutMix, temperature hoặc confidence threshold.

## 6. Data Audit

Trước training chính thức E0–E3 phải hoàn thành:

- thống kê số ảnh theo class;
- thống kê số ảnh Official Train / Official Test;
- kiểm tra ảnh lỗi hoặc không đọc được;
- kiểm tra đường dẫn / file bất thường;
- kiểm tra class mapping;
- kiểm tra số lượng ảnh sau Train / Validation / Calibration split;
- kiểm tra split không overlap;
- quan sát kích thước và aspect ratio;
- xem một số ảnh đại diện của mỗi class.

Có thể ghi nhận thêm các trường hợp phục vụ Error Analysis: complex background, blur, occlusion, dog too small, unusual pose, poor lighting.

Data Audit không được dùng để loại một class chỉ vì class đó khó phân loại.

## 7. Data Leakage Rules

```text
Train ∩ Validation = ∅
Train ∩ Calibration = ∅
Train ∩ Official Test = ∅
Validation ∩ Calibration = ∅
Validation ∩ Official Test = ∅
Calibration ∩ Official Test = ∅
```

Mọi E0–E9 sử dụng cùng class mapping và fixed split.

## 8. Version Control

Các file cần version-controlled:

```text
data/metadata/class_mapping.json
data/splits/
configs/data.yaml
docs/protocol/dataset_protocol.md
```

Dataset ảnh gốc không push lên GitHub.

## 9. Dataset Freeze v1

**Status: COMPLETED**

Dataset chính thức của DogBreedID đã được chuẩn bị và kiểm tra với:

- Số classes: 30
- Split seed: 42
- Selected Official Train: 3000 images
- Train: 2400 images
- Validation: 300 images
- Calibration: 300 images
- Selected Official Test: 1934 images
- Tổng số ảnh được audit: 4934
- Missing/unreadable images: 0
- Data leakage giữa các split: 0

Mỗi class trong Official Train có:

```text
80 Train
10 Validation
10 Calibration
