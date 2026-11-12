# DogBreedID — So sánh CenterCrop và Padding trên tập Validation (E4, seed 42)

## 1. Mục tiêu

Đánh giá ảnh hưởng của **phương pháp tiền xử lý ảnh khi suy luận** đối với hiệu quả phân loại giống chó của **cùng một checkpoint E4 MobileNetV2 Frozen + CutMix (seed 42)**. Hai chế độ được so sánh trên **cùng 300 ảnh Validation thuộc 30 lớp** (10 ảnh/lớp), giữ nguyên mô hình, trọng số, thứ tự ảnh, nhãn và các tham số E6/E7.

Đây là phép đối chứng **CenterCrop so với Padding tại thời điểm suy luận**, **không huấn luyện lại mô hình** và **không sử dụng Official Test**.

## 2. Thiết kế thực nghiệm

| Thành phần | Giá trị cố định / cách thực hiện |
| --- | --- |
| Checkpoint | E4 MobileNetV2 Frozen + CutMix, seed 42 |
| SHA-256 checkpoint | `73df0cc1c5b16866d4facc910f3e4d3ab481e3f12f1681a59fa7d61e3845c47b` |
| Tập dữ liệu | `data/splits/validation.csv`, 300 ảnh, 30 lớp |
| Chạy suy luận | CUDA, batch size 8, cùng checkpoint cho hai chế độ |
| CenterCrop | RGB → Resize(256, giữ tỷ lệ; cạnh ngắn bằng 256) → CenterCrop(224) → ToTensor → ImageNet Normalize |
| Padding (`fit`) | RGB → resize bicubic giữ tỷ lệ sao cho cạnh dài bằng 224 → đặt giữa canvas 256×256 màu RGB `(124, 116, 104)` → dùng lại Resize(256) → CenterCrop(224) → ToTensor → ImageNet Normalize |
| Temperature E6 | Giữ nguyên `0.80246283961825`; không hiệu chỉnh lại |
| Ngưỡng E7 | Giữ nguyên `0.5011672918550012`; không chọn lại |
| Mã thực hiện | Script nghiên cứu local `scripts/compare_center_crop_padding_validation.py`; sử dụng hàm `prepare_image(..., mode="fit")` của ứng dụng Gradio |
| Run ID | `e4-crop-padding-val-20261111-01` |

**Lưu ý về Padding:** Đây là **chế độ `fit` hiện có trong ứng dụng**, không phải một biến thể padding được thiết kế mới. Ảnh được đặt vào vùng giữa 224×224 của canvas 256×256 để CenterCrop chuẩn sau đó giữ lại đầy đủ nội dung ảnh đã thu nhỏ.

## 3. Kết quả định lượng

| Chỉ số | CenterCrop | Padding | Nhận xét |
| --- | ---: | ---: | --- |
| Top-1 Accuracy | **81,67%** (245/300) | 74,00% (222/300) | Padding −7,67 điểm phần trăm |
| Top-5 Accuracy | **98,67%** (296/300) | 98,33% (295/300) | Padding −0,33 điểm phần trăm |
| Macro-F1 | **0,815099** | 0,735005 | Padding −0,080094 |
| ECE sau E6 cố định | 0,079632 | **0,058136** | Padding có ECE thấp hơn; không đồng nghĩa có độ chính xác cao hơn |
| Số ảnh E7 chấp nhận | 261 | 253 | Padding ít hơn 8 ảnh |
| Coverage E7 | **87,00%** | 84,33% | Padding −2,67 điểm phần trăm |
| Selective Accuracy E7 | **84,29%** | 81,03% | Padding −3,26 điểm phần trăm |

**Kiểm tra baseline:** CenterCrop tái hiện đúng kết quả E4 Validation đã biết: Top-1 **245/300**, Top-5 **296/300**, Macro-F1 **0,815099216821925**.

## 4. Đối chiếu ghép cặp và kiểm định thống kê

| Kết quả hai chế độ trên cùng ảnh | Số ảnh |
| --- | ---: |
| Cả hai đúng | 214 |
| Chỉ CenterCrop đúng | 31 |
| Chỉ Padding đúng | 8 |
| Cả hai sai, dự đoán cùng lớp | 38 |
| Cả hai sai, dự đoán khác lớp | 9 |
| **Tổng** | **300** |

- Có **39 ảnh bất đồng về đúng/sai**, trong đó CenterCrop đúng riêng 31 ảnh và Padding đúng riêng 8 ảnh.
- Chênh lệch Top-1 (Padding − CenterCrop): **−7,6667 điểm phần trăm**.
- **McNemar exact hai phía:** `p = 0,0002940768754342571`.
- **Paired bootstrap:** 20.000 lần lấy mẫu có hoàn lại, seed 42; khoảng tin cậy 95% cho chênh lệch Top-1: **[−11,67; −3,67] điểm phần trăm**.

Với ngưỡng ý nghĩa 0,05, có bằng chứng về **khác biệt Top-1 trên 300 ảnh Validation**, theo hướng CenterCrop đạt kết quả tốt hơn. Kiểm định chỉ phản ánh cặp cấu hình suy luận được đánh giá, không chứng minh CenterCrop ưu việt trong mọi tập dữ liệu hoặc mọi checkpoint.

## 5. Phân tích theo giống chó

Mỗi giống có **10 ảnh Validation**. Trong 30 giống, số ảnh dự đoán đúng của Padding **giảm ở 15 giống**, **không đổi ở 12 giống** và **tăng ở 3 giống** so với CenterCrop.

- Giảm nhiều nhất: **Siberian Husky** (5 → 1/10) và **Standard Schnauzer** (9 → 5/10), cùng giảm 4 ảnh đúng.
- Các giống giảm 2 ảnh đúng: **Standard Poodle**, **Miniature Poodle**, **English Setter**, **Norwich Terrier**, **Shih-Tzu**.
- Tăng 1 ảnh đúng: **Greater Swiss Mountain Dog** (7 → 8/10), **Miniature Schnauzer** (9 → 10/10), **Border Collie** (8 → 9/10).

Bảng số liệu đầy đủ của 30 giống nằm trong [`e4_center_crop_vs_padding_per_class.csv`](e4_center_crop_vs_padding_per_class.csv). Các chênh lệch theo lớp chỉ có tính chất **mô tả/thăm dò** vì mỗi lớp chỉ có 10 ảnh; chưa kiểm định riêng theo lớp.

## 6. Phân tích theo tỷ lệ khung hình

Phân nhóm theo `width / height` của ảnh gốc: ảnh dọc `< 0,8`; gần vuông `0,8–<1,25`; ảnh ngang `1,25–<1,8`; ảnh rất rộng `≥ 1,8`.

| Nhóm ảnh | Số ảnh | CenterCrop đúng | Padding đúng | Top-1 CenterCrop | Top-1 Padding | Chênh lệch Padding − CenterCrop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ảnh dọc | 67 | 49 | 45 | 73,13% | 67,16% | −5,97 điểm % |
| Gần vuông | 59 | 49 | 44 | 83,05% | 74,58% | −8,47 điểm % |
| Ảnh ngang | 173 | 146 | 132 | 84,39% | 76,30% | −8,09 điểm % |
| Ảnh rất rộng | 1 | 1 | 1 | 100,00% | 100,00% | 0,00 điểm % |

Padding giảm Top-1 ở cả ba nhóm có số lượng quan sát đáng kể. Nhóm **rất rộng chỉ có 1 ảnh**, không thể rút ra kết luận riêng cho nhóm này. Tỷ lệ khung hình tự nó không giải thích được nguyên nhân của các lỗi: còn cần xét vị trí, kích thước chủ thể và nội dung ảnh.

## 7. Kiểm chứng và giới hạn

- Run đã hoàn thành cả hai lượt suy luận, mỗi lượt **300 ảnh Validation**.
- Checkpoint E4 trùng SHA-256 đã khóa và nhánh CenterCrop tái hiện đúng baseline.
- Tệp `manifest.json` và các artifact đầu ra đã được người thực thi kiểm tra lại bằng SHA-256, kết quả **`VALIDATION ARTIFACT HASH CHECK: PASS`**.
- Đã tạo **39 hình đối chiếu trực quan** (31 CenterCrop thắng, 8 Padding thắng) để lưu nội bộ; số lượng này **không có nghĩa đã hoàn thành việc phân loại nguyên nhân bằng mắt**.
- Chưa đánh giá lại trên seed/checkpoint khác; không diễn giải quan hệ nhân quả của kích thước chủ thể, hậu cảnh hay thông tin bị cắt khi chưa có mã hóa quan sát.
- ECE và E7 của Padding chỉ được tính với **Temperature/Threshold cố định** vốn chọn theo pipeline chuẩn; không tái hiệu chỉnh hoặc tối ưu trên Validation.
- Không dùng kết quả Validation này để thay thế kết quả **Official Test** của E4 đã khóa. Không dùng Validation nhiều lần để chọn tinh chỉnh Padding rồi báo cáo cải thiện như một ước lượng độc lập.

**Phạm vi công bố:** Chỉ xuất bản báo cáo và số liệu **tổng hợp**. Raw predictions (`.npz`), `paired_predictions.csv`, ảnh Validation gốc, 39 hình so sánh và các artifact provenance nội bộ được giữ ngoài repository Public.

## 8. Kết luận

Trên **300 ảnh Validation**, với **cùng E4 MobileNetV2 Frozen + CutMix seed 42**, chế độ **CenterCrop** đạt **81,67% Top-1**, cao hơn **Padding 74,00%**. Chênh lệch Padding − CenterCrop bằng **−7,67 điểm phần trăm**, có ý nghĩa thống kê theo McNemar exact (`p = 0,000294`) và khoảng tin cậy bootstrap ghép cặp 95% không chứa 0.

Padding giúp 8 ảnh được sửa đúng so với CenterCrop, nhưng làm 31 ảnh đang đúng trở thành sai. Do đó, **tiếp tục giữ CenterCrop làm chế độ mặc định**, còn Padding là **tùy chọn thử nghiệm** cho các trường hợp muốn giữ trọn bố cục; không mô tả Padding là phương pháp làm tăng độ chính xác tổng thể.

Báo cáo này **kết thúc đối chứng CenterCrop–Padding trên Validation** ở phạm vi hiện tại. Không cần chạy lại inference hay mở rộng sang Official Test.
