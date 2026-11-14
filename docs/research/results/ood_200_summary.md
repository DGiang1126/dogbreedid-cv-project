# OOD-200 — Unknown Breed & Non-Dog Detection (Task 07)

Người thực hiện: Nguyễn Anh Kiệt · Ngày: 14/11/2026 · Branch: `research/ood-200-detection`

> Đây là thí nghiệm nghiên cứu bổ sung (research). Không thay đổi E4/E6/E7, class mapping, các split
> Train/Validation/Calibration hay kết quả Official. Không truy cập Official Test.
> OOD-200 không được dùng để huấn luyện, chọn ngưỡng hay tinh chỉnh bộ phát hiện.

## 1. Bộ dữ liệu OOD-200 (nội bộ, không public)

| Nhóm | Số ảnh | Nội dung |
|---|---:|---|
| unknown_breed | 100 | 10 giống ngoài 30 lớp × 10 ảnh: Samoyed, Bernese Mountain Dog, Flat-Coated Retriever, Australian Shepherd, Cairn Terrier, Dobermann, Dalmatian, Pug, Chihuahua, French Bulldog |
| human | 30 | Chân dung, toàn thân, nhóm người, hoạt động ngoài trời — không có chó |
| cat | 30 | Nhiều giống/màu lông/tư thế |
| other_animal | 20 | Sói, cáo, gấu, ngựa, cừu, … (có cả near-OOD giống chó như sói, cáo) |
| object_scene | 20 | Đồ vật, nội thất, phong cảnh, đường phố |

- Nguồn: Wikimedia Commons, chỉ giấy phép tự do (CC0, Public domain, CC BY, CC BY-SA). Lưu nguồn/giấy phép/tác giả trong manifest nội bộ.
- Data audit (`scripts/audit_ood_200.py`): **PASS, 0 lỗi** — đủ 200 ảnh và đúng số lượng từng nhóm; đọc được mọi ảnh; SHA-256 và pHash khớp manifest; không trùng chính xác / gần trùng (pHash Hamming ≤ 6) trong OOD-200 và so với 600 ảnh Validation + Calibration; nhãn unknown_breed không thuộc 30 lớp; mọi ảnh nhóm không-chó đã được rà soát bằng mắt (một ảnh có chó bị phát hiện và thay thế trước khi khóa).
- Khóa bộ dữ liệu: `LOCK.json` lúc `20261114T055427Z`, manifest SHA-256 `b47a560c…bf8f2`. Runner đánh giá từ chối chạy nếu manifest thay đổi sau khi khóa.
- Không public ảnh, ảnh người, manifest chi tiết hay URL cấp ảnh.

## 2. Phương pháp

- Mô hình: E4 (MobileNetV2 Frozen + CutMix, seed 42), checkpoint SHA-256 `73df0cc1…5c47b`, giữ nguyên.
- Energy score trên 30 logits thô của E4: `E(x) = −T · logsumexp(z / T)`, **T = 1.0** chọn trước (theo Liu et al., 2020), độc lập với E6 (T = 0.8025) và không tinh chỉnh.
- Ngưỡng chọn **chỉ trên Calibration (300 ảnh)** ở mức chấp nhận ID ≥ 90%: lấy giá trị nhỏ thứ k, k = ⌈0.9·N⌉ → **threshold = −0.680118**, đạt đúng 270/300 = 90.0%.
- Luật quyết định: `OOD_REJECT` nếu `energy > threshold`; bằng ngưỡng thì cho qua.
- Artifact khóa: `outputs/metrics/OOD/energy_threshold.json` (SHA-256 `55b711ac…6645e`, `ood_200_used: false`).
- Ba phương pháp được so sánh:
  - **E7 Baseline**: chấp nhận khi confidence đã hiệu chỉnh ≥ 0.50117.
  - **Energy only**: chấp nhận khi qua kiểm tra Energy.
  - **Pipeline (tích hợp vào Gradio)**: kiểm tra Energy trước, sau đó áp E7 → `OOD_REJECT` / `E7_REJECT` / `ACCEPT`.
- AUROC: OOD là lớp dương. Điểm số dùng `1 − confidence` cho E7 và `energy` cho Energy.

## 3. Kết quả chính — CenterCrop (preprocessing chuẩn E4)

ID = 300 ảnh Validation (Top-1 toàn bộ 81.67%), OOD = OOD-200.

| Phương pháp | ID Coverage | ID Selective Acc. | ID Selective Risk | OOD Rejection | OOD False Accept |
|---|---:|---:|---:|---:|---:|
| E7 Baseline | 87.00% | 84.29% | 15.71% | 60.50% | 39.50% |
| Energy only | 88.33% | 84.15% | 15.85% | 70.00% | 30.00% |
| **Pipeline (Energy → E7)** | **83.00%** | **85.14%** | **14.86%** | **78.50%** | **21.50%** |

| AUROC (OOD vs ID) | Giá trị |
|---|---:|
| E7 confidence (1 − conf) | 0.8487 |
| Energy score | **0.8948** |

### Tỷ lệ từ chối theo nhóm (CenterCrop)

| Nhóm | n | E7 Baseline | Energy only | Pipeline |
|---|---:|---:|---:|---:|
| unknown_breed | 100 | 48.0% | 57.0% | 67.0% |
| human | 30 | 90.0% | 93.3% | 96.7% |
| cat | 30 | 63.3% | 63.3% | 76.7% |
| other_animal | 20 | 60.0% | 90.0% | 100.0% |
| object_scene | 20 | 75.0% | 90.0% | 90.0% |

## 4. Kết quả bổ sung — Padding (chế độ "Giữ trọn ảnh" của Gradio)

Dùng **cùng ngưỡng** chọn trên Calibration CenterCrop, không chọn lại. ID Top-1 toàn bộ 74.00% (khớp báo cáo E4 CenterCrop vs Padding).

| Phương pháp | ID Coverage | ID Selective Acc. | ID Selective Risk | OOD Rejection | OOD False Accept |
|---|---:|---:|---:|---:|---:|
| E7 Baseline | 84.33% | 81.03% | 18.97% | 58.50% | 41.50% |
| Energy only | 82.33% | 80.16% | 19.84% | 70.00% | 30.00% |
| Pipeline | 77.00% | 82.68% | 17.32% | 77.50% | 22.50% |

AUROC: E7 confidence 0.8079, Energy 0.8482. Theo nhóm (Pipeline): unknown_breed 63%, human 100%, cat 86.7%, other_animal 80%, object_scene 100%.

**Lưu ý:** Ở chế độ Padding, Energy chỉ giữ được 82.3% ảnh ID, thấp hơn mục tiêu 90% vì ngưỡng được chọn với CenterCrop. Coverage của Pipeline giảm còn 77%.

## 5. Nhận xét

1. Energy tách OOD tốt hơn confidence E7 (AUROC 0.895 so với 0.849). Kết hợp Energy với E7 tăng tỷ lệ từ chối OOD từ 60.5% lên 78.5%. Độ chính xác trên ảnh ID được chấp nhận được giữ nguyên: 84.3% → 85.1%, mức chênh nằm trong nhiễu của 300 ảnh nên không coi là cải thiện. Đổi lại, coverage ID giảm 4 điểm phần trăm (87% → 83%, tức thêm 12 ảnh chó hợp lệ bị từ chối).
2. Người, động vật khác và đồ vật/cảnh vật bị từ chối tốt (90–100%). Điểm yếu là **near-OOD**: chó giống lạ chỉ bị từ chối 67%, mèo 77%. Ví dụ ảnh Dobermann bị gán thành Gordon Setter, ảnh Australian Shepherd thành Shetland Sheepdog, ảnh mèo thành Pekinese. Lý do là các ảnh này có đặc trưng rất gần 30 lớp, nên mô hình vẫn cho logits lớn.
3. Đây là bộ phát hiện post-hoc, không cần huấn luyện. Muốn cải thiện near-OOD cần các hướng khác như outlier exposure hoặc phương pháp dựa trên feature (Mahalanobis/KNN). Những hướng này nằm ngoài phạm vi Task 07.

## 6. Tích hợp Gradio

- Bộ phát hiện chạy **tự động** mỗi lần bấm Dự đoán, không có công tắc bật/tắt. Ngưỡng nạp từ artifact đã khóa và được kiểm tra metadata. Nếu bị sửa (ngưỡng khác, split khác Calibration, `ood_200_used`, sai checkpoint hoặc E6/E7) thì app báo lỗi và không chạy.
- **OOD REJECT**: ẩn Top-1, độ tin cậy, Top-5 và chặn Grad-CAM. Hiển thị thông báo: *"Ảnh có thể nằm ngoài phạm vi 30 giống chó được hỗ trợ. Hệ thống không đưa ra kết luận phân loại cho ảnh này."*
- **E7 REJECT** giữ nguyên ý nghĩa cũ: ảnh đã qua kiểm tra OOD nhưng dự đoán chưa đủ tin cậy, Top-5 vẫn hiển thị để tham khảo. **ACCEPT** giữ nguyên.
- Kiểm thử thật qua HTTP Gradio API với checkpoint thật, 6 nhóm × 2 chế độ:

  | Nhóm | CenterCrop | Padding |
  |---|---|---|
  | Chó thuộc 30 lớp (Eskimo Dog) | ACCEPT, đúng giống | ACCEPT, đúng giống |
  | Giống lạ | ACCEPT (sai giống) | ACCEPT (sai giống) |
  | Người | OOD REJECT | OOD REJECT |
  | Mèo | ACCEPT (sai) | OOD REJECT |
  | Động vật khác | OOD REJECT | OOD REJECT |
  | Đồ vật/cảnh | OOD REJECT | E7 REJECT |

  Kết quả phù hợp với số liệu tổng hợp ở trên, kể cả điểm yếu near-OOD.

## 7. Tái lập

```bash
python scripts/audit_ood_200.py --root data/ood_200 --images-dir data/raw/stanford_dogs/Images
python scripts/evaluate_ood_200.py select-threshold --images-dir data/raw/stanford_dogs/Images --device cpu
python scripts/evaluate_ood_200.py evaluate --images-dir data/raw/stanford_dogs/Images --device cpu
python -m pytest tests/test_ood_detector.py tests/test_ood_evaluation.py tests/test_gradio_app.py -q
```

Ghi chú tái lập: chạy trên CPU (PyTorch 2.14.0+cpu), số liệu Validation khớp chính xác kết quả chính thức của E7 (coverage 87.00%, selective accuracy 84.29%). Riêng trên Calibration, E7 chấp nhận 269/300 thay vì 271/300 như bản chính thức. Nguyên nhân là 2 ảnh có confidence 0.5004, sát ngưỡng 0.50117, bị lệch số học nhỏ giữa CPU/GPU và phiên bản thư viện. Ngưỡng E7 không bị thay đổi.
