# DogBreedID — BIÊN BẢN KHÓA CẤU HÌNH ĐÁNH GIÁ CUỐI CÙNG

**Trạng thái:** ĐÃ KHÓA CẤU HÌNH

**Ngày chuẩn bị:** 19/10/2026

**Ngày phê duyệt Configuration Lock:** 26/10/2026 — Trưởng nhóm DogBreedID. OFFICIAL EVALUATION AUTHORIZATION CHƯA ĐƯỢC CẤP.

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

**Configuration Lock đã được trưởng nhóm phê duyệt ngày 26/10/2026. Official Test vẫn bị khóa: nghiêm cấm truy cập cho đến khi có quyết định Official Evaluation Authorization riêng và gate thực thi hợp lệ.**

Configuration Lock Approval và Official Evaluation Authorization là hai quyết định độc lập. Phê duyệt Lock không tự cấp quyền chạy Official Test; mỗi lượt Official Evaluation còn cần authorization riêng, claim một lần và output reservation hợp lệ.

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

Bằng chứng bổ sung Task 05 do trưởng nhóm xác nhận: checkpoint E4 thật có SHA-256 khớp giá trị tại bảng trên; smoke test một synthetic tensor trên CUDA **PASS**, logits có shape `(1,30)` và mọi giá trị hữu hạn. Đây là kiểm tra checkpoint bằng dữ liệu giả, không phải inference Validation hoặc Official Test. Không chạy lại smoke test khi hoàn thiện tài liệu và records phê duyệt này.

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
| SHA-256 checkpoint                                      | ĐÃ XÁC MINH — bằng chứng Validation replay tại mục 7.1 và smoke của trưởng nhóm tại mục 2.1 |
| Source revision S được phê duyệt cho Official Final Evaluation | `02483f801277482fbe24c25048e6cc769a16295f` — ĐÃ PHÊ DUYỆT CẤU HÌNH, CHƯA CẤP QUYỀN THỰC THI |
| Runtime profile được phê duyệt cho Official Final Evaluation | ĐỦ 16 TRƯỜNG tại mục 7.4 — ĐÃ PHÊ DUYỆT |
| Tính toàn vẹn dữ liệu và artifacts cho Official Final Evaluation | Canonical reference đã được xác nhận tại mục 7.3; chưa truy cập dữ liệu Official thật. Kiểm tra dữ liệu thực tế chỉ sau authorization gate, claim và reservation |
| Đối chiếu checkpoint với Google Drive                   | ĐÃ HOÀN THÀNH — SHA-256 TRÙNG KHỚP |
| Kiểm thử tích hợp backend Final Evaluation | B2.2-B: 52 synthetic integration tests PASS; Final Source Review: READY FOR SOURCE FREEZE. Trưởng nhóm đã chấp nhận bằng chứng cho Lock Approval; chưa chạy Official Evaluation |

Git commit `77612d7` xác định phiên bản repository sau khi tích hợp Task 03. Đây không mặc nhiên là commit sẽ được sử dụng để chạy Final Evaluation.

### 7.1. Bằng chứng Task 05 — Validation Reproducibility

Đã đối chiếu báo cáo `docs/final_evaluation/task05_validation_reproducibility.md`, file `task05_reproducibility_evidence_01.txt` và các artifacts replay hiện có. Phạm vi là **Validation, 300 ảnh, 30 lớp**; không sử dụng Official Test.

| Thành phần của phiên Validation replay | Giá trị đã ghi nhận |
| --- | --- |
| Source commit thực thi | `5d3b393666c73b699671a2f420d1c6f3996ff992` |
| Python | `3.12.4` |
| PyTorch | `2.6.0+cu124` |
| Torchvision | `0.21.0+cu124` |
| NumPy | `2.5.3` |
| CUDA build / thiết bị inference | `12.4` / `cuda` |
| Checkpoint E4 SHA-256 | `73df0cc1c5b16866d4facc910f3e4d3ab481e3f12f1681a59fa7d61e3845c47b` — đã tính lại từ file và khớp evidence |
| NPZ lịch sử | `outputs/predictions/task04_validation_review/validation_raw_predictions.npz` |
| NPZ replay | `outputs/predictions/task05_validation_replay_01/validation_raw_predictions.npz` |
| Đối chiếu predictions | Cùng schema, dtype, shapes; 300 Image IDs và labels khớp nguyên thứ tự; **9.000/9.000 logits trùng tuyệt đối**, maximum absolute difference = `0.0` |
| SHA-256 chung của hai NPZ | `5aae305b3a29e842fad313b5f3eae6edc1e2d3ca31b6e9228f4bc9c17442682a` |

**E4/E6/E7 trên Validation đã tái lập khớp bằng chứng đã ghi nhận:** metrics JSON E4/E6 lịch sử và replay giống từng byte; E7 tính từ NPZ replay khớp kết quả Validation đã nghiệm thu trong báo cáo `docs/task_reports/kiet/22_10_task.md`.

| Chỉ số Validation | Kết quả |
| --- | ---: |
| E4 Top-1 theo số đếm | `0.8166666666666667` (245/300) |
| E4 `classification.top1_accuracy` trong JSON | `0.8166666626930237` |
| E6 ECE sau hiệu chỉnh | `0.07963164098033844` |
| E7 ACCEPT / REJECT | 261 / 39 |
| E7 Coverage | `0.87` |
| E7 Selective Accuracy | `0.842911877394636` |
| E7 Selective Risk | `0.15708812260536398` |
| E7 AURC tham chiếu, grouped right-step | `0.09093638678902743` |

Hai giá trị Top-1 trên là cùng tỷ lệ 245/300 với biểu diễn độ chính xác khác nhau; giữ nguyên số liệu JSON. E7 replay được lưu tại `outputs/predictions/task05_e7_validation_replay_01/`. Không refit Temperature E6, không chọn lại threshold E7 và không thay đổi cấu hình E4/E6/E7/E8.

**Giới hạn provenance:** tái lập chính xác predictions không xác minh ngược toàn bộ source hoặc thay đổi local của phiên lịch sử có metadata Git SHA `891edf9e69a6b724bcdc20b02143888e8f35bbd8`. Artifacts lịch sử được giữ nguyên. Kết luận tái lập tuyệt đối áp dụng cho các artifacts và môi trường replay đã ghi nhận, chưa chứng minh trên môi trường khác.

### 7.2. Kiểm thử Task 05 và các điều kiện Official Final Evaluation còn mở

Kết quả integration/regression Task 05 đã ghi nhận trước giai đoạn B2: **770 passed, 7 skipped, 1 warning**, không có failed tests. Warning NumPy `longdouble` overflow thuộc test E7 đã tồn tại. Giữ nguyên bằng chứng lịch sử này; không coi đây là một lượt full regression mới của B2.

Các bằng chứng nghiệm thu tiếp theo đã được xác nhận:

| Giai đoạn / bằng chứng | Kết quả và phạm vi |
| --- | --- |
| Phase A | ACCEPTED — artifact-based Validation preflight; tham chiếu `docs/final_evaluation/task05_phase_a_preflight_acceptance.md` |
| Phase B1 | ACCEPTED — E4/E6/E7/E8 trên Validation; tham chiếu `docs/final_evaluation/task05_phase_b1_validation_acceptance.md` |
| Phase B2.1 | Design approved; không phải Configuration Lock Approval hoặc quyền sử dụng Official Test |
| Phase B2.2-A | Authorization component ACCEPTED |
| Phase B2.2-B | 52 synthetic integration tests PASS; không phải thực thi trên Official Test |
| Final Source Review | READY FOR SOURCE FREEZE; source được phê duyệt tại S ở mục 7.3 |
| Checkpoint smoke | Trưởng nhóm xác nhận checkpoint thật SHA MATCH, một synthetic forward trên CUDA PASS, logits `(1,30)` hữu hạn |

Các kết quả B2, source review và smoke ở trên được ghi theo bằng chứng trưởng nhóm cung cấp và được trưởng nhóm chấp nhận trong quyết định ngày 26/10/2026; không chạy inference, preflight, pytest, smoke test hoặc Validation replay khi lập records phê duyệt. Fixtures không thay thế bằng chứng checkpoint thật; smoke không chứng minh kết quả trên Official Test hoặc khả năng tái lập trên mọi môi trường. Snapshot vẫn theo từng file, không phải snapshot nguyên tử toàn dataset; consumer cần Git objects, source phù hợp và claim ledger theo contract đã nghiệm thu.

**Git SHA và môi trường ở mục 7.1 chỉ thuộc phiên Validation replay.** S và runtime profile tại mục 7.3–7.4 được phê duyệt riêng theo quyết định Configuration Lock ngày 26/10/2026, không suy ra từ việc một số phiên bản thư viện trùng nhau. Configuration Lock đã LOCKED; Official Test chưa được phép sử dụng vì chưa có Official Evaluation Authorization.

Khi đánh giá chính thức, phải ghi nhận đầy đủ:

- Git commit thực sự được sử dụng.
- SHA-256 của checkpoint.
- Phiên bản Python, PyTorch và các thư viện quan trọng.
- CPU/GPU và môi trường thực nghiệm.
- Cấu hình và lệnh thực thi.
- Các chỉ số, dữ liệu dự đoán và thời gian chạy.

### 7.3. Source và canonical dataset reference được phê duyệt

| Thành phần | Giá trị được trưởng nhóm phê duyệt ngày 26/10/2026 |
| --- | --- |
| Source revision S | `02483f801277482fbe24c25048e6cc769a16295f` |
| Dataset freeze revision D | `d0b6931250cc51088e7beded24737959ef9464cc` |
| Canonical reference | `data/splits/official_test.csv` |
| Canonical Git blob ID | `25af4d79d3b255df9a30db8f899f400d19910baa` |
| `source_split` | `official_test` |
| Schema CSV | `relative_path,class_id,breed_name,source_split` |
| Image ID / nhãn | Giữ nguyên `relative_path`; `class_id` và `breed_name` đối chiếu `data/metadata/class_mapping.json`, 30 lớp cố định |
| Expected sample count | `1934` — dự kiến theo tài liệu, chưa đọc Official Test để xác minh |

S và D có vai trò riêng, không tự đồng nhất hai revision. Blob ID là Git object ID, không phải SHA-256 của checkpoint hoặc SHA-256 CSV. Các giá trị dataset reference tại bảng được ghi theo xác nhận của trưởng nhóm; lượt cập nhật này không đọc, liệt kê, stat, hash, mở hoặc đếm dữ liệu Official Test, kể cả nội dung Git blob canonical. Đối chiếu CSV/ảnh thực tế chỉ được thực hiện sau authorization gate, claim và output reservation hợp lệ.

### 7.4. Runtime profile đầy đủ — được phê duyệt

Object 16 trường dưới đây do trưởng nhóm cung cấp và phê duyệt ngày 26/10/2026, giữ nguyên giá trị và kiểu dữ liệu theo contract `observe_runtime()` trong `src/final_evaluation/official_execution.py`. Đây không phải kết quả đo môi trường của tiến trình đang cập nhật tài liệu. Profile được phê duyệt cho cấu hình; việc chạy Official vẫn cần authorization riêng.

```json
{
  "batch_size": 32,
  "cuda_build": "12.4",
  "device": "cuda",
  "matplotlib": "3.11.2",
  "num_workers": 0,
  "numpy": "2.5.3",
  "os": "Windows-11-10.0.26300-SP0",
  "pandas": "3.0.6",
  "pillow": "12.3.0",
  "python": "3.12.4",
  "pytorch": "2.6.0+cu124",
  "pyyaml": "6.0.3",
  "scikit_learn": "1.9.1",
  "scipy": "1.18.1",
  "torch_num_threads": 4,
  "torchvision": "0.21.0+cu124"
}
```

Evidence phần cứng do trưởng nhóm xác nhận: **NVIDIA GTX 1650**, driver **555.97**, CUDA khả dụng. Không thêm GPU, driver hoặc `cuda_available` vào runtime object vì observer không trả các trường này. Authorization so sánh toàn bộ runtime object chính xác; không tự thay phiên bản, kiểu dữ liệu hoặc bỏ qua mismatch.

### 7.5. Configuration Lock Approval records và authorization chưa được cấp

Theo quyết định trực tiếp của trưởng nhóm ngày 26/10/2026, chỉ tạo ba records Configuration Lock Approval dưới `docs/final_evaluation/controls/`: `protocol.json`, `lock_decision.txt`, `lock_approval.json`. Các records này tuân thủ schema và fixed roles hiện có trong `authorization.py` / `official_execution.py` và không cấp quyền thực thi:

| File | Trạng thái và vai trò |
| --- | --- |
| `protocol.json` | ĐÃ TẠO — nội dung khoa học khớp chính xác `approved_protocol()`; giữ checkpoint, Temperature, threshold, preprocessing và E8 severity |
| `lock_decision.txt` | ĐÃ TẠO — lưu quyết định phê duyệt trực tiếp của trưởng nhóm, bao gồm giới hạn KHÔNG cấp authorization; role `leader-lock` |
| `lock_approval.json` | ĐÃ TẠO — schema `dogbreedid.configuration-approval.v1`, status `APPROVED`; ràng buộc S, source hashes, chín frozen references, protocol, bản Lock cuối, acceptance evidence, runtime và dataset descriptor D |
| `run_authorization.json` | CHƯA TẠO, CHƯA CẤP QUYỀN — cần quyết định riêng cho một run ID, scope E4/E6/E7/E8, output và claim root; `max_runs=1`, `automatic_retry=false` |
| `grant_decision.txt` | CHƯA TẠO — chưa có quyết định Official Evaluation Authorization |
| `b2_release.json` | CHƯA TẠO — chưa có execution release; không có trusted release revision R cho một lượt chạy |

`lock_approval.json` dùng SHA-256 của bytes bản Lock cuối này sau quyết định phê duyệt; `source_sha256` và `frozen_sha256` đúng catalog của schema. Acceptance roles `phase-a` và `phase-b1` trỏ tới hai tài liệu nghiệm thu đã có, giữ nguyên bằng chứng lịch sử. Dataset descriptor chỉ có `reference`, `revision`, `expected_samples`, `num_classes`; Git blob ID và schema CSV được ghi ở hồ sơ/evidence, không thêm key ngoài schema.

Giữ S cố định; các hashes được chuẩn bị theo thứ tự protocol/Lock/evidence → approval. Chỉ sau quyết định authorization riêng mới chuẩn bị authorization → release index. Reviewed release revision R sau này phải được trưởng nhóm pin từ bên ngoài sau commit, không chèn SHA R hoặc hash tự thân vào chính các file thuộc R. Không bắt buộc ba commit S/A/B riêng; thứ tự commit hồ sơ phụ thuộc thời điểm hai quyết định độc lập. Hiện chưa có execution release, trust anchors cho một lượt chạy hoặc claim nào được tạo từ hồ sơ này.

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

### 9.2. Theo dõi các điều kiện trước khi phê duyệt

- [x] Hợp đồng dữ liệu được kiểm chứng bằng integration tests và nghiệm thu sau Task 04.
- [x] Mã nguồn Final Evaluation và các thư viện phụ thuộc được kiểm tra.
- [x] Các kiểm thử tích hợp Final Evaluation đạt yêu cầu.
- [x] Trưởng nhóm phê duyệt Configuration Lock.

Điều kiện Data Contract được cập nhật theo kết luận nghiệm thu trong `docs/final_evaluation/data_contract.md`. Ba điều kiện còn lại được trưởng nhóm xác nhận và phê duyệt ngày 26/10/2026 trên cơ sở source review, 52 integration fixtures, checkpoint smoke và runtime profile đầy đủ tại mục 7. Không coi các checkbox này là bằng chứng đã chạy Official Test hoặc đã cấp Official Evaluation Authorization.

Trưởng nhóm đã phê duyệt S/D/canonical blob, phạm vi expected count, runtime 16 trường, cấu hình khoa học và các bằng chứng nghiệm thu trong quyết định ngày 26/10/2026. Việc chưa truy cập dữ liệu Official thật là tuân thủ gate; expected sample count vẫn chưa được kiểm đếm từ Official Test. Chưa có quyết định authorization riêng cho bất kỳ lượt chạy nào.

**Quyết định hiện tại: ĐÃ KHÓA CẤU HÌNH — LOCKED / CONFIGURATION APPROVED.**

**Official Test tiếp tục bị khóa và chưa được phép sử dụng.**

## 10. Phê duyệt khóa cấu hình

| Hạng mục                                   | Trạng thái    |
| ------------------------------------------ | ------------- |
| Trạng thái phê duyệt                       | ĐÃ PHÊ DUYỆT CONFIGURATION LOCK — LOCKED |
| Ngày phê duyệt                             | 26/10/2026 |
| Git commit Final Evaluation được phê duyệt | S: `02483f801277482fbe24c25048e6cc769a16295f` |
| Người phê duyệt                            | Trưởng nhóm DogBreedID — quyết định trực tiếp trong hội thoại |
| Cho phép sử dụng Official Test             | CHƯA CHO PHÉP |

Trạng thái hiện tại của hồ sơ, không phải execution manifest:

```json
{
  "official_test_used": false,
  "configuration_lock_approved": true,
  "official_evaluation_authorized": false,
  "official_final_evaluation": false
}
```

**Configuration Lock đã được trưởng nhóm phê duyệt và ba Configuration Lock Approval records đã được tạo. KHÔNG có Official Evaluation Authorization, grant, reviewed execution release hoặc claim. Official Test chưa được sử dụng và chưa được phép sử dụng.**

Việc chuyển trạng thái LOCKED ở tài liệu này ghi nhận đúng quyết định Configuration Lock ngày 26/10/2026; không mở quyền Official Test. Chỉ một quyết định Official Evaluation Authorization riêng, được kiểm chứng qua gate và claim/reservation hợp lệ, mới có thể cho phép một lượt chạy trong tương lai.
