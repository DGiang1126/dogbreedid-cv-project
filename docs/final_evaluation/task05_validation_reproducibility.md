# DogBreedID — Task 05: Kiểm chứng tái lập Validation

**Ngày lập báo cáo:** 25/10/2026

**Repository:** `DGiang1126/dogbreedid-cv-project`

**Branch:** `task05-final-evaluation-preflight`

**Phạm vi:** Đối chiếu artifacts Validation đã có; không chạy lại inference trong bước lập báo cáo.

## 1. Kết luận và phạm vi xác nhận

Hai bộ predictions Validation lịch sử và replay có cùng schema, dtype, shapes và thứ tự dữ liệu. Toàn bộ 300 Image IDs, 300 labels và 9.000 logits trùng tuyệt đối; maximum absolute difference của logits là `0.0`. Hai file NPZ còn có SHA-256 trùng nhau.

Phiên replay được trưởng nhóm xác nhận thực thi từ Git SHA `5d3b393666c73b699671a2f420d1c6f3996ff992`. SHA này khớp metadata inference, `source_git_sha`/`upstream_git_sha` của E7 và commit hiện có trong repository. Kết quả chứng minh predictions Validation lịch sử đã được tái lập chính xác bằng phiên replay đã ghi nhận.

Việc tái lập predictions không xác minh ngược được đầy đủ source hoặc trạng thái working tree của phiên lịch sử. Giới hạn provenance cũ được giữ nguyên tại mục 8.

**Đây không phải Official Final Evaluation. Official Test không được sử dụng. Configuration Lock vẫn là BẢN NHÁP — CHƯA KHÓA CẤU HÌNH; báo cáo này không thay đổi trạng thái phê duyệt.**

## 2. Nguồn bằng chứng

| Nguồn | Vai trò |
| --- | --- |
| `task05_reproducibility_evidence_01.txt` | Bản ghi Git SHA, branch, môi trường, phạm vi Validation và SHA-256 checkpoint cùng bảy output replay; ghi lúc `2026-10-25T22:06:30.3729968+07:00`. |
| `outputs/predictions/task04_validation_review/` | Bốn artifacts Validation lịch sử, giữ nguyên để đối chiếu. |
| `outputs/predictions/task05_validation_replay_01/` | NPZ, metrics, metadata, reliability diagram và marker hoàn tất của replay E4/E6. |
| `outputs/predictions/task05_e7_validation_replay_01/` | `selective_metrics.json`, `risk_coverage.csv`, `risk_coverage_curve.png` của replay E7. |
| Lịch sử PowerShell PSReadLine trên máy trưởng nhóm | Lệnh CLI ở mục 4; file evidence TXT không chứa lệnh chạy. |
| `configs/data.yaml`, `data/splits/validation.csv`, `data/metadata/class_mapping.json`, artifacts E6/E7 và Configuration Lock | Đối chiếu preprocessing, ID–nhãn và tham số đã phê duyệt; chỉ đọc. |

SHA-256 của chính file evidence, tính khi lập báo cáo:

`9474df57f7b633cb7bc85eb3d9821170bdb7a16d501f3dcee34fe04c29e186de`

Thời điểm ghi trong metadata inference replay: `2026-10-25T14:59:33.314533+00:00`. E7 ghi `2026-10-25T15:03:22.781515+00:00`.

## 3. Cấu hình và môi trường phiên replay

| Thành phần | Giá trị đã ghi nhận |
| --- | --- |
| Git SHA thực thi | `5d3b393666c73b699671a2f420d1c6f3996ff992` |
| Model | E4 — MobileNetV2 Frozen + CutMix, seed `42`, `30` lớp |
| Checkpoint | `outputs/checkpoints/E4/seed_42/best.pt` |
| Checkpoint SHA-256 | `73df0cc1c5b16866d4facc910f3e4d3ab481e3f12f1681a59fa7d61e3845c47b` |
| Split | Validation, `data/splits/validation.csv`, `300` ảnh, `30` lớp |
| Thư mục ảnh trong lệnh chạy | `data/raw/stanford_dogs/Images` |
| Python | `3.12.4` |
| PyTorch | `2.6.0+cu124` |
| Torchvision | `0.21.0+cu124` |
| NumPy | `2.5.3` |
| CUDA build / khả dụng | `12.4` / `True` |
| Thiết bị inference | `cuda` |
| Batch size / num_workers | `32` / `0` |
| Thiết bị E7 | `CPU`; `inference_performed=false`, chỉ xử lý predictions |
| Temperature E6 | `0.80246283961825`, từ `outputs/metrics/E6/temperature.json` |
| Threshold E7 | `0.5011672918550012`, từ `outputs/metrics/E7/threshold.json` |
| Quy tắc ACCEPT | `confidence >= threshold`, confidence sau E6 |

Đây là môi trường của phiên replay theo evidence/metadata, không phải kết quả đo lại môi trường của tiến trình đang lập báo cáo. Không refit Temperature, không chọn lại threshold.

Preprocessing trong metadata replay khớp `configs/data.yaml`: RGB → Resize(256) → CenterCrop(224) → ToTensor → ImageNet Normalization, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`, tensor `[3, 224, 224]`.

## 4. Lệnh replay đã ghi trong lịch sử PowerShell

Lệnh được đọc từ `%APPDATA%\Microsoft\Windows\PowerShell\PSReadLine\ConsoleHost_history.txt`, các dòng 11102–11111 và 11120–11124 tại thời điểm đối chiếu. Dưới đây chuẩn hóa các dòng continuation thành một dòng, giữ nguyên thứ tự và giá trị đối số. PSReadLine là bản ghi lệnh, không phải transcript đầy đủ stdout/stderr hoặc exit code.

Các lệnh chạy từ repository root. Đây là ghi nhận lịch sử; không thực thi lại trong bước lập báo cáo và không ghi đè các thư mục output đã tồn tại.

### 4.1. E4 inference và E6 calibration metrics

```powershell
python scripts/run_final_inference.py --split validation --split-csv "data/splits/validation.csv" --images-dir "data/raw/stanford_dogs/Images" --checkpoint "outputs/checkpoints/E4/seed_42/best.pt" --temperature-artifact "outputs/metrics/E6/temperature.json" --batch-size 32 --num-workers 0 --device cuda --output-dir "outputs/predictions/task05_validation_replay_01"
```

### 4.2. E7 từ predictions replay

```powershell
python scripts/run_final_selective.py --split validation --predictions "outputs/predictions/task05_validation_replay_01/validation_raw_predictions.npz" --metadata "outputs/predictions/task05_validation_replay_01/validation_metadata.json" --output-dir "outputs/predictions/task05_e7_validation_replay_01"
```

## 5. Đối chiếu trực tiếp hai NPZ

- NPZ lịch sử: `outputs/predictions/task04_validation_review/validation_raw_predictions.npz`.
- NPZ replay: `outputs/predictions/task05_validation_replay_01/validation_raw_predictions.npz`.
- Đọc cả hai bằng `numpy.load(..., allow_pickle=False)`; không gọi model hoặc runner inference.

| Trường | Dtype cũ / mới | Shape cũ / mới | Đối chiếu |
| --- | --- | --- | --- |
| `logits` | `float32` / `float32` | `(300, 30)` / `(300, 30)` | Cả 9.000 giá trị bằng tuyệt đối; bytes mảng trùng nhau; tất cả hữu hạn |
| `labels` | `int64` / `int64` | `(300,)` / `(300,)` | 300 giá trị và thứ tự trùng nhau |
| `image_ids` | `<U55` / `<U55` | `(300,)` / `(300,)` | 300 chuỗi Unicode và thứ tự trùng nhau; 300 ID duy nhất |

Schema của mỗi NPZ gồm đúng `logits`, `labels`, `image_ids`. Labels có đủ 30 lớp. IDs và labels của replay khớp nguyên thứ tự `relative_path`/`class_id` trong CSV Validation canonical.

Phép kiểm tra `numpy.array_equal` và so sánh bytes của từng mảng đều đạt. Maximum absolute difference của logits, đối chiếu sau khi chuyển cả hai mảng sang float64: `0.0`.

SHA-256 chung của hai file NPZ:

`5aae305b3a29e842fad313b5f3eae6edc1e2d3ca31b6e9228f4bc9c17442682a`

## 6. Kết quả Validation

### 6.1. E4 và E6

| Chỉ số | Giá trị trong bằng chứng/artifact |
| --- | ---: |
| Số mẫu / số lớp | 300 / 30 |
| Số dự đoán Top-1 đúng | 245 / 300 |
| E4 Top-1 theo số đếm, đồng thời là `calibration.*.top1_accuracy` | 0.8166666666666667 |
| `classification.top1_accuracy` trong metrics JSON | 0.8166666626930237 |
| Top-5 Accuracy | 0.9866666793823242 |
| Macro-F1 | 0.815099216821925 |
| Temperature | 0.80246283961825 |
| Số bins ECE | 15 |
| NLL trước / sau E6 | 0.6620433771121585 / 0.6412068263814583 |
| ECE trước / sau E6 | 0.10925347267077187 / 0.07963164098033844 |
| Multiclass Brier trước / sau E6 | 0.30574911689549933 / 0.29675198232729383 |

`classification.top1_accuracy=0.8166666626930237` là biểu diễn float32 của cùng tỷ lệ `245/300`. Báo cáo giữ nguyên trường này và giá trị `0.8166666666666667` ở phần calibration/số đếm; không sửa artifact hoặc coi đây là hai kết quả dự đoán khác nhau. Metrics JSON lịch sử và replay có nội dung bytes và SHA-256 trùng nhau. Metrics JSON ghi Temperature Scaling không đổi Top-1 prediction.

### 6.2. E7

| Chỉ số | Giá trị từ `selective_metrics.json` |
| --- | ---: |
| Tổng số mẫu | 300 |
| ACCEPT | 261 |
| REJECT | 39 |
| ACCEPT đúng | 220 |
| Coverage | 0.87 |
| Selective Accuracy | 0.842911877394636 |
| Selective Risk | 0.15708812260536398 |
| AURC tham chiếu trên Validation | 0.09093638678902743 |

Quy ước AURC ghi trong artifact: `grouped right-step over confidence tie groups`. `risk_coverage.csv` có 301 hàng dữ liệu; cả hai PNG replay đã được kiểm tra đọc hợp lệ.

E7 ghi `official_final_evaluation=false`, `synthetic=false`, split `validation`, `threshold_reselected=false`, `input_order_matches_csv=true`, `mapping_hash_mode=exact`. `temperature_hash_mode=newline_variant_lf` chỉ mô tả đối chiếu bytes LF/CRLF của reference; giá trị Temperature không thay đổi.

Cả 11 SHA-256 trong `input_sha256` của E7 đã được đối chiếu với các file hiện có: NPZ, metadata, CSV Validation, mapping, E6 Temperature, E7 threshold, configs E4/E6/E7, data config và model selection đều khớp. E7 không nạp lại checkpoint; xác minh checkpoint của E7 vẫn được ghi là `metadata-only`.

## 7. SHA-256 của checkpoint và artifacts replay

Tám dòng hash dưới đây chép đầy đủ từ `task05_reproducibility_evidence_01.txt`, giữ cách viết chữ hoa của file nguồn. Tất cả đã được tính lại từ bytes file hiện có và trùng khớp; chữ hoa/thường không làm thay đổi giá trị SHA-256.

| File | SHA-256 |
| --- | --- |
| `outputs/checkpoints/E4/seed_42/best.pt` | `73DF0CC1C5B16866D4FACC910F3E4D3AB481E3F12F1681A59FA7D61E3845C47B` |
| `outputs/predictions/task05_validation_replay_01/validation_raw_predictions.npz` | `5AAE305B3A29E842FAD313B5F3EAE6EDC1E2D3CA31B6E9228F4BC9C17442682A` |
| `outputs/predictions/task05_validation_replay_01/validation_metrics.json` | `1C0F46C05268D59AE4258E58D5E460823D36D0B2775815E636A51431AEB7D2CE` |
| `outputs/predictions/task05_validation_replay_01/validation_metadata.json` | `9D06D20B6FDB0B9EE39E9DB5CDEDD1181BE695D5CDDC43474C44CC73DAD63AC4` |
| `outputs/predictions/task05_validation_replay_01/validation_reliability_diagram.png` | `45CD20FF74D7BA75952FEE123C6AB48B3C2071F50B229E1B198B15599F83A376` |
| `outputs/predictions/task05_e7_validation_replay_01/selective_metrics.json` | `EB0E0FCA1BB5B9B49B8B0205E7914A2E5C73374E4FE9A537397F4BFBF9707463` |
| `outputs/predictions/task05_e7_validation_replay_01/risk_coverage.csv` | `F3E009CB54207436411CCD46E608068888BC40E87C8D1AA50702C6F712C872F4` |
| `outputs/predictions/task05_e7_validation_replay_01/risk_coverage_curve.png` | `DF564F3CADE488813D084BE02A714584E4C8A9E66345AF4C925F122C4502DDF0` |

Đối với bộ E4/E6 replay, `.complete` và `.inference_run` có token trùng nhau, không có `.incomplete`; cả bốn artifacts hiện diện. Kiểm tra này xác nhận trạng thái hoàn tất theo cơ chế output safety hiện hành.

## 8. Provenance lịch sử và giới hạn xác nhận

| Nội dung | Kết luận |
| --- | --- |
| Tái lập predictions Validation | ĐẠT: cùng arrays, dtype, shape, thứ tự và bytes NPZ; max absolute difference `0.0`. |
| Nguồn phiên replay | Git SHA thực thi được trưởng nhóm xác nhận, khớp metadata/artifact và commit `5d3b393666c73b699671a2f420d1c6f3996ff992`. E7 ghi `required source and reference files tracked and clean at HEAD`. |
| Source thực sự của phiên lịch sử | CHƯA XÁC MINH ĐẦY ĐỦ: metadata cũ ghi commit nền `891edf9e69a6b724bcdc20b02143888e8f35bbd8`; chưa chứng minh toàn bộ source và thay đổi local khi chạy. |
| Checkpoint E4 | SHA-256 file hiện có khớp evidence và Configuration Lock; không dùng việc này để xác nhận độc lập toàn bộ quá trình huấn luyện lịch sử. |
| Tái lập trên môi trường khác | Chưa được chứng minh; kết luận chính xác tuyệt đối áp dụng cho hai artifacts và môi trường replay đã ghi nhận. |
| Official Final Evaluation | CHƯA THỰC HIỆN; không sử dụng Official Test. |

SHA-256 metadata lịch sử: `07e179e125f5d5f5c15e64e087fdf82036351a2695e405b1e286d55d4997eb77`. Metadata replay có SHA-256 khác vì ghi Git SHA, thời điểm và output path của phiên mới; không sửa hoặc gắn lại provenance cho metadata cũ.

Artifact E7 vẫn giữ `upstream_provenance.status=historical execution not independently verified` và checkpoint verification dạng `metadata-only`. Báo cáo bổ sung bằng chứng replay riêng; không sửa các tuyên bố thận trọng đó. Việc logits trùng tuyệt đối không biến metadata lịch sử thành bằng chứng source đã được xác minh.

Giới hạn lưu trữ: khi lập báo cáo, evidence TXT và các thư mục output lịch sử/replay vẫn là file local untracked. Việc có bảng hash trong báo cáo không đồng nghĩa các file đó đã được đính kèm Git hoặc được lưu trữ lâu dài; trưởng nhóm cần quyết định nơi lưu bằng chứng.

## 9. Phạm vi thao tác khi lập báo cáo

- Chỉ đọc artifacts, đối chiếu hash/NPZ/CSV/metadata, đọc lịch sử lệnh và tạo tài liệu này.
- Không chạy lại inference, model evaluation, notebook hoặc regression tests; không tuyên bố có CI hay log exit code mới.
- Không thay model, checkpoint, configs, splits, mapping, Data Contract, Configuration Lock hoặc bất kỳ artifact lịch sử/replay nào.
- Không git add, commit, push hoặc merge trong bước này.
- Kết quả là bằng chứng phục vụ trưởng nhóm review preflight; phê duyệt Configuration Lock và cơ chế cấp quyền Official Test vẫn là bước riêng.
