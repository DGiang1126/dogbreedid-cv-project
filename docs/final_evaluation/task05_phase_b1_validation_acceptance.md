# DogBreedID — Task 05: Ghi nhận nghiệm thu Phase B1 Validation

**Ngày lập báo cáo:** 26/10/2026

**Repository:** `DGiang1126/dogbreedid-cv-project`

**Branch:** `task05-final-evaluation-preflight`

**Source commit thực thi B1:** `a6d6b9b91ad81f4b7a40cc38b3d92571e1d949ea`

**Phạm vi:** Nghiệm thu bundle Phase B1 đã thực thi trên 300 ảnh Validation thật, 30 lớp. Khi lập tài liệu chỉ đọc artifacts, gọi consumer và đối chiếu bằng chứng; không chạy lại inference, training hoặc Official Test.

## 1. Mục đích và nguồn bằng chứng

Phase B1 tích hợp thực thi E4/E6/E7/E8 trên Validation và xuất bundle có provenance cùng cơ chế publish an toàn. Báo cáo này ghi nhận kết quả kiểm chứng độc lập của bundle đã lưu, đối chiếu với Validation Replay Task 05 và E8 committed đã nghiệm thu.

| Nguồn | Vai trò |
| --- | --- |
| `outputs/final_evaluation_b1/validation/b1_validation_01/` | Bundle B1: NPZ, metrics E4/E6/E7/E8, metadata, protocol, CSV, mapping, biểu đồ và markers |
| `outputs/predictions/task05_validation_replay_01/` | NPZ, metrics và metadata Validation Replay E4/E6 |
| `outputs/predictions/task05_e7_validation_replay_01/` | Metrics và risk-coverage CSV/PNG của E7 Replay |
| `outputs/robustness/task04/validation/e8_validation_committed.json` | Artifact E8 Validation đã nghiệm thu, source `cf522ffa0b7d582ddb6557b6ff15b891e51fc117` |
| `docs/final_evaluation/task05_validation_reproducibility.md` | Bằng chứng Replay từ commit `5d3b393666c73b699671a2f420d1c6f3996ff992` và giới hạn provenance lịch sử |
| `docs/final_evaluation/configuration_lock.md` | Tham số cố định và trạng thái phê duyệt hiện hành |
| Git objects tại source commit B1 | Nguồn canonical để consumer kiểm tra source, frozen references, split CSV và mapping |

Đây là kết quả **Validation**, không phải Official Final Evaluation. Việc kiểm tra bundle khi lập báo cáo không tạo predictions mới.

## 2. CLI và môi trường thực thi đã ghi nhận

Manifest ghi thời điểm `2026-10-26T05:30:55.652847+00:00`, tương ứng `2026-10-26T12:30:55.652847+07:00`. Đây là timestamp của manifest, không phải bằng chứng riêng về thời điểm bắt đầu inference.

Lệnh dưới đây chép từ `manifest.command`, trình bày theo cú pháp PowerShell tại thư mục repository `D:\HK1_IUH_22646851_Final\TGMT\dogbreedid-cv-project`:

```powershell
& "D:\HK1_IUH_22646851_Final\TGMT\dogbreedid-cv-project\.venv\Scripts\python.exe" `
  "D:\HK1_IUH_22646851_Final\TGMT\dogbreedid-cv-project\scripts\run_final_evaluation.py" `
  --split validation `
  --output-dir outputs/final_evaluation_b1/validation/b1_validation_01 `
  --batch-size 32 `
  --device auto
```

Đây là lệnh lịch sử, không được thực thi lại khi lập báo cáo. Output của phiên này đã tồn tại.

| Thuộc tính trong manifest | Giá trị |
| --- | --- |
| Python | `3.12.4` |
| PyTorch | `2.6.0+cu124` |
| Torchvision | `0.21.0+cu124` |
| NumPy | `2.5.3` |
| CUDA build | `12.4` |
| `cuda_available` | `true` |
| Device thực thi | `cuda` (`--device auto` đã chọn CUDA) |
| Batch size | `32` |
| DataLoader workers | `0` |
| Torch threads | `4` |

Các giá trị này thuộc phiên inference B1 đã ghi nhận. Việc consumer đọc lại bundle trong lượt nghiệm thu không phải một phiên inference CUDA mới.

## 3. Consumer, completion markers và SHA-256

Đã gọi API thực tế, không mock hoặc bỏ qua validators:

```python
from scripts.run_final_evaluation import load_execution_bundle

bundle = load_execution_bundle(
    "outputs/final_evaluation_b1/validation/b1_validation_01"
)
```

Consumer trả về thành công: **PASS**. Các components `E4`, `E6`, `E7`, `E8`, `data_contract`, `weights_unchanged` và `clean_consistency` đều `PASS`. Manifest ghi `execution_status=complete`, `split=validation`, `synthetic=false`, `scientific_status=validation_only`.

| Kiểm tra | Kết quả |
| --- | --- |
| Đủ artifacts bắt buộc | PASS: 12 payloads, `.inference_run` và `.complete`; tổng cộng 14 file, không thiếu/thừa |
| Marker lỗi | PASS: không có `.incomplete` |
| Format hoàn tất | PASS: `dogbreedid.execution.b1.v1` |
| Run token | PASS: `.inference_run` là token 33 bytes, gồm newline cuối; khớp `run_token` trong `.complete` |
| Hash trong `.complete` | PASS: 12/12 payload hashes, gồm `manifest.json`, khớp bytes thực tế |
| Hash trong manifest | PASS: 11/11 payload hashes còn lại khớp bytes thực tế |
| Semantic validation | PASS: schema, canonical provenance, protocol, arrays, metrics và PNG được consumer kiểm tra |

`.complete` chứng minh publish đã hoàn tất với các hashes tương ứng. Kết luận dữ liệu hợp lệ còn phụ thuộc semantic validation của consumer; sự tồn tại của marker không tự chứng minh nghiệm thu khoa học hoặc cấp quyền Official Evaluation.

SHA-256 tính trực tiếp từ bytes:

| File B1 | SHA-256 |
| --- | --- |
| `clean_predictions.npz` | `5aae305b3a29e842fad313b5f3eae6edc1e2d3ca31b6e9228f4bc9c17442682a` |
| `e4_e6_metrics.json` | `3d4913a1dcc305f4a2c805e0e949114bf2f76ede71fb929375d8d17b465a1ade` |
| `metadata.json` | `4493215ce7d445d4d408e25718b332af88915a5a7e1c5be6eabe23d166b86b22` |
| `selective_metrics.json` | `c5593c2ff5fa3bc3fc2b36b0a2ff18e964dd5e3cd4c3412651e716e6c7de921f` |
| `risk_coverage.csv` | `f3e009cb54207436411ccd46e608068888bc40e87c8d1aa50702c6f712c872f4` |
| `risk_coverage_curve.png` | `df564f3cade488813d084be02a714584e4c8a9e66345af4c925f122c4502ddf0` |
| `e8_metrics.json` | `36158cb4f502dc6c89ff946de62247626b9dc392a511dba90010a1d066f82a46` |
| `reliability_diagram.png` | `47aa37dd95c6655bcf911035549ec4f5c3f6db349e73c6a421161601acf203a4` |
| `split.csv` | `f3d73eb820b13b6b941f6a21cd3fd0455f9d2efd6b5c51c3f134c1956a72c0d9` |
| `class_mapping.json` | `79f08f8da0d65b3cb9cf256ac3b37688e112ce84b0352e5a600861b9b2327542` |
| `protocol.json` | `fe9cc12d1fa6feab6bec3dd37e5618c4ab99b7276774792581d512376f171b73` |
| `manifest.json` | `4084b9c40ae57f9e1585cab8a8b010a8c614d4454f54c4d93de053b774a5342a` |
| `.complete` | `45ef4c0d980e4bf472c9a8b264a0ea71c035cd3cbd1bd6b6e0d099f04de46cd9` |

Hash của `.complete` ở bảng là hash kiểm chứng trực tiếp khi lập báo cáo; marker này không chứa hash của chính nó.

## 4. Đối chiếu NPZ với Validation Replay

Hai file được đọc từ bytes với `allow_pickle=False`, cùng schema chính xác `logits`, `labels`, `image_ids`:

- B1: `outputs/final_evaluation_b1/validation/b1_validation_01/clean_predictions.npz`.
- Replay: `outputs/predictions/task05_validation_replay_01/validation_raw_predictions.npz`.

| Array | Shape cả hai | Dtype cả hai | `np.array_equal` | Số phần tử khác nhau |
| --- | --- | --- | --- | --- |
| `image_ids` | `(300,)` | `<U55` | `True` | `0` |
| `labels` | `(300,)` | `int64` | `True` | `0` |
| `logits` | `(300, 30)` | `float32` | `True` | `0/9.000` |

Toàn bộ 300 IDs và 300 labels trùng theo đúng thứ tự; không sắp xếp lại để tạo kết quả so sánh. Maximum absolute difference của logits, tính sau khi chuyển sang `float64` để đối chiếu, là **`0.0`**. Arrays trả về từ consumer cũng khớp arrays đọc từ NPZ đã hash.

Cả hai NPZ có SHA-256 **`5aae305b3a29e842fad313b5f3eae6edc1e2d3ca31b6e9228f4bc9c17442682a`** và trùng toàn bộ bytes. Việc so file hash và so giá trị arrays đã được thực hiện riêng; trong trường hợp này cả hai đều trùng tuyệt đối.

## 5. Metrics E4/E6/E7/E8

### 5.1. E4 và E6 so với Validation Replay

Đối chiếu `e4_e6_metrics.json` của B1 với `validation_metrics.json` của Replay. Toàn bộ cấu trúc JSON, per-class metrics và confusion matrix trùng giá trị. Trong lượt kiểm chứng độc lập, 1.073 trường số được kiểm tra với `rtol=1e-9`, `atol=1e-12`; counts nguyên so chính xác. Maximum absolute difference thực tế là `0.0`.

| Chỉ số | B1 | Validation Replay | Kết quả |
| --- | --- | --- | --- |
| Top-1 classification | `0.8166666626930237` | `0.8166666626930237` | PASS |
| Top-5 | `0.9866666793823242` | `0.9866666793823242` | PASS |
| Macro-F1 | `0.815099216821925` | `0.815099216821925` | PASS |
| Temperature E6 | `0.80246283961825` | `0.80246283961825` | PASS, so chính xác |
| NLL trước E6 | `0.6620433771121585` | `0.6620433771121585` | PASS |
| NLL sau E6 | `0.6412068263814583` | `0.6412068263814583` | PASS |
| ECE trước E6 | `0.10925347267077187` | `0.10925347267077187` | PASS |
| ECE sau E6 | `0.07963164098033844` | `0.07963164098033844` | PASS |
| Brier trước E6 | `0.30574911689549933` | `0.30574911689549933` | PASS |
| Brier sau E6 | `0.29675198232729383` | `0.29675198232729383` | PASS |

E4 có 245 dự đoán đúng trên 300 ảnh. Top-1 tính từ counts hoặc calibration metrics là `0.8166666666666667`. Chênh lệch biểu diễn so với Top-1 classification ở bảng là sai số float đã tồn tại giống nhau ở Replay, không phải khác biệt giữa B1 và Replay. Consumer có tolerance riêng cho classification so với counts: relative `1e-7`, absolute `1e-8`.

### 5.2. E7 so với E7 Validation Replay

| Chỉ số | B1 | E7 Replay | Kết quả |
| --- | --- | --- | --- |
| Threshold | `0.5011672918550012` | `0.5011672918550012` | PASS, so chính xác |
| Tổng mẫu | `300` | `300` | PASS |
| ACCEPT | `261` | `261` | PASS |
| REJECT | `39` | `39` | PASS |
| Đúng trong ACCEPT | `220` | `220` | PASS |
| Coverage | `0.87` | `0.87` | PASS |
| Selective Accuracy | `0.842911877394636` | `0.842911877394636` | PASS |
| Selective Risk | `0.15708812260536398` | `0.15708812260536398` | PASS |
| AURC | `0.09093638678902743` | `0.09093638678902743` | PASS |

E7 dùng Temperature `0.80246283961825`, phép nhận `>=`, `threshold_reselected=false`, quy ước AURC `grouped right-step over confidence tie groups`. Tất cả trường metrics chung trùng chính xác với Replay; `risk_coverage.csv` trùng bytes. Temperature và threshold được đối chiếu chính xác với frozen artifacts E6/E7, không áp dụng tolerance metrics để chấp nhận thay đổi tham số giao thức.

### 5.3. E8 so với artifact Validation committed

Nguồn tham chiếu là `outputs/robustness/task04/validation/e8_validation_committed.json`, SHA-256 **`1014bd8bfd4bd67136230122858d67a6654e55e6fe2cf709f882b53b29caac3f`**.

| Điều kiện | Counts B1 | Counts tham chiếu suy ra | Accuracy B1 | Accuracy tham chiếu | Accuracy Drop cả hai |
| --- | --- | --- | --- | --- | --- |
| Clean | `245/300` | `245/300` | `0.8166666666666667` | `0.8166666666666667` | Không áp dụng |
| Gaussian Blur | `195/300` | `195/300` | `0.65` | `0.65` | `0.16666666666666663` |
| Gaussian Noise | `204/300` | `204/300` | `0.68` | `0.68` | `0.1366666666666666` |
| Occlusion | `228/300` | `228/300` | `0.76` | `0.76` | `0.05666666666666664` |

Cả bốn điều kiện PASS. Artifact tham chiếu không có trường `correct_counts`; counts tham chiếu ở bảng được suy ra từ accuracy và `n_samples=300`, khớp counts tường minh của B1. Đây không phải counts thu được từ một lượt inference mới.

300 Image IDs của B1 E8 trùng thứ tự với artifact tham chiếu và NPZ sạch. Labels B1 E8 khớp NPZ. Checkpoint hash, preprocessing, severity và Accuracy Drops khớp artifact tham chiếu. Clean Accuracy E8 khớp `245/300` của E4. Blur gây giảm accuracy lớn nhất và Occlusion giảm ít nhất trong ba corruption ở severity đang xét.

## 6. Model và tham số cố định

| Thành phần | Giá trị đối chiếu |
| --- | --- |
| E4 | MobileNetV2 Frozen + CutMix, seed `42`, `30` lớp |
| Checkpoint | `outputs/checkpoints/E4/seed_42/best.pt` |
| Checkpoint SHA-256 | `73df0cc1c5b16866d4facc910f3e4d3ab481e3f12f1681a59fa7d61e3845c47b` |
| Evaluation preprocessing | RGB, Resize `256`, CenterCrop `224`, ImageNet normalization |
| ImageNet mean / std | `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` |
| E6 Temperature | `0.80246283961825` |
| E7 threshold | `0.5011672918550012`, phép nhận `>=` |
| E8 seed | `42`, dùng shared API deterministic theo Image ID |
| Gaussian Blur | Kernel `7`, sigma `1.5` |
| Gaussian Noise | Mean `0.0`, std `0.05`, clamp `[0.0, 1.0]` |
| Occlusion | Square, `side_fraction=0.30`, fill RGB `[0.5, 0.5, 0.5]`, `deterministic_per_image` |

Protocol, metadata, frozen artifacts và kết quả E8 được đối chiếu nhất quán. Metadata B1 ghi `model_weights_unchanged=true`; consumer yêu cầu cờ này hợp lệ. Lượt lập báo cáo không nạp checkpoint để chạy model, không sửa weights, không refit Temperature, không chọn lại threshold hoặc chỉnh severity.

## 7. Phạm vi source và reference integrity

Manifest, metadata, `source.head`, `frozen_references.head` và Git HEAD tại lượt kiểm chứng cùng SHA **`a6d6b9b91ad81f4b7a40cc38b3d92571e1d949ea`**. `source.status=PASS`, `frozen_references.status=PASS`; cả hai evidence ghi `modified=[]`, `untracked=[]` trong phạm vi audit.

25 source files được audit:

```text
scripts/run_final_evaluation_preflight.py
tests/test_final_evaluation_preflight.py
scripts/run_final_inference.py
scripts/run_final_selective.py
scripts/run_final_robustness.py
src/inference/predictor.py
src/evaluation/metrics.py
src/calibration/calibration_metrics.py
src/calibration/temperature_scaling.py
src/selective/config.py
src/selective/threshold.py
src/selective/risk_coverage.py
src/robustness/config.py
src/robustness/corruptions.py
src/robustness/robustness_eval.py
src/data/dataset.py
src/data/transforms.py
src/models/model_factory.py
src/models/mobilenet_v2.py
src/utils/config.py
src/calibration/__init__.py
src/inference/__init__.py
src/robustness/__init__.py
scripts/run_final_evaluation.py
tests/test_final_evaluation_execution.py
```

14 frozen references được audit:

```text
configs/data.yaml
configs/model_selection.yaml
configs/experiments/e4_cutmix.yaml
configs/experiments/e6_calibration.yaml
configs/experiments/e7_selective_prediction.yaml
configs/experiments/e8_robustness.yaml
outputs/metrics/E6/temperature.json
outputs/metrics/E7/threshold.json
data/metadata/class_mapping.json
docs/final_evaluation/data_contract.md
docs/final_evaluation/configuration_lock.md
docs/final_evaluation/task05_validation_reproducibility.md
docs/task_reports/duc-anh/21_10_task.md
data/splits/validation.csv
```

Consumer đối chiếu `source_sha256` và content bindings với immutable Git blobs tại commit ghi nhận, không chỉ tin `git diff HEAD` hoặc chuỗi trạng thái PASS. Biến thể LF/CRLF chỉ được chấp nhận sau đối chiếu bytes; không bỏ qua thay đổi code hoặc metadata. Git blobs được đọc với `--no-replace-objects`.

Manifest có **315 input snapshot records**: 14 references, 300 ảnh và 1 checkpoint. Packaged CSV/mapping phải khớp snapshot records và canonical Git blobs; NPZ IDs/labels phải khớp canonical Validation CSV. Consumer hash và parse cùng các buffers đã xác minh, không dùng CSV tự đóng gói làm nguồn tin cậy duy nhất.

Phạm vi PASS là danh sách source/reference nêu trên cùng các ràng buộc bundle. Các thư mục output untracked ngoài phạm vi audit không được mô tả là source đã commit. Trạng thái này cũng không chứng minh commit đã push/merge hoặc CI đã chạy.

## 8. Giới hạn và điều kiện tái lập

- Snapshot theo từng file: manifest ghi `private_snapshot_verified=true`, `cross_file_atomic=false`. Bytes được sao chép riêng và kiểm tra trước/sau execution; không có bảo đảm snapshot nguyên tử toàn bộ filesystem tại cùng một thời điểm. Canonical references được ràng buộc với Git, còn cơ chế này không chứng minh mọi ảnh nguồn đã đồng thời tồn tại ở cùng một thời điểm.
- Source integrity ràng buộc bytes trong phạm vi audit với Git; không chứng thực bytecode đã nạp, toàn bộ dependency binaries hoặc chống tác nhân có quyền thay đổi tiến trình/private staging. Private staging dựa trên ranh giới tin cậy cùng người dùng; markers/hashes không phải chữ ký số.
- Consumer cần Git objects của commit ghi nhận. Thiếu canonical blobs hoặc evidence thiết yếu sẽ FAIL/BLOCKED; `.complete` không thay thế các điều kiện này.
- Tính trùng tuyệt đối 9.000 logits được xác nhận cho B1 và Replay hiện có. Không suy rộng thành bảo đảm bitwise reproducibility trên mọi GPU, driver, phiên bản thư viện hoặc hệ điều hành; chưa có lượt inference đối chứng môi trường khác trong báo cáo này.
- Metadata môi trường là evidence của phiên thực thi đã lưu. Lượt lập báo cáo chỉ đọc bundle, tính lại/đối chiếu metrics qua consumer và kiểm tra hashes; không lặp lại 300 ảnh để xác minh độc lập phần cứng CUDA.
- Việc B1 khớp Replay không xác minh ngược được đầy đủ source/working tree của phiên predictions lịch sử Task 04. Giới hạn provenance lịch sử trong báo cáo Validation Reproducibility tiếp tục được giữ nguyên.
- Không chạy lại targeted/full regression hoặc CI khi lập tài liệu này; không nhận kết quả kiểm tra artifacts làm kết quả chạy tests mới.

## 9. Kết luận nghiệm thu Validation

**Phase B1 đạt nghiệm thu trong phạm vi Validation: 300 ảnh thật, 30 lớp.** Consumer thực tế PASS; markers và hashes hợp lệ; source/reference provenance đúng phạm vi audit; predictions và metrics khớp các bằng chứng đã nghiệm thu. Không phát hiện sai lệch cần xử lý giữa các bộ artifacts được đối chiếu.

Đây là ghi nhận nghiệm thu kỹ thuật để trưởng nhóm review. Không chỉnh sửa source, configs, checkpoint, splits, mapping hoặc artifacts trong bước lập báo cáo.

## 10. Trạng thái Official Evaluation

Manifest và metadata B1 đều ghi:

```json
{
  "official_test_used": false,
  "official_final_evaluation": false,
  "configuration_lock_approved": false,
  "official_evaluation_authorized": false
}
```

**Official Test chưa được sử dụng trong phiên B1 và trong lượt nghiệm thu này. Configuration Lock vẫn là BẢN NHÁP — CHƯA KHÓA CẤU HÌNH (DRAFT). Chưa có quyền Official Evaluation.**

Nghiệm thu B1 Validation không phê duyệt Configuration Lock, không mở Official Test và không triển khai Phase B2. Mọi bước Official Evaluation tiếp theo cần hoàn tất các điều kiện còn mở trong Configuration Lock, được trưởng nhóm phê duyệt và có cơ chế cấp quyền riêng. Báo cáo này không thay đổi các điều kiện đó.
