# DogBreedID — Task 05: Ghi nhận nghiệm thu Phase A Preflight

**Ngày lập báo cáo:** 26/10/2026

**Repository:** `DGiang1126/dogbreedid-cv-project`

**Branch:** `task05-final-evaluation-preflight`

**Phạm vi:** Ghi nhận và kiểm tra bằng chứng Validation preflight đã lưu; trình trưởng nhóm review. Không chạy lại preflight, inference, training hoặc Official Evaluation khi lập tài liệu.

## 1. Kết luận và các phiên thực thi cần phân biệt

Bộ `outputs/preflight/task05_phase_a_validation_02/` ghi nhận **10 PASS, 0 FAIL, 0 BLOCKED**, `overall_status=PASS`, `execution_status=complete`, mode `validation`, `synthetic=false`. Cả 10 checks đều là bắt buộc (`required=true`).

Git SHA của phiên Phase A là **`2ca1f9d7042908e90ad0414b43c075ce529de113`**, khớp summary, manifest và bằng chứng Git trong checks. Thời điểm ghi báo cáo là `2026-10-26T02:04:47.418138+00:00`, tương ứng `2026-10-26T09:04:47.418138+07:00`.

Đây là **preflight dựa trên artifacts**: đọc predictions và bằng chứng đã có, tính lại metrics từ logits, kiểm tra protocol, hash và provenance. Phiên Phase A không tạo predictions mới, không chạy mô hình trên ảnh và không thực hiện Official Final Evaluation.

| Phiên | Git SHA | Vai trò |
| --- | --- | --- |
| Phase A Validation preflight `_02` | `2ca1f9d7042908e90ad0414b43c075ce529de113` | Kiểm tra artifacts và phạm vi source/reference được liệt kê trong manifest |
| Validation replay E4/E6/E7 | `5d3b393666c73b699671a2f420d1c6f3996ff992` | Source sản xuất predictions replay và kết quả E7 đã nghiệm thu |
| Validation E8 committed | `cf522ffa0b7d582ddb6557b6ff15b891e51fc117` | Source thực thi E8 gắn với artifact và báo cáo nghiệm thu Đức Anh |

Summary ghi rõ:

```json
{
  "official_test_used": false,
  "configuration_lock_approved": false,
  "official_evaluation_authorized": false,
  "inference_performed": false
}
```

Manifest cũng ghi `official_test_used=false` và `configuration_lock_approved=false`; hai cờ còn lại ở trên thuộc summary. **Configuration Lock vẫn là BẢN NHÁP — CHƯA KHÓA CẤU HÌNH. Official Test chưa được phép sử dụng.**

## 2. Nguồn bằng chứng

| Nguồn | Nội dung đối chiếu |
| --- | --- |
| `outputs/preflight/task05_phase_a_validation_02/preflight_summary.json` | Trạng thái tổng thể, số checks, Git SHA, thời điểm, metrics và các cờ an toàn |
| `outputs/preflight/task05_phase_a_validation_02/preflight_checks.json` | ID, trạng thái, tính bắt buộc và bằng chứng của từng check |
| `outputs/preflight/task05_phase_a_validation_02/preflight_manifest.json` | Runtime, input hashes, phạm vi Git audit, snapshot binding và E8 provenance |
| `.complete`, `.preflight_run` trong cùng thư mục `_02` | Token phiên và hash bytes của ba báo cáo |
| `docs/final_evaluation/configuration_lock.md` | Model, checkpoint, preprocessing, E6/E7/E8 cố định và các điều kiện phê duyệt còn mở |
| `docs/final_evaluation/task05_validation_reproducibility.md` | Bằng chứng replay Validation và giới hạn provenance lịch sử |
| `outputs/predictions/task05_validation_replay_01/` | NPZ, metadata, metrics, reliability diagram và marker E4/E6 |
| `outputs/predictions/task05_e7_validation_replay_01/` | Metrics, risk-coverage CSV và PNG E7 |
| `outputs/robustness/task04/validation/e8_validation_committed.json` | Kết quả và metadata E8 đã nghiệm thu |
| `docs/task_reports/duc-anh/21_10_task.md` | Ràng buộc artifact E8 với hash và source commit thực thi |

Khi lập tài liệu, đã đọc các báo cáo từ bytes, tính SHA-256 trực tiếp, đối chiếu inputs hiện có với manifest và đối chiếu source/reference với Git object tại commit Phase A. Metrics trong summary được đối chiếu trực tiếp với các JSON E4/E6/E7/E8 gốc. Không dùng tên thư mục làm bằng chứng duy nhất.

## 3. Kết quả cụ thể của 10 checks

Các ID và trạng thái dưới đây lấy nguyên từ `preflight_checks.json`. Những dòng chỉ ghi `detail="Evidence validated"` được diễn giải bằng manifest, input artifacts và logic validator tại commit Phase A, không coi chuỗi detail đó là bằng chứng độc lập đầy đủ.

| Check bắt buộc | Kết quả | Bằng chứng và phạm vi xác nhận |
| --- | --- | --- |
| `source_integrity` | PASS | `evidence.head` khớp SHA Phase A; `modified=[]`, `untracked=[]` trong 18 file source được audit. Danh sách tại mục 5. |
| `frozen_reference_integrity` | PASS | 14 references được Git audit, không modified/untracked; `snapshot_content_bound_to_git=true`. Snapshot references khớp nội dung Git commit, cho phép khác LF/CRLF. |
| `input_snapshot_stable` | PASS | Manifest ghi `private_snapshot_verified=true`, `capture_errors=[]`, hash từ cùng bytes được sao chép và validate. Khi lập báo cáo, 25 captured inputs và 18 observed source hashes khớp file hiện có; 7 marker ghi missing vẫn không tồn tại. |
| `protocol_references` | PASS | Frozen E4 là backbone Frozen + CutMix; E6/E7 lấy từ artifact canonical, Temperature `0.80246283961825`, threshold `0.5011672918550012`; Configuration Lock vẫn DRAFT. Configs/reference được ràng buộc với Git. |
| `prediction_bundle` | PASS | Bundle replay Validation gồm NPZ `logits`, `labels`, `image_ids`; 300 mẫu, 30 lớp; ID–nhãn đối chiếu canonical `data/splits/validation.csv`, metadata/preprocessing và reference hashes. Reliability diagram đọc hợp lệ; marker E4/E6 hoàn tất, không `.incomplete`. |
| `checkpoint_sha256` | PASS | Hash bytes của `outputs/checkpoints/E4/seed_42/best.pt` khớp SHA đã chốt tại mục 6. Check này xác minh hash, không tuyên bố Phase A đã nạp weights hoặc chạy model. |
| `e4_e6_metrics` | PASS | Tính lại bằng `predictor.compute_task1_metrics` từ logits replay và Temperature frozen, so sánh metrics JSON; kết quả tại mục 6 khớp báo cáo tái lập Validation. |
| `e7_metrics` | PASS | Tái sử dụng calibrated predictions, accept mask, selective metrics và risk-coverage/AURC của E7; đối chiếu metrics, input hashes, CSV và PNG. Temperature/threshold so sánh chính xác, counts nguyên chính xác. |
| `e8_artifact` | PASS | Artifact committed khớp checkpoint, 300 IDs, canonical Validation CSV, preprocessing, seed 42, bốn conditions và corruption config E8. Clean khớp logits replay; Accuracy Drop bằng Clean trừ Corrupted. Không tính lại predictions corrupted. |
| `e8_execution_provenance` | PASS | Artifact hash `1014bd8bfd4bd67136230122858d67a6654e55e6fe2cf709f882b53b29caac3f` và source `cf522ffa0b7d582ddb6557b6ff15b891e51fc117` khớp báo cáo nghiệm thu Đức Anh; commit tồn tại trong Git. Đây là đối chiếu bằng chứng đã nghiệm thu, không chạy lại E8. |

Manifest quy định tolerance metrics: relative `1e-9`, absolute `1e-12`; integer counts và protocol parameters dùng so sánh chính xác. Không dùng tolerance metrics để chấp nhận Temperature/threshold khác giá trị frozen; không refit hay lựa chọn lại tham số.

## 4. Marker hoàn tất và SHA-256 tính từ bytes thực tế

Đã đọc `.preflight_run` gồm 33 bytes: token ASCII `c774a26f1ed948879a6d8956f3afd8d8` kèm một byte LF cuối. `run_token` trong `.complete` trùng toàn bộ chuỗi này, kể cả LF. Không có `.incomplete` trong thư mục `_02`.

| File trong `_02` | Bytes | SHA-256 tính lại |
| --- | ---: | --- |
| `preflight_summary.json` | 2518 | `063e58dcbfe30b553bf9d8b5e06de8d6bc96aaa987cb65ffe4690140b7afe488` |
| `preflight_checks.json` | 3302 | `b6c0923fe3aff3ebbeb5e10b54d2ba5adf286eb12f546dce2f822700e1b92fda` |
| `preflight_manifest.json` | 24627 | `84849f4e6a0b76198b003d0a3dd91c267417b6885c8ed4fabcc8c69a9810f133` |
| `.complete` | 352 | `cb90cd447224d60c5b99898efc03a001f9850f2cc2f44e8dfecedec726d8659b` |
| `.preflight_run` | 33 | `acb201c907e2b18a1fd7eddc2815616edf9c05fa68434fcd32acddf6164f0146` |

Ba SHA-256 JSON khớp chính xác `output_sha256` trong `.complete`. JSON được parse từ các buffers đã đối chiếu hash; counts tính lại từ checks là 10/0/0, không chỉ tin trường `overall_status`.

Đã gọi consumer chỉ đọc `load_preflight_report()` trên bộ `_02`: kết quả `overall_status=PASS`, counts `10/0/0`. Consumer xác minh completion, parse chính buffers đã kiểm tra hash, đối chiếu trạng thái các checks bắt buộc và snapshot binding; không chạy lại preflight hoặc inference.

`.complete` xác nhận publish báo cáo hoàn tất và nhất quán, **không tự mang nghĩa PASS**. PASS ở đây còn dựa trên toàn bộ checks bắt buộc và snapshot evidence. Các hash/token là cơ chế kiểm tra tính toàn vẹn, không phải chữ ký xác thực chống người có quyền thay cả report và marker.

## 5. Source/reference integrity và giới hạn audit

### 5.1. Source: 18 file được audit

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
```

### 5.2. Frozen references: 14 file được audit

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

Khi lập báo cáo, HEAD vẫn là `2ca1f9d7042908e90ad0414b43c075ce529de113`. Đã đối chiếu nội dung hiện có của 18 source và 14 references với `git show <SHA>:<path>` sau chuẩn hóa CRLF thành LF; cả 32 file khớp, không có diff tracked trong các path audit. Các raw-byte hashes trong manifest cũng khớp file hiện có.

Manifest phân biệt 25 file `captured`, 18 file `observed_source` và 7 marker `missing`. Bảy marker thiếu là trạng thái được quan sát, không phải bảy artifacts bắt buộc bị thiếu: `.incomplete` ở E4/E6; `.complete`, `.incomplete`, `.inference_run` ở E7; ba marker tương tự ở thư mục E8. E7/E8 được kiểm tra theo định dạng artifacts hiện hành; không suy diễn có completion protocol giống E4/E6.

Snapshot có chiến lược `private read-only per-file snapshot`, hash basis `same bytes copied and validated`. `cross_file_atomic=false`; kiểm tra drift file gốc chỉ là bổ sung. Không coi đây là snapshot nguyên tử của toàn filesystem, attestation bytecode đã import, kiểm tra toàn repository hay toàn dependencies. Phạm vi PASS source/reference chỉ bao gồm các file nêu trên. Các thư mục output/evidence local untracked ngoài phạm vi đó không biến thành tracked/committed nhờ check này.

## 6. Đối chiếu E4/E6/E7/E8 trên Validation

Model và checkpoint theo Configuration Lock: **E4 MobileNetV2 Frozen + CutMix, seed 42, 30 lớp**. Checkpoint `outputs/checkpoints/E4/seed_42/best.pt` có SHA-256 đã tính lại từ bytes:

`73df0cc1c5b16866d4facc910f3e4d3ab481e3f12f1681a59fa7d61e3845c47b`

Preprocessing metadata khớp cấu hình: RGB → Resize(256) → CenterCrop(224) → ToTensor → ImageNet Normalization, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`. E8 thực hiện corruption trên tensor `[0,1]` trước normalization theo protocol đã nghiệm thu; Phase A đối chiếu config/metadata, không chạy lại pipeline ảnh.

| E4/E6, 300 mẫu / 30 lớp | Giá trị |
| --- | ---: |
| Top-1 theo số đếm / calibration | `0.8166666666666667` (245/300) |
| `classification.top1_accuracy` | `0.8166666626930237` |
| Top-5 Accuracy | `0.9866666793823242` |
| Macro-F1 | `0.815099216821925` |
| Temperature E6 | `0.80246283961825` |
| ECE bins | 15 |
| NLL trước / sau E6 | `0.6620433771121585` / `0.6412068263814583` |
| ECE trước / sau E6 | `0.10925347267077187` / `0.07963164098033844` |
| Multiclass Brier trước / sau E6 | `0.30574911689549933` / `0.29675198232729383` |

Hai cách ghi Top-1 biểu diễn cùng 245/300 với độ chính xác số khác nhau; giữ nguyên các trường JSON. Summary ghi `top1_prediction_unchanged=true` sau Temperature Scaling.

| E7 | Giá trị |
| --- | ---: |
| Threshold frozen | `0.5011672918550012` |
| ACCEPT / REJECT | 261 / 39 |
| ACCEPT đúng | 220 |
| Coverage | `0.87` |
| Selective Accuracy | `0.842911877394636` |
| Selective Risk | `0.15708812260536398` |
| AURC, grouped right-step over confidence tie groups | `0.09093638678902743` |

E7 dùng confidence sau E6 và quy tắc `confidence >= threshold`; không chọn lại threshold. Các kết quả E4/E6/E7 khớp báo cáo Validation Reproducibility.

| Điều kiện E8 đã nghiệm thu | Accuracy | Accuracy Drop so với Clean |
| --- | ---: | ---: |
| Clean | `0.8166666666666667` | Không áp dụng |
| Gaussian Blur | `0.65` | `0.16666666666666663` |
| Gaussian Noise | `0.68` | `0.1366666666666666` |
| Occlusion | `0.76` | `0.05666666666666664` |

Severity không thay đổi: Blur kernel 7, sigma 1.5; Noise mean 0, std 0.05, clamp `[0,1]`; Occlusion vuông, cạnh bằng 0.30 cạnh ảnh, fill `[0.5,0.5,0.5]`; global seed 42. Drop giữ dấu và bằng Clean Accuracy trừ Corrupted Accuracy. Blur giảm accuracy nhiều nhất, Occlusion ít nhất trong ba corruption ở severity này. Phase A đối chiếu artifact E8 đã nghiệm thu, không tạo logits corrupted mới hoặc chứng minh lại thực thi CUDA lịch sử.

Các hash chính đã đối chiếu với manifest và bytes hiện có:

| Input | SHA-256 |
| --- | --- |
| NPZ replay E4 | `5aae305b3a29e842fad313b5f3eae6edc1e2d3ca31b6e9228f4bc9c17442682a` |
| Canonical Validation CSV | `f3d73eb820b13b6b941f6a21cd3fd0455f9d2efd6b5c51c3f134c1956a72c0d9` |
| E6 `temperature.json` | `338a09892fc7a2928ee932c8faa562dae39cfa47e20389e43daade2125c3aa92` |
| E7 `threshold.json` | `b9bc8c895ae075a87e4fc5c84679cbe1a12f305da1eac31c4f8bc788da98ff27` |
| E8 `e8_validation_committed.json` | `1014bd8bfd4bd67136230122858d67a6654e55e6fe2cf709f882b53b29caac3f` |

Báo cáo Validation Reproducibility đã xác nhận NPZ lịch sử/replay có 9.000/9.000 logits trùng tuyệt đối, maximum absolute difference `0.0`. Đây là bằng chứng từ bước trước; lượt lập tài liệu này không chạy lại phép inference hay gắn lại provenance lịch sử. Source và thay đổi local của phiên lịch sử với metadata SHA `891edf9e69a6b724bcdc20b02143888e8f35bbd8` vẫn chưa được xác minh đầy đủ.

## 7. Môi trường của phiên preflight theo manifest

| Thành phần | Giá trị `manifest.runtime` |
| --- | --- |
| Python | `3.12.4` |
| PyTorch | `2.6.0+cu124` |
| Torchvision | `0.21.0+cu124` |
| NumPy | `2.5.3` |
| CUDA build | `12.4` |
| CUDA available | `true` |

Đây là runtime được ghi của phiên preflight `_02`, không phải phép đo lại môi trường khi lập báo cáo. CUDA khả dụng không có nghĩa Phase A đã chạy inference trên GPU. `inference_producer_metadata` ghi riêng phiên replay E4/E6: Git SHA `5d3b393666c73b699671a2f420d1c6f3996ff992`, thời điểm `2026-10-25T14:59:33.314533+00:00`, device `cuda`, batch size 32, num_workers 0. Hai phiên trùng một số version không làm chúng thành cùng một lần thực thi.

Các Git SHA và runtime Validation/preflight này chưa được phê duyệt làm source hoặc môi trường Official Final Evaluation.

## 8. Các điều kiện còn thiếu trước Official Evaluation

Configuration Lock mục 9.2 hiện chỉ đánh dấu hoàn tất điều kiện Data Contract đã được kiểm chứng và nghiệm thu sau Task 04. Ba điều kiện sau vẫn chưa tích và báo cáo này không thay đổi chúng:

- [ ] Kiểm tra source Final Evaluation và dependencies cho phạm vi Official Evaluation, chốt Git SHA và môi trường thực thi chính thức.
- [ ] Hoàn thành integration tests cho Official Final Evaluation runners và cơ chế cấp quyền riêng; Phase A PASS không thay thế các tests đó.
- [ ] Trưởng nhóm phê duyệt Configuration Lock.

Trước khi được phép dùng Official Test còn cần trưởng nhóm review bằng chứng Phase A, xác định phạm vi kiểm tra tính toàn vẹn dữ liệu/artifacts chính thức, quy trình output độc quyền và lưu bằng chứng, cùng cơ chế authorization có kiểm soát. Không xóa guard Task 04 để mở quyền. Việc triển khai/kiểm thử Phase B và phê duyệt sử dụng Official Test là công việc riêng sau bước review này.

Giới hạn còn giữ nguyên: source audit không bao phủ toàn môi trường/dependency supply chain; snapshot không nguyên tử xuyên nhiều file; hash marker không phải chữ ký; provenance lịch sử không được suy ngược từ logits trùng nhau; chưa chứng minh tái lập trên hệ điều hành/môi trường CUDA khác. Artifacts/evidence local cần phương án lưu trữ do trưởng nhóm quyết định; tài liệu này không chứng minh chúng đã được đưa lên GitHub hoặc CI đã chạy.

## 9. Phạm vi thao tác khi lập báo cáo

Chỉ tạo tài liệu này sau khi đọc và đối chiếu bằng chứng hiện có; không sửa metrics, artifacts, configs, Data Contract hoặc Configuration Lock. Không chạy inference/training, không truy cập Official Test và không thực hiện git add/commit/push/merge. Trạng thái nghiệm thu Phase A nêu trên được trình trưởng nhóm review, không phải quyết định khóa cấu hình hoặc cấp quyền Official Evaluation.
