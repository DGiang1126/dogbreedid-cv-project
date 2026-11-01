# DogBreedID — bộ tổng hợp E0–E9 (Task 06, Đức Anh)

Ngày: 30/10/2026. Nhánh: `duc-anh-task06-experiment-analysis`.
Nguồn repository tại commit `6af357405d38a7ae20fd4e87b820875abaac4aa3`. Đây là bộ bảng/hình phục vụ báo cáo, không phải bản nháp Chương 3–4.

## Cách đọc và đơn vị

- CSV UTF-8 BOM để mở bằng Excel; số thập phân dùng dấu chấm. Giá trị `fraction` nhân 100 khi trình bày phần trăm; giữ đầy đủ độ chính xác từ nguồn. Ô trống là không áp dụng.
- Tất cả mô hình phân loại 30 lớp, đầu vào 224×224. E0–E4: Validation 300 ảnh, development seed 42. Official Test: 1.934 ảnh. Calibration: 300 ảnh. Split seed 42; Train 2.400 ảnh theo Dataset Protocol.
- E5: E1/E4, ba seeds 42/123/2026; độ lệch chuẩn mẫu, ddof=1. Mỗi seed dùng cùng Validation 300 ảnh; không phải 900 ảnh khác nhau.
- E6: ECE 15 bins; NLL và multiclass Brier không có đơn vị; Brier là tổng theo lớp, không chia số lớp. Temperature giữ nguyên `0.80246283961825`.
- E7: confidence sau E6, ACCEPT khi `>= 0.5011672918550012`. Selective Accuracy chỉ trên ACCEPT; AURC theo grouped right-step/ties, không phải phần trăm accuracy. REJECT không có nghĩa Unknown Breed.
- E8: Accuracy Drop = Clean − Corrupted, đơn vị điểm phần trăm; các corruption ở sau crop, trước normalization. Không chạy lại E8.
- Efficiency: serialized state_dict MiB; latency ms/image, Windows CPU, float32, batch 1, 20 warmup và 100 timed runs; chỉ model forward, không tính disk I/O/DataLoader/preprocessing. Artifact không ghi tên CPU cụ thể. Không gán benchmark cho GTX 1650 hoặc so trực tiếp với độ trễ Gradio.
- E9: bảy file có tiền tố `e9_official_` được sao chép nguyên bytes từ ZIP được bàn giao; giữ nguyên schema/nhãn split `official` trong JSON nguồn. Ở bảng tổng hợp, `official` được chuẩn hóa tên thành `official_test`; không đổi kết quả.

## Danh mục bảng

| File | Nội dung và nguồn |
|---|---|
| `experiment_methods.csv` | E0–E9, cấu hình/phương pháp, seed, preprocessing, mục đích và config; E5 được xác nhận bằng artifact kết quả thay vì comment YAML còn cũ. |
| `validation_comparison.csv` | E0–E4, Top-1/Top-5/Macro-F1/loss/best epoch; đọc trực tiếp validation JSON. |
| `seed_stability.csv` | Sáu lượt seed E1/E4 và hai dòng mean ± sample SD, nguồn E5 JSON. |
| `efficiency_comparison.csv` | E0–E4: Macro-F1 chính xác từ JSON; params/size/latency giữ nguyên độ chính xác benchmark CSV, kèm môi trường. |
| `e6_e7_results_by_split.csv` | E6 trước/sau và E7 trên Calibration, Validation, Official; mỗi dòng có nguồn, đơn vị và split. |
| `official_results_summary.csv` | E4/E6/E7/E8/E9 trên Official, dạng mỗi metric một dòng. |
| `robustness_summary.csv` | E8 clean/blur/noise/occlusion, số đúng, accuracy, drop và severity. |
| `e9_official_confusion_matrix.csv` | Ma trận đếm 30×30, hàng thật/cột dự đoán, mapping canonical. |
| `e9_official_per_class_metrics.csv` | Precision/Recall/F1/Support của cả 30 lớp. |
| `e9_official_top5_confusion_pairs.csv` | Năm hướng nhầm True → Predicted, không gộp hai chiều; có support và tỷ lệ theo support. |
| `e9_official_run_summary.json`, `e9_official_e7_error_breakdown.json` | Summary E9 và bốn nhóm đúng/sai × ACCEPT/REJECT. |

## Nguồn và giới hạn provenance

1. Nguồn chính Official: `docs/final_evaluation/results/official_metrics_summary.json` và `official_test_summary.md`; các giá trị được chép từ JSON, không từ số phần trăm đã làm tròn.
2. E6/E7 Calibration: `outputs/metrics/E6/calibration_metrics.json`, `outputs/metrics/E7/selective_metrics.json`. Trường `official_calibration_execution` của E7 vẫn là split Calibration, không phải Official Test.
3. E6/E7 Validation: mục 6.1–6.2 của `docs/final_evaluation/task05_validation_reproducibility.md`. Raw replay artifacts ở máy trưởng nhóm, không nằm trong checkout này; các giá trị là trích từ báo cáo đã commit, không khai là kiểm chứng trực tiếp raw artifact. Giới hạn provenance lịch sử ở mục 8 của báo cáo nguồn vẫn giữ nguyên.
4. E9: `Official_E9_Aggregate_Handoff.zip`, SHA-256 `d268711cc2f5f1643e885feec74836ea67395c6d5154bd43d50f35d0ea2cd3fa`. Quyền bàn giao bảy aggregate được ghi ở mục G `docs/task_reports/thuan/29_10_task.md`, đã merge qua PR #23; tin nhắn nhóm xác nhận bàn giao thực tế cho Đức Anh. Không có Grad-CAM Official trong ZIP; không bổ sung ảnh hoặc kết luận nguyên nhân lỗi khi không có minh chứng.
5. E9 so với E4 có khác biệt float32: `0.8148914167528438` (1576/1934) và `0.8148913979530334` (Official classification JSON). Sai khác ~1,88e-8 là biểu diễn số; giữ nguyên cả hai. Macro-F1 trùng `0.8111032621745409`.
6. Task 06 ngày 27/10 và mục G report E9 ngày 30/10 là tham chiếu trạng thái mới; những dòng Task 04/Configuration Lock hoặc phần A–F cũ không thay thế kết quả hoàn tất Task 05 và E9. Không sửa các tài liệu lịch sử.
7. Thông tin training cấu hình chung: AdamW, weight decay 1e-4, effective batch 32, tối đa 40 epochs, early stopping patience 7, chọn checkpoint theo Validation Macro-F1; nguồn `configs/base.yaml`. E2 train features[14:]; E3 train layer4/fc. Augmentation chung: RandomResizedCrop 224 scale [0.85,1.0], ratio [0.9,1.1], flip 0.5, ColorJitter 0.15/0.15/0.15/0.05, rotation 10°. E4 thêm CutMix alpha 1, p 0.5. Cấu hình khai báo không thay thế metadata lịch sử từng run.

## Danh mục hình và chú thích dùng khi ghép báo cáo

Các PNG mới xuất 300 dpi. Chưa gán số hình Chương 3–4 để người biên tập đặt theo thứ tự bản báo cáo chung.

| File | Chú thích đề xuất / nguồn |
|---|---|
| `figures/validation_comparison.png` | So sánh Top-1, Top-5 và Macro-F1 E0–E4 trên Validation, seed 42, 300 ảnh/30 lớp; nguồn validation JSON E0–E4. |
| `figures/seed_stability.png` | Macro-F1 trung bình ± độ lệch chuẩn mẫu E1/E4; các ký hiệu là từng seed 42/123/2026, cùng 300 ảnh Validation; trục y phóng 76–85%; nguồn E5 JSON. |
| `figures/accuracy_efficiency.png` | Validation Macro-F1 và median forward latency E0–E4 trên Windows CPU, batch 1; nguồn benchmark.csv/environment.json. |
| `figures/robustness_comparison.png` | E8 Official Top-1, E4 seed 42, 1.934 ảnh, 30 lớp; nguồn Official metrics JSON. |
| `figures/official_reliability_diagram.png` | E6 Reliability Diagram Official, E4 seed 42, 1.934 ảnh, ECE 15 bins, T khóa; sao nguyên bytes từ results/reliability_diagram.png. |
| `figures/official_risk_coverage_curve.png` | E7 Risk–Coverage Official, E4 seed 42, 1.934 ảnh; nguồn results/risk_coverage_curve.png đã sửa PR #20, giữ nguyên bytes. Ngưỡng khóa áp dụng cho điểm operating point. |
| `e9_official_confusion_matrix.png` | E9 Official confusion matrix đếm; E4 seed 42, 1.934 ảnh/30 lớp; hàng thật, cột dự đoán; nguồn ZIP. |
| `e9_official_confusion_matrix_normalized.png` | E9 Official confusion matrix chuẩn hóa theo hàng; cùng tập/mô hình; nguồn ZIP. |
| `figures/e9_top5_confusion_pairs.png` | Năm hướng nhầm nhiều nhất E9 Official; E4 seed 42, 1.934 ảnh; số đếm cần đọc cùng support lớp thật; nguồn top5_confusion_pairs.csv trong ZIP. |

## Các kết luận được số liệu hỗ trợ

- E4 có Validation Macro-F1 cao nhất ở seed 42 (81,51%); E5 mean Macro-F1 E4 80,28% ± 1,17 điểm %, E1 79,95% ± 0,22 điểm %. Không suy ra khác biệt có ý nghĩa thống kê từ ba seeds.
- Official Top-1 81,49%, Top-5 99,53%, Macro-F1 81,11%. Mục tiêu Top-1 ≥85% chưa đạt, thiếu khoảng 3,51 điểm %. Không dùng Selective Accuracy 85,14% để tuyên bố đạt mục tiêu toàn tập.
- E6 Official ECE giảm 7,49% → 2,29%; không thay nhãn Top-1. E7 ACCEPT 1.736/1.934, coverage 89,76%, accuracy trong ACCEPT 85,14%.
- E8 giảm mạnh nhất ở blur: 16,29 điểm %, so với noise 8,95 và occlusion 5,07 điểm %. Đây là phạm vi các severity đã thử.
- E9 có 358 lỗi. Ba hướng nhầm đầu thuộc nhóm Northern Spitz; đây là thống kê hậu kiểm, không chứng minh nguyên nhân thị giác. Siberian Husky → Eskimo Dog: 28/92; Malamute → Siberian Husky: 21/78; Eskimo Dog → Siberian Husky: 14/50.
- E7 vẫn ACCEPT 258 dự đoán sai; REJECT 100 sai và 98 đúng. Grad-CAM là minh họa định tính và không phải bằng chứng nhân quả.

## Kiểm tra bộ tổng hợp

Đã thực hiện 50 kiểm tra số liệu trên artifact tổng hợp: mapping/30×30, tổng/đường chéo, metrics 30 lớp từ ma trận, Macro-F1, thứ tự và tỷ lệ top-five, E7 bốn nhóm, T/threshold, E5 ddof=1 và E8 count/accuracy/drop. Các bản sao E9 và hai biểu đồ Official được kiểm tra trùng bytes. Không chạy lại E9 CLI, Official inference, training hoặc toàn bộ regression của project.

### SHA-256 các nguồn đã đọc

Hash của bytes checkout tại lúc tổng hợp (có thể khác hash ở máy khác do LF/CRLF); ZIP hash là bytes file nhận được. Không hash hay mở raw Official predictions/dataset/checkpoint.

| Nguồn | SHA-256 |
|---|---|
| `configs/base.yaml` | `cc6639298eea1b9ab28fe7cd63bb0525a0345966d015d4052da6023f30fd9b09` |
| `configs/data.yaml` | `f4f710fb5f10a4b250df899c0198505ae0d16dbe4f2d22c0b72e80b1b080a960` |
| `configs/experiments/e0_custom_cnn.yaml` | `aceb0cc02d478dd20452b100280df854ee004c3172181ab428f6797f0ad81ce9` |
| `configs/experiments/e1_mobilenet_frozen.yaml` | `bd0a39f4ca516d08eb7fb51f961d9ed2911d4a018518da038055c3836be86ac8` |
| `configs/experiments/e2_mobilenet_partial.yaml` | `1a7f2a1723a824417117a158f3fbf200bf96f8dd5ce59057b1941ea42482b837` |
| `configs/experiments/e3_resnet18_partial.yaml` | `ee2143471347e8796a0d76198f9fb249d1b6b3192b184903b148a7a09df2f22f` |
| `configs/experiments/e4_cutmix.yaml` | `a5988f231a96783c13a69319364271652cf5c39b5597b0c633c09ddbf856d3cf` |
| `configs/experiments/e5_seed_stability.yaml` | `9c679c5cce7376afa97ffd45a55850a8694fb846016a33d62957e20089049e05` |
| `configs/experiments/e6_calibration.yaml` | `78e9650c6565e5add042f8eb40b20c5b5c73e2e6e0bd4a615f26000b5025557d` |
| `configs/experiments/e7_selective_prediction.yaml` | `a17ad79c89b16365f812cc52097c8fc419f369dcac54d06fa8e955066d5466b3` |
| `configs/experiments/e8_robustness.yaml` | `07a8fbc95494838eb2c8cd402e85a00d912e671c49d14c7321d218ff6c15bd56` |
| `configs/experiments/e9_error_analysis.yaml` | `52114a1b32b17e829216368870ed3162d59e3ef14c8989c3afef8c8d319c8377` |
| `data/metadata/class_mapping.json` | `79f08f8da0d65b3cb9cf256ac3b37688e112ce84b0352e5a600861b9b2327542` |
| `docs/final_evaluation/results/official_metrics_summary.json` | `e729313d7aa67c45b12e9ca7743b559470684696fdb98174131f2cf66f8e5825` |
| `docs/final_evaluation/results/reliability_diagram.png` | `b6b5ba0780467e99d028ef83badd73775efac2d2092f38505a98be99d8a16b44` |
| `docs/final_evaluation/results/risk_coverage_curve.png` | `5fc5f83f6a3a8b4341e0abd7eea1c52297bb6fcade4c094d9f476337083eee2c` |
| `docs/final_evaluation/task05_validation_reproducibility.md` | `0531b5f7c0d0e3e95efb961ef91a75a2c1534a6fc141a574400f5cc998438a6d` |
| `docs/protocol/dataset_protocol.md` | `a05e3b844e40414eb67a381eda67606f9f093e30cb3efc36b8e54ce498ae2f59` |
| `docs/protocol/evaluation_protocol.md` | `42edc40e693a3323f5b44a3b9889b4df97193acf2ec0ca1fa79208fb79e188ad` |
| `docs/task_reports/thuan/29_10_task.md` | `4310949944c15045b2816cb4b9ba69554a2c2e24a0738a9b97847b9a630c40c6` |
| `docs/team_tasks/27_10_task.md` | `a90b30c7207bfc2b33d2dd45271108b3674ca045737f171447f59f79082d8f0f` |
| `outputs/metrics/E0/seed_42/validation.json` | `6a4eabeb7d35543e575e7e82e5ef94f6ba0290c896c9caa246776a911ca3807e` |
| `outputs/metrics/E1/seed_42/validation.json` | `84e46d3bdd4d109553d576d5ad16647f37c17b3bcce12df584fe4b9dab31aa34` |
| `outputs/metrics/E2/seed_42/validation.json` | `bb6dc9e873888b8009e7e43a64a7f902c46f67f10d861227e946d4b26183474b` |
| `outputs/metrics/E3/seed_42/validation.json` | `df54247d08344ac0d4d1e63997e1850a04abe985790e287d93af80ab17bb7cea` |
| `outputs/metrics/E4/seed_42/validation.json` | `b303761bfe7028b80a4f88c82412c99afc23fc6f9f0a1da906a96834e071cc75` |
| `outputs/metrics/E5/seed_stability_summary.json` | `bc7afcf5b3888f3629d00313e6147b23c691068fa67088582eaa48f32d8bf585` |
| `outputs/metrics/E6/calibration_metrics.json` | `c7831bfb5a88e5ff9d82f875d036664dc12759d88c205375ae653d2df81ea6f5` |
| `outputs/metrics/E7/selective_metrics.json` | `d79e11bd1fac656ad5fbac8221ade7847b067af1f0c34128132abf5b374984f7` |
| `outputs/metrics/efficiency/benchmark.csv` | `5c4b904e5b3272dfb480b54df2c96f099c39bcf9309bed1a356e7519049b2805` |
| `outputs/metrics/efficiency/environment.json` | `56696f224eb91fd580393fa5ddc6e4e36ecd332fba162d445c022dd6e6de8858` |
