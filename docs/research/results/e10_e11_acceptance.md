# Nghiệm thu Task 07 E10/E11 — Lê Hồ Đức Anh

Nghiệm thu ngày 17/11/2026 trên PR #29, branch `research/e10-e11-frozen-cutmix` → `main`. Head bàn giao `901d60acdf2f7f40c5c09bb291b6220977d45b55`. `origin/main` lúc đối chiếu là `d43d9847a9cda75e5058d9daf7d1148c4039a0fe`. [PR #29](https://github.com/DGiang1126/dogbreedid-cv-project/pull/29)

- **VERIFIED** — đã kiểm tra trực tiếp trên source, artifact hoặc lần chạy nghiệm thu.
- **REPORTED** — số liệu từ báo cáo gốc cùng nguồn đối chiếu.
- **NOT VERIFIED** — nội dung nằm ngoài phép kiểm đã thực hiện.
- **BLOCKED** — phép kiểm thiếu điều kiện thực hiện. Đợt nghiệm thu này có đủ checkpoint, dữ liệu Validation và dependencies cho các phép kiểm bắt buộc.

## Phạm vi và bàn giao

**VERIFIED** — đã rà đủ 15 file gốc trong diff ba chấm với `origin/main`. Các file gồm 4 cấu hình E10-A/B và E11-A/B, 2 runner train/evaluate, 4 module trong `src/research/`, 2 bộ test, 1 báo cáo nhiệm vụ và 2 bảng tổng hợp Markdown/CSV. Merge thử bằng `git merge-tree` với `origin/main` cho kết quả sạch.

| Cặp | Kiến trúc | Backbone | CutMix A / B | Head trainable |
|---|---|---|---|---:|
| E10-A/B | torchvision EfficientNet-B0 | Frozen, ở chế độ eval khi train | Tắt / bật | 38.430 |
| E11-A/B | timm ViT-Tiny patch16-224 | Frozen, ở chế độ eval khi train | Tắt / bật | 5.790 |

**VERIFIED** — bốn cấu hình dùng 30 lớp và seed 42. Optimizer chỉ cập nhật classification head. Biến thể B dùng CutMix cho batch huấn luyện với `alpha=1.0` và xác suất `0.5`. Train runner chỉ lấy Train/Validation, evaluator chỉ lấy Validation và helper development từ chối `official_test`. Các thay đổi được nghiệm thu giới hạn trong Task 07 E10/E11. E4/E6/E7/E8, class mapping, split và Official outputs nằm ngoài diff PR.

**REPORTED** — tác giả đã chạy bốn lượt huấn luyện Frozen Backbone trên Train 2.400 ảnh, chọn best epoch theo Validation Macro-F1 và bàn giao checkpoint, kết quả đánh giá cùng báo cáo sáu cấu hình. [Báo cáo Đức Anh](../../task_reports/duc-anh/14_11_task.md) · [So sánh gốc](e10_e11_validation_comparison.md)

## Nguồn bằng chứng và checkpoint

**VERIFIED** — tải bốn tệp `{experiment}/seed_42/best.pt` từ Drive vào output root tạm. SHA-256 tính trên bytes tải về khớp hash người nghiệm thu cung cấp và trường `checkpoint_sha256` trong bốn `evaluation_summary.json` trên Drive. Các giá trị metric trong bốn summary khớp CSV trong PR. [Thư mục Drive bàn giao](https://drive.google.com/drive/folders/1qGdVpEiICNrqdtNWChSTPSixO-nZGlFB) · [Bảng CSV của PR](e10_e11_validation_metrics.csv)

| Run | SHA-256 thực tế, trùng expected | Integrity | Strict load | Epoch |
|---|---|---|---|---:|
| E10-A | `eab83d856587be31f663e510dbaff9576a7e77f4a80e7736608e3bdc8a4bac89` | VERIFIED | PASS | 12 |
| E10-B | `4df2c3d8ac6637dda8208ae2d7a79d200df34c799e701a0180f0f85e6b09f7b4` | VERIFIED | PASS | 7 |
| E11-A | `082eba3588944d6e259a1a239782b5bdeeecd8fa03869b29d335b8f38c7d4391` | VERIFIED | PASS | 6 |
| E11-B | `bf32076d3b818add172d0e607e087ae343c0cd428d0e2e2b49c39d18fda82f68` | VERIFIED | PASS | 8 |

**VERIFIED** — cả bốn checkpoint mở trên CPU với `torch.load(..., weights_only=True)`. Config lưu trong checkpoint khớp cấu hình đã resolve từ source PR. Replay dựng đúng kiến trúc với `force_no_pretrained=True`, gọi `load_state_dict(..., strict=True)` và kiểm tra forward hữu hạn với 30 logits. Bộ kiểm chứng chặn đường nạp pretrained của timm và đường tải weights của torchvision. SHA-256 của bản sao được kiểm lại sau replay.

## Validation

Manifest `data/splits/validation.csv` có SHA-256 `f3d73eb820b13b6b941f6a21cd3fd0455f9d2efd6b5c51c3f134c1956a72c0d9`. **VERIFIED** — 300 đường dẫn duy nhất thuộc `official_train`, 10 ảnh mỗi lớp và đủ cả 300 tệp. Replay dùng `evaluate_single` với `output_root` tạm, checkpoint bản sao và batch size 32. Kết quả replay ghi vào thư mục tạm đó. Không chạy Official Test hoặc retrain.

| Run | Trạng thái của số liệu | Top-1 | Top-5 | Macro-F1 | Val loss gốc | Val loss replay | Δ loss (replay − gốc) |
|---|---|---:|---:|---:|---:|---:|---:|
| E1 | REPORTED, Task 06 | 80,33% | 99,00% | 0,799640 | 0,62612492 | — | — |
| E4 | REPORTED, Task 06 | 81,67% | 98,67% | 0,815099 | 0,66204336 | — | — |
| E10-A | VERIFIED, replay PASS | 84,33% | 99,67% | 0,842268 | 0,43969146 | 0,43966950 | −2,1963 × 10⁻⁵ |
| E10-B | VERIFIED, replay PASS | 84,33% | 99,67% | 0,842181 | 0,53517858 | 0,53526188 | +8,3305 × 10⁻⁵ |
| E11-A | VERIFIED, replay PASS | 81,00% | 98,67% | 0,808029 | 0,58562289 | 0,58562971 | +6,8148 × 10⁻⁶ |
| E11-B | VERIFIED, replay PASS | 79,00% | 98,33% | 0,790218 | 0,67154765 | 0,67153997 | −7,6787 × 10⁻⁶ |

Các chỉ số E1/E4 khớp bảng Validation đã công bố. **NOT VERIFIED** — E1/E4 nằm ngoài replay lần này. E10/E11 có Δ Top-1, Δ Top-5 và Δ Macro-F1 bằng 0 ở độ chính xác ghi trong JSON. Sai số tuyệt đối cho phép của phép đối chiếu là `1e-7` cho ba chỉ số phân loại ở dạng tỷ lệ và `1e-4` cho loss. [Validation Task 06](../../task06/experiment_summary/validation_comparison.csv)

Replay chạy trên Windows 11, Python 3.12.4, torch 2.6.0+cu124, torchvision 0.21.0+cu124, timm 1.0.31 và NVIDIA GeForce GTX 1650 qua CUDA 12.4, tắt TF32. Loss lệch nhẹ trên hai môi trường thực thi. Nguyên nhân riêng của từng sai số chưa được đo tách biệt.

| Run | Median gốc (ms/ảnh) | Thiết bị gốc | Median replay (ms/ảnh) |
|---|---:|---|---:|
| E1 | 28,45 | CPU, Task 06 | — |
| E4 | 26,29 | CPU, Task 06 | — |
| E10-A | 26,78 | CUDA, RTX 4060 Laptop GPU | 25,62 |
| E10-B | 14,73 | CUDA, RTX 4060 Laptop GPU | 23,48 |
| E11-A | 8,49 | CUDA, RTX 4060 Laptop GPU | 10,87 |
| E11-B | 15,24 | CUDA, RTX 4060 Laptop GPU | 11,87 |

Benchmark replay trên GTX 1650 dùng batch 1, 20 warm-up và 100 lượt đo đồng bộ CUDA. Các cột thiết bị chỉ mô tả từng phép đo. [Efficiency Task 06](../../task06/experiment_summary/efficiency_comparison.csv)

## Phát hiện và sửa source

1. **VERIFIED** — builder E11 ở head bàn giao gọi tên không có tag. Registry timm 1.0.31, đúng phiên bản báo cáo, thực tế phân giải mặc định đó tới cùng ID `timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k`. Sửa builder dùng tag tường minh, dùng cùng định danh cho validator và provenance, đồng thời kiểm tra giá trị weights E10 theo enum `IMAGENET1K_V1`. Test xác nhận tag được chuyển vào timm, khớp ID Hugging Face và chạy được khi tắt pretrained. [Model card của timm](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k)
2. **VERIFIED** — evaluator cũ dùng `weights_only=False`. Chuyển sang `weights_only=True` và giữ strict-load. Cập nhật test checkpoint round-trip theo cùng chế độ nạp. Fixture CutMix và freeze khai báo weights hợp lệ để mỗi test kiểm tra đúng guard. Dữ liệu ảnh tổng hợp của test round-trip cố định theo chỉ số mẫu.
3. Dependency nghiên cứu là `timm==1.0.31`. Cài riêng cho môi trường nghiệm thu cùng `safetensors`. Khi tái lập từ một môi trường mới đã cài các dependency chung, chạy `python -m pip install timm==1.0.31` trước các test E10/E11.

## Kiểm thử đã chạy

| Lệnh | Passed | Failed | Skipped |
|---|---:|---:|---:|
| `python -m pytest tests/test_models.py tests/test_training.py tests/test_extended_models.py tests/test_extended_training.py -v` | 12 | 0 | 0 |
| `python -m pytest tests/test_cutmix.py tests/test_efficiency.py tests/test_final_inference.py tests/test_gradcam.py -q -rs` | 117 | 0 | 0 |

Lượt đầu sau khi thêm guard weights có 1 lỗi fixture thiếu trường đó. Bảng dùng kết quả chạy lại đầy đủ sau khi sửa fixture.

## Giới hạn khoa học và đề xuất merge

E10-A có Macro-F1 cao nhất trong sáu cấu hình trên Validation. Trong cùng kiến trúc, CutMix giữ nguyên Top-1 của E10 và giảm Top-1 của E11 2 điểm phần trăm ở seed 42. Kết quả đo trên một seed và 300 ảnh đã dùng để chọn epoch. Nguồn pretraining và preprocessing khác nhau giữa ViT-Tiny và các CNN. Vì vậy kết quả chỉ hỗ trợ mô tả các cấu hình đã chạy. Khả năng tổng quát trên seed khác, Official Test, External-600 và việc đối chiếu trực tiếp tensor backbone với bản pretrained upstream tại thời điểm chạy gốc là **NOT VERIFIED**.

**Nghiệm thu VERIFIED.** Đề xuất merge PR #29 vào `main` cho phạm vi mở rộng nghiên cứu E10/E11 dựa trên source review, 129 test đạt, 4/4 checkpoint khớp hash và strict-load, và 4/4 Validation replay PASS. [PR #29](https://github.com/DGiang1126/dogbreedid-cv-project/pull/29)
