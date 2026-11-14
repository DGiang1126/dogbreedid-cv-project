# E10/E11 Validation & Efficiency Comparison (Frozen Backbone + CutMix)

## 1. Bối cảnh và thiết lập thí nghiệm

Nhiệm vụ mở rộng kiến trúc theo góp ý của GVHD: bổ sung **EfficientNet-B0 (E10)** và **ViT-Tiny patch16-224 (E11)** ở chế độ **Frozen Backbone** (chỉ train classification head 30 lớp) với hai biến thể không CutMix (`-A`) và có CutMix (`-B`, `alpha=1.0`, `probability=0.5`), đối chứng với **E1** và **E4** (MobileNetV2 Frozen) từ số liệu đã công bố.

- **Dữ liệu:** Split `train` (2,400 ảnh) và `validation` (300 ảnh, 30 lớp), `seed=42`. **Không truy cập Official Test.**
- **Siêu tham số chung:** AdamW (`classifier_lr=1e-3`, `weight_decay=1e-4`), CrossEntropyLoss, effective batch size `32`, max epochs `40`, early stopping patience `7` theo Validation Macro-F1, `ReduceLROnPlateau` (`factor=0.5`, `patience=2`, `min_lr=1e-6`).
- **Backbone freeze policy:** Toàn bộ backbone đặt `requires_grad=False` và giữ ở chế độ `eval()` trong suốt quá trình huấn luyện để không làm thay đổi BatchNorm running stats, Dropout hoặc Stochastic Depth.
- **Provenance & Preprocessing:**
  - **E1/E4 (MobileNetV2):** `torchvision` `MobileNet_V2_Weights.IMAGENET1K_V1`, ImageNet normalization `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
  - **E10-A/B (EfficientNet-B0):** `torchvision==0.28.0+cu126`, `EfficientNet_B0_Weights.IMAGENET1K_V1`, ImageNet normalization `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
  - **E11-A/B (ViT-Tiny):** `timm==1.0.31`, `timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k`, normalization theo pretrained config của model là `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`. Cả E11-A và E11-B dùng cùng preprocessing này.

---

## 2. Bảng tổng hợp 6 cấu hình (Validation N=300)

| Exp | Kiến trúc | CutMix | Best Epoch | Top-1 Acc | Top-5 Acc | Macro-F1 | Val Loss | Total Params | Trainable Params | Size (MiB) | Latency Median (ms) | Device Benchmark |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **E1** | MobileNetV2 | Không | 18 | 80.33% | 99.00% | 0.7996 | 0.6261 | 2,262,302 | 38,430 | 8.86 | 28.45 | `cpu` (Task 06) |
| **E4** | MobileNetV2 | Có | 25 | 81.67% | 98.67% | 0.8151 | 0.6620 | 2,262,302 | 38,430 | 8.86 | 26.29 | `cpu` (Task 06) |
| **E10-A** | EfficientNet-B0 | Không | 12 | **84.33%** | **99.67%** | **0.8423** | **0.4397** | 4,045,978 | 38,430 | 15.72 | 26.78 | `cuda` (RTX 4060) |
| **E10-B** | EfficientNet-B0 | Có | 7 | **84.33%** | **99.67%** | 0.8422 | 0.5352 | 4,045,978 | 38,430 | 15.72 | 14.73 | `cuda` (RTX 4060) |
| **E11-A** | ViT-Tiny (`patch16_224`) | Không | 6 | 81.00% | 98.67% | 0.8080 | 0.5856 | 5,530,206 | 5,790 | 21.15 | 8.49 | `cuda` (RTX 4060) |
| **E11-B** | ViT-Tiny (`patch16_224`) | Có | 8 | 79.00% | 98.33% | 0.7902 | 0.6715 | 5,530,206 | 5,790 | 21.15 | 15.24 | `cuda` (RTX 4060) |

---

## 3. Phân tích tác động của CutMix theo từng kiến trúc

| Kiến trúc | Cặp so sánh | Δ Top-1 (Có - Không) | Δ Macro-F1 (Có - Không) | Δ Val Loss | Nhận xét |
|---|---|---:|---:|---:|---|
| **MobileNetV2** | E4 vs E1 | +1.33% | **+0.0155** | +0.0359 | CutMix giúp cải thiện nhẹ khả năng phân biệt trên lớp tuyến tính MobileNetV2 (+1.55 điểm % Macro-F1). |
| **EfficientNet-B0** | E10-B vs E10-A | 0.00% | **-0.0001** | +0.0955 | Top-1 (84.33%) và Top-5 (99.67%) giữ nguyên, Macro-F1 gần như không đổi (-0.0001), nhưng validation loss tăng từ 0.4397 lên 0.5352 do nhãn mềm làm giảm độ tự tin logits. |
| **ViT-Tiny** | E11-B vs E11-A | -2.00% | **-0.0178** | +0.0859 | CutMix làm giảm Macro-F1 (-0.0178) và Top-1 (-2.00%). Với backbone ViT-Tiny bị đóng băng hoàn toàn và head rất nhỏ (5,790 params = `192 × 30 + 30`), việc ghép patch trộn ảnh có thể gây nhiễu biểu diễn token đầu ra mà head tuyến tính không đủ dung lượng thích nghi. |

---

## 4. Nhận xét tổng hợp & Giới hạn

1. **So sánh kiến trúc (Frozen Backbone):** Trên tập Validation (N=300), **EfficientNet-B0 (E10-A: Macro-F1 0.8423, Top-1 84.33%)** vượt trội cả MobileNetV2 (E4: 0.8151 / 81.67%) và ViT-Tiny (E11-A: 0.8080 / 81.00%), đồng thời giữ nguyên số tham số trainable ở head là `38,430` (bằng MobileNetV2, gấp ~6.6× so với head `5,790` của ViT-Tiny).
2. **Chi phí mô hình & Latency:** EfficientNet-B0 nặng gấp ~1.77× MobileNetV2 (15.72 MiB vs 8.86 MiB), còn ViT-Tiny nặng gấp ~2.39× (21.15 MiB). Lưu ý rằng latency của E1/E4 trong bảng Task 06 được đo trên `cpu`, còn E10/E11 được đo trên `cuda` (RTX 4060 Laptop GPU, Windows WDDM); sự chênh lệch latency giữa hai cặp giống hệt kiến trúc (E10-A 26.78 ms vs E10-B 14.73 ms; E11-A 8.49 ms vs E11-B 15.24 ms) phản ánh dao động lập lịch GPU laptop khi đo batch=1, không phải khác biệt kiến trúc.
3. **Giới hạn đánh giá:** Kết quả mới đo trên 1 seed (`42`) và tập Validation (`N=300`). Không thể kết luận mở rộng sang Official Test hay External-600 khi chưa đánh giá thêm; cũng cần lưu ý ViT-Tiny dùng trọng số `augreg_in21k_ft_in1k` (pretrain ImageNet-21k + fine-tune ImageNet-1k) nên nguồn pretraining khác với ImageNet-1k của MobileNetV2 và EfficientNet-B0.
