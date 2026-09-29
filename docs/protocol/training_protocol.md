# Training Protocol — DogBreedID

Tài liệu này quy định preprocessing, augmentation và training rules dùng chung để E0–E3 có thể so sánh công bằng.

## 1. Shared Training Rules

| Hạng mục | Giá trị |
|---|---|
| Input size | 224 × 224 |
| Loss | CrossEntropyLoss |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Effective batch size | 32 |
| Max epochs | 40 |
| Early stopping patience | 7 |
| Checkpoint metric | Validation Macro-F1 |
| Scheduler | ReduceLROnPlateau |
| Scheduler mode | max |
| Scheduler factor | 0.5 |
| Scheduler patience | 2 |
| Minimum LR | 1e-6 |
| Development seed E0–E4 | 42 |

Mixed precision được bật khi GPU hỗ trợ.

Official Test không tham gia training, early stopping hoặc hyperparameter selection.

## 2. Effective Batch Size

Mục tiêu chung:

```text
effective_batch_size = 32
```

Ví dụ:

```text
batch_size = 32, accumulation = 1
```

hoặc khi thiếu VRAM:

```text
batch_size = 16, accumulation = 2
```

Runtime settings như device, num_workers và path có thể khác giữa Colab/local; experimental settings không được tự ý thay đổi.

## 3. Training Preprocessing / Basic Augmentation

```text
RGB
→ RandomResizedCrop(224)
→ RandomHorizontalFlip
→ ColorJitter
→ RandomRotation
→ ImageNet Normalization
```

Freeze v1:

```yaml
RandomResizedCrop:
  size: 224
  scale: [0.85, 1.0]
  ratio: [0.9, 1.1]

RandomHorizontalFlip:
  p: 0.5

ColorJitter:
  brightness: 0.15
  contrast: 0.15
  saturation: 0.15
  hue: 0.05

RandomRotation:
  degrees: 10
```

ImageNet normalization:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

## 4. Validation / Calibration / Official Test

Cả ba dùng deterministic preprocessing:

```text
RGB
→ Resize(256)
→ CenterCrop(224)
→ ImageNet Normalization
```

Không áp dụng random augmentation.

Gradio inference phải dùng cùng preprocessing deterministic này.

## 5. E0 — Custom CNN

```text
Input 224×224
↓
Conv(3→32, 3×3, padding=1) + BN + ReLU + MaxPool(2)
↓
Conv(32→64, 3×3, padding=1) + BN + ReLU + MaxPool(2)
↓
Conv(64→128, 3×3, padding=1) + BN + ReLU + MaxPool(2)
↓
Conv(128→256, 3×3, padding=1) + BN + ReLU + MaxPool(2)
↓
Adaptive Global Average Pooling
↓
Dropout(0.3)
↓
Linear(256 → 30)
```

Training:

```text
learning_rate = 1e-3
optimizer = AdamW
weight_decay = 1e-4
```

## 6. E1 — MobileNetV2 Frozen

- Torchvision MobileNetV2 pretrained ImageNet weights.
- Freeze toàn bộ feature backbone.
- Chỉ train classifier 30 classes.

```text
classifier_lr = 1e-3
```

## 7. E2 — MobileNetV2 Partial Fine-tuning

```text
features[:14] → frozen
features[14:] → trainable
classifier    → trainable
```

Differential learning rate:

```text
backbone_lr   = 1e-4
classifier_lr = 5e-4
```

## 8. E3 — ResNet18 Partial Fine-tuning

```text
conv1, bn1, layer1, layer2, layer3 → frozen
layer4                              → trainable
fc                                  → trainable
```

Differential learning rate:

```text
layer4_lr     = 1e-4
classifier_lr = 5e-4
```

## 9. Checkpoint / Early Stopping

- Tính Validation metrics sau mỗi epoch.
- Best checkpoint = epoch có Validation Macro-F1 cao nhất.
- Early stopping nếu Macro-F1 không cải thiện trong 7 epoch liên tiếp.
- Scheduler theo Validation Macro-F1.
- Lưu best checkpoint, config, seed, git commit/hash và training/validation history.

## 10. Reproducibility

Development seed E0–E4:

```text
42
```

E5 stability seeds:

```text
[42, 123, 2026]
```

Seed phải được áp dụng cho Python, NumPy và PyTorch khi phù hợp.

## 11. Environment Rules

Code phải chạy được trên Google Colab và local GPU.

Không hard-code:

```text
/content/...
cuda:0
đường dẫn Google Drive cá nhân
```

Nếu official run chạy trên máy khác với máy phát triển, phải giữ nguyên experimental config và ghi hardware/runtime thực tế.

## 12. Chưa Freeze ở Foundation v1

Sẽ freeze trước experiment tương ứng:

- E4: CutMix alpha / probability;
- E6: chi tiết Temperature Scaling;
- E7: final confidence-threshold selection rule;
- E8: corruption severity;
- E9: quy tắc chọn mẫu Grad-CAM.
