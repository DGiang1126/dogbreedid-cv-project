# Evaluation Protocol — DogBreedID

Tài liệu này quy định metric, model selection và efficiency measurement để các experiment E0–E9 có thể so sánh công bằng.

## 1. Development vs Final Evaluation

- Train: cập nhật weights.
- Validation: checkpoint/model/hyperparameter selection.
- Calibration: calibration và selective prediction.
- Official Test: chỉ dùng sau khi configuration đã khóa.

## 2. Classification Metrics

Metric chính:

1. **Macro-F1** — ưu tiên để chọn checkpoint/model.
2. **Top-1 Accuracy** — metric chất lượng thứ hai.

Metric bổ sung:

- Top-5 Accuracy;
- Precision / Recall / F1 theo class;
- Confusion Matrix.

## 3. Checkpoint Selection

```text
best_checkpoint = epoch có Validation Macro-F1 cao nhất
```

Official Test không tham gia checkpoint selection.

## 4. E0–E5 Model Selection

E0–E4 dùng development seed 42.

Sau E0–E4:

- chọn Top-2 configuration theo Validation Macro-F1;
- chạy lại Top-2 bằng 3 seeds: **[42, 123, 2026]**;
- báo cáo mean ± standard deviation của Macro-F1 và Top-1.

Best Classification Model sau E5:

1. ưu tiên mean Validation Macro-F1;
2. nếu chênh lệch Macro-F1 ≤ 0.005, so mean Top-1 Accuracy;
3. nếu vẫn gần nhau, dùng Parameter Count, Model Size và Inference Latency để phá hòa.

## 5. Efficiency Metrics

Báo cáo:

- total parameter count;
- trainable parameter count;
- model size;
- inference latency.

### Model Size

Đo kích thước serialized `state_dict` và báo cáo bằng MiB.

### Latency Protocol

```text
batch_size = 1
warmup_runs = 20
timed_runs = 100
mode = model.eval() + torch.inference_mode()
```

Quy tắc:

- dùng cùng device/runtime cho các model được so sánh;
- input tensor đã preprocessing sẵn;
- không tính DataLoader / disk I/O;
- CUDA synchronize trước/sau đoạn timing;
- báo cáo median ms/image và mean ± std;
- ghi rõ CPU/GPU, precision và PyTorch version.

Không so trực tiếp latency đo từ hai phần cứng khác nhau.

## 6. Calibration — E6

Best Classification Model được đánh giá trước/sau Temperature Scaling.

Metrics:

- Expected Calibration Error (ECE);
- Multiclass Brier Score.

ECE mặc định:

```text
n_bins = 15
```

Temperature chỉ được fit trên Calibration Set.

## 7. Selective Prediction — E7

Dùng calibrated confidence và Calibration Set để khảo sát:

- confidence threshold;
- Coverage;
- Selective Accuracy;
- Selective Risk;
- Risk–Coverage Curve;
- AURC.

Final threshold selection rule sẽ được freeze trước E7.

Official Test không được dùng để chọn threshold.

REJECT nghĩa là:

```text
"Mô hình chưa đủ tự tin để đưa ra kết luận."
```

REJECT không đồng nghĩa Unknown Breed.

## 8. Robustness — E8

Đánh giá Best Model đã khóa trên:

- Clean;
- Gaussian Blur;
- Gaussian Noise;
- Occlusion.

Metric chính:

```text
Accuracy Drop = Clean Accuracy - Corrupted Accuracy
```

Severity cụ thể sẽ được freeze trước E8.

## 9. Error Analysis / Grad-CAM — E9

E9 sử dụng Official Test sau final evaluation để:

- tạo Confusion Matrix;
- lấy Top-5 confusion pairs;
- chọn mẫu đúng/sai tiêu biểu;
- ghi True Label, Predicted Label và confidence;
- phân tích Grad-CAM;
- ghi nhận failure pattern thực tế.

Grad-CAM là công cụ giải thích định tính hỗ trợ Error Analysis, không phải bằng chứng duy nhất để kết luận nguyên nhân.

## 10. Raw Outputs

Mỗi experiment lưu tối thiểu:

- experiment ID;
- config;
- seed;
- git commit/hash;
- best checkpoint;
- training/validation history;
- metrics;
- runtime;
- raw predictions khi chạy final evaluation.

Raw predictions của Official Test phải được lưu để có thể tính lại metric mà không cần rerun inference.

## 11. Foundation Freeze v1

Đã freeze:

- checkpoint metric = Validation Macro-F1;
- classification metrics;
- E5 seeds;
- Best Model selection rule;
- efficiency measurement protocol;
- ECE bins = 15;
- Official Test lock policy.

E7 threshold policy và E8 corruption severity sẽ được freeze trước phase tương ứng.
