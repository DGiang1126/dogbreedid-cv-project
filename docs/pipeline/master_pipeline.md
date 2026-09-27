# DogBreedID — Master Pipeline

> Tài liệu nội bộ để toàn bộ thành viên thống nhất pipeline, thứ tự thực nghiệm và nguyên tắc đánh giá của đề tài DogBreedID.
>
> Khi thực hiện task, thành viên phải tuân theo pipeline này, các file trong `docs/protocol/` và config được nhóm trưởng cung cấp.

---

## 1. Mục tiêu đề tài

DogBreedID xây dựng hệ thống **phân loại giống chó từ hình ảnh**.

Input:

```text
1 ảnh RGB
→ chủ yếu chứa 1 con chó
```

Output:

```text
1 trong 30 giống chó đã chọn cố định
+
Top-1 prediction
+
Top-5 predictions
+
Confidence
+
Inference latency
```

Mô hình chính được nghiên cứu:

```text
Custom CNN
MobileNetV2
ResNet18
```

Đề tài không chỉ so sánh độ chính xác mà còn đánh giá:

```text
Classification Quality
+
Efficiency
+
Confidence Calibration
+
Selective Prediction
+
Robustness
+
Error Analysis
```

Mô hình cuối cùng được tích hợp vào ứng dụng **Gradio**.

---

# 2. Pipeline tổng thể

```text
Stanford Dogs Dataset
        ↓
Select Fixed 30 Breeds
        ↓
Class Mapping
        ↓
Data Audit
        ↓
Official Train / Official Test
        ↓
Split Official Train
        ↓
Train / Validation / Calibration
        ↓
Preprocessing + Basic Augmentation
        ↓
┌─────────────────────────────────────┐
│       MODEL DEVELOPMENT             │
│                                     │
│ E0 — Custom CNN                     │
│ E1 — MobileNetV2 Frozen             │
│ E2 — MobileNetV2 Partial            │
│ E3 — ResNet18 Partial               │
│ E4 — Best Transfer Model + CutMix   │
│ E5 — Top-2 Configurations × 3 Seeds │
└─────────────────────────────────────┘
        ↓
BEST CLASSIFICATION MODEL
        ↓
┌─────────────────────────────────────┐
│      RELIABILITY DEVELOPMENT        │
│                                     │
│ E6 — Temperature Scaling            │
│ E7 — Selective Prediction           │
└─────────────────────────────────────┘
        ↓
LOCK CONFIGURATION
        ↓
OFFICIAL FINAL TEST
        ↓
┌─────────────────────────────────────┐
│          FINAL EVALUATION           │
│                                     │
│ Classification                      │
│ Efficiency                          │
│ Calibration                         │
│ Risk–Coverage                       │
│ E8 — Robustness                     │
│ E9 — Error Analysis + Grad-CAM      │
└─────────────────────────────────────┘
        ↓
Gradio Demo
```

Pipeline được chia thành **3 pha chính**:

```text
Pha 1 — Model Development
Train + Validation
→ xây dựng và lựa chọn model.

Pha 2 — Reliability Development
Calibration Set
→ học Temperature và confidence threshold.

Pha 3 — Final Evaluation
Official Test
→ chỉ chạy sau khi toàn bộ configuration đã được khóa.
```

Mục tiêu của cách chia này là tránh sử dụng Official Test trong quá trình lựa chọn mô hình.

---

# 3. Dataset Preparation

## Step 1 — Chọn 30 giống chó

Nguồn dữ liệu chính:

```text
Stanford Dogs Dataset
```

Nhóm chỉ sử dụng:

```text
30 giống chó cố định
```

Danh sách 30 giống phải được chốt **trước khi huấn luyện**.

Mỗi giống có:

```text
class_id
breed_name
```

Mapping chung được lưu tại:

```text
data/metadata/class_mapping.json
```

Tất cả thành viên phải sử dụng cùng class mapping.

Không được tự thay đổi danh sách lớp theo kết quả model.

---

## Step 2 — Data Audit

Trước khi training cần kiểm tra dataset.

Các nội dung cần audit:

```text
Số lượng ảnh mỗi class
Số lượng ảnh mỗi split
Ảnh lỗi / không đọc được
Đường dẫn bất thường
Image size
Aspect ratio
Class mapping
```

Ngoài ra cần ghi nhận một số trường hợp ảnh khó:

```text
Complex background
Blur
Occlusion
Dog too small
Unusual pose
Poor lighting
```

Các trường hợp này có thể được sử dụng lại trong phần Error Analysis.

Code liên quan:

```text
src/data/audit.py
```

Notebook nếu cần:

```text
notebooks/exploration/
```

---

# 4. Data Split

Stanford Dogs có:

```text
Official Train
Official Test
```

Chỉ **Official Train** được chia tiếp.

```text
Official Train
│
├── Train       = 80%
├── Validation  = 10%
└── Calibration = 10%
```

Vai trò:

### Train

```text
Dùng để cập nhật model weights.
```

### Validation

```text
Dùng để:
- chọn checkpoint
- chọn model
- chọn hyperparameter
- so sánh E0–E5
```

### Calibration

```text
Dùng để:
- học Temperature Scaling
- chọn confidence threshold
```

### Official Test

```text
Chỉ dùng cho Final Evaluation.
```

Split phải sử dụng:

```text
Stratified Split
+
Fixed Random Seed
```

Split sau khi tạo phải được lưu tại:

```text
data/splits/
```

Tất cả thành viên sử dụng **cùng một split**.

---

# 5. Quy tắc Official Test

Trong quá trình E0–E7:

```text
KHÔNG sử dụng Official Test
để lựa chọn hoặc thay đổi:
```

```text
model
epoch
learning rate
backbone
augmentation
CutMix
checkpoint
temperature
confidence threshold
```

Official Test chỉ được sử dụng sau bước:

```text
LOCK CONFIGURATION
```

Mục tiêu là tránh:

```text
Test Leakage
```

---

# 6. Preprocessing

Tất cả model sử dụng input resolution:

```text
224 × 224
```

## Training

```text
RGB
↓
RandomResizedCrop(224 × 224)
↓
HorizontalFlip
↓
ColorJitter nhẹ
↓
Rotation nhẹ
↓
ImageNet Normalization
```

Training có stochastic augmentation.

---

## Validation

```text
RGB
↓
Resize(256)
↓
CenterCrop(224 × 224)
↓
ImageNet Normalization
```

Deterministic.

---

## Calibration

Giống Validation:

```text
RGB
↓
Resize(256)
↓
CenterCrop(224 × 224)
↓
ImageNet Normalization
```

---

## Official Test

Giống Validation và Calibration:

```text
RGB
↓
Resize(256)
↓
CenterCrop(224 × 224)
↓
ImageNet Normalization
```

Inference sau này cũng phải sử dụng pipeline deterministic này.

Code chung:

```text
src/data/transforms.py
```

Config chung:

```text
configs/data.yaml
```

Không thành viên nào tự tạo preprocessing riêng cho model của mình.

---

# 7. Shared Training Pipeline

Các model khác nhau nhưng phải sử dụng chung:

```text
Dataset Loader
Preprocessing
Trainer
Validator
Checkpoint Logic
Metrics
Logging
Seed
```

Các file chung:

```text
src/data/dataset.py
src/data/transforms.py

src/training/trainer.py
src/training/validator.py
src/training/checkpoint.py

src/evaluation/metrics.py

src/utils/config.py
src/utils/seed.py
src/utils/logger.py
```

Không xây mỗi model thành một pipeline training độc lập.

Notebook Colab chỉ dùng để:

```text
clone repository
↓
load config
↓
import src/
↓
run experiment
↓
show result
↓
save outputs
```

Không copy toàn bộ model/training/evaluation logic vào notebook.

---

# 8. Model Development — E0 đến E5

## E0 — Custom CNN

Model:

```text
Custom CNN
```

Training:

```text
Train From Scratch
```

Kiến trúc dự kiến:

```text
3–4 Convolution Blocks
+
Classifier
```

Mục tiêu:

```text
Tạo baseline CNN huấn luyện hoàn toàn từ đầu.
```

Dùng để trả lời:

> Transfer Learning cải thiện bao nhiêu so với model học từ đầu?

Code:

```text
src/models/custom_cnn.py
```

Config:

```text
configs/experiments/e0_custom_cnn.yaml
```

Kết quả chính:

```text
Top-1 Accuracy
Macro-F1
Top-5 Accuracy
Parameter Count
Model Size
Inference Latency
```

---

## E1 — MobileNetV2 Frozen

Model:

```text
MobileNetV2 pretrained
```

Chiến lược:

```text
Freeze Backbone
↓
Train Classification Head
```

Mục tiêu:

```text
Đánh giá Transfer Learning khi pretrained features được giữ cố định.
```

Code:

```text
src/models/mobilenet_v2.py
```

Config:

```text
configs/experiments/e1_mobilenet_frozen.yaml
```

---

## E2 — MobileNetV2 Partial Fine-tuning

Model:

```text
MobileNetV2 pretrained
```

Chiến lược:

```text
Freeze phần đầu backbone
+
Unfreeze một phần các layer cuối
+
Fine-tune
```

Mục tiêu:

```text
So sánh MobileNetV2 Frozen
vs
MobileNetV2 Partial Fine-tuning
```

Qua đó đánh giá tác động của fine-tuning.

Code:

```text
src/models/mobilenet_v2.py
```

Config:

```text
configs/experiments/e2_mobilenet_partial.yaml
```

---

## E3 — ResNet18 Partial Fine-tuning

Model:

```text
ResNet18 pretrained
```

Chiến lược:

```text
Partial Fine-tuning
```

Mục tiêu:

```text
So sánh một backbone lớn hơn với MobileNetV2
trong cùng protocol.
```

Code:

```text
src/models/resnet18.py
```

Config:

```text
configs/experiments/e3_resnet18_partial.yaml
```

So sánh chính:

```text
MobileNetV2 Partial
vs
ResNet18 Partial
```

Các yếu tố cần xét:

```text
Top-1 Accuracy
Macro-F1
Parameters
Model Size
Inference Latency
```

---

# 9. E4 — CutMix Ablation

Sau E0–E3:

```text
chọn Best Transfer Model
```

Sau đó chạy:

```text
Best Transfer Model
+
CutMix
```

So sánh:

```text
Basic Augmentation
vs
Basic Augmentation + CutMix
```

Mục tiêu:

```text
Kiểm tra CutMix có thực sự cải thiện model hay không.
```

Config:

```text
configs/experiments/e4_cutmix.yaml
```

Nếu CutMix không cải thiện Validation performance:

```text
vẫn lưu kết quả
```

và model cuối có thể tiếp tục sử dụng:

```text
Basic Augmentation
```

Không được bỏ kết quả âm.

---

# 10. E5 — Seed Stability

Sau E0–E4:

```text
Chọn Top-2 configurations
```

Mỗi configuration chạy:

```text
3 random seeds
```

Ví dụ:

```text
Config A
├── Seed 1
├── Seed 2
└── Seed 3

Config B
├── Seed 1
├── Seed 2
└── Seed 3
```

Báo cáo:

```text
Top-1 Accuracy → mean ± std
Macro-F1       → mean ± std
```

Mục tiêu:

```text
Không chọn model chỉ vì một lần chạy may mắn.
```

Config:

```text
configs/experiments/e5_seed_stability.yaml
```

Sau E5:

```text
→ chọn BEST CLASSIFICATION MODEL
```

---

# 11. Lựa chọn Best Classification Model

Việc lựa chọn không chỉ dựa vào một metric.

Cần xem đồng thời:

```text
Top-1 Accuracy
Macro-F1
Top-5 Accuracy
Seed Stability
Parameter Count
Model Size
Inference Latency
```

Nếu hai model có classification quality gần nhau:

```text
ưu tiên model nhẹ hơn
và có inference latency thấp hơn.
```

Mục tiêu trong đề cương đối với model tốt nhất:

```text
Top-1 Accuracy ≥ 0.85
```

trên Official Test.

Đây là mục tiêu của đề tài, không áp đặt cho Custom CNN baseline.

---

# 12. E6 — Temperature Scaling

Sau khi Best Classification Model đã được chọn:

```text
Best Model
↓
Logits
↓
Temperature Scaling
↓
Calibrated Probability
```

Temperature `T` chỉ được học trên:

```text
Calibration Set
```

Không học `T` trên Official Test.

So sánh:

```text
Before Temperature Scaling
vs
After Temperature Scaling
```

Metrics:

```text
ECE
Brier Score
```

Temperature Scaling không nhằm thay đổi Top-1 class prediction.

Mục tiêu:

```text
Confidence phản ánh tốt hơn khả năng dự đoán đúng của model.
```

Code:

```text
src/calibration/temperature_scaling.py
src/calibration/calibration_metrics.py
```

Config:

```text
configs/experiments/e6_calibration.yaml
```

---

# 13. E7 — Selective Prediction

Sau calibration:

```text
Calibrated Softmax
↓
Confidence
↓
Confidence Threshold
```

Decision:

```text
confidence >= threshold
→ ACCEPT

confidence < threshold
→ REJECT
```

Nếu ACCEPT:

```text
hiển thị prediction bình thường
```

Nếu REJECT:

```text
"Mô hình chưa đủ tự tin để đưa ra kết luận."
```

Lưu ý:

```text
REJECT ≠ Unknown Breed
```

DogBreedID không phải Open-set Recognition system.

Threshold chỉ được chọn trên:

```text
Calibration Set
```

Evaluation:

```text
Coverage
Selective Accuracy
Selective Risk
Risk–Coverage Curve
AURC
```

Có thể quan sát các coverage point như:

```text
100%
90%
80%
```

nhưng threshold cuối cùng phải được chọn trước khi chạy Official Test.

Code:

```text
src/selective/threshold.py
src/selective/risk_coverage.py
```

Config:

```text
configs/experiments/e7_selective_prediction.yaml
```

---

# 14. Lock Configuration

Sau E7, nhóm khóa toàn bộ:

```text
30 classes
class mapping
data split
preprocessing
model architecture
checkpoint
augmentation
temperature
confidence threshold
```

Từ thời điểm này:

```text
không điều chỉnh configuration
dựa trên Official Test.
```

Sau đó mới bắt đầu:

```text
OFFICIAL FINAL TEST
```

---

# 15. Official Final Test

Best Model đã khóa được chạy trên Official Test sạch.

Các kết quả cần lưu:

```text
Top-1 Accuracy
Macro-F1
Top-5 Accuracy
Per-class Precision
Per-class Recall
Per-class F1
Confusion Matrix

Parameter Count
Model Size
Inference Latency

ECE
Brier Score

Coverage
Selective Accuracy
Risk–Coverage
AURC
```

Raw predictions cũng phải được lưu để có thể tính lại metric mà không cần inference lại.

---

# 16. E8 — Robustness Evaluation

Từ Official Test sạch tạo ba corrupted test set:

```text
Gaussian Blur
Gaussian Noise
Occlusion
```

Nguyên tắc:

```text
Cùng ảnh gốc
Cùng nhãn
Corruption deterministic
Một severity trung bình cố định
```

Pipeline:

```text
Official Test
├── Clean
├── Gaussian Blur
├── Gaussian Noise
└── Occlusion
```

Best Model đã khóa được chạy trên cả 4 điều kiện.

Metric chính:

```text
Clean Accuracy
Corrupted Accuracy
Accuracy Drop
```

Trong đó:

```text
Accuracy Drop
=
Clean Accuracy - Corrupted Accuracy
```

Nếu đủ thời gian có thể tính thêm:

```text
ECE under corruption
```

Mục tiêu của E8 là **stress testing**, không phải huấn luyện một robust model mới.

Code:

```text
src/robustness/corruptions.py
src/robustness/robustness_eval.py
```

Config:

```text
configs/experiments/e8_robustness.yaml
```

---

# 17. E9 — Error Analysis + Grad-CAM

Bắt đầu từ:

```text
Confusion Matrix
```

trên Official Test sạch.

Sau đó tìm:

```text
Top-5 Confusion Pairs
```

Ví dụ:

```text
Breed A → Breed B
Breed C → Breed D
...
```

Từ đó chọn các trường hợp:

```text
Correct predictions
Wrong predictions
High-confidence errors
Low-confidence samples
```

Mỗi failure case nên ghi:

```text
True Label
Predicted Label
Confidence
Image Condition
Possible Failure Category
```

Các nhóm lỗi dự kiến:

```text
Fine-grained similarity
Complex background
Blur
Occlusion
Dog too small
Unusual pose
Poor lighting
```

Sau đó dùng:

```text
Grad-CAM
```

để quan sát vùng ảnh ảnh hưởng đến prediction.

Grad-CAM chỉ dùng làm bằng chứng hỗ trợ.

Không kết luận nguyên nhân lỗi chỉ dựa trên heatmap.

Code:

```text
src/evaluation/confusion_matrix.py
src/explainability/gradcam.py
```

Config:

```text
configs/experiments/e9_error_analysis.yaml
```

---

# 18. Experiment Matrix

| ID | Experiment | Mục tiêu |
|---|---|---|
| E0 | Custom CNN + Basic Augmentation | Baseline train from scratch |
| E1 | MobileNetV2 Frozen + Basic Augmentation | Transfer Learning baseline |
| E2 | MobileNetV2 Partial + Basic Augmentation | Đánh giá Partial Fine-tuning |
| E3 | ResNet18 Partial + Basic Augmentation | So sánh backbone và efficiency |
| E4 | Best Transfer Model + CutMix | Augmentation ablation |
| E5 | Top-2 Configurations × 3 Seeds | Stability / reproducibility |
| E6 | Before vs After Temperature Scaling | Confidence Calibration |
| E7 | Confidence Threshold / Risk–Coverage | Selective Prediction |
| E8 | Clean vs Blur / Noise / Occlusion | Robustness stress test |
| E9 | Confusion Matrix + Grad-CAM | Error Analysis |

E0–E5 là phần training/model selection chính.

Sau E5:

```text
Best Classification Model
```

được khóa.

E6–E9 chủ yếu phục vụ reliability và evaluation.

---

# 19. Output của mỗi experiment

Mỗi experiment phải xác định được:

```text
experiment_id
config
git commit/hash
seed
timestamp
```

Cần lưu:

```text
best checkpoint
training history
validation history
runtime
metrics
predictions nếu cần
```

Output được tổ chức theo experiment ID.

Ví dụ:

```text
outputs/
├── checkpoints/
│   └── E0/
├── logs/
│   └── E0/
├── metrics/
│   └── E0/
├── predictions/
│   └── E0/
└── figures/
    └── E0/
```

Tương tự cho:

```text
E1
E2
...
E9
```

Checkpoint lớn không nhất thiết phải commit trực tiếp lên GitHub.

---

# 20. Notebook Colab

Notebook không được tạo sẵn.

Khi task được giao, thành viên tạo notebook trên Google Colab và lưu lại theo tên được nhóm trưởng quy định.

Ví dụ:

```text
notebooks/training/E0_custom_cnn.ipynb
notebooks/training/E1_mobilenet_frozen.ipynb
notebooks/training/E2_mobilenet_partial.ipynb
notebooks/training/E3_resnet18_partial.ipynb
```

Notebook là nơi:

```text
setup environment
load config
run source code
train
evaluate
visualize results
```

Logic chính vẫn phải nằm trong:

```text
src/
```

---

# 21. Gradio Deployment

Sau khi model, calibration và threshold đã được khóa:

```text
Upload Image
↓
Input Validation
↓
Preprocessing
↓
Best Classification Model
↓
Logits
↓
Temperature Scaling
↓
Calibrated Softmax
↓
Confidence Threshold
↓
ACCEPT / REJECT
```

Nếu ACCEPT:

```text
Top-1 Breed
Top-1 Confidence
Top-5 Predictions
Inference Latency
```

Nếu REJECT:

```text
"Mô hình chưa đủ tự tin để đưa ra kết luận."
```

Có thể thêm:

```text
Grad-CAM
```

như tính năng giải thích tùy chọn.

Code:

```text
src/inference/predictor.py
app/gradio_app.py
```

Config:

```text
configs/inference.yaml
```

---

# 22. Các nội dung không thuộc core scope

Không mở rộng sang các bài toán sau nếu chưa hoàn thành pipeline chính:

```text
Object Detection
Segmentation
Dog Localization
Multiple Dogs Recognition
Mixed-breed Classification
Pure-breed Verification
Individual Dog Recognition
Open-set Recognition
Vision Transformer
EfficientNet
Few-shot Learning
```

Dataset ngoài Stanford Dogs cũng không phải core experiment.

Ưu tiên hoàn thành:

```text
E0 → E9 → Gradio
```

trước khi mở rộng.

---

# 23. Nguyên tắc làm việc chung

Toàn bộ nhóm phải sử dụng chung:

```text
1 Repository
1 Class Mapping
1 Data Split
1 Preprocessing Pipeline
1 Training Pipeline
1 Evaluation Pipeline
```

Không thực hiện theo kiểu:

```text
Member A → tự viết pipeline riêng
Member B → tự tạo config riêng
Member C → tự chia dataset riêng
Member D → cuối kỳ ghép lại
```

Mỗi thành viên chỉ thực thi các file được ghi trong:

```text
docs/team_tasks/<ngay>_task.md
```

Nếu cần:

```text
tạo file .py mới
tạo folder mới
tạo config mới
thay đổi shared pipeline
```

phải trao đổi với nhóm trưởng trước.

---

# 24. Pipeline tóm tắt

```text
DATA
Stanford Dogs
→ Fixed 30 Breeds
→ Audit
→ Fixed Split
→ Preprocessing

MODEL DEVELOPMENT
→ E0 Custom CNN
→ E1 MobileNetV2 Frozen
→ E2 MobileNetV2 Partial
→ E3 ResNet18 Partial
→ E4 CutMix
→ E5 Top-2 × 3 Seeds
→ Best Classification Model

RELIABILITY
→ E6 Temperature Scaling
→ E7 Selective Prediction

LOCK
→ Model
→ Temperature
→ Threshold

FINAL EVALUATION
→ Official Test
→ E8 Robustness
→ E9 Error Analysis + Grad-CAM

APPLICATION
→ Gradio Demo
```

Đây là pipeline chính thức mà nhóm sử dụng trong quá trình triển khai DogBreedID.
