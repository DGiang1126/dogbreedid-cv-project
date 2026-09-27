# DogBreedID — Project Structure

Tài liệu này mô tả cấu trúc repository và quy ước lưu file của project **DogBreedID**.

Pipeline tổng thể xem tại:

`docs/pipeline/master_pipeline.md`

---

## 1. Cấu trúc project

```text
dogbreedid-cv-project/
│
├── README.md
├── CONTRIBUTING.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── base.yaml
│   ├── data.yaml
│   ├── inference.yaml
│   └── experiments/
│       ├── e0_custom_cnn.yaml
│       ├── e1_mobilenet_frozen.yaml
│       ├── e2_mobilenet_partial.yaml
│       ├── e3_resnet18_partial.yaml
│       ├── e4_cutmix.yaml
│       ├── e5_seed_stability.yaml
│       ├── e6_calibration.yaml
│       ├── e7_selective_prediction.yaml
│       ├── e8_robustness.yaml
│       └── e9_error_analysis.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── metadata/
│       └── class_mapping.json
│
├── notebooks/
│   ├── exploration/
│   ├── training/
│   ├── evaluation/
│   └── demo/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── calibration/
│   ├── selective/
│   ├── robustness/
│   ├── explainability/
│   ├── inference/
│   └── utils/
│
├── scripts/
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── metrics/
│   ├── predictions/
│   └── figures/
│
├── app/
├── tests/
│
└── docs/
    ├── project_structure.md
    ├── protocol/
    ├── pipeline/
    ├── team_tasks/
    └── task_reports/
        ├── thuan/
        ├── duc-anh/
        └── kiet/
```

---

## 2. `configs/`

Chứa toàn bộ config của project.

```text
base.yaml       → cấu hình chung
data.yaml       → dataset, preprocessing, split
inference.yaml  → inference và Gradio

experiments/
→ config riêng cho E0–E9
```

Config chung do nhóm trưởng quản lý.

Thành viên không tự tạo config mới nếu chưa thống nhất.

---

## 3. `data/`

```text
raw/        → dữ liệu Stanford Dogs gốc
processed/  → dữ liệu trung gian nếu cần
splits/     → Train / Validation / Calibration / Official Test
metadata/   → metadata dùng chung
```

`class_mapping.json` lưu mapping cố định giữa:

```text
class_id ↔ breed_name
```

Tất cả thành viên phải sử dụng cùng split và class mapping.

---

## 4. `notebooks/`

Chứa notebook Google Colab.

```text
exploration/  → Data Audit / khám phá dữ liệu
training/     → E0–E5
evaluation/   → E6–E9
demo/         → thử Gradio / inference
```

Notebook không tạo sẵn.

Khi giao task, tên notebook sẽ được quy định cụ thể.

Ví dụ:

```text
E0_custom_cnn.ipynb
E1_mobilenet_frozen.ipynb
```

Notebook chỉ dùng để chạy experiment. Logic chính phải nằm trong `src/`.

---

## 5. `src/`

Chứa source code chính.

```text
data/           → dataset, preprocessing, split, audit
models/         → Custom CNN, MobileNetV2, ResNet18
training/       → training, validation, checkpoint
evaluation/     → metrics, confusion matrix, efficiency
calibration/    → Temperature Scaling
selective/      → confidence threshold, Risk–Coverage
robustness/     → Blur, Noise, Occlusion
explainability/ → Grad-CAM
inference/      → pipeline prediction cuối cùng
utils/          → config, seed, logger, device, path
```

Không tự tạo thêm file `.py` hoặc module mới nếu task chưa quy định.

---

## 6. `scripts/`

Chứa các file dùng để khởi chạy pipeline.

Ví dụ:

```text
train.py
evaluate.py
calibrate.py
robustness_evaluate.py
run_demo.py
```

`scripts/` chỉ gọi lại code trong `src/`, không viết một pipeline riêng.

---

## 7. `outputs/`

Chứa kết quả của các experiment.

```text
checkpoints/  → model weights
logs/         → training log
metrics/      → kết quả metric
predictions/  → raw predictions
figures/      → biểu đồ, Confusion Matrix, Grad-CAM
```

Kết quả nên được tổ chức theo experiment ID:

```text
E0/
E1/
...
E9/
```

---

## 8. `app/`

Chứa ứng dụng Gradio cuối cùng.

```text
app/gradio_app.py
```

Gradio sử dụng pipeline inference chung trong `src/inference/`.

---

## 9. `tests/`

Chứa test cho các module của project.

Ví dụ:

```text
test_dataset.py
test_models.py
test_training.py
test_metrics.py
test_inference.py
```

Khi giao task, nhóm trưởng sẽ quy định file test cần chỉnh sửa.

---

## 10. `docs/`

Chứa tài liệu chung của nhóm.

```text
protocol/
→ quy định dataset, training, evaluation và Git

pipeline/
→ pipeline tổng thể E0–E9

team_tasks/
→ nhiệm vụ do nhóm trưởng giao

task_reports/
→ báo cáo công việc của từng thành viên
```

File giao việc đặt tên theo ngày:

```text
docs/team_tasks/27_09_task.md
```

Sau khi hoàn thành, thành viên viết report trong folder cá nhân.

Ví dụ Thuận hoàn thành ngày 30/09:

```text
docs/task_reports/thuan/30_09_task.md
```

Không tạo file `.md` báo cáo công việc ở các vị trí khác.

---

## 11. Quy tắc chung

```text
Code       → src/
Config     → configs/
Notebook   → notebooks/
Output     → outputs/
Test       → tests/
Tài liệu   → docs/
Ứng dụng   → app/
```

Khi được giao task, thành viên phải:

- Chỉ sửa các file được chỉ định.
- Dùng đúng config và file chung được yêu cầu.
- Lưu notebook và output đúng vị trí.
- Không tự tạo thêm `.py`, config hoặc folder mới nếu chưa trao đổi với nhóm trưởng.

Mục tiêu là toàn bộ nhóm làm việc trên **một cấu trúc và một pipeline thống nhất**.
