# Master Pipeline

Pipeline chính của DogBreedID:

1. Chọn cố định 30 giống và class mapping.
2. Audit Stanford Dogs.
3. Tạo/freeze Train - Validation - Calibration; giữ Official Test.
4. Xây shared preprocessing/training/evaluation pipeline.
5. E0 - Custom CNN.
6. E1 - MobileNetV2 Frozen.
7. E2 - MobileNetV2 Partial Fine-tuning.
8. E3 - ResNet18 Partial Fine-tuning.
9. E4 - CutMix ablation trên best transfer model.
10. E5 - Top configurations x 3 random seeds; khóa Best Classification Model.
11. E6 - Temperature Scaling.
12. E7 - Selective Prediction.
13. E8 - Robustness evaluation.
14. E9 - Error Analysis + Grad-CAM.
15. Tích hợp Gradio.

Chi tiết file, config, notebook, output và Definition of Done của từng task sẽ được ghi trong `docs/team_tasks/<ngay>_task.md`.
