# DogBreedID V2 — Experiment Registry

**APPROVED FOR IMPLEMENTATION PLANNING — EXECUTION PENDING**

**Task 08.01 · Phase 2.3 · Hybrid CNN–VLM with Knowledge-Augmented Visual Understanding**

Registry ghi thiết kế, người phụ trách, điều kiện đầu vào và evidence cần có cho E12–E15. Baseline đối chiếu là `v1.0.0-cnn-stable` tại `b7d880506b96168b0e335f3a455be3a7b2bdf77e` trên `research/vlm-protocol`. Research Protocol đã được duyệt để lập kế hoạch triển khai và còn chờ Protocol Lock. Cả bảy experiment đang `PLANNED`. [Architecture](architecture.md), [Research Protocol](research_protocol.md)

## 1. Phạm vi ID và cách đọc hồ sơ

| ID đã có | Phạm vi đối chiếu |
| --- | --- |
| E0–E5 | E0 Custom CNN, E1 MobileNetV2 Frozen, E2 MobileNetV2 Partial, E3 ResNet18 Partial, E4 MobileNetV2 Frozen + CutMix, E5 Seed Stability. [Experiment Methods](../../task06/experiment_summary/experiment_methods.csv), [Evaluation Protocol](../../protocol/evaluation_protocol.md) |
| E6–E9 | E6 Temperature Scaling, E7 Selective Prediction, E8 Robustness, E9 Error Analysis và Grad-CAM. [Evaluation Protocol](../../protocol/evaluation_protocol.md), [E6 config](../../../configs/experiments/e6_calibration.yaml), [E7 config](../../../configs/experiments/e7_selective_prediction.yaml), [E8 config](../../../configs/experiments/e8_robustness.yaml), [E9 config](../../../configs/experiments/e9_error_analysis.yaml) |
| E10-A/B, E11-A/B | EfficientNet-B0 và ViT-Tiny, mỗi model có biến thể không/có CutMix. [E10-A](../../../configs/experiments/e10a_efficientnet_no_cutmix.yaml), [E10-B](../../../configs/experiments/e10b_efficientnet_cutmix.yaml), [E11-A](../../../configs/experiments/e11a_vit_no_cutmix.yaml), [E11-B](../../../configs/experiments/e11b_vit_cutmix.yaml), [So sánh Validation E10/E11](../results/e10_e11_validation_comparison.md) |

Đã đối chiếu trường `experiment.id` của các config hiện có và phạm vi nghiên cứu trong hai tài liệu thiết kế. ID đăng ký mới gồm E12-A, E12-B, E13-A, E13-B, E14-A, E14-B và E15. Các ID bàn giao ở mục 4 thuộc Task 08.04.

- **Task ID.** Phân công thực thi theo bảng dưới đây. Phê duyệt registry áp dụng cho lập kế hoạch triển khai. Các gate và trạng thái thực thi của từng nhóm chưa thay đổi.

| Task ID | Owner | Phạm vi |
| --- | --- | --- |
| 08.01 | Trưởng nhóm | Architecture, Research Protocol và E15 Integration |
| 08.02 | Đức Anh | E12-A/B và E13-A/B |
| 08.03 | Kiệt | E14-A/B |
| 08.04 | Thuận | Bảy deliverables dữ liệu và retrieval ở mục 4 |
- **Thông số chưa xác minh.** `TBD` cần bổ sung bằng bằng chứng hoặc phê duyệt. `PENDING PILOT` cần đo khả thi trước khi đề xuất khóa. Mọi đường dẫn config, entry point và output trong hồ sơ đều là **proposed**.
- **Ranh giới dữ liệu.** Pilot dùng development subset được phép, tách Validation-300 và VLM-Eval formal. Formal chỉ dùng cohort, artifacts và deliverables đã duyệt và khóa. Official Test, predictions và protected bundles nằm ngoài mọi lượt chạy, audit và so sánh Task 08. Ground-truth labels và answers chỉ phục vụ scorer. [Research Protocol](research_protocol.md)

## 2. Hồ sơ thí nghiệm

Các mục **Metrics** tham chiếu định nghĩa và mẫu số trong Research Protocol. Evidence chung cho từng experiment ở mục 6. Mọi hồ sơ dưới đây hiện chưa có revision, hash hoặc kết quả mới từ thực nghiệm.

### E12-A — SigLIP 2 — Name Prompt

| Trường | Hồ sơ |
| --- | --- |
| Experiment ID và tên | `E12-A` · SigLIP 2 — Name Prompt |
| Task ID | `08.02` |
| Research question | RQ-VLM-01, RQ-VLM-02, RQ-VLM-03 |
| Objective | Đo baseline zero-shot SigLIP 2 chỉ dùng tên giống. |
| Hypothesis | Tên giống có thể cho tín hiệu xếp hạng. Mức độ khác E4 và khác biệt giữa các nhóm giống cần đo. |
| Owner và contributors | Đức Anh. Contributors `TBD`. |
| Candidate model ID | `google/siglip2-base-patch16-224` |
| Model revision | `PENDING PILOT`, revision pinning tại Protocol Lock. |
| Dataset/cohort và hash | Validation-300. Dataset/manifest hash `TBD` tại khóa cấu hình. |
| Input configuration | Ảnh nguồn chung, native processor, một name prompt cho mỗi canonical class, ranking đủ 30 ID. |
| Independent variable | Name Prompt Bank, điều kiện đối chứng của E12-B. |
| Controlled variables | Trong cùng encoder, khóa model revision, processor/tokenizer, ảnh, class order, scoring/aggregation, device/dtype và metrics. Native preprocessing theo adapter. |
| Evaluation metrics | Top-1/Top-5, Macro-F1, per-class Precision/Recall/F1, confusion matrix, failure/abstention và paired disagreement. Kiểm định và CI theo Research Protocol mục 5.3. Benchmark theo mục 7. |
| Baseline/comparison experiments | E4, E12-B và E13-A. |
| Required upstream deliverables | `08.04-NAME` và `08.04-AUDIT` trong phạm vi cohort/mapping được duyệt. Đối sánh E4 cần predictions đã xác minh hoặc replay được phép. |
| Pilot requirements | Khả năng nạp local, processor/tokenizer, score hữu hạn đủ 30 ID, prompt token budget, tie-break tái lập, latency và peak RAM/VRAM. |
| Protocol Lock requirements | Pin model/processor, môi trường, cohort/mapping/prompt hashes, native input, scoring và tie-break, plan thống kê, lỗi và resource profile. |
| Required evidence | Checklist chung ở mục 6. Score vector 30 lớp, Top-5 IDs, prompt/tokenization log và pairing theo Image ID. |
| Proposed config path | `configs/experiments/e12a_siglip2_name.yaml` |
| Proposed source entry point | `scripts/evaluate_vlm_zero_shot.py` |
| Proposed output directory | `outputs/research/vlm/E12-A/<run_id>/` |
| Acceptance dependencies | Gate an ninh nếu áp dụng, Protocol Lock và review evidence. Paired comparison với E4 cần evidence ghép cặp hợp lệ. |
| Current status | `PLANNED` |
| Blocker/fallback policy | Feasibility hoặc cohort không đạt thì ghi `BLOCKED` và lý do. Giữ nguyên model và ID. |

### E12-B — SigLIP 2 — Morphology Prompt

| Trường | Hồ sơ |
| --- | --- |
| Experiment ID và tên | `E12-B` · SigLIP 2 — Morphology Prompt |
| Task ID | `08.02` |
| Research question | RQ-VLM-01, RQ-VLM-02, RQ-VLM-03 |
| Objective | Đo tác động của morphology prompts trên SigLIP 2. |
| Hypothesis | Thuộc tính có nguồn có thể đổi ranking hoặc tăng nhầm lẫn. Hiệu ứng có thể thay đổi theo giống. |
| Owner và contributors | Đức Anh. Contributors `TBD`. |
| Candidate model ID | `google/siglip2-base-patch16-224` |
| Model revision | `PENDING PILOT`, revision pinning tại Protocol Lock. |
| Dataset/cohort và hash | Validation-300. Dataset/manifest hash `TBD` tại khóa cấu hình. |
| Input configuration | Cùng pipeline E12-A, một morphology prompt giữ tên giống và thêm thuộc tính có nguồn cho mỗi canonical class. |
| Independent variable | Chỉ đổi từ Name Prompt Bank sang Morphology Prompt Bank so với E12-A. |
| Controlled variables | Giữ nguyên model/revision, processor/tokenizer, scoring/aggregation, class order, device/dtype, metrics và dataset của E12-A. |
| Evaluation metrics | Top-1/Top-5, Macro-F1, per-class Precision/Recall/F1, confusion matrix, failure/abstention và paired disagreement. Kiểm định và CI theo Research Protocol mục 5.3. Benchmark theo mục 7. |
| Baseline/comparison experiments | E4, E12-A và E13-B. |
| Required upstream deliverables | `08.04-MORPH`, `08.04-KB` cho nguồn thuộc tính, `08.04-AUDIT` và cấu hình chung E12-A. |
| Pilot requirements | Pilot encoder như E12-A, kiểm giới hạn token cho morphology prompts và giữ nguyên thông tin cần thiết. |
| Protocol Lock requirements | Pin model/processor, môi trường, cohort/mapping/prompt hashes, native input, scoring và tie-break, plan thống kê, lỗi và resource profile. |
| Required evidence | Checklist chung ở mục 6. Score vector 30 lớp, Top-5 IDs, prompt/source provenance và bằng chứng chỉ đổi prompt bank trong cặp E12. |
| Proposed config path | `configs/experiments/e12b_siglip2_morphology.yaml` |
| Proposed source entry point | `scripts/evaluate_vlm_zero_shot.py` |
| Proposed output directory | `outputs/research/vlm/E12-B/<run_id>/` |
| Acceptance dependencies | Gate của E12-A, prompt bank và nguồn thuộc tính đã duyệt, pairing E12-A/B và review evidence. |
| Current status | `PLANNED` |
| Blocker/fallback policy | Prompt không qua audit/token gate thì chờ sửa và duyệt trước khóa. Ghi `BLOCKED` nếu thiếu điều kiện formal. |

### E13-A — FG-CLIP 2 — Name Prompt

| Trường | Hồ sơ |
| --- | --- |
| Experiment ID và tên | `E13-A` · FG-CLIP 2 — Name Prompt |
| Task ID | `08.02` |
| Research question | RQ-VLM-01, RQ-VLM-02, RQ-VLM-03 |
| Objective | Đo baseline zero-shot FG-CLIP 2 với name prompts và đối chiếu SigLIP 2. |
| Hypothesis | Encoder và native processor có thể tạo ranking khác. Chưa giả định chiều hiệu ứng. |
| Owner và contributors | Đức Anh. Contributors `TBD`. |
| Candidate model ID | `qihoo360/fg-clip2-base` |
| Model revision | `PENDING PILOT`, revision pinning tại Protocol Lock. |
| Dataset/cohort và hash | Validation-300. Dataset/manifest hash `TBD` tại khóa cấu hình. |
| Input configuration | Ảnh nguồn chung, native processor, một name prompt cho mỗi canonical class. Ưu tiên cùng nội dung với E12-A. |
| Independent variable | Name Prompt Bank, điều kiện đối chứng của E13-B. Đối chiếu E12-A theo cùng loại prompt. |
| Controlled variables | Trong cùng encoder, khóa model revision, processor/tokenizer, ảnh, class order, scoring/aggregation, device/dtype và metrics. Native preprocessing theo adapter. |
| Evaluation metrics | Top-1/Top-5, Macro-F1, per-class Precision/Recall/F1, confusion matrix, failure/abstention và paired disagreement. Kiểm định và CI theo Research Protocol mục 5.3. Benchmark theo mục 7. |
| Baseline/comparison experiments | E4, E13-B và E12-A. |
| Required upstream deliverables | `08.04-NAME`, `08.04-AUDIT`, xác minh native interface và gate custom code nếu cần. |
| Pilot requirements | Khả năng nạp local, processor/tokenizer, score hữu hạn đủ 30 ID, prompt token budget, tie-break tái lập, latency và peak RAM/VRAM. Xác định có cần custom code và gate khả thi hay không. |
| Protocol Lock requirements | Pin model/processor, môi trường, cohort/mapping/prompt hashes, native input, scoring và tie-break, plan thống kê, lỗi và resource profile. Duyệt source/revision/môi trường nếu yêu cầu custom code. |
| Required evidence | Checklist chung ở mục 6. Score vector 30 lớp, Top-5 IDs, pairing và ghi rõ khác biệt native interface hoặc bằng chứng phê duyệt code nếu áp dụng. |
| Proposed config path | `configs/experiments/e13a_fg_clip2_name.yaml` |
| Proposed source entry point | `scripts/evaluate_vlm_zero_shot.py` |
| Proposed output directory | `outputs/research/vlm/E13-A/<run_id>/` |
| Acceptance dependencies | Feasibility và security gate, Protocol Lock, mapping/cohort hợp lệ và review evidence. |
| Current status | `PLANNED` |
| Blocker/fallback policy | Nếu FG-CLIP 2 không qua feasibility/security gate, ghi `BLOCKED` cho cả E13-A và E13-B cùng lý do. Giữ model và ID đã đăng ký. |

### E13-B — FG-CLIP 2 — Morphology Prompt

| Trường | Hồ sơ |
| --- | --- |
| Experiment ID và tên | `E13-B` · FG-CLIP 2 — Morphology Prompt |
| Task ID | `08.02` |
| Research question | RQ-VLM-01, RQ-VLM-02, RQ-VLM-03 |
| Objective | Đo tác động của morphology prompts trên FG-CLIP 2. |
| Hypothesis | Thông tin hình thái có thể đổi ranking. Hiệu ứng có thể khác cặp E12. |
| Owner và contributors | Đức Anh. Contributors `TBD`. |
| Candidate model ID | `qihoo360/fg-clip2-base` |
| Model revision | `PENDING PILOT`, revision pinning tại Protocol Lock. |
| Dataset/cohort và hash | Validation-300. Dataset/manifest hash `TBD` tại khóa cấu hình. |
| Input configuration | Cùng pipeline E13-A, một morphology prompt có tên và thuộc tính có nguồn mỗi lớp, ưu tiên nội dung giống E12-B. |
| Independent variable | Chỉ thay Name Prompt Bank bằng Morphology Prompt Bank so với E13-A. |
| Controlled variables | Giữ nguyên model/revision, processor/tokenizer, scoring/aggregation, class order, device/dtype, metrics và dataset của E13-A. |
| Evaluation metrics | Top-1/Top-5, Macro-F1, per-class Precision/Recall/F1, confusion matrix, failure/abstention và paired disagreement. Kiểm định và CI theo Research Protocol mục 5.3. Benchmark theo mục 7. |
| Baseline/comparison experiments | E4, E13-A và E12-B. |
| Required upstream deliverables | `08.04-MORPH`, `08.04-KB` cho nguồn thuộc tính, `08.04-AUDIT` và cấu hình chung E13-A. |
| Pilot requirements | Pilot encoder như E13-A, kiểm morphology token budget. Dùng cùng gate custom code nếu áp dụng. |
| Protocol Lock requirements | Pin model/processor, môi trường, cohort/mapping/prompt hashes, native input, scoring và tie-break, plan thống kê, lỗi và resource profile. Dùng cùng phê duyệt custom code như E13-A nếu cần. |
| Required evidence | Checklist chung ở mục 6. Score vector 30 lớp, Top-5 IDs, nguồn thuộc tính và bằng chứng chỉ đổi prompt bank trong cặp E13. |
| Proposed config path | `configs/experiments/e13b_fg_clip2_morphology.yaml` |
| Proposed source entry point | `scripts/evaluate_vlm_zero_shot.py` |
| Proposed output directory | `outputs/research/vlm/E13-B/<run_id>/` |
| Acceptance dependencies | Gate của E13-A, prompt bank đã duyệt, pairing E13-A/B và review evidence. |
| Current status | `PLANNED` |
| Blocker/fallback policy | Feasibility/security gate không đạt thì E13-A/B đều `BLOCKED`. Thiếu prompt hợp lệ thì ghi lý do, chờ sửa và duyệt trước formal. |

### E14-A — Generative VLM without KB

| Trường | Hồ sơ |
| --- | --- |
| Experiment ID và tên | `E14-A` · Generative VLM without KB |
| Task ID | `08.03` |
| Research question | RQ-VLM-04, RQ-VLM-05 |
| Objective | Đo thuộc tính thị giác và VQA tiếng Việt khi KB context rỗng. |
| Hypothesis | Model có thể trả lời đúng hoặc từ chối phù hợp. Kết quả có thể khác theo category và bằng chứng ảnh. |
| Owner và contributors | Kiệt. Contributors `TBD`. |
| Candidate model ID | Chính `Qwen/Qwen3-VL-2B-Instruct`. Dự phòng `HuggingFaceTB/SmolVLM-500M-Instruct`. |
| Model revision | `PENDING PILOT`, revision pinning tại Protocol Lock. |
| Dataset/cohort và hash | VLM-Eval, dùng sau audit và Protocol Lock. Dataset/manifest hash `TBD` tại khóa cấu hình. |
| Input configuration | Ảnh, câu hỏi, pipeline trợ lý đã khóa và slot KB rỗng. Retrieval là `not_applicable`. |
| Independent variable | KB context rỗng, điều kiện đối chứng của E14-B. |
| Controlled variables | Khóa cùng model, ảnh/câu hỏi, processor, pipeline, template, context limit, generation, seed, schema và validation. CNN context hoặc thuộc tính trung gian nếu dùng phải dùng cùng bản ghi. Scorer giữ ground truth. |
| Evaluation metrics | VQA Accuracy toàn cohort là chính. Theo dõi Attribute Accuracy, JSON/schema, hallucination, failure/coverage, groundedness, abstention, tiếng Việt và efficiency theo Research Protocol mục 6–7. KB citation metrics là `N/A`. |
| Baseline/comparison experiments | E14-B trên cùng cặp Image ID và Question ID. |
| Required upstream deliverables | `08.04-VLM-EVAL`, `08.04-RUBRIC` và `08.04-AUDIT`. Phối hợp snapshot KB và retrieval của E14-B để khóa thiết kế cặp. |
| Pilot requirements | Nạp local và kiểm tra tiếng Việt, structured output, unknown/not_visible, single-pass/multi-pass, context/token budget, generation latency và bộ nhớ. |
| Protocol Lock requirements | Pin một model và pipeline, manifest/mapping, KB/rubric revisions, cùng context limit, generation, rubric nội dung, chính sách offline/runtime và mẫu số của toàn cohort. |
| Required evidence | Checklist chung ở mục 6. Raw model output, validated model output, controller-generated response, policy-blocked status, trạng thái retrieval, annotation, hồ sơ hòa giải và inter-rater agreement. Ghi context rỗng và coverage theo question category. |
| Proposed config path | `configs/experiments/e14a_vlm_without_kb.yaml` |
| Proposed source entry point | `scripts/evaluate_vlm_assistant.py` |
| Proposed output directory | `outputs/research/vlm/E14-A/<run_id>/` |
| Acceptance dependencies | Gate model và pipeline, VLM-Eval/rubric đã duyệt và khóa, cùng chính sách chấm E14-A/B và review evidence. |
| Current status | `PLANNED` |
| Blocker/fallback policy | Nếu model chính không khả thi, pilot model dự phòng và khóa trước khi chạy cả cặp. Nếu không có model đủ điều kiện, ghi `BLOCKED`. |

### E14-B — Generative VLM with KB

| Trường | Hồ sơ |
| --- | --- |
| Experiment ID và tên | `E14-B` · Generative VLM with KB |
| Task ID | `08.03` |
| Research question | RQ-VLM-04, RQ-VLM-05 |
| Objective | Đo tác động của KB context lên độ đúng nội dung, hallucination và khả năng cung cấp nguồn. |
| Hypothesis | Tri thức truy xuất có thể cải thiện câu hỏi kiến thức hoặc gây suy diễn không đúng về ảnh. |
| Owner và contributors | Kiệt. Contributors `TBD`. |
| Candidate model ID | Cùng model và revision đã chọn cho E14-A từ `Qwen/Qwen3-VL-2B-Instruct` hoặc `HuggingFaceTB/SmolVLM-500M-Instruct`. |
| Model revision | `PENDING PILOT`, revision pinning tại Protocol Lock. |
| Dataset/cohort và hash | Cùng VLM-Eval của E14-A, dùng sau audit và Protocol Lock. Dataset/manifest hash `TBD` tại khóa cấu hình. |
| Input configuration | Cùng ảnh, câu hỏi và pipeline E14-A. Slot KB chứa context và source IDs theo snapshot, chính sách retrieval và budget đã khóa. |
| Independent variable | Chỉ thay KB context so với E14-A. |
| Controlled variables | Khóa cùng model, ảnh/câu hỏi, processor, pipeline, template, context limit, generation, seed, schema và validation. CNN context hoặc thuộc tính trung gian nếu dùng phải dùng cùng bản ghi. Scorer giữ ground truth. Giữ nguyên input chung sau tokenization. Chỉ phần KB có thể được rút gọn theo quy tắc đã khóa. |
| Evaluation metrics | VQA Accuracy toàn cohort là chính. Theo dõi Attribute Accuracy, JSON/schema, hallucination, failure/coverage, groundedness, abstention, tiếng Việt và efficiency theo Research Protocol mục 6–7. Báo riêng Citation Correctness và KB Citation Coverage khi áp dụng. |
| Baseline/comparison experiments | E14-A trên cùng cặp Image ID và Question ID. |
| Required upstream deliverables | `08.04-KB`, `08.04-RETRIEVAL`, `08.04-VLM-EVAL`, `08.04-RUBRIC`, `08.04-AUDIT` và cấu hình chung E14-A. |
| Pilot requirements | Nạp local và kiểm tra tiếng Việt, structured output, unknown/not_visible, single-pass/multi-pass, context/token budget, generation latency và bộ nhớ. Thử lookup, budget, miss/ambiguous và kiểm phần chung được giữ nguyên. |
| Protocol Lock requirements | Pin một model và pipeline, manifest/mapping, KB/rubric revisions, cùng context limit, generation, rubric nội dung, chính sách offline/runtime và mẫu số của toàn cohort. Pin KB snapshot, quy tắc retrieval và xử lý miss/ambiguous. |
| Required evidence | Checklist chung ở mục 6. Raw model output, validated model output, controller-generated response, policy-blocked status, trạng thái retrieval, annotation, hồ sơ hòa giải và inter-rater agreement. Kèm context/source IDs, mẫu số toàn cohort và tiêu chí retrieval-hit subgroup nếu báo cáo. |
| Proposed config path | `configs/experiments/e14b_vlm_with_kb.yaml` |
| Proposed source entry point | `scripts/evaluate_vlm_assistant.py` |
| Proposed output directory | `outputs/research/vlm/E14-B/<run_id>/` |
| Acceptance dependencies | Các dependency E14-A, KB/retrieval được duyệt, kiểm E14-A/B chỉ khác KB context và review evidence. |
| Current status | `PLANNED` |
| Blocker/fallback policy | Dùng cùng quy tắc fallback model của E14-A. Giữ những mẫu miss, ambiguous, lỗi hoặc bị policy chặn trong E14-B theo protocol. |

### E15 — Hybrid CNN–VLM System Integration and Regression

| Trường | Hồ sơ |
| --- | --- |
| Experiment ID và tên | `E15` · Hybrid CNN–VLM System Integration and Regression |
| Task ID | `08.01` |
| Research question | RQ-VLM-06 |
| Objective | Kiểm tra tích hợp VLM tùy chọn, giữ CNN decisions, Grad-CAM và contract `/predict` của hệ thống baseline. |
| Hypothesis | Controller có thể cô lập lỗi, tài nguyên và trạng thái VLM trong khi hợp đồng CNN vẫn tương đương baseline. |
| Owner và contributors | Trưởng nhóm. Contributors Đức Anh, Kiệt, Thuận. |
| Candidate model ID | E4 MobileNetV2 cố định. Encoder/profile và generative VLM lấy từ nghiệm thu E12–E14, các model IDs được chọn còn `TBD`. |
| Model revision | E4 theo baseline và checkpoint đã khóa. Revisions của VLM chờ formal acceptance và deployment gates. |
| Dataset/cohort và hash | Fixture và development inputs non-Official cho contract/regression, chờ duyệt kịch bản và cohort. Dataset/fixture manifest hash `TBD` khi khóa cấu hình. |
| Input configuration | Hai mode ảnh hiện hành, chuỗi tác vụ theo session, VLM tùy chọn, KB snapshot/retrieval và fault injection được duyệt. |
| Independent variable | Trạng thái VLM tắt/bật/bận/lỗi và các kịch bản policy, session, stale, tài nguyên hoặc fallback. |
| Controlled variables | Giữ checkpoint E4, E6/E7/Energy OOD, preprocessing, canonical mapping, Grad-CAM và baseline contract. Dùng cùng fixture và cấu hình CNN cho đối chứng. |
| Evaluation metrics | Contract parity, pass/fail của case regression, session isolation, policy/fallback, khả dụng, latency/tài nguyên khi có tải và recovery. Theo Research Protocol mục 7 và 9.2. |
| Baseline/comparison experiments | CNN baseline cùng fixture, với VLM tắt so với bật/bận/lỗi. Kiểm tra hành vi system E15 và truy vết riêng model output E14. |
| Required upstream deliverables | Profile được nghiệm thu từ các nhánh E12–E14 nằm trong phạm vi tích hợp, qua deployment eligibility. Cần module/snapshot Task 08.04 tương ứng, kiến trúc đã duyệt và tiêu chí E15 ở phase sau. |
| Pilot requirements | Thử adapter và controller bằng fixture khi được phép, kiểm tùy chọn dependencies, cold/warm, concurrency, timeout/OOM và khả năng phục hồi. |
| Protocol Lock requirements | Duyệt model/profile đã chọn, fixture, kịch bản, môi trường, resource limits, tiêu chí parity/regression và phạm vi evidence trước acceptance run. |
| Required evidence | Checklist chung ở mục 6. Trace raw/validated model output, controller response và policy status. Integration tests, API compatibility, session isolation, stale/fallback và resource tests. |
| Proposed config path | `configs/experiments/e15_hybrid_integration.yaml` |
| Proposed source entry point | `scripts/verify_vlm_integration.py` |
| Proposed output directory | `outputs/research/vlm/E15/<run_id>/` |
| Acceptance dependencies | Profile đủ điều kiện trước tích hợp, dependency Task 08.04 theo phạm vi, phạm vi đầy đủ hoặc reduced scope được trưởng nhóm duyệt, và evidence hồi quy sau tích hợp theo tiêu chí E15. |
| Current status | `PLANNED` |
| Blocker/fallback policy | Cô lập nhánh thiếu dependency và báo unavailable. CNN tiếp tục phục vụ. Nếu thiếu điều kiện cho toàn phạm vi, trình trưởng nhóm duyệt reduced scope trước nghiệm thu. |

## 3. Comparison Registry

### 3.1. E12/E13 — Tám đối sánh Top-1 định trước

**Unit và chiều hiệu ứng.** Join theo paired Image ID trên cùng Validation-300 đã khóa. Kiểm đủ ID, nhãn và canonical mapping theo protocol. Hiệu ứng là Top-1 bên thứ nhất trừ bên thứ hai. Dấu dương nghiêng về bên thứ nhất. Endpoint chính là **Top-1 Accuracy trên toàn cohort** với failure/abstention theo protocol.

| Bên thứ nhất | Bên thứ hai | Research question | Hiệu ứng báo cáo |
| --- | --- | --- | --- |
| E12-A | E4 | RQ-VLM-01 | E12-A − E4 |
| E12-B | E4 | RQ-VLM-01 | E12-B − E4 |
| E13-A | E4 | RQ-VLM-01 | E13-A − E4 |
| E13-B | E4 | RQ-VLM-01 | E13-B − E4 |
| E12-B | E12-A | RQ-VLM-02 | E12-B − E12-A |
| E13-B | E13-A | RQ-VLM-02 | E13-B − E13-A |
| E13-A | E12-A | RQ-VLM-03 | E13-A − E12-A |
| E13-B | E12-B | RQ-VLM-03 | E13-B − E12-B |

Theo Research Protocol mục 5.3, đề xuất McNemar exact hai phía cho cặp đúng/sai, Holm family gồm tám phép so sánh ở `alpha=0.05` và paired bootstrap percentile CI 95% theo Image ID phân tầng theo lớp. Lưu effect size, discordant counts và CI theo chiều đã đăng ký. Seed, replicates và thiết lập phân tích giữ **proposed** tới Protocol Lock. Nếu thiếu evidence ghép cặp hoặc một nhánh bị chặn, ghi `BLOCKED` cho đối sánh chịu ảnh hưởng và áp dụng quy tắc Holm trong protocol. [Research Protocol](research_protocol.md)

### 3.2. E14-B versus E14-A

Ghép cặp theo Image ID và Question ID, báo hiệu ứng **E14-B − E14-A**. Primary endpoint là **VQA Accuracy toàn cohort**, dùng cùng rubric nội dung. Báo thêm Attribute Accuracy, hallucination cùng failure/coverage, groundedness, abstention quality, Vietnamese quality, KB citation metrics của E14-B khi áp dụng và efficiency. Mẫu số và cách tính theo mục 6.3 và 7 của protocol. Báo metrics tổng hợp và theo question category. Nếu dùng CI, resample cặp theo image ID để giữ tất cả câu hỏi của ảnh cùng nhau. Giữ mẫu `policy_blocked` trong cohort và ghi riêng. Phản hồi do controller hoặc fallback template sinh chỉ phục vụ đánh giá hệ thống E15. [Research Protocol](research_protocol.md)

## 4. Dependency Registry — Task 08.04

Thuận là owner của cả bảy deliverables. Các mã dưới đây là mã bàn giao đề xuất để truy vết. Đức Anh và Kiệt có thể phát triển bằng fixture/sample inputs được phép trước khi nhận dữ liệu chính thức. Với pilot có mẫu thật, phê duyệt và audit development subset trước khi chạy. Formal Evaluation chỉ dùng deliverables đã duyệt và khóa.

| Deliverable ID | Deliverable | Owner | Consumer | Cần trước | Version/hash yêu cầu | Yêu cầu nghiệm thu | Current status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `08.04-KB` | Breed Knowledge Base | Thuận | Đức Anh (E12-B/E13-B), Kiệt (E14-B), E15 | Trước formal đối với morphology banks và E14-B. Trước tích hợp E15. | Snapshot version và SHA-256 `TBD` | Đủ 30 ID canonical, thuộc tính và nguồn đã review, tách KB runtime khỏi annotation/answers | `PLANNED` |
| `08.04-NAME` | Name Prompt Bank | Thuận | Đức Anh (E12-A/E13-A), E15 nếu dùng profile tương ứng | Trước formal | Version, content SHA-256 và mapping hash `TBD` | Một prompt mỗi lớp, đúng class ID, giữ template và đã kiểm token budget | `PLANNED` |
| `08.04-MORPH` | Morphology Prompt Bank | Thuận | Đức Anh (E12-B/E13-B), E15 nếu dùng profile tương ứng | Trước formal | Version, content SHA-256 và KB/source provenance `TBD` | Một prompt mỗi lớp, giữ tên giống, thuộc tính có nguồn và đã kiểm token budget | `PLANNED` |
| `08.04-RETRIEVAL` | Knowledge Retrieval | Thuận | Kiệt (E14-B), E15 | Trước formal E14-B và trước tích hợp E15 | Source revision và version/hash cấu hình lookup `TBD` | Lookup theo class ID/alias, nhiều giống, xử lý hit, miss, ambiguous và budget, tách ground truth | `PLANNED` |
| `08.04-VLM-EVAL` | VLM-Eval dataset | Thuận | Kiệt (E14-A/B) | Trước formal E14 | Dataset revision và manifest SHA-256 `TBD` | Audit trước khóa, hợp lệ về quyền sử dụng, ID, nhóm ảnh và phân bổ câu hỏi. Mục tiêu 80 ảnh/240 cặp chờ duyệt | `PLANNED` |
| `08.04-RUBRIC` | Annotation Rubric | Thuận | Kiệt (E14-A/B), người chấm | Trước formal E14 | Rubric version và SHA-256 `TBD` | Cùng thước đo nội dung E14-A/B, KB citations riêng, hướng dẫn abstention và hòa giải | `PLANNED` |
| `08.04-AUDIT` | Data Audit | Thuận | Đức Anh (E12/E13), Kiệt (E14), E15 với development inputs | Audit pilot subset trước pilot. Audit cohort và khóa evidence trước formal. | Report revision, manifest và report hashes `TBD` | Kiểm scope, quyền dùng, canonical mapping khi áp dụng, trùng/gần trùng, split isolation và giới hạn pretraining-overlap audit | `PLANNED` |

## 5. Run identification và lifecycle

### 5.1. Định danh và lưu trữ

| Trường | Quy ước |
| --- | --- |
| Experiment ID | Thiết kế cố định trong registry. Các biến thể A/B giữ ID riêng. |
| Run ID | Một lần thực thi. Format đề xuất `<experiment_id>_<phase>_<YYYYMMDDTHHMMSSZ>_<uuidv4>`, thời gian UTC và UUID v4. Cấp ID mới cho retry hoặc cấu hình mới. |
| Phase | `pilot`, `formal` hoặc `exploratory`. `formal` với E15 là run nghiệm thu tích hợp theo phạm vi đã duyệt. |
| Output | `outputs/research/vlm/<experiment_id>/<run_id>/`. Tạo mới độc quyền, giữ output và metadata của run cũ. Cặp A/B dùng group ID chung trong metadata. |

### 5.2. Research lifecycle

| Trạng thái | Điều kiện |
| --- | --- |
| `PLANNED` | Đã đăng ký thiết kế và phụ thuộc. |
| `PILOT_IN_PROGRESS` | Pilot được phép đang chạy trên input đúng phạm vi. |
| `PILOT_REVIEW` | Evidence pilot sẵn sàng để đánh giá khả thi. |
| `READY_FOR_LOCK` | Đã gom hồ sơ và xử lý điều kiện cần thiết, chờ trưởng nhóm khóa. |
| `LOCKED` | Trưởng nhóm đã phê duyệt revision/config/cohort và quy tắc formal. |
| `FORMAL_IN_PROGRESS` | Formal Evaluation hoặc E15 acceptance run đã được phép và đang thực hiện. |
| `RESULTS_READY` | Evidence và phân tích sẵn sàng để review. |
| `ACCEPTED` | Trưởng nhóm nghiệm thu thiết kế thực thi và evidence theo tiêu chí đã khóa. |
| `BLOCKED` | Thiếu điều kiện khả thi, an ninh, dữ liệu, quyền hoặc evidence. Ghi điều kiện chặn và điểm quay về khi gỡ chặn. |
| `REJECTED` | Review kết luận chưa đạt tiêu chí nghiệm thu. Ghi lý do và giữ evidence. |

Luồng dự kiến đi từ `PLANNED` đến `RESULTS_READY` theo các gate ở bảng, rồi trưởng nhóm quyết định `ACCEPTED` hoặc `REJECTED`. Có thể ghi `BLOCKED` khi gate thất bại. Chuyển trạng thái dựa trên evidence và giữ lịch sử. E12–E15 hiện đều `PLANNED`.

### 5.3. Staged Protocol Lock

Cho phép trưởng nhóm phê duyệt lock riêng cho bốn nhóm **E12-A/B**, **E13-A/B**, **E14-A/B** và **E15**. Mỗi nhóm phải hoàn tất toàn bộ mục Research Protocol Lock áp dụng cho mình, ghi revision/config/input hashes và phê duyệt phạm vi trước lượt formal. Hai biến thể trong một cặp dùng cùng hồ sơ lock cho các biến kiểm soát. E15 khóa profile đủ điều kiện, phạm vi, fixture, tiêu chí và điều kiện tích hợp trước lượt nghiệm thu hệ thống.

Các phép đối chứng dùng chung phải đối chiếu đúng cohort/Image IDs, canonical mapping và metric definitions. Đối chiếu prompt banks và cấu hình theo loại đối chứng đã đăng ký, ghi rõ khác biệt native bắt buộc đã duyệt. Shared inputs đổi thì đánh giá lại phép so sánh chịu ảnh hưởng trước khi ghép evidence. Staged lock giữ nguyên mọi gate nghiên cứu. Nếu formal results đã được quan sát, áp dụng quy tắc version mới và exploratory khi dùng lại cohort theo Research Protocol. [Research Protocol](research_protocol.md)

### 5.4. Run execution status

| Trạng thái | Điều kiện |
| --- | --- |
| `RUNNING` | Run đã bắt đầu và đang thực thi. |
| `COMPLETED` | Run đã kết thúc và hoàn tất ghi evidence theo cấu hình. |
| `FAILED` | Lỗi kỹ thuật làm run dừng hoặc output không hoàn thành. |
| `ABORTED` | Run bị dừng theo quyết định vận hành hoặc người thực hiện. |

`COMPLETED` chỉ xác nhận một lượt thực thi đã kết thúc. Nghiệm thu experiment thuộc research lifecycle. Với gate chặn trước lúc chạy, ghi blocker ở hồ sơ experiment và lý do trước khi có run. Các nhãn tình trạng evidence `complete`, `failed`, `blocked` trong Research Protocol mục 9.1 dùng để theo dõi đầu ra hoặc lý do thiếu evidence. Với run thực sự đã chạy, metadata dùng bốn execution statuses trong bảng cùng run ID.

## 6. Evidence checklist

Checklist chung áp dụng cho **từng experiment** E12-A, E12-B, E13-A, E13-B, E14-A, E14-B và E15. Evidence đặc thù ghi thêm trong từng hồ sơ. Trường không áp dụng cần ghi `N/A` kèm lý do. Checklist dưới đây ghi yêu cầu của lượt thực thi tương lai.

- [ ] Source code và Git SHA của entry point, adapter, scorer hoặc controller đã dùng.
- [ ] Config đã khóa, command, experiment ID, run ID, phase và trạng thái thực thi.
- [ ] Model provenance, model/processor/tokenizer revisions, weight fingerprint và phê duyệt security gate nếu áp dụng.
- [ ] Environment report và profile tài nguyên.
- [ ] Dataset/fixture, manifest và canonical mapping hashes, phạm vi input được phép.
- [ ] Prompt/KB versions và hashes theo điều kiện.
- [ ] Per-sample results hoặc per-case traces, join IDs và status của lỗi.
- [ ] Metrics theo protocol, counts, mẫu số và bằng chứng ghép cặp khi áp dụng.
- [ ] Benchmark và định nghĩa phạm vi đo.
- [ ] Error logs và blocker/fallback reasons.
- [ ] Analysis report, provenance checks và các giới hạn diễn giải.
- [ ] Tests phù hợp với phạm vi và reproduction instructions có version.

**E14-A/B**

- [ ] Raw model output, validated model output, controller-generated response và policy-blocked status được lưu riêng.
- [ ] Retrieval statuses, source IDs, annotation records, inter-rater agreement và evidence hòa giải.
- [ ] Kiểm tra giống nhau của input chung sau processor/tokenizer, giữ chung rubric nội dung và đối chiếu toàn cohort.

**E15**

- [ ] Integration tests và API compatibility cho `/predict`, mode và decision parity kể cả khi VLM lỗi.
- [ ] Session isolation, stale result, fallback, optional dependencies, timeout/OOM, recovery và resource tests.
- [ ] Full regression evidence trong phạm vi an toàn được duyệt, kèm môi trường và kết quả chi tiết.

## 7. Model selection và quyết định còn mở

- **Research acceptance.** Xét tuân thủ protocol, đủ evidence, minh bạch về failures và chất lượng phân tích. Kết quả âm tính vẫn có thể được nghiệm thu. Vượt Accuracy của E4 không phải điều kiện nghiên cứu.
- **Deployment eligibility trước tích hợp.** Sau Formal Evaluation và nghiệm thu nghiên cứu, xét quality, failure, tài nguyên, tương thích và evidence về readiness của profile theo các gate đã khóa. Chọn tối đa một encoder/profile theo Research Protocol mục 8. Đây là điều kiện đầu vào E15. Nghiệm thu E15 dựa trên evidence tích hợp và hồi quy sau đó. E4 giữ quyền quyết định phân loại.
- **Cô lập blocker.** E13 `BLOCKED` chỉ chặn nhánh đó và các đối sánh cần E13. E12/E14 có thể tiếp tục nếu tự đáp ứng gate. E15 dùng các profile đã nghiệm thu và đủ deployment eligibility. Nếu thiếu profile hoặc dependency làm hẹp phạm vi, trưởng nhóm duyệt reduced scope trước nghiệm thu E15 và báo phần thiếu. Các experiment bị chặn giữ `BLOCKED` đến khi có bằng chứng gỡ blocker.
- **Evidence E4.** Thiếu predictions E4 hợp lệ chỉ chặn những paired comparisons cần chúng. Standalone metrics hợp lệ của encoder trên cohort đã khóa vẫn được báo cáo cùng trạng thái của các đối sánh bị chặn.
- **Model không khả thi.** Nếu FG-CLIP 2 không qua feasibility/security gate, đánh dấu cả E13-A/B `BLOCKED`, giữ ID và model. Nếu Qwen không khả thi, SmolVLM phải qua pilot và được khóa trước khi chạy cả E14-A/B. Không hoán đổi model trong cặp.
- **Phạm vi.** Mandatory scope chỉ gồm các ID trong mục 2. Fine-tuning, Linear Probe và Fusion nằm ngoài phạm vi.
- **PENDING PILOT.** Revision/model khả thi, processor, scoring/token budgets, single-pass hoặc multi-pass, cách dùng CNN context, schema, dtype/device, concurrency, timeout, recovery và chi phí tài nguyên.
- **TBD trước Protocol Lock.** Scope pilot, quyền dùng hoặc replay đối chứng E4, dataset/manifest và VLM-Eval sau audit, versions/hashes Task 08.04, custom code approval nếu cần, bootstrap settings, acceptance thresholds, output access và cấu hình E15.
- **Chờ trưởng nhóm.** Duyệt từng hồ sơ Protocol Lock và cohort/tiêu chí trước formal. Xác nhận nghiệm thu nghiên cứu, quyết định profile đủ điều kiện triển khai và duyệt reduced scope trước nghiệm thu E15 nếu cần. [Architecture](architecture.md), [Research Protocol](research_protocol.md)
