# DogBreedID V2 — Research Protocol

**APPROVED FOR IMPLEMENTATION PLANNING — PROTOCOL LOCK PENDING**

**Task 08.01 · Phase 2.2 · Hybrid CNN–VLM with Knowledge-Augmented Visual Understanding**

Phê duyệt này áp dụng cho thiết kế phương pháp nghiên cứu và lập kế hoạch triển khai E12–E15. Cấu hình thực nghiệm cụ thể và quyền chạy Formal Evaluation chờ Research Protocol Lock do trưởng nhóm duyệt. Checklist Protocol Lock vẫn để trống. Các thông số PENDING PILOT và các mục chờ trưởng nhóm duyệt giữ nguyên. Mốc đối chiếu thiết kế là tag `v1.0.0-cnn-stable` tại commit `b7d880506b96168b0e335f3a455be3a7b2bdf77e` trên nhánh `research/vlm-protocol`. [Kiến trúc DogBreedID V2](architecture.md)

## 1. Câu hỏi và giả thuyết nghiên cứu

| ID | Câu hỏi | Giả thuyết cần kiểm tra |
| --- | --- | --- |
| RQ-VLM-01 | Zero-shot VLM khác supervised CNN E4 thế nào trên cùng Validation-300? | Hai phương pháp có thể khác về Top-1, Macro-F1 và nhóm giống hay nhầm. So sánh ghép cặp theo ảnh, không giả định chiều chênh lệch. |
| RQ-VLM-02 | Morphology prompts ảnh hưởng thế nào so với name prompts? | Thuộc tính thị giác có thể giúp phân biệt một số giống hoặc làm loãng tín hiệu tên giống. Hiệu ứng có thể khác theo encoder. |
| RQ-VLM-03 | SigLIP 2 và FG-CLIP 2 khác nhau thế nào ở cùng loại prompt? | Hai hệ encoder, processor và scoring có thể tạo thứ hạng khác nhau. Đối chiếu riêng trong từng điều kiện prompt. |
| RQ-VLM-04 | Generative VLM quan sát thuộc tính và trả lời VQA tiếng Việt tốt đến đâu? | Độ đúng có thể thay đổi theo khả năng nhìn thấy thuộc tính, loại câu hỏi và nhóm ảnh. Model có thể trả lời chính xác hoặc từ chối phù hợp khi thiếu căn cứ. |
| RQ-VLM-05 | KB context ảnh hưởng thế nào đến chất lượng trả lời và hallucination? | Kiến thức truy xuất có thể giúp câu hỏi cần kiến thức giống, nhưng cũng có thể gây suy diễn thuộc tính điển hình lên ảnh. Hiệu ứng được đo trên toàn cohort. |
| RQ-VLM-06 | VLM có đáp ứng chi phí tài nguyên, độ tin cậy và yêu cầu tích hợp với CNN không? | Lazy loading và giới hạn tác vụ có thể giúp giữ hợp đồng CNN. Độ trễ và tài nguyên cần được đo trên máy mục tiêu cả khi chạy riêng và đồng thời. |

Kết quả âm tính, kết quả hỗn hợp và khoảng tin cậy rộng đều được báo cáo khi thực nghiệm tuân thủ protocol. Kết luận giới hạn ở cohort, revision và cấu hình đã đo.

## 2. Phạm vi và các mốc phê duyệt

| ID | Điều kiện bắt buộc | Câu hỏi |
| --- | --- | --- |
| E12-A | `google/siglip2-base-patch16-224`, name prompt | RQ-VLM-01–03 |
| E12-B | `google/siglip2-base-patch16-224`, morphology prompt | RQ-VLM-01–03 |
| E13-A | `qihoo360/fg-clip2-base`, name prompt | RQ-VLM-01–03 |
| E13-B | `qihoo360/fg-clip2-base`, morphology prompt | RQ-VLM-01–03 |
| E14-A | Generative VLM, KB context rỗng | RQ-VLM-04–05 |
| E14-B | Cùng generative VLM và pipeline, bật truy xuất KB | RQ-VLM-04–05 |
| E15 | System Integration and Regression | RQ-VLM-06 |

- **Ngoài phạm vi.** CNN retraining, VLM fine-tuning, LoRA/QLoRA, Linear Probe, score fusion, threshold tuning, complex vector database, separate LLM và Official Test execution.
- **Bảo toàn baseline.** Giữ E4 checkpoint seed 42, canonical mapping, preprocessing và kết quả đã khóa. E6 giữ temperature `0.80246283961825`. E7 giữ ngưỡng `0.5011672918550012` và toán tử `>=`. Energy OOD giữ temperature `1.0` và `OOD_REJECT` khi energy thô vượt `-0.6801182939636002`. Energy dùng logits thô E4 và quyết định OOD có ưu tiên cao hơn E7. Giữ E8 corruption config, kết quả E8, Grad-CAM và hành vi `/predict`. [Kiến trúc V2](architecture.md), [Configuration Lock hiện có](../../final_evaluation/configuration_lock.md), [Phương pháp Energy OOD](../results/ood_200_summary.md)
- **Trình tự.** Pilot kiểm tra khả thi trên development subset riêng. Protocol Lock phê duyệt input, cấu hình và kế hoạch phân tích trước Formal Evaluation. Formal Evaluation thực hiện E12–E14 trên cohort đã khóa. Sau nghiệm thu formal, chọn tối đa một encoder/profile triển khai và kiểm tra E15. Pilot và formal benchmark được ghi thành các giai đoạn riêng.
- **Điều kiện chặn.** Ứng viên hoặc cohort không qua gate thì ghi `BLOCKED` kèm lý do. Giữ experiment ID và phạm vi bị thiếu trong báo cáo. Thay đổi phạm vi cần phê duyệt trước khi thực thi điều kiện đã điều chỉnh.

## 3. Dataset và ranh giới truy cập

| Cohort | Vai trò trong Task 08 | Quy tắc |
| --- | --- | --- |
| Train-2400 | Nguồn có thể lấy development subset để pilot | Dùng tập con được duyệt. Ghi manifest hoặc danh sách ID ảnh pilot, nguồn và audit. Mandatory scope chỉ chạy pretrained inference. |
| Validation-300 | Formal Evaluation cho E12/E13 và đối chiếu E4 | Dùng đủ 300 ảnh, 30 lớp, 10 ảnh/lớp theo split canonical. Giữ canonical IDs `0–29` và thứ tự ánh xạ. Cohort từng dùng để chọn checkpoint/model CNN nên phép đo thuộc development evaluation, với giới hạn về tính độc lập. |
| Calibration-300 | Giữ nguyên vai trò lịch sử E6/E7 và Energy OOD | Không dùng để tuning hoặc hiệu chỉnh VLM. Đối chiếu trùng lặp chỉ ở phạm vi development inputs được duyệt riêng. |
| Official Test | Ngoài phạm vi Task 08 | Không truy cập ảnh, nhãn, manifests, predictions hoặc protected bundles để đối chiếu, audit hay phát triển VLM. Không chạy Official Evaluation. |
| VLM-Eval-80 | Cohort đề xuất cho E14-A/B | Mục tiêu gồm 60 ảnh thuộc 30 giống (2 ảnh/giống), 10 ảnh chó ngoài phạm vi, 10 ảnh không phải chó. Dự kiến 3 câu hỏi/ảnh, tương ứng 240 cặp ảnh–câu hỏi nếu đủ dữ liệu. Số lượng là mục tiêu chờ audit, duyệt quyền sử dụng và Protocol Lock. |

Split phát triển của Stanford Dogs được mô tả trong tài liệu Dataset Freeze với tỷ lệ `80/10/10`, stratified theo lớp và seed 42. Dùng mapping chính thức và kiểm tra hash của bản mapping tại thời điểm khóa. Task 08 giữ split hiện có. [Dataset Protocol](../../protocol/dataset_protocol.md), [Canonical class mapping](../../../data/metadata/class_mapping.json), [Cấu hình dữ liệu](../../../configs/data.yaml)

### 3.1. Thiết kế và audit VLM-Eval

- Manifest phải có **Sample ID, Image ID, Source URL, license hoặc usage permission, collection date, canonical class ID khi áp dụng, SHA-256, perceptual hash, audit status, question IDs và annotation provenance**. Một ảnh có Image ID ổn định. Mỗi đơn vị đánh giá có Sample ID duy nhất và ánh xạ được đến cặp Image ID và Question ID. Annotation provenance ghi người hoặc vai trò annotate/review, thời điểm và revision của rubric. Nhóm ngoài 30 giống hoặc không phải chó dùng class ID rỗng và group riêng trong metadata chấm.
- Kiểm tra quyền sử dụng, danh tính ảnh, nhãn nhóm và khả năng đọc. Đối chiếu SHA-256 để phát hiện trùng chính xác, perceptual hash theo thuật toán và ngưỡng được duyệt để gắn cờ gần trùng. Review cờ gần trùng và ghi quyết định. Audit trong VLM-Eval, với tập pilot và các development datasets được phép. Ghi phạm vi, phương pháp và giới hạn audit. Official Test nằm ngoài mọi phép kiểm trùng.
- Hoàn tất audit trước khi khóa cohort. Nếu thiếu dữ liệu hợp lệ hoặc không thể đạt độ phủ, ghi `BLOCKED` hoặc trình trưởng nhóm duyệt quy mô, phân tầng và giới hạn kết luận đã điều chỉnh trước khi chạy. Nếu phát hiện lỗi dữ liệu sau khóa, ghi nhận và xin quyết định tái khóa theo mục 8.
- Audit chỉ phát hiện overlap trong phạm vi nguồn đã đối chiếu. Có thể tồn tại overlap với pretraining corpora hoặc nguồn ngoài phạm vi kiểm tra. Báo cáo nguy cơ và bằng chứng sẵn có, không suy ra đã loại trừ leakage.
- Tách KB runtime, prompt banks, manifest chấm, annotation rubric và ground-truth answers theo snapshot, đường dẫn và quyền truy cập. Runtime nhận ảnh được phép, câu hỏi và ngữ cảnh hợp lệ. Trường label và câu trả lời chuẩn chỉ chuyển sang scorer. Không dùng annotation/review của cohort formal để viết morphology prompts hoặc cập nhật KB cho phiên formal.

### 3.2. Bằng chứng đối chứng đã có

| Đối chứng | Bằng chứng Validation đã công bố | Cách dùng trong Task 08 |
| --- | --- | --- |
| E4 seed 42 | Top-1 245/300, Top-5 296/300, Macro-F1 0,8151. Hồ sơ Validation Replay ghi nhận `validation_raw_predictions.npz` với `image_ids`, `labels` và `logits`. | Đối chứng supervised chính. Ưu tiên predictions đã lưu, chỉ lấy bản không protected và đã được cấp quyền. |
| E1 seed 42 | Top-1 241/300, Top-5 297/300, Macro-F1 0,7996. | Bổ sung kết quả nếu xác minh được cùng cohort. Chỉ kiểm định ghép cặp khi có predictions theo ảnh. |
| E10-A/B, E11-A/B | Báo cáo Validation N=300, 30 lớp, seed 42. Có số liệu tổng hợp theo cấu hình, kèm các profile benchmark khác thiết bị. | Đối chứng bổ sung nếu bằng chứng xác nhận cùng cohort. Phân tích theo ảnh cần predictions tương ứng. Latency CPU/GPU lịch sử giữ nhãn riêng. |

Đối chiếu ID không trùng, đủ 300 ảnh, nhãn khớp theo ID, mapping, fingerprint của cohort, checkpoint/config và provenance trước khi join. Tính lại metric phân loại bằng cùng quy tắc ranking trên raw outputs đã xác minh. Dùng E4 standard preprocessing và ranking 30 lớp, trước bộ quyết định E7/Energy OOD. Tỷ lệ chấp nhận, selective accuracy và OOD rejection là hành vi vận hành riêng. Nếu thiếu baseline ghép cặp, báo phần kiểm định đó `BLOCKED`. Chỉ đề xuất replay read-only trên development data được trưởng nhóm cho phép và sau khi kiểm SHA-256 checkpoint E4 đã khóa. Các số tổng hợp đã công bố dùng để mô tả tham chiếu với nguồn, không cấp dữ liệu cho phép kiểm định theo ảnh. [Bảng Validation E1/E4](../../task06/experiment_summary/validation_comparison.csv), [Số liệu Validation E10/E11](../results/e10_e11_validation_comparison.md), [Validation Reproducibility](../../final_evaluation/task05_validation_reproducibility.md), [Nghiệm thu Validation B1](../../final_evaluation/task05_phase_b1_validation_acceptance.md)

## 4. Pilot

Dùng ảnh phát triển được duyệt, có ID đối chiếu và audit tách biệt khỏi Validation-300 và VLM-Eval formal. Kết quả phục vụ khả thi, chọn cấu hình đề xuất và ước lượng tài nguyên. Bộ cấu hình được đề xuất qua pilot chỉ có hiệu lực formal sau Protocol Lock.

| Pilot | Kiểm tra | Đầu ra cần ghi |
| --- | --- | --- |
| Encoder | Nạp revision, tương thích processor/tokenizer, scoring đủ 30 lớp, prompt token limits và nguy cơ truncate, ngữ nghĩa/chiều score, ranking và tie-break tái lập | Revision và tương thích, tính hợp lệ của score/ranking, thời gian, peak RAM/VRAM, lỗi hoặc lý do `BLOCKED` |
| Generative | Qwen trước rồi cân nhắc SmolVLM, VQA tiếng Việt, structured output và schema, `unknown`/`not_visible`, single-pass/multi-pass, context/token limits | Cấu hình chung đề xuất cho E14-A/B, parse/schema failures, latency generation, peak RAM/VRAM |
| Resource và regression | CPU/GPU/RAM/VRAM thực tế, OS/driver/dependency, cold/warm, CNN/Grad-CAM dưới tải VLM, timeout, OOM và phục hồi | Một profile benchmark có thông số đo, giới hạn concurrency, queue và timeout dự kiến, nhu cầu worker/environment riêng |

Với model yêu cầu custom/remote code, áp dụng gate trong kiến trúc đã duyệt. Xác minh và pin revision, review source, xin trưởng nhóm phê duyệt trước khi thực thi trong môi trường biệt lập không truy cập protected data hoặc credentials. Khả năng chạy trên phần cứng nhóm và các cấu hình ở mục 10 vẫn **PENDING PILOT**. [Gate thực thi model code](architecture.md)

## 5. E12/E13 — Zero-shot classification

### 5.1. Thiết kế kiểm soát

- Mỗi cặp A/B khóa model revision, processor/tokenizer, ảnh, class order, scoring và aggregation, precision/device profile và metrics. Chỉ thay nội dung prompt. Dùng native model processor để đưa cùng ảnh nguồn vào từng encoder. Ghi riêng input resolution, crop/resize, normalization và các bước native đã kiểm.
- Mỗi bank có một prompt chính cho mỗi canonical class. Name bank dùng tên giống theo template. Morphology bank giữ cùng tên, bổ sung thuộc tính thị giác có nguồn đã duyệt. Giữ phần khung template nhất quán, kiểm số token theo từng tokenizer, ghi limit và kiểm hết 30 prompt trước khi khóa. Truncation trái với cấu hình đã khóa làm output không hợp lệ. Nếu prompt vượt giới hạn, sửa và audit lại trước Protocol Lock.
- Ưu tiên dùng cùng nội dung hai bank cho hai encoder. Ghi rõ và khóa khác biệt do native interface bắt buộc. Nếu content hoặc scoring phải khác ngoài mức cần thiết, diễn giải E12 so với E13 là so sánh hai cấu hình đầu cuối. Ở scope một prompt/lớp, aggregation mặc định là sử dụng trực tiếp score của prompt tương ứng. Cố định quy tắc adapter trước formal.
- Lưu vector score hữu hạn đủ 30 lớp theo canonical ID, class ID Top-1 và 5 ID xếp đầu. Ghi loại score thực dùng, chiều xếp hạng, scaling/normalization theo adapter và tie-break theo class ID tăng dần sau khi cùng quy ước score cao hơn xếp trước. Similarity/logit là điểm xếp hạng, không dùng như calibrated confidence. Lưu ID ảnh, status, lý do failure/abstention và provenance. Nếu run đã bắt đầu rồi gặp lỗi, giữ đủ ảnh đã lên lịch và trạng thái từng ảnh để giải trình. Với run bị chặn trước thực thi, báo `BLOCKED` và lý do.

### 5.2. Metric và mẫu số

Gọi `N = 300` là số ảnh đã khóa, `C = 30` là số lớp. Mỗi ảnh góp đúng một dòng theo từng cấu hình. Label thật dùng tại scorer. Output thiếu, NaN/Inf, sai contract hoặc inference không hoàn thành ghi `FAIL`. Kết quả từ chối có chủ đích ghi `ABSTAIN` nếu phát sinh. Hai trạng thái đều có cờ đúng `0` với Top-1/Top-5.

| Metric | Định nghĩa chính |
| --- | --- |
| Top-1 Accuracy | Số ảnh có ID hạng 1 bằng nhãn thật chia `N`. |
| Top-5 Accuracy | Số ảnh có nhãn thật trong 5 ID đầu chia `N`. |
| Per-class Precision / Recall / F1 | Với lớp `c`, `P=TP/(TP+FP)`, `R=TP/(TP+FN)`, `F1=2TP/(2TP+FP+FN)`. Mẫu số bằng 0 dùng giá trị 0 theo quy ước source. FAIL/ABSTAIN đóng góp FN cho lớp thật. |
| Macro-F1 | Tổng F1 của đúng 30 canonical classes chia `C`. FAIL/ABSTAIN là trạng thái dự đoán bổ sung, không phải lớp tính trung bình. |
| Confusion matrix | 30 hàng nhãn thật, 30 cột canonical dự đoán và hai cột `FAIL`, `ABSTAIN`. Báo counts và có thể thêm tỷ lệ chuẩn hóa theo hàng. Mỗi hàng có 10 ảnh ở cohort đầy đủ. |
| Failure rate và abstention rate | Số ảnh theo từng trạng thái chia `N`. Ghi counts và nguyên nhân theo nhóm. |
| Paired disagreement | Tỷ lệ ảnh có Top-1 ID hoặc status khác nhau giữa hai cấu hình, chia `N`. Báo thêm bốn ô đúng/sai ghép cặp và số lượt mà ít nhất một bên FAIL/ABSTAIN. |
| Successful-only bổ sung | Tính trên số ảnh có ranking hợp lệ, ghi rõ mẫu số và quy tắc tuyển mẫu. Khi đối chiếu ghép cặp bổ sung, dùng giao hai tập thành công và ghi số ảnh bị loại. |

Source CNN `classification_metrics()` tính Top-1/Top-5 theo tỉ lệ và Macro-F1 trên tập ID cố định với `zero_division=0`. Protocol này bổ sung trạng thái failure/abstention và tie-break xác định cho scorer chung. [Source classification metrics](../../../src/evaluation/metrics.py), [Evaluation Protocol CNN](../../protocol/evaluation_protocol.md)

### 5.3. Phép so sánh định trước

1. **RQ-VLM-01.** Từng điều kiện E12-A, E12-B, E13-A và E13-B so với E4 trên cùng Validation-300.
2. **RQ-VLM-02.** E12-B so với E12-A, E13-B so với E13-A.
3. **RQ-VLM-03.** E13-A so với E12-A, E13-B so với E12-B.

- **Endpoint kiểm định.** Top-1 đúng/sai theo ảnh. Với `b` ảnh chỉ cấu hình thứ nhất đúng và `c` ảnh chỉ cấu hình thứ hai đúng, McNemar exact hai phía dùng `X ~ Binomial(b+c, 0.5)` và `p = min(1, 2 × P[X ≤ min(b,c)])`. Nếu `b+c=0` thì `p=1`. Báo `b`, `c` và chênh lệch Top-1 `(b-c)/N` theo thứ tự đã đăng ký.
- **Multiple comparisons.** Đề xuất một family gồm tám so sánh Top-1 và Holm correction ở `alpha=0.05`. Nếu phép so sánh bị chặn, ghi `BLOCKED`, không báo p-value quan sát cho phép đó. Giữ kích thước family đã đăng ký bằng cách xử lý phép bị chặn như `p=1` chỉ trong bước tính ngưỡng Holm. Đối chiếu E1/E10/E11 và phân tích theo lớp là bổ sung, ghi rõ phạm vi.
- **Độ bất định.** Đề xuất paired bootstrap percentile 95% cho chênh lệch Top-1, Top-5 và Macro-F1. Resample Image ID có hoàn lại trong từng lớp, giữ 10 ảnh/lớp và giữ cặp outputs cho mỗi draw. Tính lại metric trên mỗi replicate. Seed và số replicate chờ Protocol Lock. Ghi CI danh nghĩa cùng effect size, chưa hiệu chỉnh đồng thời. Kết quả không có ý nghĩa thống kê hoặc CI chứa 0 không xác lập tương đương.

## 6. E14 — Structured Visual Understanding và VQA

### 6.1. Cố định pipeline và can thiệp KB

Ứng viên chính là `Qwen/Qwen3-VL-2B-Instruct`, dự phòng là `HuggingFaceTB/SmolVLM-500M-Instruct`. Pilot dùng ảnh riêng để đề xuất một model và pipeline single-pass hoặc multi-pass. Trưởng nhóm duyệt revision và toàn bộ pipeline trước formal. [Kiến trúc trợ lý ảnh](architecture.md)

| Nhánh | Can thiệp |
| --- | --- |
| E14-A | Giữ slot KB rỗng, retrieval `not_applicable`. |
| E14-B | Chèn phần tri thức và source IDs lấy từ snapshot KB đã khóa. Lookup canonical ID hoặc alias trong câu hỏi, giữ thứ tự xác định và budget cố định khi có nhiều giống. Ghi trạng thái truy xuất riêng. |

- Hai biến thể dùng cùng model/revision, processor/tokenizer, ảnh, câu hỏi, pipeline, prompt template, generation settings, seed policy, CNN context nếu có, schema và validation policy. Khi structured visual attributes là input cho retrieval hoặc VQA, tạo bản ghi chung từ đầu vào giống nhau trước khi chèn KB và dùng nguyên bản ghi cho cả hai biến thể. Khi pipeline single-pass trực tiếp xuất thuộc tính cùng câu trả lời, chấm thuộc tính như output riêng.
- Khóa cùng trường, thứ tự và quyền dùng ngữ cảnh CNN. Nếu dùng ID CNN đã `ACCEPT`, đọc từ kết quả E4 phù hợp với chính sách E15. Giữ nguyên giới hạn của `E7_REJECT` và `OOD_REJECT`. Không dùng ground-truth breed labels hoặc expected answers để tạo retrieval keys, CNN context, intermediate attributes hay model prompts. Tên giống được nêu trong câu hỏi kiến thức hoặc so sánh là phần input đã khóa của cả hai biến thể, không tự chèn từ nhãn ảnh.
- Dùng chung quy tắc parse, schema và citation validation. E14-B giữ định danh cả khi `miss`, `ambiguous`, `partial_hit` hoặc `retrieval_error`. Ghi trạng thái theo từng giống. Formal giữ nguyên câu hỏi. Cách chạy với context rỗng hoặc thiếu một phần phải được khóa trước và áp dụng nhất quán, không đổi model hoặc rerun theo kết quả.
- Giữ cùng context limit và ngân sách ảnh, câu hỏi, thông tin chung giữa E14-A/B. Trước khóa, đối chiếu đầu vào sau processor/tokenizer để xác minh ảnh, câu hỏi và thông tin chung được giữ nguyên, không bị truncate ở nhánh nào. Chỉ E14-B điền KB context trong phần ngân sách đã dành riêng. Nếu vượt ngân sách, áp quy tắc rút gọn đã khóa cho riêng phần KB hoặc ghi lỗi theo protocol, giữ các thành phần chung.

### 6.2. Câu hỏi và annotation

| Nhóm ảnh | Loại câu hỏi phù hợp | Bằng chứng chấm |
| --- | --- | --- |
| 30 giống canonical | Visible attributes, visibility and uncertainty, breed knowledge, breed comparison | Quan sát trực tiếp để chấm thuộc tính nhìn thấy. Nguồn đã duyệt để chấm kiến thức chung. Câu hỏi so sánh nêu rõ các giống cần xét. |
| Chó ngoài phạm vi | Visible attributes, visibility and uncertainty, out-of-scope handling | Mô tả phần ảnh có thể quan sát và phản hồi bất định. Chấm hành vi khi thiếu căn cứ đưa ra giống thuộc 30 lớp. |
| Không phải chó | General visual observation, visibility and uncertainty, non-dog handling | Nhận ra không đủ căn cứ cho thuộc tính chó, tránh bịa giống/đặc điểm. Có thể mô tả chi tiết ảnh phù hợp với câu hỏi. |

**Phân bổ mục tiêu.** Đề xuất 80 ảnh và 240 cặp ảnh–câu hỏi, dự kiến ba câu mỗi ảnh. Phân bổ chờ Task 08.04 audit và trưởng nhóm phê duyệt.

| Nhóm ảnh | Mục tiêu ảnh | Câu hỏi theo category | Mục tiêu cặp ảnh–câu hỏi |
| --- | ---: | --- | ---: |
| Chó thuộc 30 giống canonical | 60 | 60 visible attributes, 60 visibility/uncertainty, 60 breed knowledge/comparison | 180 |
| Chó ngoài phạm vi | 10 | 10 visible attributes, 10 visibility/uncertainty, 10 out-of-scope handling | 30 |
| Không phải chó | 10 | 10 general visual observation, 10 visibility/uncertainty, 10 non-dog handling | 30 |
| **Tổng** | **80** | — | **240** |

Báo metrics tổng hợp và theo question category, kèm số lượng và mẫu số của mỗi nhóm. Dataset và annotation được chuẩn bị ở Task 08.04 sau bước thiết kế này.

Mỗi câu hỏi khai báo category chính, yêu cầu trả lời hoặc abstain, trường thuộc tính và citation cần chấm nếu áp dụng. Phân bổ category theo nhóm ảnh và quy tắc xử lý câu hỏi đa ý được duyệt trước khóa. Task 08.04 chuẩn bị rubric chi tiết và tiêu chí chấm từng mức để trưởng nhóm duyệt. Ground truth có thể đánh dấu `unknown`/`not_visible` khi bằng chứng ảnh thiếu.

### 6.3. Metrics và đối chứng ghép cặp

Gọi `Q` là số cặp ảnh–câu hỏi đã khóa. Mỗi biến thể có `Q` trạng thái. Mẫu số các metric chính theo câu hỏi giữ cả lỗi thực thi, lỗi schema và mẫu bị policy chặn. Giữ mẫu bị chặn trong cohort, ghi `policy_blocked` và báo counts riêng. Mẫu bị chặn không được tính là model success. Dùng `N/A` khi metric không áp dụng hoặc mẫu số bằng 0, ghi rõ nguyên nhân.

| Metric | Đơn vị và cách tính |
| --- | --- |
| JSON parse rate | Số output parse được thành JSON chia `Q`. Lỗi inference hoặc output thiếu không được tính thành công. |
| Schema valid rate | Số output parse được và đúng schema/validation policy chia `Q`. Tách lỗi parse, schema và chính sách ở bảng counts. |
| Attribute accuracy | Số mục thuộc tính có giá trị hoặc trạng thái nhìn thấy đúng chia số mục cần chấm đã khóa, nhận diện theo Image ID, Question ID nếu gắn với câu hỏi, và Attribute ID. Thiếu output/từ chối sai tính không đúng. |
| VQA accuracy | Dùng cùng rubric về độ đúng nội dung cho E14-A/B. Số câu đáp ứng rubric hoặc abstain đúng yêu cầu chia `Q`. Điểm nội dung của E14-A không bị trừ chỉ vì thiếu citation từ KB. Điểm partial có thể trình bày riêng. |
| Hallucination rate | Tỷ lệ câu có ít nhất một claim sai hoặc không có căn cứ trong số câu có output đủ để chấm claim. Báo đồng thời số câu có hallucination trên `Q`, tỷ lệ output không chấm được và failure rate. |
| Groundedness | Số claim có căn cứ đúng từ ảnh hoặc nguồn đã kiểm theo loại claim, chia số claim chấm được. Báo số claim và số câu đủ điều kiện để tách ảnh hưởng của output ngắn hoặc lỗi. |
| Abstention quality | Abstain đúng trên số câu rubric yêu cầu từ chối/unknown, và abstain sai trên số câu có đủ căn cứ trả lời. Báo số lượng mỗi nhóm. Lỗi inference và output thiếu là failure, không được tính như abstain đúng. |
| Citation Correctness (KB) | Báo riêng cho E14-B khi áp dụng. Số citation hợp lệ, có nguồn KB và hỗ trợ claim chia số citation KB đã sinh trong output chấm được. Mẫu số 0 ghi `N/A`. Metric này là `N/A` cho E14-A. |
| KB Citation Coverage | Báo riêng cho E14-B. Số câu yêu cầu nguồn có ít nhất một citation thuộc nguồn KB đã truy xuất chia số câu yêu cầu nguồn được xác định trước, kể cả câu miss, bị chặn hoặc lỗi. Metric này là `N/A` cho E14-A. |
| Vietnamese quality | Chấm tiếng Việt theo rubric, gồm độ rõ, đúng nghĩa và bám câu hỏi. Báo phân bố mức hoặc mean trên số câu chấm được, kèm tỷ lệ đủ điều kiện trên `Q`. |
| Retrieval hit/miss/ambiguous rates | Cho E14-B, báo số hit đầy đủ, partial hit, miss, ambiguous, lỗi và not applicable. Tỷ lệ chính chia số câu đã xác định cần truy xuất, báo kèm counts trên `Q`. Ghi chi tiết theo giống và khóa cách tổng hợp trạng thái khi câu hỏi có nhiều giống. E14-A tắt retrieval, ghi `N/A`. |
| Inference failure rate | Số câu không hoàn thành inference, timeout hoặc OOM chia `Q`. Báo thêm failure do invalid output và nguyên nhân upstream riêng. |

**Phạm vi chấm.** E14 chấm output do generative VLM thực sự sinh ra. VQA accuracy đo độ đúng nội dung, còn Citation Correctness và KB Citation Coverage đo khả năng cung cấp nguồn. Chỉ tính câu trả lời hoặc từ chối do model sinh và đủ điều kiện chấm vào kết quả đúng của E14. E15 chấm kết quả hệ thống sau CNN policy, validators, fallback và Integration Controller. Thông báo cố định do controller tạo và fallback template thuộc kết quả E15. Khóa phạm vi offline model evaluation, runtime policy và quy tắc xử lý `policy_blocked` trước Formal Evaluation.

Một câu trả lời `unknown` hoặc `not_visible` đúng cần phù hợp với ảnh và rubric. Confidence do generative model tự khai chỉ dùng để mô tả bất định, không dùng như xác suất đã hiệu chỉnh. ROUGE-L hoặc Token-F1 có thể là phân tích bổ sung nếu được đăng ký, bên cạnh chấm groundedness và độ đúng theo rubric.

Đề xuất VQA accuracy toàn cohort làm endpoint chính của E14 và hallucination rate kèm failure/coverage để phân tích RQ-VLM-05. Báo chênh lệch E14-B − E14-A trên cùng cohort. Giữ riêng các kết quả có hallucination, không phát hiện hallucination trong output chấm được và output không chấm được để thấy ảnh hưởng của missingness. Với metrics chỉ chấm được trên output đủ điều kiện, trình bày mẫu số từng nhánh, loss of coverage và ghi rõ phân tích ghép cặp nếu chỉ dùng cặp chấm được ở cả hai nhánh. Phân tích bổ sung nhóm retrieval hit dùng đúng các cặp mà E14-B truy xuất đủ giống cần thiết, ghi kích thước nhóm trên `Q` và tiêu chí lọc. Với CI, resample theo Image ID, mang theo mọi câu hỏi và cả hai nhánh của ảnh để giữ phụ thuộc trong ảnh. Có thể phân tầng theo ba nhóm ảnh nếu được định trước.

Dự kiến hai người chấm độc lập trên cùng rubric, dùng ID ẩn và thứ tự output xáo trộn khi khả thi. Ghi hạn chế blinding khi citation hoặc format cho thấy điều kiện. Tính percent agreement và Cohen’s kappa trên nhãn hai người chấm trước hòa giải, dùng weighted kappa nếu rubric là thang thứ bậc và có quy tắc trọng số được khóa. Ghi mẫu số và trường hợp không xác định. Bất đồng được thảo luận theo evidence, chuyển người phân xử được chỉ định nếu chưa thống nhất, lưu nhãn ban đầu, nhãn cuối và lý do.

## 7. Efficiency benchmark

| Đại lượng | Đo và báo cáo |
| --- | --- |
| Loading time | Từ bắt đầu load từ local cache đến model sẵn sàng trên device. Ghi trạng thái cache và worker/process. |
| Cold inference | Request đầu sau khi model sẵn sàng, ghi rõ compile/init và cache text có thuộc phép đo hay không. Nếu đo cold start end-to-end có load, báo riêng. |
| Warm inference | Các request sau warm-up. Báo p50/p95, số đo hợp lệ, số lần chạy, timeout và định nghĩa đoạn timing. Đồng bộ GPU ở ranh giới cần thiết khi đo stage device. |
| End-to-end latency | Từ tiếp nhận request đến output validated, gồm queue, preprocessing, retrieval, forward/generation và hậu xử lý. Ghi trạng thái queue và profile cold/warm. |
| Retrieval và generation latency | Đo riêng KB lookup/context assembly và generate. Ghi số input/output tokens, stop rule và giới hạn đã khóa. |
| Text embedding cache | Đo thời gian tạo, kích thước, cache hit/miss và latency tương ứng cho encoder. Cố định cache key theo model, processor, mapping, prompt hash và scoring profile. |
| Peak RAM / VRAM | Đo riêng lúc load và inference, ghi phạm vi process/worker, RAM RSS, VRAM theo allocator hoặc công cụ thiết bị, phương pháp lấy mẫu và peak khi chạy cùng CNN. |
| Model storage size | Tổng MiB của weights và các tệp local cần cho profile, ghi phạm vi tính. Khi dẫn kích thước `state_dict` CNN lịch sử, ghi đây là cách tính khác. |
| Failure rate | Số request lỗi chia số request đã lên lịch trong từng profile, tách startup failure, timeout, OOM, schema failure và tài nguyên không khả dụng. |

Ghi CPU/GPU/RAM/VRAM, OS/driver, software versions, device, dtype, quantization hoặc `none`, image resolution, batch size, max input/output tokens, concurrency và điều kiện chạy. Đo CNN/Grad-CAM riêng và khi chạy chung VLM để định lượng tác động tích hợp lên latency và tỷ lệ lỗi. So sánh latency trực tiếp trên cùng phần cứng, tải, batch size, điều kiện cache và ranh giới đo, với những khác biệt native được ghi rõ. Các profile khác thiết bị được trình bày riêng. Số warm-up runs, measured runs, định nghĩa percentile estimator, p50/p95 reporting và resource limits được đề xuất sau pilot và khóa trước formal benchmark. [Convention benchmark CNN](../../protocol/evaluation_protocol.md), [Source measure_latency()](../../../src/evaluation/efficiency.py)

## 8. Research Protocol Lock

**Gate áp dụng cho Task 08.** Configuration Lock của Official Evaluation đã có phạm vi và quy trình cấp quyền riêng. Research Protocol Lock không thay đổi Configuration Lock hay điều kiện truy cập Official Test. [Configuration Lock hiện có](../../final_evaluation/configuration_lock.md)

Trước Formal Evaluation, trưởng nhóm xác nhận revision của protocol và duyệt checklist sau.

- [ ] Git commit của source và cấu hình, run entry point, dependency versions và environment profile.
- [ ] Revisions của model, processor/tokenizer, weight provenance và hash theo artifact, phê duyệt security gate nếu áp dụng.
- [ ] Mapping hash, dataset/manifest hash, tập pilot, kết quả audit, phạm vi và điều kiện sử dụng.
- [ ] Name/Morphology Prompt Bank hashes, template, KB snapshot hash, revision nguồn và rubric version.
- [ ] Preprocessing, class order, tie-break, scoring/aggregation, generation settings, seeds, precision và schema/validation policy. Chốt phạm vi đánh giá model offline, runtime policy và quy tắc giữ đầu vào chung của E14-A/B.
- [ ] Metrics, mẫu số, failure/abstention handling, quy tắc retrieval và cách chấm khi thiếu output, CI và seed/replicates.
- [ ] Primary comparisons, endpoint family, multiple comparison policy, acceptance thresholds và resource limits.
- [ ] Model selection rule và cách xử lý khi ứng viên bị chặn, thiếu evidence hoặc không đạt yêu cầu.
- [ ] Nơi lưu, quyền truy cập, run ID policy, yêu cầu provenance và điều kiện formal run được phép bắt đầu.

**Quy tắc lựa chọn đề xuất.** Sau formal và nghiệm thu, chỉ xét encoder/profile đủ evidence và qua các gate chất lượng, failure rate, tài nguyên, tương thích và E15 readiness theo tiêu chí đã khóa. Trong số profile đủ điều kiện, ưu tiên Macro-F1 toàn cohort, rồi Top-1, rồi chi phí warm end-to-end theo điều kiện đo chung. Chốt ngưỡng tối thiểu, biên hòa và thứ tự tie-break trước formal. Nếu không có profile đạt, ghi `BLOCKED` hoặc unavailable. Lựa chọn trên Validation cần ghi nhận selection bias và không được diễn giải như ước lượng trên tập độc lập.

Mọi sửa đổi sau khi thấy formal results phải có protocol version và run ID mới, nêu rõ lý do, thành phần thay đổi và phê duyệt. Nếu dùng lại cohort đã quan sát, ghi **exploratory** và giữ nguyên báo cáo cũ. Công bố các biến thể đã thực hiện và các phép bị chặn. Không ghi đè, cherry-pick hoặc thay tiêu chí theo kết quả.

## 9. Reproducibility, evidence và E15

### 9.1. Evidence cần có khi thực thi

Vị trí dự kiến `outputs/research/vlm/<experiment_id>/<run_id>/`, dưới quyền truy cập riêng cho Task 08. Dùng ID đầy đủ của từng nhánh, đánh dấu `pilot`, `formal` hoặc `exploratory` trong metadata, liên kết run cặp bằng ID chung. Tạo run directory riêng và ghi trạng thái complete/failed/blocked để phân biệt output hoàn chỉnh và dang dở.

- **Cấu hình và provenance.** Run configuration, Git SHA và tình trạng workspace, protocol revision, model/weight provenance, processor/tokenizer revisions, environment report, mapping/dataset/prompt/KB hashes, rubric revision, seeds và invocation đã ghi nhận.
- **Per-sample evidence.** Score vector/ranking và status E12/E13. Lưu riêng raw model output, validated model output, controller-generated response và policy-blocked status theo cặp Image ID/Question ID hoặc request ID. Nếu không gọi model, raw và validated output ghi `N/A`. Giữ schema status, source IDs, retrieval statuses và snapshot IDs cho E14. Ground truth và annotation chấm lưu theo quyền riêng. Join bằng Image ID/Question ID theo schema đã khóa.
- **Kết quả và phân tích.** Metrics JSON/CSV, counts và denominators, phân tích paired, benchmark results, failure logs, inter-rater agreement, analysis report và file hashes. Chỉ công bố dữ liệu có quyền sử dụng, bỏ credentials và metadata nhạy cảm khỏi log hoặc bản chia sẻ.

Các conventions lịch sử gồm pin Git SHA, checkpoint hash và environment, đối chiếu ID/label/logits với cohort canonical và kiểm integrity từng artifact. Acceptance dựa trên cả provenance lẫn semantic validation. Tài liệu phase này không tạo run artifacts hay kết quả giả lập. [Validation Reproducibility](../../final_evaluation/task05_validation_reproducibility.md), [Nghiệm thu Validation B1](../../final_evaluation/task05_phase_b1_validation_acceptance.md), [Nghiệm thu hồi quy cuối](../results/final_regression_acceptance.md)

### 9.2. E15 — System Integration and Regression

- Giữ E4 checkpoint, E6 temperature, E7 threshold, Energy OOD behavior, Grad-CAM, hai mode preprocessing hiện có và hành vi phân loại. So với baseline trên fixture hoặc development data đã duyệt theo trạng thái `ACCEPT`, `E7_REJECT`, `OOD_REJECT`. VLM bất đồng, timeout hoặc unavailable không đổi quyết định CNN.
- Kiểm tra `/predict` dùng input và default mode hiện hành, đúng thứ tự bảy output công khai, trạng thái reject và xử lý ảnh lỗi. Grad-CAM giữ quyền truy cập theo trạng thái CNN cùng kiểm tra mode, digest và revision. Đo CNN parity khi VLM tắt, bận hoặc lỗi.
- Kiểm tra session isolation, stale results, đổi ảnh/mode, Clear, hai request trả kết quả đảo thứ tự, timeout, OOM, phục hồi worker, model unavailable, KB miss/ambiguous, output sai schema và optional dependencies. Dùng fixture/fault injection trong phạm vi cho phép để kiểm tra ranh giới dữ liệu và giữ endpoint CNN khả dụng.
- Tiêu chí và fixture nghiệm thu chi tiết sẽ được định nghĩa trong `acceptance_criteria.md` ở phase sau. Evidence nghiệm thu hồi quy hiện có là tham chiếu cho baseline.

[Kiến trúc E15 và hợp đồng AS-IS](architecture.md), [Source controller và binding /predict](../../../app/gradio_app.py), [Kiểm thử hợp đồng Gradio](../../../tests/test_gradio_app.py)

## 10. Các mục cần pilot và trưởng nhóm quyết định

| Trạng thái | Hạng mục |
| --- | --- |
| **PENDING PILOT** | Model/processor revisions khả thi, scoring và prompt token limits theo encoder, device/dtype/quantization, bộ nhớ và latency trên máy nhóm. |
| **PENDING PILOT** | Model trợ lý, single-pass/multi-pass, chọn CNN context, schema, generation/context limits, phân bổ cache và cold/warm benchmark profile. |
| **PENDING PILOT** | Số warm-up/measured runs, concurrency, queue limits, timeout, OOM recovery và nhu cầu tách worker hoặc environment. |
| **PENDING LEADER REVIEW / PROTOCOL LOCK** | Quyền dùng development subset, quyền đọc predictions Validation hoặc replay nếu cần, manifest và quy mô VLM-Eval, audit policy, KB/prompt/rubric revisions, chấm độc lập và quyền truy cập artifacts. |
| **PENDING LEADER REVIEW / PROTOCOL LOCK** | Xác nhận endpoints/comparisons, Holm family, bootstrap settings, cách tính mẫu số khi thiếu output, acceptance thresholds và model selection rule trước formal. |
| **PENDING FORMAL ACCEPTANCE** | Lựa chọn encoder/profile triển khai sau khi có số liệu đã nghiệm thu. |

## 11. Tham chiếu triển khai

| Đường dẫn | Phần đối chiếu |
| --- | --- |
| [src/inference/predictor.py](../../../src/inference/predictor.py) | `verify_checkpoint_hash()`, `load_e4_model()`, `validate_raw_predictions()` và `load_raw_predictions()` xác minh checkpoint và schema logits, labels, image IDs. |
| [src/evaluation/metrics.py](../../../src/evaluation/metrics.py) | `classification_metrics()` định nghĩa Top-1, Top-5, Macro-F1 CNN. |
| [src/calibration/calibration_metrics.py](../../../src/calibration/calibration_metrics.py), [cấu hình E6](../../../configs/experiments/e6_calibration.yaml) | `compute_calibration_metrics()` và Calibration scope của E6. |
| [src/selective/threshold.py](../../../src/selective/threshold.py), [src/selective/risk_coverage.py](../../../src/selective/risk_coverage.py), [cấu hình E7](../../../configs/experiments/e7_selective_prediction.yaml) | `calibrated_predictions()`, `accept_mask()`, `selective_metrics()` và `compute_aurc()`. Coverage là accepted/total. Selective accuracy và risk tính trên tập accepted. |
| [src/ood/energy_detector.py](../../../src/ood/energy_detector.py), [app/inference_service.py](../../../app/inference_service.py) | `energy_score()`, `is_ood()`, `InferenceService.predict_image()` và `explain()`. Điều kiện reject, contract kết quả và Grad-CAM của baseline. |
| [Dataset Protocol](../../protocol/dataset_protocol.md), [Evaluation Protocol](../../protocol/evaluation_protocol.md), [Configuration Lock](../../final_evaluation/configuration_lock.md) | Development split, metric baseline và các ràng buộc của cấu hình đã khóa. |
