# DogBreedID V2 — Interface Contract

**APPROVED FOR IMPLEMENTATION PLANNING — PILOT AND PROFILE LOCK PENDING**

**Task 08.01 · Phase 2.4 · Hybrid CNN–VLM with Knowledge-Augmented Visual Understanding**

Phê duyệt áp dụng cho thiết kế và lập kế hoạch triển khai. Cấu hình pilot, profile và phạm vi formal vẫn chờ các gate tương ứng. Các hợp đồng dưới đây dùng baseline `v1.0.0-cnn-stable` tại `b7d880506b96168b0e335f3a455be3a7b2bdf77e`. Chữ ký Python, kiểu, đường dẫn và case kiểm thử mới đều là proposed. Mọi JSON trong tài liệu là **illustrative fixture**, dùng tên canonical thật khi cần đối chiếu. Revision, hash, nội dung tri thức, điểm số, latency và dữ liệu chấm trong fixture chỉ là dữ liệu minh họa cho hợp đồng. Với ví dụ dùng hash hoặc revision lặp ký tự, test harness cung cấp snapshot giả tương ứng cho kiểm tra shape. Ca kiểm hash bytes thực phải tự tính hash của fixture được dùng. [Architecture](architecture.md), [Research Protocol](research_protocol.md), [Experiment Registry](experiment_registry.md)

## 1. Baseline đã đối chiếu

| Bề mặt | Hợp đồng hiện có |
| --- | --- |
| Ảnh và canonical mapping | `read_image()` trả `(PIL RGB image, digest)` sau khi kiểm định dạng JPEG/PNG/WebP, 10 MiB và 25 triệu pixel. Digest SHA-256 lấy trên bytes upload trước decode. `_artifacts()` đọc object có khóa chuỗi `"0"` đến `"29"` và xếp tên theo ID số. [inference_service.py](../../../app/inference_service.py), [class_mapping.json](../../../data/metadata/class_mapping.json) |
| CNN | `InferenceService.predict_image()` trả `status`, `accepted`, `latency_ms`, `image_digest`, `mode`, `ood`. `class_id` được ép `int`. `ACCEPT` và `E7_REJECT` có `class_id`, `name`, `confidence` và `top5` keyed by canonical name. `OOD_REJECT` lược bốn trường phân loại. Energy OOD có ưu tiên cao hơn E7. `explain()` kiểm mode và digest, từ chối OOD. [inference_service.py](../../../app/inference_service.py) |
| Gradio `/predict` | `build_app()` gắn `DemoController.predict()` cho file ảnh và mode tùy chọn `standard` mặc định hoặc `fit`. Bảy public outputs theo thứ tự gồm status Markdown, giống hiển thị, confidence đã hiệu chỉnh dạng text, latency dạng text, Top-5 `gr.Label`, ảnh Grad-CAM và chú thích Grad-CAM. Hai output Grad-CAM rỗng sau predict. Output callback thứ tám là `gr.State` nội bộ. Khi OOD, UI không hiển thị giống, confidence, Top-5 hoặc Grad-CAM. [gradio_app.py](../../../app/gradio_app.py), [test_gradio_app.py](../../../tests/test_gradio_app.py) |
| Phiên | `DemoController._revision()` đọc `request.session_hash`. Predict, Clear và selection có thể đổi revision. Cùng path/mode cần xác minh digest. Controller dùng `gr.skip()` cho kết quả lỗi thời và kiểm revision trước Grad-CAM. [gradio_app.py](../../../app/gradio_app.py) |
| Evidence CNN | `validate_raw_predictions()` kiểm logits `[N,30]` hữu hạn, labels nguyên trong `0–29` và `image_ids` không rỗng, duy nhất. `load_raw_predictions()` đọc đúng ba trường NPZ `logits`, `labels`, `image_ids`. `load_e4_model()` xác minh checkpoint đã khóa. `classification_metrics()` và tests baseline là tham chiếu cho Top-1/Top-5 và Macro-F1. [predictor.py](../../../src/inference/predictor.py), [metrics.py](../../../src/evaluation/metrics.py), [test_metrics.py](../../../tests/test_metrics.py) |

## 2. Quy ước dùng chung

| Quy ước | Đặc tả proposed |
| --- | --- |
| Envelope và phiên bản | Mỗi object gốc của artifact, dòng evidence hoặc payload qua boundary có `schema_version: str`, dạng `major.minor`. `"1.0"` là version khởi đầu đề xuất cho implementation, formal vẫn cần lock theo Research Protocol. Object con dùng version của envelope. Tên schema được xác định bởi artifact/operation và `record_type` với evidence đa loại. |
| Required và nullability | **R** luôn phải có key. **O** cho phép bỏ key. Chỉ dùng `null` ở trường ghi rõ nullable. Giữ phân biệt mảng rỗng, chuỗi rỗng, missing và null. JSON chỉ nhận keys đã khai báo cho version đang đọc, từ chối key trùng khi parse. |
| Canonical ID | `ClassId` là integer JSON `0–29`, không nhận bool, float hoặc chuỗi. Chỉ map keyed by ID mới dùng object keys `"0"`…`"29"` không có zero padding. `canonical_name` khớp chính xác chuỗi trong `class_mapping.json`. Normalize alias chỉ trong retrieval. |
| Dữ liệu cơ bản | UTF-8, không BOM. `str` không rỗng nếu không ghi riêng. Number phải hữu hạn, loại `NaN`/`Infinity`. `*_ms` là số không âm, millisecond theo monotonic clock cho duration, mô tả ranh giới đo. Giới hạn kích thước payload, text, queue và thời gian chờ theo profile đã duyệt. |
| Hash | `Sha256` là 64 ký tự hex thường. `image_digest` là SHA-256 bytes ảnh gốc chưa decode hoặc resize, trước khi tạo RGB. `mapping_sha256`, `kb_sha256`, `prompt_bank_sha256`, `config_sha256`, `manifest_sha256` là SHA-256 bytes của artifact tương ứng. `context_sha256` hash UTF-8 của chính context đã truyền, kể cả chuỗi rỗng. Hash nằm ở tham chiếu hoặc downstream record, tránh hash tự tham chiếu. |
| Model và profile | `ModelRef` gồm `model_id: str`, `model_revision: str` và `processor_revision: str`. Với repository model/processor dùng full immutable commit SHA tương ứng, không dùng branch/tag động. `profile_id: str` tra profile bất biến gồm ModelRef, tokenizer/chat template khi áp dụng, config hash, device/dtype, giới hạn, scoring hoặc generation và approval reference đúng phạm vi. Server chọn profile từ allowlist. |
| Experiment và run | `EvalRef` gồm `experiment_id: str` và `run_id: str` theo Registry `<experiment_id>_<phase>_<YYYYMMDDTHHMMSSZ>_<uuidv4>`. Hai trường xuất hiện cùng nhau trong offline evidence. Runtime payload bỏ cả hai. Profile ID là định danh cấu hình có thể liên kết nhiều run. |
| Compatibility | Thêm trường optional chỉ ở minor mới, giữ bộ đọc cũ bằng downgrade/negotiation có kiểm chứng hoặc migration đã duyệt. Required field, type, enum, semantic hoặc breaking change cần leader review và major version mới. Từ chối version chưa hỗ trợ trước xử lý. Hợp đồng public `/predict` giữ nguyên hiện trạng. |

`ValidatedRgb` là kiểu Python nội bộ do server tạo sau xác thực. Chứa buffer RGB chỉ đọc, kích thước và digest lấy từ cùng lần đọc ảnh. JSON fixture dùng metadata, không truyền path hoặc pixel của ảnh thật. Nếu pilot chọn process riêng, worker nhận buffer/bản sao có vòng đời do server quản lý và metadata ràng buộc, giữ digest của bytes upload thay vì hash lại buffer RGB. UI, raw model payload và model text đều là input cần kiểm tra.

**Lỗi và namespace trạng thái.** `ErrorInfo` gồm `code: ErrorCode`, `message_vi: str` đã xử lý, và `retryable: bool`. Các code đề xuất gồm `INVALID_INPUT`, `UNSUPPORTED_SCHEMA`, `HASH_MISMATCH`, `MAPPING_MISMATCH`, `INVALID_SNAPSHOT`, `UNREVIEWED_SOURCE`, `TOKEN_TRUNCATION`, `BUDGET_EXCEEDED`, `MODEL_UNAVAILABLE`, `POLICY_BLOCKED`, `TIMEOUT`, `RESOURCE_EXHAUSTED`, `INVALID_OUTPUT`, `RETRIEVAL_ERROR` và `INTERNAL_ERROR`. Loader/request validator có thể trả structured error trước dispatch. Adapter result dùng terminal status chung ở Contract 05, kèm error khi thất bại và không điền dữ liệu dự đoán thay thế. `stale` chỉ dùng nội bộ. Research lifecycle, run execution status, policy evidence và retrieval status có enum riêng.

## 3. Contract 01 — Breed Knowledge Base và Prompt Banks

**Owner** Thuận. **Consumers** Đức Anh, Kiệt và E15. Input là JSON snapshot và canonical mapping đã pin, đầu ra là snapshot chỉ đọc đã xác thực, có index theo ClassId và alias. Dùng một schema cho mọi consumer của cùng artifact.

| Artifact | Proposed path | Trường và constraint |
| --- | --- | --- |
| `BreedKnowledgeBase` | `data/knowledge/vlm/breed_kb.json` | R `schema_version`, `kb_version: str`, `mapping_sha256: Sha256`, `breeds: BreedRecord[30]`, `sources: SourceRecord[]`. Breeds theo ID tăng dần, đúng một record mỗi ID. |
| `BreedRecord` | Nested trong KB | R `class_id: ClassId`, `canonical_name: str`, `aliases: {vi: str[], en: str[]}`, `attributes: MorphAttribute[]`. Các alias trim, không rỗng, không trùng sau normalize trong cùng danh sách. Alias trùng giữa nhiều lớp được giữ để Contract 02 báo ambiguous. |
| `MorphAttribute` | Nested trong breed | R `attribute_id: str` theo vocabulary hình thái được duyệt, `value: str`, `source_ids: str[]` có ít nhất một nguồn. O `variability_note: str`, `uncertainty_note: str`. Mỗi attribute ID duy nhất trong breed. |
| `SourceRecord` | Nested trong KB | R `source_id: str` duy nhất, `title: str`, `url: str` HTTP(S), `review_status: enum(pending, approved, rejected)`. O `review_ref: str`. Snapshot được duyệt chỉ sử dụng nguồn `approved` có evidence review, tới cấp từng attribute. URL là provenance, runtime lookup chỉ dùng dữ liệu local. |
| `PromptBank` | `data/knowledge/vlm/name_prompt_bank.json` và `data/knowledge/vlm/morphology_prompt_bank.json` | R `schema_version`, `bank_version: str`, `prompt_type: enum(name, morphology)`, `mapping_sha256`, `entries: PromptEntry[30]` theo ID tăng dần. R riêng morphology `kb_version` và `kb_sha256`. Hai trường KB bỏ khỏi name bank. |
| `PromptEntry` | Nested trong bank | R `class_id`, `canonical_name`, `prompt_text: str` có nội dung. Với morphology, R `attribute_ids: str[]` và `source_ids: str[]`, đều không rỗng, tham chiếu cùng breed trong KB đã pin. Name entry chỉ chứa ba trường chung. |
| `PromptTokenAudit` | `data/knowledge/vlm/prompt_tokenization_audit.json` | R `schema_version` và `audits: TokenAudit[]`. Mỗi audit ràng buộc `prompt_bank_sha256`, `profile_id`, `tokenizer_revision`, `token_limit: int` dương, `entries[30]` có `class_id`, `token_count: int` dương và `truncated: bool`. |

**Vocabulary KB `kb-morphology-v1`.** Trưởng nhóm sở hữu allowlist. Caller truyền `src.vlm.knowledge_base.KB_MORPHOLOGY_V1` vào `allowed_attribute_ids` và ghi `KB_MORPHOLOGY_VERSION` trong cấu hình bàn giao. Tập gồm 8 ID Visual Observation hiện tại và 4 ID dành riêng cho KB là `breed_size`, `head_shape`, `facial_hair` và `coat_fullness`. Contract 04 tiếp tục dùng vocabulary Visual Observation v1 gồm 8 ID. Với `breed_size`, ghi rõ hệ chuẩn và yêu cầu tỷ lệ hoặc số đo trong `value` để điều kiện đến được context. Schema KB và các trường wire hiện tại giữ nguyên.

Kiểm file hash trước khi dùng, xác nhận đủ tập ID, tên exact, source refs tồn tại, nguồn reviewed, mô tả morphology giữ tên giống và token audit khớp bank/profile. Bất kỳ prompt rỗng, sai type, duplicate/missing ID, hash mismatch, source thiếu hoặc `truncated=true` đều từ chối snapshot cho profile đó. Lỗi trả `INVALID_SNAPSHOT`, `MAPPING_MISMATCH`, `HASH_MISMATCH`, `UNREVIEWED_SOURCE` hoặc `TOKEN_TRUNCATION`. Runtime KB/prompt schema cấm trường label, rubric, expected answers và eval annotations.

**Illustrative fixture hợp lệ.** Ví dụ record con `BreedRecord` và `PromptEntry` morphology, trong snapshot giả đủ 30 lớp. Nguồn `src-fixture` là source giả đã cấp trạng thái approved trong fixture, chỉ dùng test schema. Tên và nguồn thuộc tính thật cần review riêng.

```json
{
  "breed_record": {
    "class_id": 7,
    "canonical_name": "Miniature Poodle",
    "aliases": {"vi": ["poodle nhỏ"], "en": ["mini poodle"]},
    "attributes": [
      {"attribute_id": "coat_texture", "value": "xoăn", "source_ids": ["src-fixture"], "variability_note": "Giá trị minh họa"}
    ]
  },
  "source_record": {
    "source_id": "src-fixture",
    "title": "Fixture source",
    "url": "https://example.invalid/fixture",
    "review_status": "approved",
    "review_ref": "review-fixture"
  },
  "morphology_prompt_entry": {
    "class_id": 7,
    "canonical_name": "Miniature Poodle",
    "prompt_text": "A photo of a Miniature Poodle with a curly coat",
    "attribute_ids": ["coat_texture"],
    "source_ids": ["src-fixture"]
  }
}
```

**Illustrative fixture hợp lệ cho snapshot đầy đủ.** Name bank gồm đúng 30 IDs. Bank và KB thật phải có provenance cùng hash đúng bytes trước khi được chấp nhận.

```json
{
  "schema_version": "1.0",
  "bank_version": "fixture-1",
  "prompt_type": "name",
  "mapping_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "entries": [
    {"class_id":0,"canonical_name":"Eskimo Dog","prompt_text":"A photo of a Eskimo Dog"},
    {"class_id":1,"canonical_name":"Malamute","prompt_text":"A photo of a Malamute"},
    {"class_id":2,"canonical_name":"Siberian Husky","prompt_text":"A photo of a Siberian Husky"},
    {"class_id":3,"canonical_name":"Greater Swiss Mountain Dog","prompt_text":"A photo of a Greater Swiss Mountain Dog"},
    {"class_id":4,"canonical_name":"Appenzeller","prompt_text":"A photo of a Appenzeller"},
    {"class_id":5,"canonical_name":"EntleBucher","prompt_text":"A photo of a EntleBucher"},
    {"class_id":6,"canonical_name":"Toy Poodle","prompt_text":"A photo of a Toy Poodle"},
    {"class_id":7,"canonical_name":"Miniature Poodle","prompt_text":"A photo of a Miniature Poodle"},
    {"class_id":8,"canonical_name":"Standard Poodle","prompt_text":"A photo of a Standard Poodle"},
    {"class_id":9,"canonical_name":"Miniature Schnauzer","prompt_text":"A photo of a Miniature Schnauzer"},
    {"class_id":10,"canonical_name":"Standard Schnauzer","prompt_text":"A photo of a Standard Schnauzer"},
    {"class_id":11,"canonical_name":"Giant Schnauzer","prompt_text":"A photo of a Giant Schnauzer"},
    {"class_id":12,"canonical_name":"English Setter","prompt_text":"A photo of a English Setter"},
    {"class_id":13,"canonical_name":"Irish Setter","prompt_text":"A photo of a Irish Setter"},
    {"class_id":14,"canonical_name":"Gordon Setter","prompt_text":"A photo of a Gordon Setter"},
    {"class_id":15,"canonical_name":"English Springer","prompt_text":"A photo of a English Springer"},
    {"class_id":16,"canonical_name":"Welsh Springer Spaniel","prompt_text":"A photo of a Welsh Springer Spaniel"},
    {"class_id":17,"canonical_name":"Cocker Spaniel","prompt_text":"A photo of a Cocker Spaniel"},
    {"class_id":18,"canonical_name":"Norfolk Terrier","prompt_text":"A photo of a Norfolk Terrier"},
    {"class_id":19,"canonical_name":"Norwich Terrier","prompt_text":"A photo of a Norwich Terrier"},
    {"class_id":20,"canonical_name":"Border Terrier","prompt_text":"A photo of a Border Terrier"},
    {"class_id":21,"canonical_name":"Collie","prompt_text":"A photo of a Collie"},
    {"class_id":22,"canonical_name":"Shetland Sheepdog","prompt_text":"A photo of a Shetland Sheepdog"},
    {"class_id":23,"canonical_name":"Border Collie","prompt_text":"A photo of a Border Collie"},
    {"class_id":24,"canonical_name":"Golden Retriever","prompt_text":"A photo of a Golden Retriever"},
    {"class_id":25,"canonical_name":"Labrador Retriever","prompt_text":"A photo of a Labrador Retriever"},
    {"class_id":26,"canonical_name":"Chesapeake Bay Retriever","prompt_text":"A photo of a Chesapeake Bay Retriever"},
    {"class_id":27,"canonical_name":"Japanese Spaniel","prompt_text":"A photo of a Japanese Spaniel"},
    {"class_id":28,"canonical_name":"Pekinese","prompt_text":"A photo of a Pekinese"},
    {"class_id":29,"canonical_name":"Shih-Tzu","prompt_text":"A photo of a Shih-Tzu"}
  ]
}
```

**Không hợp lệ.** `{"class_id":"7"}` trong record, ID lặp hoặc thiếu, tên `"Mini Poodle"` ở `canonical_name`, reference nguồn chưa approved, prompt whitespace-only, audit truncation hoặc bank không khớp mapping hash.

**Proposed Python**

```python
def load_knowledge_base(raw: bytes, *, mapping: CanonicalMapping, expected_sha256: Sha256,
                        reviewed_sources: Mapping[str, str],
                        allowed_attribute_ids: Collection[str]) -> KnowledgeSnapshot: ...
def load_prompt_bank(raw: bytes, *, mapping: CanonicalMapping, expected_sha256: Sha256,
                     knowledge: KnowledgeSnapshot | None) -> PromptSnapshot: ...
def validate_prompt_token_audit(audit: PromptTokenAudit, *, bank: PromptSnapshot, profile: ApprovedProfile) -> None: ...
```

**Contract tests đề xuất.** Thuận cung cấp fixture sinh đủ 30 IDs và source links. Kiểm name/morphology hợp lệ, Unicode alias, duplicate/missing ID, case mismatch, thiếu source, wrong hash và prompt truncate. Đức Anh kiểm consumer từ chối profile/bank sai binding. Kiệt và E15 kiểm cùng snapshot version dùng cho retrieval.

## 4. Contract 02 — Knowledge Retrieval

**Owner** Thuận. **Consumers** Kiệt và E15. Input là `RetrievalRequest` và KB snapshot đã pin. Output `RetrievalResult` chỉ chứa context/metadata cần cho assistant. Dùng input contract của pipeline E14/E15, bỏ trường ground truth.

| Record | Trường và constraint |
| --- | --- |
| `RetrievalRequest` | R `schema_version`, `question: str`, `explicit_class_ids: ClassId[]`, `cnn_context: AllowedCNNContext hoặc null`, `kb_version: str`, `kb_sha256: Sha256`, `context_budget: ContextBudget`, `lookup_applicable: bool`. Explicit IDs do caller có thẩm quyền xác nhận, có thể rỗng. CNN context chỉ có class ID/name khi `ACCEPT` theo Contract 04 và Contract 05. |
| `ContextBudget` | R `max_tokens: int` không âm, `tokenizer_revision: str` đã pin. Đếm token của slot KB bằng tokenizer của profile, tính cả dấu phân cách và source refs đưa vào slot. Caller dựng template rồi xác minh lại ngân sách đầy đủ sau tokenization. |
| `RetrievalResult` | R `schema_version`, `status: RetrievalStatus`, `per_breed: BreedLookup[]`, `resolved_class_ids: ClassId[]` theo thứ tự lookup, `entries: RetrievedEntry[]`, `context: str` có thể rỗng, `source_ids: str[]`, `kb_version`, `kb_sha256`, `truncation: TruncationInfo`, `latency_ms: number`. O `error: ErrorInfo` chỉ với `retrieval_error`. |
| `BreedLookup` | R `mention: str hoặc null`, `class_id: ClassId hoặc null`, `status: enum(hit, miss, ambiguous)`. R `candidate_class_ids: ClassId[]`, chỉ có dữ liệu ở ambiguous. O `reason: str` như `no_alias_match` hoặc `budget_exhausted`. |
| `RetrievedEntry` | R `class_id`, `canonical_name`, `attributes: MorphAttribute[]` đã giữ theo budget. Sources đủ và đúng theo entries đã dùng trong context. |
| `TruncationInfo` | R `applied: bool`, `budget_tokens: int`, `used_tokens: int`, `omitted_attribute_count: int` không âm. `used_tokens <= budget_tokens`. |

1. Với policy không cho truy xuất hoặc câu hỏi không cần tra giống, trả `not_applicable`. Với lookup, ưu tiên explicit IDs, kiểm ID và loại trùng theo thứ tự caller. Normalize câu hỏi và alias bằng NFC, casefold, NFC lại, trim và gộp khoảng trắng, giữ dấu tiếng Việt. Match cụm từ trọn vẹn theo vị trí trái sang phải, chọn cụm dài nhất tại vị trí chồng lấn.
2. Thêm các giống từ câu hỏi theo thứ tự xuất hiện, gộp những mention cùng ID. Alias map tới nhiều ID thì giữ `ambiguous` và candidates theo ID tăng dần. Nếu câu hỏi cần tra và không có đối tượng xác định, ghi `miss`. Chỉ dùng ID CNN hợp lệ làm fallback khi câu hỏi về ảnh không nêu giống và policy cho phép. Alias collision hoặc đối tượng được hỏi nhưng không khớp không được thay bằng ID CNN. Query so sánh có thể có nhiều mục, giữ mục thiếu căn cứ khi không xác định hết.
3. Build context theo thứ tự giống và thứ tự attribute của KB, đưa nguồn cùng claim. Cắt theo đơn vị attribute, giữ nguyên source ref, ghi phần bỏ. `hit` nghĩa là có ít nhất một attribute của breed trong context. Breed đã giải được ID nhưng budget không chứa attribute có `miss` kèm `budget_exhausted`. Tất cả mục hit cho `hit`, có hit và có mục không hit cho `partial_hit`, không có hit nhưng có collision cho `ambiguous`, còn lại `miss`. Snapshot/lookup lỗi trả `retrieval_error` kèm error. `resolved_class_ids` giữ các IDs đã giải được, kể cả ID bị bỏ hết do budget. Các trạng thái không có hit trả entries, source_ids và context rỗng. `not_applicable` và `retrieval_error` có per_breed và resolved_class_ids rỗng.
4. Kiệt nhận `context` nguyên văn, `source_ids` là tập nguồn được phép, cùng provenance KB và status. Kiểm mapping, policy và snapshot binding ở biên caller. Không dùng vector DB hoặc LLM truy xuất riêng. `RetrievalStatus` gồm `hit`, `partial_hit`, `miss`, `ambiguous`, `not_applicable` và `retrieval_error`.

**Illustrative fixture hợp lệ.** Output `partial_hit` với một mục có tri thức và một alias mơ hồ.

```json
{
  "schema_version": "1.0",
  "status": "partial_hit",
  "per_breed": [
    {"mention": "mini poodle", "class_id": 7, "status": "hit", "candidate_class_ids": []},
    {"mention": "poodle", "class_id": null, "status": "ambiguous", "candidate_class_ids": [6, 7, 8]}
  ],
  "resolved_class_ids": [7],
  "entries": [
    {"class_id": 7, "canonical_name": "Miniature Poodle", "attributes": [
      {"attribute_id": "coat_texture", "value": "xoăn", "source_ids": ["src-fixture"]}
    ]}
  ],
  "context": "Miniature Poodle: lông xoăn [src-fixture]",
  "source_ids": ["src-fixture"],
  "kb_version": "fixture-1",
  "kb_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "truncation": {"applied": false, "budget_tokens": 64, "used_tokens": 12, "omitted_attribute_count": 0},
  "latency_ms": 0
}
```

**Không hợp lệ.** Chọn ID 7 từ collision mà không có evidence, `hit` khi context rỗng, nguồn không có trong retrieved entries, budget âm hoặc đầu vào có `expected_answer`. Request sai trả `INVALID_INPUT` trước lookup, sai snapshot/hash trả `retrieval_error` với code tương ứng.

**Proposed Python**

```python
def retrieve_knowledge(request: RetrievalRequest, *, knowledge: KnowledgeSnapshot,
                       count_context_tokens: ContextTokenCounter) -> RetrievalResult: ...
```

**Contract tests đề xuất.** Thuận kiểm Unicode Việt tổ hợp, longest-match, cạnh từ, multi-breed order, ambiguity, miss, not_applicable, rút budget và snapshot mismatch. Kiệt kiểm assistant tiêu thụ context và chỉ dùng đúng source IDs. E15 kiểm lookup bị policy chặn.

## 5. Contract 03 — Zero-shot VLM Classifier

**Owner** Đức Anh. **Consumers** E15 và offline evaluator. Một adapter interface dùng chung cho SigLIP 2 và FG-CLIP 2. Preprocessing, processor/tokenizer và score transform do profile đã duyệt định nghĩa.

| Record | Trường và constraint |
| --- | --- |
| `ZeroShotRequest` | R `schema_version`, `image: ValidatedRgb`, `image_digest: Sha256`, `profile_id: str`, `prompt_bank_sha256: Sha256` và `mapping_sha256: Sha256`. Offline thêm `eval_ref: EvalRef`. `image_digest` phải khớp metadata từ server. `ApprovedProfile` và `PromptSnapshot` được resolver nội bộ đưa vào adapter, không load từ path của client. |
| `ZeroShotResult` | R `schema_version`, `status: BranchOutcome`, `image_digest`, `profile_id`, `model: ModelRef`, `mapping_sha256`, `prompt_bank_sha256`, `calibrated: bool` luôn `false`, `latency_ms: number hoặc null`. R riêng `ok` gồm `score_type`, `scores: number[30]`, `top1_class_id: ClassId`, `top5_class_ids: ClassId[5]`. Outcome lỗi R `error: ErrorInfo`. Offline echo `eval_ref`. |
| `score_type` | `cosine_similarity`, `scaled_similarity`, `similarity` hoặc `logit`. Loại thực tế, native score direction, phép scaling và aggregation nằm trong profile/provenance. Type và công thức còn chờ pilot. Mỗi cặp A/B dùng cùng phương pháp đã khóa. |

Chỉ nhận đúng 30 số hữu hạn theo ID `0–29`. Chuẩn hóa quy ước score cao hơn xếp trước theo phép đã khóa, xếp giảm dần theo score và phá hòa bằng ID tăng dần. Top-1 là phần tử đầu của Top-5, Top-5 có năm IDs duy nhất và tái lập được từ vector. Score phụ thuộc profile, không đưa vào confidence E6 hoặc dùng ghi đè status CNN.

Runtime trả payload nội bộ để E15 kiểm và render projection theo policy, có thể chỉ hiển thị Top-5/loại điểm. Offline evidence lưu 30 scores, ranking, hashes, statuses và ID ảnh từ manifest, có run binding riêng. Khi lỗi, score_type, scores, Top-1 và Top-5 vắng mặt. Adapter báo `invalid_output`, `unavailable`, `timeout` hoặc `resource_exhausted` và error phù hợp.

**Illustrative fixture hợp lệ.** `cosine_similarity` giả với điểm bằng nhau tại IDs 1 và 2, ưu tiên ID 1.

```json
{
  "schema_version": "1.0",
  "status": "ok",
  "image_digest": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "profile_id": "fixture-encoder-1",
  "model": {
    "model_id": "google/siglip2-base-patch16-224",
    "model_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "processor_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "mapping_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "prompt_bank_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
  "score_type": "cosine_similarity",
  "scores": [0.1, 0.7, 0.7, 0.4, 0.3, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "top1_class_id": 1,
  "top5_class_ids": [1, 2, 3, 4, 5],
  "calibrated": false,
  "latency_ms": 0
}
```

**Không hợp lệ.** Vector 29 scores hoặc có NaN/Inf, `top5_class_ids: [2, 1, 3, 4, 5]` với fixture trên, ID `"1"`, `calibrated: true` hoặc `status: "unavailable"` kèm ranking.

**Proposed Python**

```python
class ZeroShotClassifier(Protocol):
    def classify(self, request: ZeroShotRequest, *, profile: ApprovedProfile,
                 prompts: PromptSnapshot) -> ZeroShotResult: ...
```

**Contract tests đề xuất.** Đức Anh chạy cùng fixture cho hai adapter, test shape, tie, nonfinite score, model/profile/hash mismatch, xử lý lỗi và offline/runtime fields. Trưởng nhóm test không cập nhật CNN decision hoặc công bố ranking trái policy.

## 6. Contract 04 — Generative VLM Assistant

**Owner** Kiệt. **Consumers** E15 và E14 evaluator. Structured visual attribute extraction và VQA là hai operation logic. Chọn single-pass hoặc multi-pass sau pilot. E14-A/B dùng cùng model và pipeline, shared inputs và validation policy đã khóa.

**Offline và runtime.** E14 đo nội dung do model sinh. Đề xuất pilot với `cnn_context=null` cho cả A và B để giảm yếu tố gây nhiễu. E14-B tra breed mentions trong câu hỏi hợp lệ hoặc nguồn context do protocol phê duyệt. Ground-truth labels và expected answers chỉ đi vào scorer. Trưởng nhóm khóa phạm vi offline, cách ghi mẫu `policy_blocked` và lựa chọn CNN context cuối cùng trước formal. E15 đo hành vi hệ thống sau policy, validator và fallback. Runtime E15 giữ chính sách CNN đã duyệt ở Contract 05.

| Input | Trường và constraint |
| --- | --- |
| `AssistantRequest` | R `schema_version`, `operation: enum(attributes, vqa, attributes_and_vqa)`, `image: ValidatedRgb`, `image_digest: Sha256`, `question: str hoặc null`, `cnn_context: AllowedCNNContext hoặc null`, `intermediate_attributes: AttributeObservation[] hoặc null`, `kb_context: str` có thể rỗng, `allowed_source_ids: str[]`, `context_provenance: ContextProvenance`, `profile_id: str`, `policy_mode: PolicyMode`. Offline thêm `eval_ref: EvalRef`. |
| `question` và dữ liệu chung | `question` bắt buộc có nội dung với `vqa` và `attributes_and_vqa`, là null cho `attributes`. Limit theo profile. Nếu dùng thuộc tính trung gian trong E14, tạo trước khi chèn KB và chia sẻ bản ghi bất biến. Đối chiếu digest và shared tokens để chỉ slot KB thay đổi. |
| `AllowedCNNContext` | R `status: enum(ACCEPT, E7_REJECT, OOD_REJECT)`. O `class_id: ClassId` và `canonical_name: str` xuất hiện cùng nhau, chỉ khi `status=ACCEPT`, `policy_mode=breed_context_allowed` và E15 cho phép. Mode, session và confidence thô không cần chuyển cho assistant. Với CNN chưa chạy hoặc lỗi dùng null. |
| `PolicyMode` | `breed_context_allowed` cho phạm vi breed context đã duyệt. `observation_only` dùng tập thuộc tính kiểm soát và không có breed context. Runtime policy do E15 tạo. Luồng offline dùng phạm vi đã được lock. |
| `ContextProvenance` | R `retrieval_status: RetrievalStatus`, `kb_version: str hoặc null`, `kb_sha256: Sha256 hoặc null`, `context_sha256: Sha256`. Version và hash cùng có hoặc cùng null. E14-A và lượt không dùng KB có `not_applicable`, cặp null, context rỗng và sources rỗng. E14-B giữ version/hash của KB đã pin cho cả miss/ambiguous/error. |

| Output | Trường và constraint |
| --- | --- |
| `AssistantResult` | R server-confirmed `schema_version`, `status: BranchOutcome`, `image_digest`, `profile_id`, `model: ModelRef`, `context_provenance`, `latency_ms: number hoặc null`. Offline echo `eval_ref`. Với `ok`, R `generated: GeneratedContent`. Outcome lỗi R `error: ErrorInfo` và bỏ `generated`. |
| `GeneratedContent` | Model-generated `attributes: AttributeObservation[]`, `answer_vi: str hoặc null`, `disposition: enum(answered, abstained, not_applicable)`, `uncertainty: enum(low, medium, high, unspecified)`, `abstention_reason: str hoặc null`, `source_ids: str[]` duy nhất. Tất cả key R trong output model. `source_ids` phải là tập con của `allowed_source_ids`. Dùng câu trả lời plain text tiếng Việt khi có VQA. Uncertainty là mô tả định tính. |
| `AttributeObservation` | R `attribute_id: str` từ vocabulary v1 hoặc version đã khóa, `visibility: enum(observed, unknown, not_visible, not_applicable)`, `value: str, str[] hoặc null` theo attribute và visibility. Mỗi ID xuất hiện tối đa một lần trong một result hoặc annotation. |
| `disposition` | `answered` cần `answer_vi` không rỗng và `abstention_reason=null`. `abstained` cần lý do và có thể kèm text tiếng Việt trong phạm vi policy. `not_applicable` dùng cho attributes-only, answer/reason null. VQA-only trả `attributes=[]` nếu không xuất thuộc tính. |

`unknown` nghĩa là thuộc tính áp dụng nhưng chưa xác định đủ chắc. `not_visible` nghĩa là bộ phận hoặc dấu hiệu bị che hoặc ngoài khung. `not_applicable` nghĩa là thuộc tính không áp dụng cho chủ thể. Không suy ra visibility chỉ từ việc thiếu value.

**Vocabulary v1 đề xuất.** Thuận và Kiệt kiểm độ nhất quán giữa model, fixture, retrieval và rubric trong pilot. Trưởng nhóm khóa vocabulary, quy tắc diễn giải và version trước formal. Values là token tiếng Anh ổn định để validate, UI dùng nhãn tiếng Việt khi render.

| Attribute ID | Dấu hiệu quan sát | Kiểu `value` khi `observed` | Controlled values đề xuất |
| --- | --- | --- | --- |
| `ear_position` | Mức dựng hoặc rủ của tai | Một `str` | `erect`, `semi_erect`, `dropped` |
| `ear_shape` | Viền tai nhìn thấy rõ | Một `str` | `triangular`, `rounded`, `elongated` |
| `coat_color` | Một hoặc nhiều màu lông nhìn thấy | `str[]` không rỗng, không trùng | `black`, `white`, `gray`, `brown`, `tan`, `cream`, `red`, `golden`, theo thứ tự này |
| `coat_length` | Độ dài lông nhìn thấy | Một `str` | `short`, `medium`, `long` |
| `coat_texture` | Dáng sợi lông biểu hiện trên ảnh | Một `str` | `straight`, `wavy`, `curly`, `wiry` |
| `muzzle_length` | Chiều dài mõm tương đối với đầu | Một `str` | `short`, `medium`, `long` |
| `tail_shape` | Dáng đuôi nhìn thấy | Một `str` | `straight`, `curved`, `curled` |
| `body_build` | Dáng thân thể thể hiện trong khung | Một `str` | `slender`, `medium`, `stocky` |

Validator nhận enum đúng case, không ép chuỗi thành mảng hoặc ngược lại. `coat_color` là thuộc tính nhiều giá trị, kể cả một màu vẫn là mảng một phần tử. Dùng thứ tự palette để lưu nhất quán. `observed` yêu cầu value hợp lệ và có căn cứ ảnh. `unknown`, `not_visible` và `not_applicable` yêu cầu `value=null`. Dùng trạng thái phù hợp khi bằng chứng thiếu hoặc vocabulary chưa bao phủ. Cùng schema áp dụng cho intermediate attributes, output model và target attributes của scorer. `MorphAttribute` ở Contract 01 vẫn là mô tả tri thức có nguồn. Thuộc tính điển hình trong KB không được dùng làm bằng chứng thị giác. Phạm vi này không bắt buộc suy ra tuổi, cân nặng, sức khỏe hoặc thuần chủng chỉ từ ảnh. Trường metadata và status do adapter/server xác nhận từ request và validator, không lấy từ raw generation. Trong pipeline multi-pass, kiểm từng stage trước khi dùng tiếp.

E15 render `observation_only` từ thuộc tính có giá trị theo allowlist và template cố định. Text tự do hoặc payload gợi ý giống không được render trong mode này. Các câu trả lời tiếng Việt tự do ở scope cho phép vẫn kiểm schema, độ dài, policy và nguồn, tách chi tiết quan sát khỏi kiến thức giống. Output sai schema/nguồn hoặc chứa enum ngoài tập trả `invalid_output`. Controller giữ thông báo riêng. Raw model output giữ riêng theo Contract 06, chỉ lưu ở luồng evidence đã được cấp quyền.

**Illustrative fixture hợp lệ.** Runtime không dùng KB có context rỗng như E14-A. SHA dưới đây là hash của chuỗi UTF-8 rỗng. Ví dụ minh họa runtime `attributes_and_vqa` ở phạm vi cho phép hiển thị câu trả lời.

```json
{
  "schema_version": "1.0",
  "status": "ok",
  "image_digest": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "profile_id": "fixture-assistant-1",
  "model": {
    "model_id": "Qwen/Qwen3-VL-2B-Instruct",
    "model_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "processor_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "context_provenance": {
    "retrieval_status": "not_applicable",
    "kb_version": null,
    "kb_sha256": null,
    "context_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  },
  "generated": {
    "attributes": [
      {"attribute_id": "ear_shape", "visibility": "not_visible", "value": null},
      {"attribute_id": "coat_color", "visibility": "observed", "value": ["black", "white"]}
    ],
    "answer_vi": "Ảnh không cho thấy rõ tai của chủ thể.",
    "disposition": "abstained",
    "uncertainty": "high",
    "abstention_reason": "Tai ở ngoài khung ảnh.",
    "source_ids": []
  },
  "latency_ms": 0
}
```

**Không hợp lệ.** `visibility: "not_visible"` nhưng có value, `observed` với value null, `coat_color` dạng chuỗi hoặc mảng trùng màu, `ear_position` dạng mảng, raw model tự khai revision, source ID ngoài allowlist, hoặc tự tăng context budget để A/B nhận token ảnh/câu hỏi khác nhau.

**Proposed Python**

```python
class GenerativeAssistant(Protocol):
    def extract_attributes(self, request: AssistantRequest, *, profile: ApprovedProfile) -> AssistantResult: ...
    def answer_question(self, request: AssistantRequest, *, profile: ApprovedProfile) -> AssistantResult: ...
    def execute(self, request: AssistantRequest, *, profile: ApprovedProfile) -> AssistantResult: ...
```

`execute()` là entry point của adapter cho E15/evaluator. Hai thao tác logic có thể là method nội bộ hoặc cùng một lượt generate theo profile. Offline capture thu raw/validated output tại boundary mà không đổi `AssistantResult` chuyển về UI.

**Contract tests đề xuất.** Kiệt kiểm từng visibility, operation, nullability, controlled values đơn/đa trị, thứ tự màu, tiếng Việt, provenance/source cross-check, truncate và binding profile. Với E14, kiểm đối sánh cùng đầu vào chung và cùng rubric. Trưởng nhóm kiểm policy không công bố thông báo controller như model success.

## 7. Contract 05 — Integration và Session

**Owner** Trưởng nhóm. **Producers và consumers** UI gửi thao tác, controller xác thực/chọn policy và tạo envelope, adapter/worker chỉ thực thi job, controller xác thực kết quả và cập nhật UI. Offline evaluators dùng adapter qua chính sách đã khóa. E15 chỉ gắn profile qua deployment eligibility và đúng phạm vi đã được duyệt.

| Record | Trường và constraint |
| --- | --- |
| `BranchRequest` | R `schema_version`, `request_id: str` mới do server cấp, `session_id: str` opaque do server bind với phiên, `image_digest: Sha256`, `revision: int` không âm, `cnn_mode: enum(standard, fit)`, `cnn_decision: CNNDecisionSnapshot`, `branch: enum(comparison, assistant)`, `profile_id: str`, `remaining_budget_ms: number` không âm khi dispatch. Đây là record nội bộ của controller. Deadline do controller theo dõi, quyền công bố và ảnh giữ ở scope của phiên. |
| `CNNDecisionSnapshot` | R `state: enum(decided, not_run, error)`, `status: CNNStatus hoặc null`, `class_id: ClassId hoặc null`. `CNNStatus` là `ACCEPT`, `E7_REJECT`, `OOD_REJECT`. `decided` có status. `ACCEPT`/`E7_REJECT` có class_id Server để xét comparison, còn OOD có class_id null. `not_run` và `error` có hai trường null. Snapshot là projection E15, contract dict CNN baseline ở mục 1 dùng quy tắc riêng. |
| `SessionState` | R `session_id`, `revision`, `image_digest: Sha256 hoặc null`, `cnn_mode`, `cnn_decision`, `latest_request_ids: {comparison: str hoặc null, assistant: str hoặc null}`, `selected_profile_ids: {comparison: str hoặc null, assistant: str hoặc null}`, `branch_states` keyed by hai branch với giá trị `BranchState`. Server là writer duy nhất. |
| `WorkerJob` | R `schema_version`, `request_id`, `session_id`, `image_digest`, `revision`, `cnn_mode`, `branch`, `profile_id`, `remaining_budget_ms` và `payload: ZeroShotRequest hoặc AssistantRequest`. Payload chỉ chứa projection từ Contract 03/04, không có CNN snapshot đầy đủ. Profile và snapshot được resolve qua dependency nội bộ đã duyệt. |
| `WorkerReply` | R `schema_version`, `request_id`, `session_id`, `image_digest`, `revision`, `cnn_mode`, `branch`, `profile_id`, `result: ZeroShotResult hoặc AssistantResult` khớp branch. Worker chỉ echo tuple identity do server gửi. Controller kiểm tất cả các identity và ràng buộc digest/profile của result với envelope. |
| `BranchState` | R `status: RuntimeStatus`, `request_id: str hoặc null`. O `error: ErrorInfo`, `controller_response_vi: str`, `model_result` đã validate và giới hạn scope. Chỉ `ok` có `model_result`. Controller có thể hiển thị phản hồi cố định khi policy hoặc fallback yêu cầu. |

**RuntimeStatus** gồm `disabled`, `pending`, `running`, `ok`, `busy`, `unavailable`, `timeout`, `invalid_output`, `resource_exhausted`. `BranchOutcome` của adapter chỉ gồm `ok`, `unavailable`, `timeout`, `invalid_output`, `resource_exhausted`. Với invalid input/policy block, controller chặn dispatch, trả trạng thái phù hợp như `disabled` và lưu lý do. `busy` xuất phát từ admission. `stale` là disposition nội bộ của controller, không là status công bố. Khi hết deadline, sử dụng `timeout` và loại reply đến muộn. Enum nghiên cứu và run được tham chiếu ở Contract 06.

| CNN state/status | Comparison | Assistant |
| --- | --- | --- |
| `ACCEPT` | Cho phép khi opt-in/profile khả dụng. Agreement hoặc disagreement theo ID, giữ quyền quyết định của E4. | VQA theo policy và có thể nhận breed context đã lọc. |
| `E7_REJECT` | Ranking tham khảo có cảnh báo. | Mô tả ảnh và bất định. Chặn suy diễn giống từ ranking hay câu hỏi, không nhận ID CNN. |
| `OOD_REJECT` | Chặn công bố breed ranking. | `observation_only`, bỏ breed context và yêu cầu xác định giống. |
| `not_run` hoặc `error` | Yêu cầu một snapshot CNN hợp lệ. | Với ảnh đã xác thực, chỉ quan sát chung, thông báo trạng thái CNN riêng. |
| Ảnh không hợp lệ | Không dispatch. | Không dispatch. |

- Đổi ảnh, bytes ảnh, cnn_mode, Clear hoặc kết quả CNN mới làm mất hiệu lực công việc phụ thuộc. Câu hỏi mới thay request Assistant trong cùng revision. Server đối chiếu request ID mới nhất của đúng branch, session, digest, revision, cnn_mode, profile, liveness và deadline trước validate và công bố. Việc so và cập nhật diễn ra dưới cùng cơ chế nhất quán theo phiên. Hủy khi khả thi, loại kết quả đã stale dù worker vẫn hoàn thành. Worker không ghi session hoặc CNN status.
- Client chỉ gửi thao tác, file và câu hỏi theo interface. Không tin session identifier, profile, cwd/path, hash hoặc chính sách gửi dưới dạng dữ liệu client. Adapter nhận projection tối thiểu và buffer chỉ đọc. Việc chọn cùng tiến trình hoặc IPC chờ pilot. Với process riêng, chỉ dùng channel local, cùng schema, quyền tối thiểu, không cấp credentials hoặc protected data. Cả hai worker tránh log ảnh hay câu hỏi thô ở runtime mặc định.
- Giữ binding `/predict` của baseline, ảnh và mode tùy chọn `standard`, bảy đầu ra, lỗi hiện có, `gr.State` nội bộ và Grad-CAM theo baseline. Không đưa branching, model availability hoặc các DTO mới vào return values của endpoint. Event mở rộng VLM dùng riêng phần UI/controller.

**Illustrative fixture hợp lệ.** Metadata dispatch Comparison sau `ACCEPT`, ảnh và policy đã được server kiểm.

```json
{
  "schema_version": "1.0",
  "request_id": "req-fixture",
  "session_id": "session-fixture",
  "image_digest": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "revision": 4,
  "cnn_mode": "standard",
  "cnn_decision": {"state": "decided", "status": "ACCEPT", "class_id": 7},
  "branch": "comparison",
  "profile_id": "fixture-encoder-1",
  "remaining_budget_ms": 250
}
```

**Không hợp lệ.** Worker đổi CNN status sang `ACCEPT`, output khác digest/session/revision, `status: "stale"` trong public branch state, hoặc client tự cấp profile. Trình xử lý bỏ reply stale và giữ UI thuộc revision hiện hành.

**Proposed Python**

```python
def submit_branch(action: UiBranchAction, *, session: SessionState,
                  image: ValidatedRgb, profiles: ApprovedProfiles) -> BranchState: ...
def accept_worker_reply(reply: WorkerReply, *, session: SessionState) -> PublishDisposition: ...
```

**Contract tests đề xuất.** Trưởng nhóm kiểm hai session, đổi bytes cùng path, đổi mode, Clear, trả kết quả đảo thứ tự, profile mismatch, timeout, OOM recovery, optional deps, guard CNN status, Gradio event binding và parity `/predict` với cả ba status. Dùng baseline cases `test_gradio_event_contract`, `test_predict_through_actual_gradio_http` và `test_inflight_result_cannot_restore_invalidated_outputs` làm tham chiếu source. [Baseline Gradio tests](../../../tests/test_gradio_app.py)

## 8. Contract 06 — Evaluation và Evidence

**Producers** Thuận cho manifest, questions và rubric inputs. Đức Anh cho E12/E13 predictions và evidence. Kiệt cho E14 output traces. Trưởng nhóm điều phối run, scorer, review và E15 evidence. **Consumers** evaluator được cấp quyền, hai reviewer độc lập, người phân xử và trưởng nhóm. Routes và metrics theo Research Protocol. [Research Protocol](research_protocol.md), [Experiment Registry](experiment_registry.md)

Mỗi dòng evidence có R `schema_version` và `record_type`. Với record gắn run, `eval_ref: EvalRef` phải khớp RunProvenance. `image_id: str` duy nhất trong manifest. `question_id: str` duy nhất trong ảnh, cùng `image_id` tạo khóa câu hỏi. `sample_id: str` duy nhất trong cohort được ghi ở manifest hoặc question record khi là đơn vị câu hỏi. Dates theo ISO 8601 và timestamp UTC. Mọi kiểu sau dùng trường R trừ khi ghi O.

| record_type và consumer | Trường và validation |
| --- | --- |
| `pilot_image`, pilot loader | `image_id`, `sample_id`, `image_ref: str` scoped vào approved dev store, `image_digest: Sha256`, `source_ref: str`, `usage_permission: str`, `audit_status: enum(pending, approved, rejected)`. O `perceptual_hash: str` theo thuật toán audit đã pin. Chỉ load approved. Kiểm không trùng giữa tập pilot và cohort formal theo protocol. |
| `vlm_eval_image`, curator và evaluator | `image_id`, `sample_id`, `image_ref`, `source_url: str`, `usage_permission`, `collection_date: str`, `class_id: ClassId hoặc null`, `group: enum(in_scope_dog, out_of_scope_dog, non_dog)`, `image_digest: Sha256`, `perceptual_hash: str`, `audit_status`, `question_ids: str[]`, `annotation_provenance: str` trỏ hồ sơ annotate/review. `class_id` có giá trị chỉ với in-scope. Runtime projection bỏ class ID, group và provenance chấm. |
| `question`, evaluator và runtime qua projection | `image_id`, `question_id`, `sample_id`, `question: str`, `category: QuestionCategory`. `QuestionCategory` gồm `visible_attributes`, `visibility_uncertainty`, `breed_knowledge`, `breed_comparison`, `general_visual_observation`, `out_of_scope_handling` và `non_dog_handling`. Cho phép aggregate knowledge/comparison theo phân bổ đã duyệt. Runtime chỉ nhận text và thuộc tính yêu cầu của operation đã khóa. |
| `annotation`, scorer/reviewer được cấp quyền | `image_id`, `question_id`, `rubric_version: str`, `target_attributes: AttributeObservation[]`, `expected_answer: str hoặc null`, `answerability: enum(answerable, abstention_expected)`, `source_required: bool`, `provenance: {annotator_id: str, reviewed_at: str, revision: str}`. O `expected_source_ids: str[]`. Đây là label riêng của evaluator. Không dùng làm retrieval key, CNN context hoặc prompt. |
| `classification_prediction`, E12/E13 scorer | `eval_ref`, `image_id`, `image_digest`, `prediction_status: enum(OK, FAIL, ABSTAIN)`, `model_result: ZeroShotResult hoặc null`. `OK` cần ranking đã validate. `FAIL`/`ABSTAIN` không có ranking, O `reason_code: ErrorCode` khi có failure/abstention. Một record mỗi ảnh mỗi run. Join nhãn ở scorer. Giữ đủ dòng cohort theo protocol, cả failures. |
| `vqa_output_trace`, E14 evaluator hoặc E15 evidence reviewer | `eval_ref`, `image_id`, `question_id`, `response_id: str` opaque, `raw_model_output: str hoặc null`, `validated_model_output: AssistantResult hoặc null` chứa kết quả model status `ok` đã xác thực, `controller_generated_response: str hoặc null`, `policy_blocked: bool`, `retrieval_status: RetrievalStatus`, `invocation_refs: InvocationRef[]` theo thứ tự stage đã dùng. O `reason_code: ErrorCode`, bắt buộc nếu blocked hoặc inference fail/invalid. Khi blocked trước call, raw và validated đều null, `invocation_refs=[]`, reason `POLICY_BLOCKED`. Phản hồi controller có thể tồn tại riêng. |
| `e14_model_invocation`, E14 evaluator | `eval_ref`, `invocation_id: str`, `stage_type: enum(attributes, vqa, attributes_and_vqa)`, `image_id`, `question_id: str hoặc null`, `model: ModelRef`, `profile_id: str`, `raw_output: str hoặc null`, `parse_status: enum(not_attempted, ok, invalid_json)`, `schema_validation_status: enum(not_attempted, valid, invalid)`, `terminal_outcome: BranchOutcome`, `error_code: ErrorCode hoặc null`, `latency_ms: number`, `context_provenance: ContextProvenance`, `validated_stage_output: AttributeObservation[] hoặc GeneratedContent hoặc null`, `produced_attributes_sha256: Sha256 hoặc null`, `consumed_attributes: AttributeInputRef hoặc null`. |
| `response_index`, evaluator riêng | `response_id`, `eval_ref`, `image_id`, `question_id`. Ánh xạ một ID opaque tới một output slot trong một run. Giữ private index để unblind sau chấm. |
| `blinded_response`, reviewer | `response_id`, `review_image_ref: str` quyền đọc phạm vi chấm, `question: str`, `content_for_review: GeneratedContent hoặc null`, `rubric_version`. Ẩn experiment/branch/model metadata khi khả thi, xáo thứ tự theo kế hoạch khóa. Dữ liệu citation cần cho rubric được giữ và ghi hạn chế blinding nếu có. |
| `reviewer_label`, reviewer độc lập/adjudicator | `reviewer_label_id: str` duy nhất, `response_id`, `reviewer_id: str`, `rubric_version`, `labels: object` theo kiểu của từng rubric field, `submitted_at: str`. O `notes: str`. Mỗi reviewer chấm độc lập trước khi xem nhãn người còn lại. |
| `adjudicated_label`, scorer | `response_id`, `rubric_version`, `reviewer_label_refs: str[2]`, `final_labels: object` cùng schema rubric, `resolved_by: str`, `resolution_note: str`. Hai refs phải trỏ tới hai người chấm khác nhau của cùng response và rubric. Giữ nhãn ban đầu để tính agreement. |
| `metric`, analyst | `eval_ref`, `metric_id: str` từ protocol/lock, `scope: str` như toàn cohort hoặc category đã duyệt, `value: number hoặc null`, `unit: str`, `n_evaluated: int` không âm. O `numerator: number`, `denominator: number` và `ci: {lower: number, upper: number, level: number}` khi áp dụng. Null đi cùng O `na_reason: str` bắt buộc. Trình bày mẫu số theo protocol. |
| `benchmark`, analyst | `eval_ref`, `profile_id`, `hardware_ref: str`, `measurement: str` theo protocol, `scenario: str`, `warmup_count: int`, `measured_count: int`, `unit: str` (ví dụ ms hoặc MiB), `values: number[]`, `failure_count: int`. Counts không âm, số giá trị cộng số lượt lỗi bằng measured_count. Ghi cache/batch/precision và ranh giới đo trong scenario/profile. Không gộp đo khác phần cứng vào đối chứng trực tiếp. |
| `run_provenance`, evaluator/lead | `eval_ref`, `phase: enum(pilot, formal, exploratory)`, `execution_status: RunExecutionStatus`, `git_sha: str`, `protocol_revision: str`, `lock_ref: str hoặc null`, `config_sha256`, `environment_ref: str`, `mapping_sha256`, `manifest_sha256`, `profiles: {profile_id: str, model: ModelRef}[]`, `artifact_hashes: map[str, Sha256]`, `started_at: str` và `finished_at: str hoặc null`. Formal cần `lock_ref` đã duyệt. O `prompt_bank_sha256`, `kb_sha256`, `rubric_version` và `paired_group_id: str` theo run. |

**Evidence theo stage E14.** `InvocationRef` gồm `eval_ref` và `invocation_id`. `AttributeInputRef` gồm `producer: InvocationRef` và `attributes_sha256: Sha256` của snapshot đã validate. Hash trên UTF-8 bytes của snapshot serialization được pin ở hồ sơ lock. Dùng reference có run binding khi hai biến thể của cùng paired group dùng chung một lượt trích xuất. Kiểm ảnh, snapshot hash, stage tương thích, scope và quyền truy cập của cả source lẫn consumer.

- Mỗi invocation thực sự gọi model có một record duy nhất. Stage `attributes_and_vqa` ghi một invocation single-pass và liên kết tới trace. Multi-pass ghi riêng stage `attributes` và `vqa`. Stage theo câu hỏi cần `question_id`, còn trích xuất dùng chung theo ảnh cho phép null. Hash output thuộc tính chỉ điền khi thực sự publish snapshot để tái sử dụng. VQA dùng thuộc tính trung gian phải link về cùng snapshot của cả A/B.
- `ok` yêu cầu parse ok, schema valid, validated output phù hợp stage và error null. Khi gọi model lỗi hoặc output sai, giữ outcome, raw bytes đã giải mã nếu có, mã lỗi, latency và context provenance. Parse lỗi hoặc không có raw thì schema ghi not_attempted. Trước khi gọi model, lưu failure hoặc `policy_blocked` trong trace với reason và không ghi invocation giả. Output fallback nằm riêng ở controller response.
- `raw_model_output` cấp cặp lấy raw của stage trực tiếp tạo câu trả lời. Nếu dừng ở stage thuộc tính trước khi gọi VQA, giữ raw cấp cặp null và lưu lỗi cùng reference của stage đã thử. Với single-pass, cùng một invocation cung cấp raw và output cho trace. Tính số lượt gọi và latency trên invocation IDs duy nhất, kể cả invocation dùng chung nhiều câu hoặc hai biến thể. Giữ đơn vị ảnh, mục thuộc tính theo rubric và cặp Image ID/Question ID theo Research Protocol. Mỗi biến thể vẫn có đủ trace cho cohort, gồm failures và policy block. Dùng liên kết để tái lập metrics, không nhân một lượt trích xuất dùng chung thành nhiều lần inference.

**Ranh giới dữ liệu.**

- **Runtime inputs** dùng RGB, câu hỏi, snapshot đã xác thực, context được policy cho phép và profile. Kiệt và adapter trợ lý không nhận expected answers, class labels hoặc rich eval metadata qua runtime projection. KB/prompt bank không lấy nội dung từ annotations.
- **Scorer-only** giữ manifest chấm, private labels, rubric/expected answers và `response_index` trong quyền riêng. Tạo `response_id` ngẫu nhiên opaque theo response slot, ghép lại sau khi chấm. Reviewer biết ảnh, câu hỏi và nội dung được duyệt để chấm, còn phép ghép với Experiment ID nằm trong index riêng. Agreement lấy nhãn ban đầu.
- **Research evidence** nằm theo `outputs/research/vlm/<experiment_id>/<run_id>/` đã đăng ký. Validation/VLM-Eval formal cần đủ IDs, audit, lock, artifact hashes và join được theo cohort. Policy-blocked row vẫn giữ trong cohort. Controller-generated response và fallback chỉ dùng E15. Khi dữ liệu không đủ cho một phép đối chứng, áp blocker isolation trong Registry. Scorer tuân theo định nghĩa metric đã duyệt.
- **Runtime logs** chỉ giữ ID vận hành tối thiểu, trạng thái, error code và timing. Raw text, ảnh, captions, expected answers, đường dẫn upload và thông tin phiên nhạy cảm tách khỏi logs mặc định. Official Test và protected bundles nằm ngoài cả runtime, worker mounts lẫn evidence loaders.

Loader từ chối hash, schema, ID hoặc scope không hợp lệ. Lỗi `INVALID_INPUT`/`INVALID_SNAPSHOT` được ghi ở evidence đã được phép. Các run ở trạng thái thực thi `RUNNING`, `COMPLETED`, `FAILED` hoặc `ABORTED`. Research lifecycle gồm `PLANNED`, `PILOT_IN_PROGRESS`, `PILOT_REVIEW`, `READY_FOR_LOCK`, `LOCKED`, `FORMAL_IN_PROGRESS`, `RESULTS_READY`, `ACCEPTED`, `BLOCKED`, `REJECTED`. `retrieval_status`, `policy_blocked` và runtime `BranchState` là trường khác. Một `COMPLETED` run có thể làm evidence cho review, còn `ACCEPTED` là quyết định nghiệm thu. [Lifecycle và staged lock](experiment_registry.md)

**Illustrative fixture hợp lệ.** E14 sample bị policy chặn trước khi gọi model. ID và thời gian trong fixture minh họa định dạng run.

```json
{
  "schema_version": "1.0",
  "record_type": "vqa_output_trace",
  "eval_ref": {
    "experiment_id": "E14-A",
    "run_id": "E14-A_formal_20990101T000000Z_123e4567-e89b-42d3-a456-426614174000"
  },
  "image_id": "fixture-image",
  "question_id": "fixture-question",
  "response_id": "response-fixture",
  "raw_model_output": null,
  "validated_model_output": null,
  "controller_generated_response": "Chỉ có thể mô tả chi tiết quan sát được.",
  "policy_blocked": true,
  "invocation_refs": [],
  "retrieval_status": "not_applicable",
  "reason_code": "POLICY_BLOCKED"
}
```

**Illustrative fixture theo stage.** Một lượt trích xuất theo ảnh. Validator chiếu output hợp lệ thành danh sách thuộc tính và ghi hash của snapshot JSON UTF-8 compact. Khi dùng chung, stage VQA giữ `AttributeInputRef` trỏ lại run, invocation ID và hash bên dưới.

```json
{
  "schema_version": "1.0",
  "record_type": "e14_model_invocation",
  "eval_ref": {
    "experiment_id": "E14-A",
    "run_id": "E14-A_pilot_20990101T000000Z_123e4567-e89b-42d3-a456-426614174000"
  },
  "invocation_id": "invocation-fixture-01",
  "stage_type": "attributes",
  "image_id": "fixture-image",
  "question_id": null,
  "model": {
    "model_id": "Qwen/Qwen3-VL-2B-Instruct",
    "model_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "processor_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "profile_id": "fixture-assistant-1",
  "raw_output": "{\"attributes\":[{\"attribute_id\":\"coat_color\",\"visibility\":\"observed\",\"value\":[\"black\",\"white\"]}],\"answer_vi\":null,\"disposition\":\"not_applicable\",\"uncertainty\":\"unspecified\",\"abstention_reason\":null,\"source_ids\":[]}",
  "parse_status": "ok",
  "schema_validation_status": "valid",
  "terminal_outcome": "ok",
  "error_code": null,
  "latency_ms": 0,
  "context_provenance": {
    "retrieval_status": "not_applicable",
    "kb_version": null,
    "kb_sha256": null,
    "context_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  },
  "validated_stage_output": [
    {
      "attribute_id": "coat_color",
      "visibility": "observed",
      "value": [
        "black",
        "white"
      ]
    }
  ],
  "produced_attributes_sha256": "582205e5f7c524429115bcbeeaa8bc88ca8a18b03ee7219d75e61c99f792c496",
  "consumed_attributes": null
}
```

**Không hợp lệ.** Blinded reviewer payload chứa `experiment_id`, expected answer chuyển sang model, cohort trùng ID, `policy_blocked=true` nhưng điền output giả do model sinh, formal provenance thiếu lock hoặc `execution_status: "ACCEPTED"`.

**Proposed Python**

```python
def load_evaluation_inputs(manifest: ApprovedManifest, *, scope: EvaluationScope) -> Iterable[RuntimeEvalInput]: ...
def make_blinded_packet(trace: VqaOutputTrace, *, rubric: ReviewRubric) -> BlindedResponse: ...
def validate_evidence(record: EvidenceRecord, *, run: RunProvenance, scope: EvaluationScope) -> None: ...
```

**Contract tests đề xuất.** Thuận kiểm manifest IDs, counts, allowed origins, hash và annotation tách biệt. Đức Anh kiểm prediction alignment và failures giữ đầy đủ. Kiệt kiểm traces, unique invocation, các liên kết giữa stages, snapshot dùng chung, policy block, failures, retrieval statuses, raw/validated layers và source allowlist. Trưởng nhóm kiểm quyền đọc, blinded index, reviewer độc lập, join sau chấm và trạng thái run. Dùng fixture hoặc approved development inputs, theo scope đã duyệt.

## 9. Ownership và ranh giới triển khai

| Contract ID | Producer | Consumer | Proposed module path | Required fields trọng tâm | Dependency | Validation owner | Test responsibility |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | Thuận | Đức Anh, Kiệt, E15 | `src/vlm/knowledge_base.py` | Version, mapping hash, 30 breed/prompt IDs, provenance | 08.04-KB/NAME/MORPH, canonical map | Thuận, consumer kiểm binding | Thuận loader/fixture, Đức Anh token audit, Kiệt/E15 snapshot |
| 02 | Thuận | Kiệt, E15 | `src/vlm/retrieval.py` | Version/hash, lookup status, entries, context, source IDs, budget | Contract 01, policy/profile tokenizer | Thuận và E15 tại policy boundary | Thuận alias, multi-breed, budget. Kiệt context. E15 policy. |
| 03 | Đức Anh | E15, offline evaluator | `src/vlm/zero_shot.py` | Model/profile, hashes, score vector/ranking, terminal status | Contract 01, ApprovedProfile, ValidatedRgb | Đức Anh, E15 pre-publish | Đức Anh adapter/ranking/value. Trưởng nhóm parity. |
| 04 | Kiệt | E14 evaluator, E15 | `src/vlm/assistant.py` | Operation, context provenance, structured observations, answer và status | Contract 02, policy DTO và ApprovedProfile | Kiệt và E15 | Kiệt schema và A/B isolation. Trưởng nhóm publication scope. |
| 05 | Trưởng nhóm | UI, adapters/workers | `app/vlm_integration.py` | Session, digest, revision, mode, CNN snapshot, branch, request/profile | CNN baseline, Contracts 03/04, selected profiles | Trưởng nhóm | Trưởng nhóm API/session/fallback với Đức Anh, Kiệt, Thuận. |
| 06 | Trưởng nhóm quản lý shared schema. Thuận, Đức Anh, Kiệt và trưởng nhóm sản xuất evidence theo lớp đã phân công. | Evaluator, scorers, reviewers, trưởng nhóm | `src/vlm/evidence.py` | Run/provenance, join IDs, separated outputs, review records | Protocol Lock, Contracts 01–05, 08.04 audit/rubric | Producer kiểm record, trưởng nhóm kiểm run/scope | Mỗi owner kiểm schema và evidence. Trưởng nhóm kiểm end-to-end. |

Trưởng nhóm sở hữu `src/vlm/contracts.py` và quản lý shared schema của `src/vlm/evidence.py`. `src/vlm/contracts.py` là module proposed dùng Python stdlib cho type aliases, enum, DTO và shared validators. Khai báo `CanonicalMapping`, `ValidatedRgb`, `ModelRef`, `ApprovedProfile`, `EvalRef`, `ErrorInfo` và các data shapes ở đây. Adapter riêng chịu trách nhiệm biến RGB thành model input và chấm điểm/generate. Shared module và fixture consumer dùng được khi thiếu model libraries. Có thể inject mock adapter, tokenizer counter và approved fixtures để phát triển song song.

**Milestone Shared Contract Foundation.** Sau khi hoàn thành và duyệt toàn bộ planning documents, trưởng nhóm mở milestone triển khai với version `1.0` dự kiến. Phạm vi gồm shared DTO/types, enums, validators, mock adapters/interfaces, synthetic fixtures và contract tests cơ bản. Kiểm import và smoke bằng fixture khi thiếu model libraries. Modules dùng chung chỉ phụ thuộc stdlib hoặc shared DTO nội bộ. Các thành viên giữ ownership module ở bảng, review cùng nhau trước handover. Giai đoạn hiện tại chỉ đăng ký milestone ở mức thiết kế.

Các **proposed source entry points** bám Registry. `scripts/evaluate_vlm_zero_shot.py` dùng chung cho E12/E13, `scripts/evaluate_vlm_assistant.py` cho E14-A/B, và `scripts/verify_vlm_integration.py` cho E15. Modules ở bảng được gọi từ các entry points hoặc controller, dùng chung contract cho runtime và offline projection. Config paths và outputs lấy theo từng experiment trong Registry. `/predict` tiếp tục là endpoint CNN hiện có. Thay đổi breaking schema cần trưởng nhóm duyệt, tăng version phù hợp và cập nhật fixture cùng producer/consumer. [Experiment Registry](experiment_registry.md)

## 10. Quyết định còn mở

- **PENDING PILOT.** Model/processor revisions khả thi và security gate, score_type/native score transform, token budget, single-pass hoặc multi-pass, taxonomy và limits có thể kiểm, timeout, concurrency, OOM/recovery và in-process hoặc separate process.
- **PENDING LEADER REVIEW.** DTO/paths đề xuất và xác nhận vocabulary sau pilot, cách cấp và thu hồi session/worker handles, phạm vi offline/runtime policy, quyền lưu dữ liệu private, và kiểm tra compatibility giữa các producer/consumer.
- **Trước formal.** Khóa từng nhóm theo Registry và Research Protocol, gồm mapping/model/dataset/prompt/KB hashes, validation policy và rubric. E15 chọn profile đã nghiệm thu và đủ deployment eligibility trước tích hợp. Phạm vi nghiệm thu E15 đầy đủ hoặc reduced scope do trưởng nhóm duyệt. [Architecture](architecture.md), [Research Protocol](research_protocol.md), [Experiment Registry](experiment_registry.md)
