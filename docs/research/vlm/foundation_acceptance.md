# Biên bản kỹ thuật nghiệm thu Shared Contract Foundation

**FOUNDATION ACCEPTED BY LEADER — PUBLICATION AND H0 HANDOVER PENDING**

**Task 08.01 · Phase 3.4 · DogBreedID V2**

## 1. Phạm vi và baseline

| Mục | Giá trị đối chiếu |
| --- | --- |
| Repository | `DGiang1126/dogbreedid-cv-project` |
| Planning revision | Planning PR #32 đã merge. Sáu tài liệu được đối chiếu tại commit khởi đầu Foundation. |
| Foundation branch và starting commit | `feature/vlm-shared-contracts` tại `c8e6f60b63238770a69ed4affc01352f31d1e87d` |
| Frozen CNN baseline | Tag `v1.0.0-cnn-stable` trỏ tới `b7d880506b96168b0e335f3a455be3a7b2bdf77e` |
| Phạm vi implementation | Shared Contract 01–06, wire schema `1.0`, DTOs, enums, validators, Protocol interfaces, fixtures, mocks và dedicated tests |
| Phạm vi kiểm | Python source/test Foundation và fixtures synthetic, public API và wire examples, import isolation, provenance và joins trong bộ nhớ |

Foundation cung cấp hợp đồng giữa các component. Model adapters và weights, native tokenizer, loader/dataset thật, lookup/renderer production, semantic scoring, evaluator, artifact store, actual lock authority và E15 controller/Gradio thuộc các mốc triển khai sau. Pilot, formal evaluation, benchmark thật, Official Test và protected artifacts nằm ngoài đợt kiểm này.

Diff giữa frozen baseline và starting commit chỉ thêm sáu planning documents. Tracked diff từ starting commit rỗng tại preflight. E4/E6/E7/E8, Energy OOD, Grad-CAM, `/predict`, canonical mapping và Configuration Lock được bảo toàn. Đây là đối chiếu Git scope, còn actual runtime data boundary enforcement thuộc E15. [Architecture](architecture.md#3-nguyên-tắc-vận-hành-và-bảo-toàn-baseline) · [Acceptance matrix](acceptance_criteria.md#7-e15-integration-test-matrix)

## 2. Planning acceptance · AC-0801-01

Năm tài liệu kỹ thuật thống nhất baseline, task/experiment IDs, ownership, data boundary, gate và terminology. Team Assignment dùng cùng quy tắc để phân công và định nghĩa H0–H6. Bảng ghi **SHA-256 của bytes Git blob** tại starting commit. Working tree Windows khác line endings và khớp nội dung sau chuẩn hóa CRLF/LF. Trạng thái dưới đây được giữ theo từng tài liệu. [AC-0801-01 và AC-0801-02](acceptance_criteria.md#6-task-0801--trưởng-nhóm)

| Tài liệu | SHA-256 tại `c8e6f60…` | Trạng thái review và quyết định mở |
| --- | --- | --- |
| [Architecture](architecture.md) | `3b5222d0f24037c1479f6614f9612db8f005292ebb945a56c7c49da82fa53361` | APPROVED FOR IMPLEMENTATION PLANNING — PENDING PILOT. Model/profile, single-pass/multi-pass, tài nguyên và worker architecture chờ pilot. |
| [Research Protocol](research_protocol.md) | `c1776eb6167f69c4acc6280481be9451c545cd32fd5ff558500a673bb15c7fd3` | APPROVED FOR IMPLEMENTATION PLANNING — PROTOCOL LOCK PENDING. Inputs, rubric, metrics và ngưỡng formal chờ khóa. |
| [Experiment Registry](experiment_registry.md) | `17c4012197fa6b135b0fba8128ac22accc837f9bc8137bbc82306b76cc54df44` | APPROVED FOR IMPLEMENTATION PLANNING — EXECUTION PENDING. E12-A/B, E13-A/B, E14-A/B và E15 vẫn `PLANNED`. |
| [Interface Contract](interface_contract.md) | `83451680454df495480da1af8ba50f222d10c856dbe280d3d87552cf757620e9` | APPROVED FOR IMPLEMENTATION PLANNING — PILOT AND PROFILE LOCK PENDING. Implementation schema `1.0` đã kiểm. Production bindings và policy còn chờ duyệt. |
| [Acceptance Criteria](acceptance_criteria.md) | `f509f62b4bd907890960f3723012232326a8fa4cc6da33f651fa926d2076bf74` | APPROVED FOR IMPLEMENTATION PLANNING — EXECUTION AND ACCEPTANCE PENDING. 57 criterion statuses giữ `NOT_EVALUATED`. |
| [Team Assignment](../../team_tasks/24_11_task.md) | `aa2494567b115459235fd0c16b17bbc7f7a5c1edf61eae402624f232be2628d2` | APPROVED FOR IMPLEMENTATION PLANNING — H0 HANDOVER PENDING. Đây là trạng thái của planning revision. Quyết định nghiệm thu Foundation và ngoại lệ xác minh được ghi tại mục 7. |

## 3. Foundation acceptance · AC-0801-02

| Requirement | Evidence đối chiếu |
| --- | --- |
| DTO và types | `contracts.py` định nghĩa DTO bất biến cho RGB, model/profile, KB/banks, retrieval, classifier, assistant và worker envelopes. Test kiểm exact types, nullable fields và defensive copying. [Contracts](../../../src/vlm/contracts.py) · [Tests](../../../tests/vlm_foundation/test_contracts.py) |
| Enums | Wire/runtime statuses, adapter outcomes, CNN policy, retrieval status và execution status có miền giá trị riêng. Negative tests từ chối sai namespace và giá trị lạ. [Contract tests](../../../tests/vlm_foundation/test_contracts.py) · [Evidence tests](../../../tests/vlm_foundation/test_evidence.py) |
| Validators | Kiểm canonical IDs, strict JSON, SHA-256 trên bytes, prompt audits, ranking, knowledge/context binding, DTO invariants, policy và trusted run/manifest joins. [Contracts](../../../src/vlm/contracts.py) · [Evidence](../../../src/vlm/evidence.py) |
| Protocol interfaces | `ZeroShotClassifier.classify`, ba methods của `GenerativeAssistant`, callable `KnowledgeRetriever` và `ContextTokenCounter`. Mock signatures được đối chiếu theo test. [Contracts](../../../src/vlm/contracts.py) · [Signature test](../../../tests/vlm_foundation/test_contracts.py#L943) |
| Synthetic fixtures | Mapping 30 tên canonical, RGB, KB/Prompt Banks, token audit, profile, retrieval, run/cohort và reviewer records được dựng trong bộ nhớ. [Contract fixtures](../../../tests/vlm_foundation/fixtures.py) · [Evidence fixtures](../../../tests/vlm_foundation/evidence_fixtures.py) |
| Mock adapters | Bốn mock classifier, generative assistant, retriever và token counter. Fixture kiểm kết quả hợp lệ, failure và trạng thái retrieval. [Mocks](../../../tests/vlm_foundation/mocks.py) |
| Shared evidence schemas | Contract 06 có 14 record types. Tests kiểm decoding, manifest, classification cohort, E14 stages, rubric/blinding, metrics, benchmark và run provenance. [Evidence](../../../src/vlm/evidence.py) · [Evidence tests](../../../tests/vlm_foundation/test_evidence.py) |
| Consumer smoke tests | Luồng mock classifier → prediction, mock retrieval/assistant → trace/blinded packet và manifest → runtime projection chạy trong subprocess cô lập. [Import và smoke tests](../../../tests/vlm_foundation/test_import_isolation.py) |
| Import isolation | Contracts, evidence và mock consumers chạy với `-I -S -B`. Import hook ghi nhận 0 lần thử import model/UI dependencies. Socket audit ghi 0 events. [Import và smoke tests](../../../tests/vlm_foundation/test_import_isolation.py) |
| Schema compatibility | Wire version `1.0`, round-trip và strict decoding được kiểm. Hai ví dụ Contract 06 round-trip. Python 3.10 syntax, UTF-8 không BOM, LF và whitespace đã đối chiếu. Runtime chạy trên Python 3.12.4 và trong review kỹ thuật trên Python 3.13.5. [Interface Contract](interface_contract.md#2-quy-ước-dùng-chung) |

## 4. Danh mục implementation

SHA-256 dưới đây tính trên bytes thực của 10 file Python Foundation trong working tree. Cả 10 khớp revision kỹ thuật đã review ở Phase 3.3B.

| Path | Vai trò | Consumer hoặc mục đích test | SHA-256 |
| --- | --- | --- | --- |
| [src/vlm/\_\_init\_\_.py](../../../src/vlm/__init__.py) | Package initializer nhẹ | Mọi consumer import package | `ef43a463f963bbf435d3517beda2b9ad821a471038077d3967b2383aaa3343bc` |
| [src/vlm/contracts.py](../../../src/vlm/contracts.py) | Shared DTO, enum, Protocol, codec và validators Contract 01–05 | Đức Anh, Kiệt, Thuận và E15 | `d8e56e7b207b4321479ad8a700909e5851346b4a125a2e3d360488aa6cdcff61` |
| [src/vlm/evidence.py](../../../src/vlm/evidence.py) | Record types, run binding, joins và projections Contract 06 | Producers, evaluator và trưởng nhóm | `7c53d4b1fda0e233e65cf1d22add5fa369810394d0105f0927f78d7407aaa863` |
| [tests/vlm_foundation/\_\_init\_\_.py](../../../tests/vlm_foundation/__init__.py) | Test package initializer | Relative imports của fixtures và tests | `f9e3cd802abf5156576ad7d05c2a061cb55375d2972c72ff23b6ee7dc18a062c` |
| [tests/vlm_foundation/evidence_fixtures.py](../../../tests/vlm_foundation/evidence_fixtures.py) | Synthetic manifests, runs, stages và review records | Evidence tests và smoke consumers | `94d6d1750ba777784be8851d178b58f920afe643cf5b0547eb4bd69a17b11709` |
| [tests/vlm_foundation/fixtures.py](../../../tests/vlm_foundation/fixtures.py) | Synthetic RGB, mapping, KB/banks, profile và request/reply fixtures | Contract tests, evidence fixtures và mock consumers | `d5d5ffd2cdb7ea9df0951d09884d1897af84aa26c08e6812728fdb4009302448` |
| [tests/vlm_foundation/mocks.py](../../../tests/vlm_foundation/mocks.py) | Bốn test doubles | Kiểm adapter handoff và failure paths | `05cd532b4698d865387eb04cfebc44a193058481bf097659e6efa4d835ad7264` |
| [tests/vlm_foundation/test_contracts.py](../../../tests/vlm_foundation/test_contracts.py) | Positive/negative tests Contract 01–05 | Kiểm schema, invariants và trusted binding | `366d9cc0f3a37e400b2475718886af1834241f096113a777607e240fff01788d` |
| [tests/vlm_foundation/test_evidence.py](../../../tests/vlm_foundation/test_evidence.py) | Tests Contract 06 và F04 | Run/cohort, capture, pairing, reviewer và formal completeness | `53c0db15b00a607c9e11b59b7b7bd0beb0d7abccc31b9bc33cea0fcf9b48e807` |
| [tests/vlm_foundation/test_import_isolation.py](../../../tests/vlm_foundation/test_import_isolation.py) | Subprocess import/smoke test | Kiểm model/UI dependency attempts và network events | `91e35cf52777d044178d50c9b1169bf416294438c23e06f8e7973ece74ec9b02` |

## 5. Bằng chứng kiểm thử

Các dòng dưới đây ghi lượt chạy cuối của từng phase. Failed/errors/skipped/warnings là số đếm của pytest. Đây là kết quả kiểm hợp đồng trên synthetic fixtures.

| Nguồn | Phase | Python / pytest | Passed | Failed / Errors / Skipped / Warnings | Pytest duration | Exit |
| --- | --- | --- | ---: | --- | ---: | ---: |
| Codex Windows | 3.2 | 3.12.4 / 8.3.2 | 187 | 0 / 0 / 0 / 0 | 1.19 s | 0 |
| Codex Windows | 3.2B | 3.12.4 / 8.3.2 | 222 | 0 / 0 / 0 / 0 | 1.26 s | 0 |
| Codex Windows | 3.3 | 3.12.4 / 8.3.2 | 368 | 0 / 0 / 0 / 0 | 1.24 s | 0 |
| Codex Windows | 3.3B | 3.12.4 / 8.3.2 | 394 | 0 / 0 / 0 / 0 | 1.72 s | 0 |
| Review kỹ thuật độc lập bằng ChatGPT | 3.3 ZIP + hai file 3.3B | 3.13.5 / 9.0.2 | 394 | 0 / 0 / 0 / 0 | 1.28 s | 0 |
| Codex Windows, chạy lại cho biên bản này | 3.4 | 3.12.4 / 8.3.2 | 394 | 0 / 0 / 0 / 0 | 2.38 s | 0 |
| Review kỹ thuật độc lập mới nhất, kết quả được cung cấp | 3.4 trước cập nhật quyết định | 3.13.5 / 9.0.2 | 394 | 0 / 0 / 0 / 0 | 1.11 s | 0 |

Codex Windows dùng interpreter `C:\Users\pc\AppData\Local\Programs\Python\Python312\python.exe`. Lượt 3.4 đo toàn lệnh 3.703 s. Lệnh Phase 3.2/3.2B dùng cùng options và hai module `test_contracts.py`, `test_import_isolation.py`. Từ Phase 3.3 thêm `test_evidence.py`. Trước khi chạy đã kiểm import graph, pytest config, conftest và plugin settings.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:PYTEST_ADDOPTS = ''
$env:PYTEST_PLUGINS = ''
python -B -m pytest --noconftest -o addopts= -p no:cacheprovider -q tests/vlm_foundation/test_contracts.py tests/vlm_foundation/test_evidence.py tests/vlm_foundation/test_import_isolation.py
```

Báo cáo độc lập cho Phase 3.3B ghi 10/10 SHA-256 khớp, giữ 368 test IDs và thêm 26 IDs. Import isolation, cú pháp Python 3.10, UTF-8 không BOM, LF và whitespace đều đạt. Báo cáo cung cấp kết quả của ba module Foundation và không ghi nguyên văn câu lệnh shell. Thông báo khởi tạo spreadsheet runtime trên stderr ở lượt này là giới hạn môi trường, với 0 pytest warnings và exit code 0.

Review kỹ thuật độc lập mới nhất được cung cấp ghi PASS cho cả hai tài liệu, 132/132 relative links và 3/3 ví dụ onboarding synthetic đạt. SHA-256 của hai Markdown trước cập nhật, sáu planning documents và mười Python files khớp. Import isolation PASS, F01–F04 CLOSED và retrieval-to-assistant binding PASS. Trưởng nhóm dùng evidence này để nghiệm thu. Independent team verification **NOT PERFORMED** theo ngoại lệ ở mục 7.

Lượt publication preflight dùng kết quả 394 tests hiện có vì toàn bộ mười file Python source/test giữ nguyên từng byte. [Workflow mục 9](acceptance_criteria.md#9-acceptance-workflow)

## 6. Kết quả xử lý findings

| Finding | Lỗi ban đầu | Bản sửa | Bằng chứng hồi quy và kết luận kỹ thuật |
| --- | --- | --- | --- |
| F01 · Trusted prompt token limit | Audit khai limit 128 và count 80 vẫn qua dù trusted profile limit 32. | Kiểm khóa nội bộ `prompt_token_limit`, kiểu số nguyên dương, audit limit bằng trusted limit và count/truncation hợp lệ. | Positive/negative tests về sai pin, thiếu limit và truncation đạt. **Closed qua review kỹ thuật.** [Tests](../../../tests/vlm_foundation/test_contracts.py#L256) |
| F02 · E14-B KB provenance | VQA và combined VQA chấp nhận request thiếu KB pin hoặc retrieval binding. | E14-B giữ pin qua cả sáu retrieval statuses, kể cả `not_applicable`. Shared extraction trước KB và E14-A giữ semantics riêng. | Kiểm đủ statuses, KB mismatch và no-KB stage hợp lệ. **Closed qua review kỹ thuật.** [Tests](../../../tests/vlm_foundation/test_contracts.py#L688) |
| F03 · Retrieval context binding | Text bị sửa vẫn qua khi entries/sources và token count khớp. | Renderer tin cậy tạo expected text từ entries đã đối chiếu KB. Chỉ sau kiểm mới trả `ValidatedRetrieval`. | Kiểm text bị sửa và context rỗng khi không hit. **Closed qua review kỹ thuật.** [Tests](../../../tests/vlm_foundation/test_contracts.py#L354) |
| Retrieval-to-assistant binding | Retrieval hợp lệ cho câu hỏi, CNN context hoặc tokenizer khác vẫn ghép được với assistant. | Giữ retrieval request và trusted policy rồi đối chiếu câu hỏi, CNN context, policy và tokenizer revision với assistant/profile. | Kiểm mismatches qua cả request và result paths. **Closed qua review kỹ thuật.** [Tests](../../../tests/vlm_foundation/test_contracts.py#L753) |
| F04 · Formal E14 completeness gate | Formal E14 có thể qua run binding khi thiếu annotations hoặc có ảnh nhưng không có question slots. | Formal E14-A/B kiểm loại ảnh, slots từng ảnh, hai artifact hashes, exact annotation coverage và rubric trên mọi execution status. | 26 IDs mới. Partial traces của FAILED/ABORTED vẫn hợp lệ nếu đầu vào đầy đủ. **Closed qua review kỹ thuật.** [Tests](../../../tests/vlm_foundation/test_evidence.py#L647) |

## 7. Quyết định nghiệm thu của trưởng nhóm

| Criterion | Quyết định | Cơ sở |
| --- | --- | --- |
| `AC-0801-01` | **PASS — Leader acceptance based on technical review** | Review thiết kế, revisions/hashes và phạm vi của sáu planning documents |
| `AC-0801-02` | **PASS — Leader acceptance based on technical review** | Contract 01–06, schema `1.0`, 394 tests, import isolation và closure của findings |

Biên bản dùng các trường của [mẫu mục 9.1](acceptance_criteria.md#91-mẫu-biên-bản-theo-thành-viên). Quyết định áp dụng cho planning và Shared Contract Foundation. Research experiments, deployment profiles và E15 có các gate nghiệm thu riêng. Criterion statuses trong Acceptance Criteria và experiment lifecycle giữ nguyên.

| Trường biên bản | Giá trị |
| --- | --- |
| Task / deliverable / gate | Task 08.01 · Shared Contract Foundation |
| Owner theo phân công | Trưởng nhóm |
| Ngày quyết định | 25/11/2026 (Asia/Saigon) |
| Reviewed revision | Starting HEAD `c8e6f60b63238770a69ed4affc01352f31d1e87d`, danh mục SHA-256 Foundation tại mục 4 và hai Markdown ở revision trước cập nhật |
| Scope / schema / evidence references | Planning và Contract 01–06 · schema `1.0` · evidence ở mục 1–6 |
| Run IDs / execution status áp dụng | Evidence kỹ thuật trên synthetic fixtures |
| Technical review evidence | **PASS**, theo các kết quả được cung cấp ở mục 5 |
| Independent team verification | **NOT PERFORMED** |
| Process deviation | **Team verification step waived by leader.** Trưởng nhóm quyết định dùng technical review evidence hiện có thay cho lượt xác minh bổ sung của thành viên. |
| Leader decision | `AC-0801-01` PASS và `AC-0801-02` PASS trong phạm vi nêu trên |
| Leader sign-off | Ghi nhận quyết định do trưởng nhóm cung cấp trong yêu cầu Phase 3.4 Leader Acceptance and Publication Preflight. |
| H0 | **PENDING** · chờ publication, Foundation PR merge, công bố common commit và xác nhận nhận bàn giao |
| Consumer acknowledgements | Đức Anh, Kiệt và Thuận xác nhận sau publication và nhận common commit/schema `1.0` |

## 8. Quyết định triển khai còn chờ

Các mục dưới đây thuộc gate của từng workstream. H0 xét shared contract và synthetic handoff. Quyền thực thi trên đầu vào thật được duyệt theo từng gate.

| Quyết định | Owner thực hiện / người duyệt | Gate |
| --- | --- | --- |
| Model, processor và tokenizer revisions thật | Đức Anh cho E12/E13, Kiệt cho E14 / trưởng nhóm | Pin đề xuất và kiểm khả thi trước pilot tương ứng. Chốt trước formal. |
| Native scoring và score transform | Đức Anh / trưởng nhóm | Pilot encoder rồi staged lock E12-A/B và E13-A/B. |
| Generation, vocabulary, single-pass/multi-pass và shared inputs | Kiệt, phối hợp Thuận / trưởng nhóm | Pilot trợ lý rồi lock E14-A/B. |
| Token limits và native token audit | Đức Anh, Kiệt, Thuận / trưởng nhóm | Chọn theo profile trong pilot. Duyệt audit và budgets trước formal. |
| Hardware, resource budgets và deployment thresholds | Các owner model và trưởng nhóm | Duyệt scope máy trước pilot, khóa ngưỡng trước formal và áp eligibility trước E15. |
| Model security approval | Owner adapter / trưởng nhóm | Review và pin custom code, duyệt môi trường biệt lập trước tải/thực thi theo gate. |
| Dataset thật, quyền dùng và audit | Thuận / trưởng nhóm | Duyệt development subset trước pilot. Chốt cohort và audit H3 trước formal. |
| Final rubric, expected answers, reviewer assignment và evaluation criteria | Thuận, Kiệt / trưởng nhóm | Calibration trên fixture pilot, H3 và lock trước E14 formal. |
| Staged Protocol Locks | Trưởng nhóm cùng các owner | E12-A/B, E13-A/B và E14-A/B trước lượt formal tương ứng. E15 lock trước lượt nghiệm thu. |
| Retrieval context renderer và nhận diện tên giống ngoài KB | Thuận phối hợp Kiệt / trưởng nhóm duyệt caller policy | Làm rõ bounded recognition và fallback trước retrieval thật. Khóa renderer, tokenizer và scope trong H2/E14-B. |
| Runtime requested-attribute projection | Thuận và Kiệt / trưởng nhóm duyệt ranh giới | H3 và lock E14. Kiểm deployment projection trước E15. API hiện chiếu RGB, operation và question. |
| Actual permissions, lock authority và artifact storage | Trưởng nhóm và producer theo scope | Duyệt quyền và store trước sử dụng inputs/outputs thật. Kiểm boundary khi tích hợp. |
| Worker architecture | Trưởng nhóm cùng Đức Anh, Kiệt | Pilot tài nguyên và fault handling. Quyết định trước profile integration. |
| E15 system policy và safe regression | Trưởng nhóm / verifier độc lập | Sau eligibility H4/H5, E15 lock và phê duyệt phạm vi hồi quy, trước H6. |

[Research Protocol Lock](research_protocol.md#8-research-protocol-lock) · [Staged locks](experiment_registry.md#53-staged-protocol-lock) · [H0–H6](../../team_tasks/24_11_task.md#vii-phụ-thuộc-và-mốc-bàn-giao-h0h6) · [Quickstart](../../team_tasks/task08_quickstart.md)
