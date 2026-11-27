# Quickstart Task 08 — DogBreedID V2

Hướng dẫn cho Đức Anh, Kiệt, Thuận và trưởng nhóm. Foundation đã được trưởng nhóm nghiệm thu kỹ thuật dựa trên technical review evidence. Team verification không thực hiện theo ngoại lệ đã ghi trong biên bản. Publication và H0 vẫn pending. Common starting commit chưa được xác định. Dùng các ví dụ synthetic để làm quen với API. Planning documents và shared source đã duyệt là nguồn đặc tả khi có khác biệt với hướng dẫn này. [Team Assignment](24_11_task.md) · [Hồ sơ Foundation](../research/vlm/foundation_acceptance.md)

## 1. Mục tiêu và thuật ngữ

DogBreedID V2 bổ sung hai chức năng độc lập cho CNN nhận diện 30 giống chó. **Comparison** đối chiếu ảnh với mô tả văn bản để nghiên cứu phân loại zero-shot. **Visual AI Assistant** mô tả thuộc tính nhìn thấy và trả lời câu hỏi tiếng Việt. CNN E4 vẫn quyết định phân loại, với E6, E7, Energy OOD, E8, Grad-CAM và `/predict` được bảo toàn. [Kiến trúc và ranh giới baseline](../research/vlm/architecture.md#3-nguyên-tắc-vận-hành-và-bảo-toàn-baseline)

| Thuật ngữ | Cách hiểu trong Task 08 |
| --- | --- |
| VLM | Vision-Language Model xử lý ảnh cùng văn bản. Encoder so độ tương hợp ảnh–text. Generative VLM có thể sinh câu trả lời và thuộc tính có cấu trúc. |
| Zero-shot | Dùng model pretrained và prompt cho từng lớp để phân loại mà không huấn luyện thêm. |
| Prompt Bank | Bộ 30 prompt gắn với canonical class IDs. Name Bank dùng tên giống. Morphology Bank giữ tên và thêm mô tả hình thái có nguồn. |
| KB và retrieval | Knowledge Base lưu tri thức theo giống có provenance. Retrieval tìm mục phù hợp theo ID hoặc alias rồi tạo context có giới hạn token và nguồn. |
| VQA | Visual Question Answering, tức trả lời câu hỏi dựa trên ảnh và context được phép. |
| DTO và validator | DTO quy định các trường và kiểu dữ liệu trao đổi. Validator kiểm cả cấu trúc và quan hệ giữa các trường, ví dụ output ghép đúng ảnh và snapshot. |
| Mock và fixture | Mock giả lập adapter với kết quả định sẵn. Fixture cung cấp đầu vào synthetic và giá trị kỳ vọng để test lặp lại được. |
| Pilot và Protocol Lock | Pilot kiểm khả thi, schema, tài nguyên và đề xuất cấu hình trên development inputs đã duyệt. Protocol Lock ghi và phê duyệt revisions, inputs, rubric, metrics và quy tắc trước formal. |
| Evidence và handover | Evidence ghi lại phạm vi, provenance, outputs và lỗi để người khác kiểm lại. Handover chuyển revision và quyền dùng đúng phạm vi cho consumer, kèm xác nhận. |

## 2. Bắt đầu từ repository root

1. Sau khi Foundation PR merge vào `main`, dùng commit chung do trưởng nhóm công bố. Placeholder trong hướng dẫn là `<FOUNDATION_MERGE_SHA>`. Diễn giải SHA này theo biên bản H0, khác với commit khởi đầu của Foundation và tag CNN baseline.
2. Mở PowerShell trong checkout hoặc worktree của mình. Đối chiếu tên branch, `HEAD` và baseline tag với biên bản. Với branch đã có commit riêng, đối chiếu commit xuất phát theo Git history.
3. Kiểm đúng ba module Foundation và fixture imports trước khi chạy. Dùng Python 3.10+ và pytest sẵn có trong môi trường của nhóm. Shared modules dùng stdlib, nên chạy bài này không cần model libraries.

```powershell
git status -sb
git branch --show-current
git rev-parse HEAD
git rev-list -n 1 v1.0.0-cnn-stable
```

Trong một PowerShell riêng, dùng lệnh giới hạn sau. `-B` tắt ghi bytecode, `--noconftest` bỏ external conftest và `-p no:cacheprovider` tắt pytest cache. [Test imports và mock consumers](../../tests/vlm_foundation/test_import_isolation.py)

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:PYTEST_ADDOPTS = ''
$env:PYTEST_PLUGINS = ''
python -B -m pytest --noconftest -o addopts= -p no:cacheprovider -q tests/vlm_foundation/test_contracts.py tests/vlm_foundation/test_evidence.py tests/vlm_foundation/test_import_isolation.py
```

Đọc theo thứ tự `fixtures.py` → `mocks.py` → test của consumer. `make_artifacts()` xây mapping synthetic với 30 canonical names, KB và hai banks trong bộ nhớ. `make_profile()` và các hàm mock tạo profile, ảnh và output minh họa. `evidence_fixtures.py` dựng run, manifest, invocation và review records synthetic. Các ví dụ PowerShell bên dưới thêm repository root và thư mục `tests` vào đường dẫn import của subprocess. [Fixtures](../../tests/vlm_foundation/fixtures.py) · [Mocks](../../tests/vlm_foundation/mocks.py) · [Evidence fixtures](../../tests/vlm_foundation/evidence_fixtures.py)

Import runtime contracts từ `src.vlm.contracts`. `ValidatedRgb` mang buffer RGB bất biến và digest của bytes upload gốc. `ApprovedProfile` mang pins và giới hạn do caller truyền. Evidence từ `src.vlm.evidence` có thêm run/cohort metadata. `project_runtime_input(...)` hiện chiếu RGB, operation và question. `EvalRef` điều phối evidence offline, còn expected answers, required sources, class labels, scorer annotations và private response index ở evaluator. Phần chiếu requested attributes theo operation đã khóa vẫn cần chốt trước khi dùng thật. [Quy ước dùng chung](../research/vlm/interface_contract.md#2-quy-ước-dùng-chung) · [Contract 06](../research/vlm/interface_contract.md#8-contract-06--evaluation-và-evidence)

Khi gặp blocker, ghi experiment hoặc handover bị ảnh hưởng, revision, fixture tái lập, lỗi hoặc quyền đang thiếu, evidence path và điều kiện gỡ chặn. Gửi trưởng nhóm hồ sơ pilot gồm mục tiêu, development inputs và audit, revisions đề xuất, nhu cầu máy, security review, kế hoạch đo và tiêu chí dừng. Chờ duyệt scope trước khi dùng input thật. [Pilot](../research/vlm/research_protocol.md#4-pilot) · [Nộp bàn giao](24_11_task.md#xi-nộp-bàn-giao-và-nghiệm-thu)

### Windows checkout and LF verification

Git có thể giữ CRLF trong canonical mapping khi cập nhật checkout Windows đã có sẵn. Blob của mapping giống nhau giữa Foundation và Task 08.04, nên Git có thể bỏ qua việc ghi lại file khi quy tắc `text eol=lf` mới có hiệu lực. Loader kiểm SHA-256 trên bytes thực tế của file.

Ở repository root, sau khi checkout revision bàn giao đã có `.gitattributes`, kiểm tra mapping trong PowerShell:

```powershell
git check-attr text eol -- data/metadata/class_mapping.json
git ls-files --eol -- data/metadata/class_mapping.json
Get-FileHash -Algorithm SHA256 -LiteralPath data/metadata/class_mapping.json
```

Checkout mới cần báo `i/lf w/lf attr/text eol=lf` và SHA-256 `d86b3a3a232babea289070299727d20d56b8f1607593ce1aaf8b174465690c3c`. Nếu checkout cũ còn `w/crlf`, chạy khối dưới đây khi đã dừng các thao tác Git và chỉnh sửa mapping trong checkout đó. Khối lệnh kiểm tra attributes, index stage 0 khớp HEAD, pin của blob và so sánh raw bytes sau khi đổi CRLF thành LF trong bộ nhớ. Khi đủ điều kiện, nó tạm cất bản cũ và yêu cầu Git ghi lại đúng một path.

```powershell
@'
from pathlib import Path
from hashlib import sha256
import subprocess
import tempfile

rel = "data/metadata/class_mapping.json"
pin = "d86b3a3a232babea289070299727d20d56b8f1607593ce1aaf8b174465690c3c"

def git(*args):
    return subprocess.check_output(["git", *args])

if git("rev-parse", "--show-prefix").strip():
    raise SystemExit("Run from the repository root.")
path = Path(rel)
if path.is_symlink() or not path.is_file():
    raise SystemExit("Expected a regular mapping file.")
if git("check-attr", "text", "eol", "--", rel).splitlines() != [
    f"{rel}: text: set".encode(), f"{rel}: eol: lf".encode()
]:
    raise SystemExit("Expected text eol=lf. Check the checkout revision and attributes.")

head = git("rev-parse", f"HEAD:{rel}").strip().decode()
entry = f"100644 {head} 0\t{rel}".encode()
if git("ls-files", "--stage", "--", rel).splitlines() != [entry]:
    raise SystemExit("Mapping has staged changes or an unresolved index entry.")
blob = git("cat-file", "blob", head)
if sha256(blob).hexdigest() != pin:
    raise SystemExit("HEAD mapping blob does not match the pinned SHA-256.")

raw = path.read_bytes()
if raw.replace(b"\r\n", b"\n") != blob:
    raise SystemExit("Mapping has local changes beyond CRLF/LF. Original preserved.")

def verify():
    eol = git("ls-files", "--eol", "--", rel).decode().split()
    if eol[:2] != ["i/lf", "w/lf"] or path.read_bytes() != blob:
        raise SystemExit("Mapping verification failed. Check the attributes and SHA-256.")
    if git("ls-files", "--stage", "--", rel).splitlines() != [entry]:
        raise SystemExit("Mapping index entry changed. Stop and check the index.")

print("SHA-256 before:", sha256(raw).hexdigest())
if raw != blob:
    backup_dir = Path(tempfile.mkdtemp(prefix=".mapping-lf-", dir=path.parent))
    backup = backup_dir / path.name
    try:
        if path.read_bytes() != raw:
            raise SystemExit("Mapping changed during verification. Original preserved.")
        path.rename(backup)
        git("checkout-index", "--index", "--", rel)
        verify()
    except BaseException:
        if backup.exists() and not path.exists() and not path.is_symlink():
            backup.rename(path)
        if backup.exists():
            print(f"Original preserved at {backup}", flush=True)
        else:
            backup_dir.rmdir()
        raise
    backup.unlink()
    backup_dir.rmdir()
else:
    verify()
print("SHA-256 after: ", sha256(path.read_bytes()).hexdigest())
'@ | python -B -
if ($LASTEXITCODE -ne 0) { throw 'Canonical mapping verification failed.' }
```

Nếu bước kiểm tra dừng hoặc hash vẫn lệch, giữ các bản hiện có, ghi lại HEAD, kết quả `git check-attr`, `git ls-files --stage`, `git ls-files --eol` của mapping và SHA-256 rồi báo trưởng nhóm. Nếu Git ghi file gặp lỗi, khối lệnh khôi phục bản cũ khi path còn trống hoặc in nơi giữ bản gốc. Có thể kiểm lại trên checkout mới của revision đã bàn giao.

Khi mapping đã ở LF và đúng pin, chạy toàn bộ suite trong PowerShell riêng:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:PYTEST_ADDOPTS = ''
$env:PYTEST_PLUGINS = ''
python -B -m pytest --noconftest -o addopts= -p no:cacheprovider -q tests/vlm_foundation
```

Ở head Task 08.04 `de338c6`, kết quả mong đợi là `413 passed`, exit code `0`.

## 3. Đức Anh — Task 08.02 · Zero-shot classifier

Thử hai pretrained image-text encoders với 30 prompt. E12-A/B dùng `google/siglip2-base-patch16-224`. E13-A/B dùng `qihoo360/fg-clip2-base`. A dùng Name Bank, B dùng Morphology Bank. Giữ cùng cohort và phương pháp scoring trong mỗi cặp. Score vector có 30 phần tử theo canonical ID `0–29`. Top-1/Top-5 xếp score giảm dần, hòa thì ID tăng dần. Output khai `calibrated=false`. [Phân công IV](24_11_task.md#iv-task-0802--lê-hồ-đức-anh)

| Đọc trước | Mục cần dùng |
| --- | --- |
| Architecture | [2.3 zero-shot](../research/vlm/architecture.md#23-phân-loại-zero-shot), [2.4 sequence](../research/vlm/architecture.md#24-zero-shot-classification-sequence) |
| Research Protocol | [3 dữ liệu](../research/vlm/research_protocol.md#3-dataset-và-ranh-giới-truy-cập), [4 pilot](../research/vlm/research_protocol.md#4-pilot), [5 E12/E13](../research/vlm/research_protocol.md#5-e12e13--zero-shot-classification), [7 benchmark](../research/vlm/research_protocol.md#7-efficiency-benchmark), [8 lock](../research/vlm/research_protocol.md#8-research-protocol-lock), [9.1 evidence](../research/vlm/research_protocol.md#91-evidence-cần-có-khi-thực-thi) |
| Registry | [E12-A](../research/vlm/experiment_registry.md#e12-a--siglip-2--name-prompt), [E12-B](../research/vlm/experiment_registry.md#e12-b--siglip-2--morphology-prompt), [E13-A](../research/vlm/experiment_registry.md#e13-a--fg-clip-2--name-prompt), [E13-B](../research/vlm/experiment_registry.md#e13-b--fg-clip-2--morphology-prompt), [dependencies](../research/vlm/experiment_registry.md#4-dependency-registry--task-0804), [evidence](../research/vlm/experiment_registry.md#6-evidence-checklist) |
| Contract và tiêu chí | [Quy ước chung](../research/vlm/interface_contract.md#2-quy-ước-dùng-chung), [01 KB/banks](../research/vlm/interface_contract.md#3-contract-01--breed-knowledge-base-và-prompt-banks), [03 classifier](../research/vlm/interface_contract.md#5-contract-03--zero-shot-vlm-classifier), [06 evidence](../research/vlm/interface_contract.md#8-contract-06--evaluation-và-evidence), [AC-0802-01–AC-0802-11](../research/vlm/acceptance_criteria.md#3-task-0802--đức-anh) |

Từ `contracts`, dùng `ZeroShotRequest` cho RGB, digest và pins, `ZeroShotResult` cho score/ranking hoặc failure, và `ZeroShotClassifier.classify(request, *, profile, prompts) -> ZeroShotResult` để thay mock bằng adapter. Hai keyword arguments là `ApprovedProfile` và `PromptSnapshot`. `rank_scores(scores)` cho thứ hạng tái lập. Từ `evidence`, dùng `ClassificationPrediction`, `validate_evidence_run_binding` và `validate_classification_cohort` để ghép prediction với run/cohort và giữ đủ failure/abstention rows. [Shared contracts](../../src/vlm/contracts.py) · [Evidence](../../src/vlm/evidence.py)

Ví dụ synthetic phân loại một ảnh và kiểm evidence của cohort fixture.

```powershell
@'
from pathlib import Path
import sys

root = Path.cwd()
sys.path[:0] = [str(root), str(root / "tests")]
from src.vlm.contracts import rank_scores, validate_zero_shot_result
from src.vlm.evidence import ClassificationPrediction, PredictionStatus, validate_classification_cohort
from vlm_foundation import evidence_fixtures as ef, fixtures as f, mocks as m

artifacts = f.make_artifacts()
profile = f.make_profile()
run, trusted = ef.make_run("E12-A", artifacts)
request = f.make_zero_shot_request(profile, artifacts.name, eval_ref=run.eval_ref)
result = m.MockZeroShotClassifier().classify(request, profile=profile, prompts=artifacts.name)
validate_zero_shot_result(result, request=request, profile=profile, prompts=artifacts.name, offline=True)
assert result.top5_class_ids == rank_scores(result.scores)[:5]
prediction = ClassificationPrediction(
    eval_ref=run.eval_ref, image_id=ef.IMAGE_ID, image_digest=request.image_digest,
    prediction_status=PredictionStatus.OK, model_result=result,
)
validate_classification_cohort((prediction,), run=run, trusted=trusted)
print("Synthetic Top-5:", result.top5_class_ids)
'@ | python -I -S -B -X utf8 -
```

1. Chạy Foundation tests, đọc mock classifier rồi thiết kế interface adapter bằng synthetic RGB.
2. Nghiên cứu native API, scoring và tài nguyên, phối hợp Thuận kiểm native tokenization theo profile và nhận từng bank qua H1. Với FG-CLIP 2, cần pin revision, review và duyệt security trước khi tải hoặc thực thi remote custom code. Nếu gate không đạt, ghi blocker E13-A/B.
3. Trình kế hoạch pilot. Sau lock riêng từng cặp, nộp `src/vlm/zero_shot.py`, `scripts/evaluate_vlm_zero_shot.py`, tests, benchmark, run evidence và profile proposal cho H4. Đường dẫn configs dự kiến gồm `configs/experiments/e12a_siglip2_name.yaml`, `configs/experiments/e12b_siglip2_morphology.yaml`, `configs/experiments/e13a_fg_clip2_name.yaml` và `configs/experiments/e13b_fg_clip2_morphology.yaml`. [Registry và đường dẫn outputs](../research/vlm/experiment_registry.md#51-định-danh-và-lưu-trữ)

## 4. Kiệt — Task 08.03 · Visual AI Assistant

Xây trợ lý trích xuất thuộc tính có kiểu và trả lời tiếng Việt. Ứng viên chính là `Qwen/Qwen3-VL-2B-Instruct`. `HuggingFaceTB/SmolVLM-500M-Instruct` là fallback sau pilot nếu được lock cho cả cặp. E14-A không có KB context. E14-B chỉ thay KB intervention, giữ cùng model, pipeline và shared inputs. [Phân công V](24_11_task.md#v-task-0803--nguyễn-anh-kiệt) · [Human evaluation IX](24_11_task.md#ix-phân-công-human-evaluation-e14)

| Đọc trước | Mục cần dùng |
| --- | --- |
| Architecture | [2.5 assistant và KB](../research/vlm/architecture.md#25-trợ-lý-ảnh-và-breed-knowledge-base), [2.6 sequence](../research/vlm/architecture.md#26-vlm-assistant-with-knowledge-retrieval-sequence) |
| Research Protocol | [3 dữ liệu](../research/vlm/research_protocol.md#3-dataset-và-ranh-giới-truy-cập), [4 pilot](../research/vlm/research_protocol.md#4-pilot), [6 E14](../research/vlm/research_protocol.md#6-e14--structured-visual-understanding-và-vqa), [7 benchmark](../research/vlm/research_protocol.md#7-efficiency-benchmark), [8 lock](../research/vlm/research_protocol.md#8-research-protocol-lock), [9.1 evidence](../research/vlm/research_protocol.md#91-evidence-cần-có-khi-thực-thi) |
| Registry | [E14-A](../research/vlm/experiment_registry.md#e14-a--generative-vlm-without-kb), [E14-B](../research/vlm/experiment_registry.md#e14-b--generative-vlm-with-kb), [ghép cặp](../research/vlm/experiment_registry.md#32-e14-b-versus-e14-a) |
| Contract và tiêu chí | [Quy ước chung](../research/vlm/interface_contract.md#2-quy-ước-dùng-chung), [01 KB/banks](../research/vlm/interface_contract.md#3-contract-01--breed-knowledge-base-và-prompt-banks), [02 retrieval](../research/vlm/interface_contract.md#4-contract-02--knowledge-retrieval), [04 assistant](../research/vlm/interface_contract.md#6-contract-04--generative-vlm-assistant), [06 evidence](../research/vlm/interface_contract.md#8-contract-06--evaluation-và-evidence), [AC-0803-01–AC-0803-12](../research/vlm/acceptance_criteria.md#4-task-0803--kiệt) |

| API công khai | Cách dùng |
| --- | --- |
| `AssistantRequest` và `AssistantResult` từ `contracts` | Request chọn `Operation.ATTRIBUTES`, `VQA` hoặc `ATTRIBUTES_AND_VQA`. Result giữ status, model/pin metadata và nội dung hợp lệ hoặc lỗi. |
| `GenerativeAssistant` từ `contracts` | `extract_attributes(request, *, profile)`, `answer_question(request, *, profile)` và `execute(request, *, profile)` đều nhận `AssistantRequest`, `ApprovedProfile` và trả `AssistantResult`. |
| `AttributeObservation`, `GeneratedContent`, `ContextProvenance` từ `contracts` | Observation mang attribute ID, visibility và value nullable theo vocabulary. GeneratedContent giữ nội dung sinh và source IDs. ContextProvenance giữ retrieval status, KB pin và SHA-256 của context UTF-8. |
| `ValidatedRetrieval` từ `contracts` | Đầu ra của `validate_retrieval_result(result, *, request, knowledge, token_counter, render_context, policy_mode=None)`. Truyền qua `retrieval=` của `validate_assistant_result(result, *, request, profile, policy_mode, mapping=None, retrieval=None, offline=False)` khi request có KB provenance. |
| `E14ModelInvocation`, `InvocationRef`, `AttributeInputRef` và `VqaOutputTrace` từ `evidence` | Một record cho mỗi invocation thực tế, một trace cho mỗi cặp ảnh–câu hỏi. `validate_e14_evidence(traces, invocations, *, runs, trusted_runs, captures, serialize_attributes=None, pair=None)` kiểm joins, snapshot links và coverage. |

E14-B giữ version/hash của KB đã duyệt ngay cả với miss, ambiguous, error hoặc `not_applicable`. Shared attributes-only extraction trước KB có thể không mang KB. Khi tái sử dụng attributes giữa A/B, dẫn cùng producer invocation và snapshot hash theo paired lock. Policy block sau extraction giữ extraction ref và VQA output null. Controller fallback giữ riêng khỏi model output khi chấm. [Stage và trace contract](../research/vlm/interface_contract.md#8-contract-06--evaluation-và-evidence) · [Tests](../../tests/vlm_foundation/test_evidence.py)

Ví dụ synthetic kết nối hai mock cho E14-B. Các pins, token count và quyền dùng trong ví dụ chỉ thuộc fixture.

```powershell
@'
from pathlib import Path
import sys

root = Path.cwd()
sys.path[:0] = [str(root), str(root / "tests")]
from src.vlm.contracts import Operation, PolicyMode, RetrievalStatus, validate_assistant_result, validate_retrieval_result
from vlm_foundation import fixtures as f, mocks as m

kb = f.make_artifacts().knowledge
policy = PolicyMode.BREED_CONTEXT_ALLOWED
lookup_request = f.make_retrieval_request(kb)
scripted = f.make_retrieval_result(kb, RetrievalStatus.HIT)
counter = m.MockContextTokenCounter({f.CONTEXT: 8})
lookup = m.MockKnowledgeRetriever(scripted)(lookup_request, knowledge=kb, count_context_tokens=counter)
bound = validate_retrieval_result(
    lookup, request=lookup_request, knowledge=kb, token_counter=counter,
    render_context=f.render_context, policy_mode=policy,
)
profile = f.make_profile("assistant")
request = f.make_assistant_request(
    profile, operation=Operation.VQA, retrieval=bound.result, eval_ref=f.make_eval_ref("E14-B"),
)
result = m.MockGenerativeAssistant().execute(request, profile=profile)
validate_assistant_result(result, request=request, profile=profile, policy_mode=policy, retrieval=bound, offline=True)
print("Synthetic retrieval:", result.context_provenance.retrieval_status.value)
'@ | python -I -S -B -X utf8 -
```

1. Chạy Foundation tests, đọc vocabulary, thử hợp lệ và sai schema bằng mock. Nghiên cứu loading/generation và trình pilot single-pass/multi-pass sau khi xác định điều kiện tài nguyên.
2. Nhận KB/retrieval theo H2. Từ H3, dùng approved images/questions qua projection của evaluator. Expected answers và nhãn chấm ở scorer. Trình lock cùng pipeline, shared inputs và scope cho A/B.
3. Nộp `src/vlm/assistant.py`, `scripts/evaluate_vlm_assistant.py`, `configs/experiments/e14a_vlm_without_kb.yaml`, `configs/experiments/e14b_vlm_with_kb.yaml`, tests và E14 evidence theo H5. KB thuộc Thuận và Gradio thuộc E15. [Phân công V](24_11_task.md#v-task-0803--nguyễn-anh-kiệt)

## 5. Thuận — Task 08.04 · Tri thức và dữ liệu đánh giá

Chuẩn bị nguồn tri thức và dữ liệu cho classifier, assistant và evaluator. Đọc [Phân công VI–IX](24_11_task.md#vi-task-0804--phạm-lê-nhật-thuận), Architecture [2.3](../research/vlm/architecture.md#23-phân-loại-zero-shot) và [2.5](../research/vlm/architecture.md#25-trợ-lý-ảnh-và-breed-knowledge-base), Research Protocol [3](../research/vlm/research_protocol.md#3-dataset-và-ranh-giới-truy-cập), [3.1](../research/vlm/research_protocol.md#31-thiết-kế-và-audit-vlm-eval), [6.2](../research/vlm/research_protocol.md#62-câu-hỏi-và-annotation) và [8](../research/vlm/research_protocol.md#8-research-protocol-lock). Tra [Registry mục 4](../research/vlm/experiment_registry.md#4-dependency-registry--task-0804), Contracts [01](../research/vlm/interface_contract.md#3-contract-01--breed-knowledge-base-và-prompt-banks), [02](../research/vlm/interface_contract.md#4-contract-02--knowledge-retrieval), [04 vocabulary](../research/vlm/interface_contract.md#6-contract-04--generative-vlm-assistant), [06](../research/vlm/interface_contract.md#8-contract-06--evaluation-và-evidence) và [AC-0804-01–AC-0804-07](../research/vlm/acceptance_criteria.md#5-task-0804--thuận).

| Deliverable | Đường dẫn dự kiến và điều kiện |
| --- | --- |
| 08.04-KB | `data/knowledge/vlm/breed_kb.json`. Đủ 30 canonical IDs, aliases và nguồn hình thái được review tới từng attribute. |
| 08.04-NAME | `data/knowledge/vlm/name_prompt_bank.json`. Một prompt mỗi lớp. |
| 08.04-MORPH | `data/knowledge/vlm/morphology_prompt_bank.json`. Giữ tên giống cùng thuộc tính đã review. Hai banks dùng `data/knowledge/vlm/prompt_tokenization_audit.json`, phối hợp Đức Anh audit bằng native tokenizer và trusted profile. |
| 08.04-RETRIEVAL | `src/vlm/retrieval.py`. Lookup nhiều giống, báo trạng thái và nguồn, giữ context trong budget. Loader dự kiến cho KB và banks là `src/vlm/knowledge_base.py`. |
| 08.04-VLM-EVAL | `vlm_eval_image` và `question` lưu trong store nội bộ được duyệt của Contract 06. Mục tiêu 80 ảnh và 240 câu hỏi chờ audit và phê duyệt. |
| 08.04-RUBRIC | Rubric có version, expected answers, evaluation criteria và `annotation` lưu ở store scorer-only được duyệt. |
| 08.04-AUDIT | Manifest và audit report ở store nội bộ được duyệt, ghi path khi giao H3. Báo usage rights, SHA-256/pHash, duplicates, near-duplicates và phạm vi kiểm. |

Dùng `CanonicalMapping`, `BreedKnowledgeBase`, `PromptBank`, `KnowledgeSnapshot` và `PromptSnapshot` từ `contracts`. Các hàm `decode_canonical_mapping`, `validate_knowledge_snapshot`, `validate_prompt_snapshot` và `validate_prompt_token_audit` kiểm bytes, provenance và profile pins. `RetrievalRequest`/`RetrievalResult` đi qua `validate_retrieval_request`/`validate_retrieval_result` để consumer nhận `ValidatedRetrieval`. Từ `evidence`, dùng `VlmEvalImage`, `Question`, `Annotation`, `EvaluationScope`, `validate_manifest`, `validate_disjoint_manifests` và `project_runtime_input`. [Shared contracts](../../src/vlm/contracts.py) · [Evidence](../../src/vlm/evidence.py)

Alias có thể trỏ đến nhiều giống. Giữ collision để báo `ambiguous`. `hit` có mục phù hợp, `miss` thiếu mục phù hợp, `partial_hit` chỉ tìm được một phần trong nhiều mục. Lookup alias đã biết không phân biệt được hết tên giống ngoài KB với câu hỏi không đề cập giống. Trước retrieval thật, thống nhất cách nhận diện trong phạm vi hỗ trợ và caller policy. Khi còn đối tượng được hỏi chưa xác định, không mặc định fallback sang CNN. [Contract 02](../research/vlm/interface_contract.md#4-contract-02--knowledge-retrieval)

Ví dụ synthetic kiểm lại KB, Morphology Bank và context do renderer fixture dựng độc lập với output retrieval.

```powershell
@'
from pathlib import Path
import sys

root = Path.cwd()
sys.path[:0] = [str(root), str(root / "tests")]
from src.vlm.contracts import PolicyMode, validate_knowledge_snapshot, validate_prompt_snapshot, validate_retrieval_result
from vlm_foundation import fixtures as f, mocks as m

artifacts = f.make_artifacts()
kb = validate_knowledge_snapshot(
    artifacts.knowledge_raw, mapping=artifacts.mapping, expected_sha256=artifacts.knowledge.kb_sha256,
    reviewed_sources=f.REVIEWED_SOURCES, allowed_attribute_ids={f.KB_ATTRIBUTE_ID},
)
bank = validate_prompt_snapshot(
    artifacts.morphology_raw, mapping=artifacts.mapping,
    expected_sha256=artifacts.morphology.prompt_bank_sha256, knowledge=kb,
)
request = f.make_retrieval_request(kb)
bound = validate_retrieval_result(
    f.make_retrieval_result(kb), request=request, knowledge=kb,
    token_counter=m.MockContextTokenCounter({f.CONTEXT: 8}), render_context=f.render_context,
    policy_mode=PolicyMode.BREED_CONTEXT_ALLOWED,
)
assert len(bank.data.entries) == 30 and bound.result.context == f.CONTEXT
print("Synthetic bank and retrieval context verified")
'@ | python -I -S -B -X utf8 -
```

Bắt đầu với [canonical mapping](../../data/metadata/class_mapping.json) và [synthetic KB fixtures](../../tests/vlm_foundation/fixtures.py). Đọc source references, dựng loader/lookup trên fixture rồi trình kế hoạch nguồn và audit. Giao từng phần H1, H2, H3 theo dependency đã review. Expected answers không đi vào KB hoặc banks. Công việc dùng model pretrained, không yêu cầu huấn luyện AI. [Phân công VI](24_11_task.md#vi-task-0804--phạm-lê-nhật-thuận)

## 6. Trưởng nhóm — Task 08.01 · Foundation và E15

Sau H0, duy trì shared contracts, review từng handover và duyệt pilot scope, security, tài nguyên và staged locks. Chỉ định reviewer, người phân xử và verifier. Kiểm provenance và research evidence trước quyết định profile. Chuẩn bị controller/Gradio và kiểm session, fallback, `/predict`, Grad-CAM cùng regression scope an toàn. Có thể chuẩn bị tích hợp song song bằng mocks, sau đó nối profile thật khi đã đủ deployment eligibility và dependency. Mỗi thành viên giữ quyền sở hữu deliverables của mình. [Phân công III](24_11_task.md#iii-task-0801--trưởng-nhóm) · [Architecture E15](../research/vlm/architecture.md#27-e15--integration-controller-và-ba-khu-vực-gradio) · [Contract 05](../research/vlm/interface_contract.md#7-contract-05--integration-và-session) · [Regression matrix](../research/vlm/acceptance_criteria.md#7-e15-integration-test-matrix)

Chuẩn bị human evaluation bằng `make_blinded_packet` và `validate_review_joins`. Kiệt quản lý private response index và packet preparation. Phân công dự kiến gồm trưởng nhóm và Thuận chấm độc lập, Đức Anh adjudicate. Trưởng nhóm xác nhận vai trò và rubric trước chấm. [Human evaluation IX](24_11_task.md#ix-phân-công-human-evaluation-e14) · [Evidence](../../src/vlm/evidence.py)

## 7. Handover H0–H6

| Mốc | Producer → consumer | Bàn giao và điều kiện | Xác nhận |
| --- | --- | --- | --- |
| H0 | Trưởng nhóm → Đức Anh, Kiệt, Thuận | Sáu tài liệu, Contract 01–06, schema version, fixture hashes và common commit sau merge. Trưởng nhóm đã nghiệm thu theo ngoại lệ xác minh. Publication và H0 còn pending. | Chờ common commit sau merge và xác nhận nhận bàn giao của ba consumer. |
| H1 | Thuận → Đức Anh | KB, banks và native token audit. Có thể giao Name Bank trước, Morphology Bank sau KB review. | Đức Anh xác nhận từng bank, trưởng nhóm duyệt revision dùng formal. |
| H2 | Thuận → Kiệt | KB pin, retrieval revision và lookup config. Có fixture thì phát triển trước phần thật. | Kiệt kiểm consumer binding, trưởng nhóm duyệt bản cho E14-B. |
| H3 | Thuận → Trưởng nhóm, E14 evaluator | Manifest, audit, rubric, expected answers và evaluation criteria theo quyền. Adapter nhận projection ảnh/câu hỏi, scorer giữ nhãn. | Trưởng nhóm duyệt/khóa, evaluator xác nhận đúng version và phạm vi. |
| H4 | Đức Anh → Trưởng nhóm, E15 | Encoder source, tests, E12/E13 evidence, benchmark và profile proposal sau lock/review của từng cặp. | Verifier kiểm evidence, trưởng nhóm quyết định nghiệm thu và eligibility. |
| H5 | Kiệt → Trưởng nhóm, E15 | Assistant source, tests, E14 evidence, human review và profile proposal sau H2/H3 và E14 lock. | Verifier kiểm evidence, trưởng nhóm quyết định nghiệm thu và eligibility. |
| H6 | Trưởng nhóm → Nhóm dự án | Tích hợp E15, profiles đủ điều kiện, policy, regression trong scope đã duyệt và system evidence. | Verifier độc lập kiểm hồi quy, trưởng nhóm ký hoặc ghi blocker. |

Bàn giao H1/H2 và phần H3 đủ điều kiện bằng reviewed Git revisions hoặc dependency PRs. Ghi artifact path hoặc ref, revision/hash, criterion IDs, quyền, blocker và xác nhận. Consumer dùng revision đã duyệt với đúng phạm vi, không cần sao chép file qua trưởng nhóm. D1–D7 là mục tiêu sau H0, phụ thuộc scope, machine time và gate riêng của từng nhánh. [Bảng bàn giao gốc](24_11_task.md#vii-phụ-thuộc-và-mốc-bàn-giao-h0h6) · [Timeline VIII](24_11_task.md#viii-timeline-đề-xuất)

## 8. Git workflow và quy tắc an toàn

| PR source | PR target | Gate |
| --- | --- | --- |
| `feature/vlm-shared-contracts` | `main` | Leader acceptance đã ghi nhận theo ngoại lệ xác minh. Chờ publication và PR merge. |
| `research/vlm-zero-shot` | `integration/hybrid-cnn-vlm` | Theo reviewed dependencies và phạm vi H4. |
| `feature/vlm-assistant` | `integration/hybrid-cnn-vlm` | Theo reviewed dependencies và phạm vi H5. |
| `research/vlm-knowledge-base` | `integration/hybrid-cnn-vlm` | Có thể nộp theo từng deliverable đủ điều kiện. |
| `integration/hybrid-cnn-vlm` | `main` | Sau E15 system acceptance và approval. |

Cả ba component branches và integration branch xuất phát từ cùng `<FOUNDATION_MERGE_SHA>` chứa planning documents lẫn Foundation. Placeholder chỉ được thay sau khi Foundation PR đã merge và SHA được công bố. [Git workflow X](24_11_task.md#x-git-workflow-đề-xuất)

- Giữ E4/E6/E7/E8, Energy OOD, Grad-CAM, `/predict`, canonical mapping và Configuration Lock. Official Test, protected bundles và protected artifacts nằm ngoài scope.
- Không training/fine-tuning ngoài scope. Chỉ chạy pilot thật khi có approval và chỉ chạy formal khi cặp tương ứng có Protocol Lock.
- Phiên formal giữ pins đã khóa. Thay prompt, KB hay config sau khi thấy formal results phải theo quy tắc version mới và exploratory trong protocol.
- Giữ failures và blockers trong evidence. Synthetic fixtures phục vụ phát triển và test contracts. Giữ data, weights và private outputs trong store được phép. Giữ credentials ngoài Git và khỏi worker mounts theo scope. [Ranh giới dữ liệu](../research/vlm/research_protocol.md#3-dataset-và-ranh-giới-truy-cập) · [Quy tắc lock](../research/vlm/research_protocol.md#8-research-protocol-lock)
