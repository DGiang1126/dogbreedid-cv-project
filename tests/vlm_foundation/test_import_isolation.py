"""Verify the Foundation and mock consumers without site packages."""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]

ISOLATED_SMOKE = r"""
from importlib.abc import MetaPathFinder
from pathlib import Path
import sys


FORBIDDEN = {"torch", "torchvision", "transformers", "timm", "accelerate", "gradio", "numpy", "PIL"}


class BlockModelImports(MetaPathFinder):
    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in FORBIDDEN:
            self.attempts.append(fullname)
            raise ModuleNotFoundError(fullname)
        return None


def reject_network(event, args):
    if event.startswith("socket."):
        network_attempts.append(event)
        raise AssertionError("network use is forbidden")


blocker = BlockModelImports()
sys.meta_path.insert(0, blocker)
network_attempts = []
sys.addaudithook(reject_network)
root = Path(sys.argv[1])
sys.path[:0] = [str(root), str(root / "tests")]

import src.vlm
from src.vlm import contracts as c, evidence as e
from vlm_foundation import evidence_fixtures as ef
from vlm_foundation import fixtures as f
from vlm_foundation import mocks as m

artifacts = f.make_artifacts()
profile = f.make_profile()
request = f.make_zero_shot_request(profile, artifacts.name)
job = f.make_worker_job(request)
for fail in (False, True):
    result = m.MockZeroShotClassifier(fail=fail).classify(request, profile=profile, prompts=artifacts.name)
    c.validate_zero_shot_result(result, request=request, profile=profile, prompts=artifacts.name)
    reply = c.decode_record(c.encode_record(f.make_worker_reply(request, result)), c.WorkerReply)
    c.validate_worker_identity(job, reply)
assert c.rank_scores(f.TIED_SCORES)[:5] == (1, 2, 0, 3, 4)

lookup_request = f.make_retrieval_request(artifacts.knowledge)
lookup = f.make_retrieval_result(artifacts.knowledge, c.RetrievalStatus.PARTIAL_HIT)
counter = m.MockContextTokenCounter({f.CONTEXT: 8})
lookup = m.MockKnowledgeRetriever(lookup)(lookup_request, knowledge=artifacts.knowledge, count_context_tokens=counter)
validated = c.validate_retrieval_result(lookup, request=lookup_request, knowledge=artifacts.knowledge,
                                        token_counter=counter, render_context=f.render_context,
                                        policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED)
assert validated.result.context == f.render_context(validated.result.entries) == f.CONTEXT

profile = f.make_profile("assistant")
request = f.make_assistant_request(profile, retrieval=validated.result)
for fail in (False, True):
    result = m.MockGenerativeAssistant(fail=fail).execute(request, profile=profile)
    c.validate_assistant_result(result, request=request, profile=profile, retrieval=validated,
                                policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED)
    if not fail:
        assert c.decode_generated_content(c.encode_record(result.generated), request=request) == result.generated

encoder, trusted_encoder = ef.make_run("E12-A", artifacts)
prediction = ef.prediction(encoder, artifacts)
e.validate_classification_cohort((e.decode_evidence_record(c.encode_record(prediction)),),
                                 run=encoder, trusted=trusted_encoder)
e14, trusted_e14 = ef.make_run("E14-B", artifacts)
answer = ef.make_stage(e14, artifacts, retrieval_status=c.RetrievalStatus.HIT)
trace = ef.trace_for(e14, answer=answer)
e.validate_e14_evidence((trace,), (answer.record,), runs=(e14,), trusted_runs={e14.eval_ref: trusted_e14},
                        captures={answer.record.ref: answer.capture})
rubric = ef.review_rubric()
index, packet, labels, final = ef.review_records(trace, rubric)
e.validate_review_joins((trace,), (index,), (packet,), labels, (final,), rubric=rubric,
                        manifest=trusted_e14.manifest, review_image_refs={ef.IMAGE_ID: ef.REVIEW_REF})
assert packet.content_for_review == answer.result.generated
projection = e.project_runtime_input(trusted_e14.manifest, image_id=ef.IMAGE_ID, image=ef.make_image(),
                                     operation=c.Operation.VQA, question_id=ef.QUESTION_ID)
assert projection.question == f.QUESTION

assert not blocker.attempts, blocker.attempts
assert not network_attempts, network_attempts
assert not FORBIDDEN.intersection(name.partition(".")[0] for name in sys.modules)
print("isolated Foundation and mock consumers passed")
"""


def test_foundation_imports_and_mock_consumers_are_isolated():
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", ISOLATED_SMOKE, str(ROOT)],
        cwd=ROOT, capture_output=True, encoding="utf-8", timeout=15, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "isolated Foundation and mock consumers passed" in completed.stdout
