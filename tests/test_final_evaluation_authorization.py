"""B2.2-A component tests only: fake Git, buffers/runtime and temp claim roots.

No real approval, output claim, dataset, checkpoint or CUDA execution is used.
Fixture anchors are deliberately manufactured as a trusted test context; they
are not evidence of leader approval for the actual repository. Source validation
uses real Git/B1 helpers, not a mocked PASS audit. Adapter calls below are fake
orchestration positive controls, not an implemented Official execution path.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import pytest

from scripts import run_final_evaluation as b1
from src.final_evaluation import authorization as auth


def raw(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE)


def commit(root):
    git(root, 'add', '.')
    git(root, '-c', 'user.name=B2 fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fake fixture')
    return git(root, 'rev-parse', 'HEAD').decode().strip()


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path / 'checkout'
    names = (*b1.SOURCE_FILES, *auth.EXTRA_SOURCE)
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'# artificial source: ' + name.encode() + b'\n')
    git(root, 'init', '-q')
    git(root, 'config', 'core.autocrlf', 'false')
    revision = commit(root)
    monkeypatch.setattr(b1, 'SOURCE_ROOT', root)
    claim_root = tmp_path / 'claims'
    claim_root.mkdir()
    output_root = tmp_path / 'results'
    request = auth.RunRequest('run-01', str(output_root / 'run-01'))
    lock = '**Trạng thái:** ĐÃ KHÓA CẤU HÌNH\n'.encode('utf-8')
    protocol = raw(auth.approved_protocol())
    evidence = {'leader-lock': b'FAKE explicit lock decision', 'leader-grant': b'FAKE explicit grant decision',
                'accepted-phase-a': b'FAKE accepted Phase A PASS', 'accepted-phase-b1': b'FAKE accepted B1 PASS'}
    ref = lambda name: dict(reference=name, sha256=digest(evidence[name]))
    runtime = dict(python='3.12.4', pytorch='2.6.0+cu124', torchvision='0.21.0+cu124',
                   numpy='2.5.3', cuda_build='12.4', device='cuda', batch_size=32, num_workers=0)
    frozen = {name: b'FAKE frozen reference: ' + name.encode() for name in auth.FROZEN_REFERENCES}
    approval = dict(schema=auth.APPROVAL_SCHEMA, approval_id='lock-01', status='APPROVED', approver='Fixture leader',
                    approval_evidence=ref('leader-lock'), source_revision=revision,
                    source_sha256={name: digest((root / name).read_bytes()) for name in names},
                    protocol_sha256=digest(protocol), lock_sha256=digest(lock),
                    frozen_sha256={name: digest(value) for name, value in frozen.items()},
                    acceptance=dict(phase_a=ref('accepted-phase-a'), phase_b1=ref('accepted-phase-b1')),
                    runtime=runtime,
                    dataset_descriptor=dict(reference='FAKE opaque canonical descriptor; never a data input',
                                            revision=revision, expected_samples=1934, num_classes=30))
    grant = dict(schema=auth.AUTHORIZATION_SCHEMA, authorization_id='grant-01', status='GRANTED', grantor='Fixture leader',
                 grant_evidence=ref('leader-grant'), approval_id=approval['approval_id'], approval_sha256=digest(raw(approval)),
                 run_id=request.run_id, scope=list(auth.SCOPE), output_dir=request.output_dir,
                 claim_root=str(claim_root), max_runs=1, automatic_retry=False)
    inputs = auth.AuthorizationInputs(lock, raw(approval), raw(grant), protocol, frozen, evidence, dict(runtime))
    anchors = auth.TrustAnchors(digest(inputs.approval), digest(inputs.authorization), revision, digest(protocol),
                               str(claim_root), str(output_root))
    return dict(root=root, inputs=inputs, anchors=anchors, request=request, approval=approval, grant=grant)


def validate(case):
    return auth.validate_authorization(case['inputs'], case['anchors'], case['request'])


def codes(decision):
    return {reason.code for reason in decision.reasons}


def reseal(case, *, approval=None, grant=None):
    """Simulate an independently reviewed fixture update, not untrusted rehash."""
    approval = deepcopy(case['approval'] if approval is None else approval)
    grant = deepcopy(case['grant'] if grant is None else grant)
    grant['approval_sha256'] = digest(raw(approval))
    case['approval'], case['grant'] = approval, grant
    case['inputs'] = replace(case['inputs'], approval=raw(approval), authorization=raw(grant))
    case['anchors'] = replace(case['anchors'], approval_sha256=digest(raw(approval)), authorization_sha256=digest(raw(grant)))


def fake_orchestration(case, adapter):
    decision = validate(case)
    if decision.status != 'PASS':
        return decision.status
    claim = auth.claim_authorization(decision)
    if claim.status == 'PASS':
        adapter()
    return claim.status


def test_positive_control_validates_claims_once_and_only_then_calls_fake_adapter(case):
    calls = []
    assert fake_orchestration(case, lambda: calls.append('FAKE input')) == 'PASS'
    assert calls == ['FAKE input']
    assert not Path(case['request'].output_dir).exists()  # No output reservation/inference component yet.
    assert fake_orchestration(case, lambda: calls.append('unexpected')) == 'BLOCKED'
    assert calls == ['FAKE input']
    receipt = json.loads((Path(case['anchors'].claim_root) / 'grant-01.claim.json').read_bytes())
    assert receipt['run_id'] == case['request'].run_id and receipt['state'] == 'CLAIMED'
    assert receipt['authorization_sha256'] == case['anchors'].authorization_sha256


@pytest.mark.parametrize('missing', ['approval', 'authorization'])
def test_missing_record_and_phase_pass_evidence_do_not_grant_access(case, missing):
    case['inputs'] = replace(case['inputs'], **{missing: None})
    decision = validate(case)
    assert decision.status == 'BLOCKED' and missing.upper() + '_MISSING' in codes(decision)
    assert fake_orchestration(case, lambda: pytest.fail('fake Official I/O before authorization')) == 'BLOCKED'
    assert list(Path(case['anchors'].claim_root).iterdir()) == []


def test_draft_lock_blocks_without_source_audit_or_fake_input(case, monkeypatch):
    case['inputs'] = replace(case['inputs'], lock_document='**Trạng thái:** BẢN NHÁP — CHƯA KHÓA CẤU HÌNH\n'.encode('utf-8'))
    monkeypatch.setattr(auth, 'capture_source_binding', lambda *a: pytest.fail('premature source audit'))
    decision = validate(case)
    assert decision.status == 'BLOCKED' and 'LOCK_DRAFT' in codes(decision)
    assert fake_orchestration(case, lambda: pytest.fail('fake Official I/O')) == 'BLOCKED'


@pytest.mark.parametrize('field', ['approval_sha256', 'authorization_sha256', 'source_revision', 'protocol_sha256'])
def test_missing_independent_anchor_blocks(case, field):
    case['anchors'] = replace(case['anchors'], **{field: None})
    assert validate(case).status == 'BLOCKED'


@pytest.mark.parametrize('record', ['approval', 'authorization'])
def test_rehashing_incoming_record_without_changing_trusted_anchor_fails(case, record):
    value = json.loads(getattr(case['inputs'], record))
    value['approver' if record == 'approval' else 'grantor'] = 'Another actor'
    case['inputs'] = replace(case['inputs'], **{record: raw(value)})
    decision = validate(case)
    assert decision.status == 'FAIL' and record.upper() + '_HASH_MISMATCH' in codes(decision)


@pytest.mark.parametrize('bad', [b'{', b'[]', b'{"status":"APPROVED","status":"REVOKED"}', b'{"value":NaN}'])
def test_malformed_or_duplicate_json_fails_even_with_anchored_bytes(case, bad):
    case['inputs'] = replace(case['inputs'], approval=bad)
    case['anchors'] = replace(case['anchors'], approval_sha256=digest(bad))
    decision = validate(case)
    assert decision.status == 'FAIL' and 'APPROVAL_MALFORMED' in codes(decision)


@pytest.mark.parametrize('change', ['source', 'protocol', 'source_hash', 'frozen_hash', 'empty_source', 'empty_frozen', 'acceptance', 'descriptor'])
def test_anchored_approval_still_needs_consistent_bindings(case, change):
    a = deepcopy(case['approval'])
    if change == 'source': a['source_revision'] = '0' * 40
    if change == 'protocol': a['protocol_sha256'] = '0' * 64
    if change == 'source_hash': a['source_sha256'][auth.EXTRA_SOURCE[1]] = '0' * 64
    if change == 'frozen_hash': a['frozen_sha256'][auth.FROZEN_REFERENCES[0]] = '0' * 64
    if change == 'empty_source': a['source_sha256'] = {}
    if change == 'empty_frozen': a['frozen_sha256'] = {}
    if change == 'acceptance': a['acceptance'] = {}
    if change == 'descriptor': a['dataset_descriptor'] = {}
    reseal(case, approval=a)
    assert validate(case).status == 'FAIL'


@pytest.mark.parametrize('parameter', ['temperature', 'threshold', 'checkpoint', 'preprocessing', 'blur', 'noise', 'occlusion', 'bool_type'])
def test_protocol_values_are_exact_even_when_record_and_anchors_are_rebound(case, parameter):
    value = auth.approved_protocol()
    if parameter in ('temperature', 'threshold'): value[parameter] = float(np.nextafter(value[parameter], 1.0))
    if parameter == 'checkpoint': value['checkpoint_sha256'] = '0' * 64
    if parameter == 'preprocessing': value['preprocessing']['resize'] = 255
    if parameter == 'blur': value['e8']['gaussian_blur']['sigma'] = 1.6
    if parameter == 'noise': value['e8']['gaussian_noise']['std'] = 0.06
    if parameter == 'occlusion': value['e8']['occlusion']['side_fraction'] = 0.31
    if parameter == 'bool_type': value['model']['backbone_frozen'] = 1
    case['inputs'] = replace(case['inputs'], protocol=raw(value))
    case['anchors'] = replace(case['anchors'], protocol_sha256=digest(raw(value)))
    a = deepcopy(case['approval']); a['protocol_sha256'] = digest(raw(value))
    reseal(case, approval=a)
    decision = validate(case)
    assert decision.status == 'FAIL' and 'FROZEN_PARAMETERS_MISMATCH' in codes(decision)


@pytest.mark.parametrize('flag', ['--assume-unchanged', '--skip-worktree'])
def test_additional_b2_source_hidden_by_git_index_flags_is_rejected(case, flag):
    name = auth.EXTRA_SOURCE[1]
    git(case['root'], 'update-index', flag, name)
    path = case['root'] / name
    path.write_bytes(path.read_bytes() + b'# changed authorization code\n')
    assert git(case['root'], 'diff', 'HEAD', '--name-only', '--', name).strip() == b''
    decision = validate(case)
    assert decision.status == 'FAIL' and 'SOURCE_INTEGRITY_MISMATCH' in codes(decision)


def test_untracked_required_source_blocks_and_unrelated_outputs_do_not(case):
    history = case['root'] / 'outputs/history/old.txt'
    history.parent.mkdir(parents=True); history.write_bytes(b'historical untracked output')
    assert validate(case).status == 'PASS'
    assert history.read_bytes() == b'historical untracked output'
    git(case['root'], 'rm', '--cached', auth.EXTRA_SOURCE[1])
    git(case['root'], '-c', 'user.name=B2 fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'untrack fake source')
    assert validate(case).status == 'BLOCKED'


def test_doc_only_head_after_pinned_source_is_accepted(case):
    (case['root'] / 'acceptance-note.md').write_bytes(b'fake documentation-only commit\n')
    later = commit(case['root'])
    assert later != case['anchors'].source_revision
    assert validate(case).status == 'PASS'


@pytest.mark.parametrize('problem', ['runtime', 'evidence', 'frozen_buffer', 'lock_hash'])
def test_supplied_actual_buffers_and_runtime_must_match_reviewed_bindings(case, problem):
    if problem == 'runtime':
        case['inputs'] = replace(case['inputs'], runtime=dict(case['inputs'].runtime, numpy='changed'))
    if problem == 'evidence':
        case['inputs'] = replace(case['inputs'], evidence=dict(case['inputs'].evidence, **{'leader-grant': b'not the decision'}))
    if problem == 'frozen_buffer':
        refs = dict(case['inputs'].frozen_references); refs[auth.FROZEN_REFERENCES[0]] += b'changed'
        case['inputs'] = replace(case['inputs'], frozen_references=refs)
    if problem == 'lock_hash':
        case['inputs'] = replace(case['inputs'], lock_document=case['inputs'].lock_document + b'changed\n')
    assert validate(case).status == 'FAIL'


@pytest.mark.parametrize('state', ['PENDING', 'REVOKED'])
def test_ungranted_status_blocks(case, state):
    grant = deepcopy(case['grant']); grant['status'] = state
    reseal(case, grant=grant)
    assert validate(case).status == 'BLOCKED'


@pytest.mark.parametrize('problem', ['run_id', 'scope', 'output_dir', 'max_runs', 'automatic_retry'])
def test_request_and_single_run_policy_cannot_be_changed(case, problem):
    grant = deepcopy(case['grant'])
    grant[problem] = {'run_id':'other-run', 'scope':['E4'], 'output_dir':str(Path(case['request'].output_dir).parent / 'other'),
                      'max_runs':2, 'automatic_retry':True}[problem]
    reseal(case, grant=grant)
    assert validate(case).status == 'FAIL'


def test_fail_has_priority_over_blocked_and_rejected_decision_cannot_claim(case):
    case['inputs'] = replace(case['inputs'], authorization=b'bad', lock_document=None)
    decision = validate(case)
    assert decision.status == 'FAIL'
    assert {'LOCK_MISSING','AUTHORIZATION_MALFORMED'} <= codes(decision)
    assert auth.claim_authorization(decision).status == 'FAIL'
    assert list(Path(case['anchors'].claim_root).iterdir()) == []


@pytest.mark.parametrize('problem', ['evidence', 'frozen_references', 'runtime', 'lock_document'])
def test_bad_binding_types_fail_closed_instead_of_crashing_or_accepting_mutable_bytes(case, problem):
    value = bytearray(case['inputs'].lock_document) if problem == 'lock_document' else None
    case['inputs'] = replace(case['inputs'], **{problem: value})
    assert validate(case).status == 'FAIL'


@pytest.mark.parametrize('kind', ['one_none', 'all_none', 'text', 'integer', 'mutable', 'empty'])
def test_frozen_regression_invalid_values_never_produce_grant_claim_or_adapter_call(case, kind):
    original = case['inputs']
    anchors = case['anchors']
    name = auth.FROZEN_REFERENCES[0]
    refs = dict(original.frozen_references)
    if kind == 'all_none':
        refs = dict.fromkeys(refs, None)
    else:
        refs[name] = {'one_none': None, 'text': refs[name].decode('utf-8'), 'integer': 0,
                      'mutable': bytearray(refs[name]), 'empty': b''}[kind]
    assert set(refs) == set(original.frozen_references)
    case['inputs'] = replace(original, frozen_references=refs)
    decision = validate(case)
    assert decision.status == 'FAIL'
    assert 'FROZEN_REFERENCE_HASH_MISMATCH' in codes(decision)
    assert decision.grant is None
    claim = auth.claim_authorization(decision)
    assert claim.status == 'FAIL' and claim.code == 'CLAIM_REQUIRES_VALIDATION'
    calls = []
    assert fake_orchestration(case, lambda: calls.append('unexpected input access')) == 'FAIL'
    assert calls == [] and list(Path(anchors.claim_root).iterdir()) == []
    assert case['inputs'].approval == original.approval
    assert case['inputs'].authorization == original.authorization
    assert case['anchors'] == anchors


def test_frozen_regression_missing_key_remains_blocked(case):
    refs = dict(case['inputs'].frozen_references)
    refs.pop(auth.FROZEN_REFERENCES[0])
    case['inputs'] = replace(case['inputs'], frozen_references=refs)
    decision = validate(case)
    assert decision.status == 'BLOCKED'
    assert 'FROZEN_REFERENCES_MISSING' in codes(decision)
    assert 'FROZEN_REFERENCE_HASH_MISMATCH' not in codes(decision)
    assert decision.grant is None
    assert auth.claim_authorization(decision).status == 'BLOCKED'
    assert list(Path(case['anchors'].claim_root).iterdir()) == []


def test_frozen_regression_empty_placeholder_is_invalid_even_with_matching_digest(case):
    name = auth.FROZEN_REFERENCES[0]
    refs = dict(case['inputs'].frozen_references)
    refs[name] = b''
    case['inputs'] = replace(case['inputs'], frozen_references=refs)
    approval = deepcopy(case['approval'])
    approval['frozen_sha256'][name] = digest(b'')
    reseal(case, approval=approval)
    decision = validate(case)
    assert decision.status == 'FAIL'
    assert 'FROZEN_REFERENCE_HASH_MISMATCH' in codes(decision)
    assert decision.grant is None


def test_dataset_descriptor_is_never_resolved_or_opened(case, monkeypatch):
    a = deepcopy(case['approval']); a['dataset_descriptor']['reference'] = 'DO_NOT_TOUCH/opaque-reference'
    reseal(case, approval=a)
    original = Path.stat
    def stat(path, *args, **kwargs):
        assert 'DO_NOT_TOUCH' not in str(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', stat)
    assert validate(case).status == 'PASS'


def test_failure_after_claim_does_not_automatically_reuse_grant(case):
    def fail():
        raise RuntimeError('fake execution failure after claim')
    with pytest.raises(RuntimeError, match='after claim'):
        fake_orchestration(case, fail)
    assert fake_orchestration(case, lambda: pytest.fail('automatic retry')) == 'BLOCKED'


def test_failed_receipt_write_spends_claim_and_preserves_file(case, monkeypatch):
    decision = validate(case)
    def fail(*args): raise OSError('fake disk failure')
    with monkeypatch.context() as patch:
        patch.setattr(os, 'fsync', fail)
        result = auth.claim_authorization(decision)
    assert result.status == 'FAIL' and result.code == 'CLAIM_WRITE_FAILED'
    old = result.claim_path.read_bytes()
    assert auth.claim_authorization(decision).code == 'AUTHORIZATION_ALREADY_CLAIMED'
    assert result.claim_path.read_bytes() == old


def test_preexisting_foreign_claim_is_never_overwritten_or_deleted(case):
    decision = validate(case)
    path = Path(case['anchors'].claim_root) / 'grant-01.claim.json'
    path.write_bytes(b'foreign writer or a previous partial receipt')
    assert auth.claim_authorization(decision).status == 'BLOCKED'
    assert path.read_bytes() == b'foreign writer or a previous partial receipt'


def test_switching_claim_root_cannot_reuse_grant(case, tmp_path):
    assert auth.claim_authorization(validate(case)).status == 'PASS'
    other = tmp_path / 'other-claims'
    other.mkdir()
    case['anchors'] = replace(case['anchors'], claim_root=str(other))
    decision = validate(case)
    assert decision.status == 'FAIL' and 'CLAIM_ROOT_BINDING_MISMATCH' in codes(decision)
    assert list(other.iterdir()) == []


def test_two_processes_cannot_claim_the_same_validated_grant(case, tmp_path):
    decision = validate(case)
    assert decision.status == 'PASS'
    receipt = tmp_path / 'controlled-test-decision.pickle'
    receipt.write_bytes(pickle.dumps(decision))  # Only self-created trusted fixture bytes.
    go = tmp_path / 'go'
    code = '''
import pickle,sys,time
from pathlib import Path
from src.final_evaluation.authorization import claim_authorization
decision=pickle.loads(Path(sys.argv[1]).read_bytes())
Path(sys.argv[3]).write_bytes(b'ready')
while not Path(sys.argv[2]).exists(): time.sleep(0.01)
result=claim_authorization(decision)
print(result.status, result.code)
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    ready = [tmp_path / f'ready-{i}' for i in range(2)]
    processes = [subprocess.Popen([sys.executable,'-B','-c',code,str(receipt),str(go),str(path)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env) for path in ready]
    try:
        deadline = time.monotonic() + 30
        while not all(path.exists() for path in ready) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert all(path.exists() for path in ready)
        go.write_bytes(b'go')
        results = [process.communicate(timeout=30) for process in processes]
        assert all(process.returncode == 0 for process in processes), results
        outputs = sorted(out.decode().strip() for out, err in results)
        assert outputs == ['BLOCKED AUTHORIZATION_ALREADY_CLAIMED', 'PASS AUTHORIZATION_CLAIMED']
        assert len(list(Path(case['anchors'].claim_root).iterdir())) == 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
