"""B2 boundary tests: independent fake Git releases, CPU fixtures, zero real Test.

The synthetic checkpoint is a dummy model fixture, not the E4 checkpoint. All
scientific protocol constants stay unchanged. Fake Official controls/claims and
results below exist exclusively in temporary fixtures, never project permission.
Authorization, source/hash validators, math, corruption and consumers run for real.
"""
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

import pytest

from src.final_evaluation import authorization as auth
from src.final_evaluation import official_execution as execution

b1 = execution.b1
REAL_ROOT = execution.REAL_ROOT


def raw(value):
    return b1._json_bytes(value)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE)


def commit(root):
    git(root, 'add', '.')
    git(root, '-c', 'user.name=B2 integration fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-qm', 'synthetic fixture only')
    return git(root, 'rev-parse', 'HEAD').decode().strip()


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def seal(case):
    root = case['root']
    approval, grant = case['approval'], case['grant']
    grant['approval_sha256'] = sha(raw(approval))
    index = dict(schema='dogbreedid.reviewed-controls.v1', source_revision=case['source_revision'],
                 approval_sha256=sha(raw(approval)), authorization_sha256=sha(raw(grant)),
                 protocol_sha256=sha(case['protocol']), claim_root=str(case['claims']),
                 output_root=str(case['output_root']), profile=case['profile'])
    for label, body in (('approval',raw(approval)),('authorization',raw(grant)),('protocol',case['protocol'])):
        write(root, execution.CONTROL_FILES[label], body)
    write(root, execution.CONTROL_INDEX, raw(index))
    case['release'] = execution.ReviewedRelease(commit(root))


@pytest.fixture(autouse=True)
def single_thread():
    old = b1.torch.get_num_threads()
    b1.torch.set_num_threads(1)
    yield
    b1.torch.set_num_threads(old)


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path / 'checkout'
    source_names = (*b1.SOURCE_FILES, *auth.EXTRA_SOURCE, *execution.INTEGRATION_SOURCE)
    for name in source_names:
        write(root, name, b'# artificial Git source fixture: ' + name.encode() + b'\n')
    frozen = {name:(REAL_ROOT / name).read_bytes() for name in auth.FROZEN_REFERENCES}
    for name, value in frozen.items():
        write(root, name, value)
    lock = '**Trạng thái:** ĐÃ KHÓA CẤU HÌNH\n'.encode('utf-8')
    write(root, execution.LOCK, lock)
    evidence = {name:('FAKE separately reviewed '+name+' decision/evidence\n').encode() for name in execution.EVIDENCE_FILES}
    for name, value in evidence.items():
        write(root, execution.EVIDENCE_FILES[name], value)
    cp = root / b1.predictor.DEFAULT_CHECKPOINT
    cp.parent.mkdir(parents=True, exist_ok=True)
    model = b1.robustness._make_synthetic_model(42)
    b1.torch.save(model.state_dict(), cp)
    cp_hash = sha(cp.read_bytes())
    git(root, 'init', '-q')
    git(root, 'config', 'core.autocrlf', 'false')
    source_revision = commit(root)
    monkeypatch.setattr(b1, 'SOURCE_ROOT', root)
    data = tmp_path / 'artificial-data'
    data.mkdir()
    mapping = json.loads(frozen['data/metadata/class_mapping.json'])
    csv_text = 'relative_path,class_id,breed_name,source_split\n'
    image_hashes = {}
    for i in range(4):
        name = f'synthetic/image_{i:03d}.png'
        path = data / 'images' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        b1.Image.new('RGB',(12,12),(i*50,80,180)).save(path)
        image_hashes[name] = sha(path.read_bytes())
        csv_text += f'{name},{i},{mapping[str(i)]},synthetic\n'
    (data / 'split.csv').write_bytes(csv_text.encode())
    declaration = raw(dict(synthetic=True,n_samples=4,split_sha256=sha(csv_text.encode()),images=image_hashes))
    (data / 'fixture_manifest.json').write_bytes(declaration)
    claims = tmp_path / 'claims'
    claims.mkdir()
    output_root = tmp_path / 'results'
    request = auth.RunRequest('run-01',str(output_root / 'run-01'))
    protocol = raw(auth.approved_protocol())
    ref = lambda name: dict(reference=name,sha256=sha(evidence[name]))
    approval = dict(schema=auth.APPROVAL_SCHEMA,approval_id='fake-lock-01',status='APPROVED',approver='Fixture leader',
        approval_evidence=ref('leader-lock'),source_revision=source_revision,
        source_sha256={name:sha((root/name).read_bytes()) for name in (*b1.SOURCE_FILES,*auth.EXTRA_SOURCE)},
        protocol_sha256=sha(protocol),lock_sha256=sha(lock),frozen_sha256={name:sha(value) for name,value in frozen.items()},
        acceptance=dict(phase_a=ref('phase-a'),phase_b1=ref('phase-b1')),runtime=execution.observe_runtime(2),
        dataset_descriptor=dict(reference='fake-canonical-descriptor',revision=source_revision,expected_samples=1934,num_classes=30))
    grant = dict(schema=auth.AUTHORIZATION_SCHEMA,authorization_id='fake-grant-01',status='GRANTED',grantor='Fixture leader',
        grant_evidence=ref('leader-grant'),approval_id=approval['approval_id'],approval_sha256=sha(raw(approval)),
        run_id=request.run_id,scope=list(auth.SCOPE),output_dir=request.output_dir,claim_root=str(claims),max_runs=1,automatic_retry=False)
    descriptor = execution.DatasetDescriptor('synthetic_fixture','fake-canonical-descriptor',source_revision,str(data))
    case = dict(root=root,source_revision=source_revision,claims=claims,output_root=output_root,request=request,
                approval=approval,grant=grant,protocol=protocol,descriptor=descriptor,data=data,checkpoint=cp,
                profile=dict(kind='synthetic_fixture',checkpoint_sha256=cp_hash,declaration_sha256=sha(declaration),n_samples=4))
    seal(case)
    calls = []
    original = execution._make_reader
    def reader(descriptor):
        output = Path(case['request'].output_dir)
        assert output.is_dir() and (output / b1.RUN_MARKER).is_file()
        assert not (output / '.complete').exists()
        assert (claims / 'fake-grant-01.claim.json').is_file()
        calls.append('reader-after-claim-and-reservation')
        return original(descriptor)
    monkeypatch.setattr(execution, '_make_reader', reader)
    case['calls'] = calls
    return case


def run(case, **changes):
    args = dict(release=case['release'],batch_size=2)
    args.update(changes)
    return execution.run_authorized_evaluation(case['request'],case['descriptor'],**args)


def assert_denied_without_data(case, result):
    assert result.status in {'FAIL','BLOCKED'}, result
    assert case['calls'] == []


def rehash(output):
    m = json.loads((output / 'manifest.json').read_bytes())
    m['output_sha256'] = {n:sha((output/n).read_bytes()) for n in execution.PAYLOADS if n!='manifest.json'}
    (output / 'manifest.json').write_bytes(raw(m))
    c = json.loads((output / '.complete').read_bytes())
    c['output_sha256'] = {n:sha((output/n).read_bytes()) for n in execution.PAYLOADS}
    (output / '.complete').write_bytes(raw(c))


def test_real_checkout_draft_default_deny_never_constructs_reader(tmp_path,monkeypatch):
    monkeypatch.setattr(b1,'SOURCE_ROOT',REAL_ROOT)
    monkeypatch.setattr(execution,'_make_reader',lambda *a:pytest.fail('input reader constructed'))
    monkeypatch.setattr(b1.predictor,'verify_checkpoint_hash',lambda *a:pytest.fail('checkpoint read before authorization'))
    result=execution.run_authorized_evaluation(auth.RunRequest('not-authorized',str(tmp_path/'result')),
        execution.DatasetDescriptor('official','UNRESOLVED','0'*40,'DO_NOT_TOUCH'))
    assert result.status=='BLOCKED' and result.code=='LOCK_DRAFT' and not result.claim_consumed
    assert not (tmp_path/'result').exists()


@pytest.mark.parametrize('problem',['missing_release','pending','revoked','control_hash'])
def test_missing_or_unapproved_controls_never_invoke_data(case,problem):
    if problem=='missing_release':
        result=run(case,release=None)
    else:
        if problem in ('pending','revoked'):
            case['grant']['status']=problem.upper()
            seal(case)
        if problem=='control_hash':
            # A reviewed index with a conflicting nested binding is still invalid.
            grant_path=case['root']/execution.CONTROL_FILES['authorization']
            grant=json.loads(grant_path.read_bytes()); grant['approval_sha256']='0'*64
            grant_path.write_bytes(raw(grant))
            index_path=case['root']/execution.CONTROL_INDEX
            index=json.loads(index_path.read_bytes()); index['authorization_sha256']=sha(raw(grant))
            index_path.write_bytes(raw(index)); case['release']=execution.ReviewedRelease(commit(case['root']))
        result=run(case)
    assert_denied_without_data(case,result)
    assert not result.claim_consumed and list(case['claims'].iterdir())==[]


@pytest.mark.parametrize('problem',['source','checkpoint','frozen_reference','protocol','runtime','descriptor'])
def test_provenance_or_frozen_science_mismatch_fails_before_data(case,problem):
    if problem=='source':
        name=execution.INTEGRATION_SOURCE[0]
        git(case['root'],'update-index','--assume-unchanged',name)
        path=case['root']/name; path.write_bytes(path.read_bytes()+b'# hidden change\n')
        assert git(case['root'],'diff','HEAD','--name-only','--',name).strip()==b''
    if problem=='checkpoint': case['checkpoint'].write_bytes(b'wrong checkpoint')
    if problem=='frozen_reference':
        path=case['root']/auth.FROZEN_REFERENCES[0]; path.write_bytes(path.read_bytes()+b'# changed frozen file\n')
    if problem=='protocol':
        p=auth.approved_protocol(); p['temperature']=0.9; case['protocol']=raw(p)
        case['approval']['protocol_sha256']=sha(case['protocol']); seal(case)
    if problem=='runtime':
        case['approval']['runtime']['numpy']='unapproved-version'; seal(case)
    if problem=='descriptor': case['descriptor']=replace(case['descriptor'],reference='other-source')
    result=run(case)
    assert_denied_without_data(case,result)
    assert result.status=='FAIL' and not result.claim_consumed


def test_positive_cpu_fixture_exercises_all_components_and_consumer(case):
    before={p:sha(p.read_bytes()) for p in case['data'].rglob('*') if p.is_file()}
    result=run(case)
    assert result.status=='PASS',result
    assert result.claim_consumed and case['calls']==['reader-after-claim-and-reservation']
    bundle=execution.load_authorized_bundle(result.output_dir,release=case['release'])
    assert bundle['logits'].shape==(4,30) and bundle['logits'].dtype==b1.np.float32
    assert bundle['labels'].dtype==b1.np.int64 and bundle['image_ids'].dtype.kind=='U'
    assert bundle['robustness']['conditions']==list(auth.SCOPE_E8)
    assert bundle['selective']['temperature']==0.80246283961825 and bundle['selective']['threshold']==0.5011672918550012
    assert bundle['metadata']['model_kind']=='synthetic_dummy'
    assert bundle['metadata']['approved_e4_checkpoint_sha256']==auth.CHECKPOINT_SHA256
    assert bundle['metadata']['executed_fixture_checkpoint_sha256']==case['profile']['checkpoint_sha256']
    assert all(bundle['manifest'][k] is False for k in execution.FALSE_FLAGS)
    assert bundle['manifest']['scientific_status']=='synthetic_fixture_only'
    assert {p:sha(p.read_bytes()) for p in before}==before
    assert not list(case['output_root'].glob('.dogbreedid-b2-publish-*'))
    second=run(case)
    assert second.status=='BLOCKED' and len(case['calls'])==1


@pytest.mark.parametrize('change',['run_id','output_dir','scope'])
def test_request_cannot_reuse_grant_for_a_different_run(case,change):
    case['request']=replace(case['request'],**{change:{'run_id':'another','output_dir':str(case['output_root']/'another'),'scope':('E4',)}[change]})
    result=run(case)
    assert_denied_without_data(case,result)
    assert not result.claim_consumed


def test_existing_output_is_untouched_and_claim_not_spent(case):
    output=Path(case['request'].output_dir); output.mkdir(parents=True)
    (output/'foreign.txt').write_bytes(b'preserve')
    result=run(case)
    assert_denied_without_data(case,result)
    assert result.status=='BLOCKED' and not result.claim_consumed
    assert (output/'foreign.txt').read_bytes()==b'preserve' and list(case['claims'].iterdir())==[]


def test_already_claimed_grant_is_rejected_even_without_an_output(case):
    index,inputs,anchors=execution._controls(case['release'],live=True,runtime=execution.observe_runtime(2))
    decision=auth.validate_authorization(inputs,anchors,case['request'])
    assert auth.claim_authorization(decision).status=='PASS'
    result=run(case)
    assert_denied_without_data(case,result)
    assert result.status=='BLOCKED' and result.code=='AUTHORIZATION_ALREADY_CLAIMED'
    assert not Path(case['request'].output_dir).exists()


@pytest.mark.parametrize('suffix',['safe/../run-01','run-01:stream'])
def test_lexical_output_alias_is_rejected_before_data(case,suffix):
    location=str(case['output_root'])+'/'+suffix
    case['request']=replace(case['request'],output_dir=location)
    case['grant']['output_dir']=location
    seal(case)
    result=run(case)
    assert_denied_without_data(case,result)
    assert result.status=='FAIL' and not result.claim_consumed


@pytest.mark.parametrize('kind',['symlink','junction'])
def test_link_output_is_rejected_before_claim_or_data(case,monkeypatch,kind):
    target=Path(case['request'].output_dir)
    method='is_symlink' if kind=='symlink' else 'is_junction'
    original=getattr(Path,method,lambda self:False)
    monkeypatch.setattr(Path,method,lambda self: True if self==target else original(self),raising=False)
    result=run(case)
    assert_denied_without_data(case,result)
    assert result.status=='FAIL' and not result.claim_consumed


def test_claim_stays_spent_if_reservation_loses_race(case,monkeypatch):
    target=Path(case['request'].output_dir)
    original=Path.mkdir
    def mkdir(path,*args,**kwargs):
        if path==target and not path.exists():
            original(path,*args,**kwargs)
            (path/'foreign.txt').write_bytes(b'foreign wins')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'mkdir',mkdir)
    result=run(case)
    assert_denied_without_data(case,result)
    assert result.claim_consumed and result.status=='BLOCKED'
    assert (target/'foreign.txt').read_bytes()==b'foreign wins'
    assert (case['claims']/'fake-grant-01.claim.json').is_file()


def test_live_lock_drift_after_claim_blocks_reader_and_consumes_grant(case,monkeypatch):
    target=Path(case['request'].output_dir); original=Path.mkdir
    def mkdir(path,*args,**kwargs):
        result=original(path,*args,**kwargs)
        if path==target:
            (case['root']/execution.LOCK).write_bytes('**Trạng thái:** BẢN NHÁP — CHƯA KHÓA CẤU HÌNH\n'.encode('utf-8'))
        return result
    monkeypatch.setattr(Path,'mkdir',mkdir)
    result=run(case)
    assert_denied_without_data(case,result)
    assert result.status=='FAIL' and result.claim_consumed
    assert (target/'.incomplete').exists() and not (target/'.complete').exists()


def test_failure_after_claim_does_not_allow_a_retry(case):
    (case['data']/'fixture_manifest.json').write_bytes(b'changed declaration')
    result=run(case)
    assert result.status=='FAIL' and result.claim_consumed
    assert len(case['calls'])==1
    output=Path(case['request'].output_dir)
    assert (output/'.incomplete').exists() and not (output/'.complete').exists()
    again=run(case)
    assert again.status=='BLOCKED' and len(case['calls'])==1


@pytest.mark.parametrize('position',[3,4])
def test_publish_failure_preserves_foreign_file_and_invalidates_bundle(case,monkeypatch,position):
    target=execution.PAYLOADS[position-1]; original=os.link
    def link(source,destination,*args,**kwargs):
        if destination.name==target: destination.write_bytes(b'foreign file')
        return original(source,destination,*args,**kwargs)
    monkeypatch.setattr(os,'link',link)
    result=run(case)
    output=Path(case['request'].output_dir)
    assert result.status!='PASS' and result.claim_consumed
    assert (output/target).read_bytes()==b'foreign file'
    assert (output/'.incomplete').exists() and not (output/'.complete').exists()
    assert not list(case['output_root'].glob('.dogbreedid-b2-publish-*'))
    with pytest.raises(ValueError): execution.load_authorized_bundle(output,release=case['release'])


def test_consumer_failure_after_publication_leaves_failure_marker(case,monkeypatch):
    with monkeypatch.context() as patch:
        def fail(*args,**kwargs): raise ValueError('injected final consumer failure')
        patch.setattr(execution,'load_authorized_bundle',fail)
        result=run(case)
    output=Path(case['request'].output_dir)
    assert result.status=='FAIL' and result.claim_consumed
    assert (output/'.complete').exists() and (output/'.incomplete').exists()
    with pytest.raises(ValueError,match='Incomplete'): execution.load_authorized_bundle(output,release=case['release'])


@pytest.mark.parametrize('problem',['approval_link','source_receipts','synthetic_flag','npz_csv_ids'])
def test_consumer_rejects_rehashed_authorization_or_data_forgery(case,problem):
    result=run(case); assert result.status=='PASS',result
    output=Path(result.output_dir)
    if problem=='approval_link':
        p=output/'controls.json'; d=json.loads(p.read_bytes()); d['approval_sha256']='0'*64; p.write_bytes(raw(d))
    if problem=='source_receipts':
        p=output/'manifest.json'; d=json.loads(p.read_bytes()); d['source_sha256']={}; p.write_bytes(raw(d))
    if problem=='synthetic_flag':
        p=output/'metadata.json'; d=json.loads(p.read_bytes()); d['synthetic']=False; p.write_bytes(raw(d))
    if problem=='npz_csv_ids':
        p=output/'clean_predictions.npz'
        with b1.np.load(io.BytesIO(p.read_bytes()),allow_pickle=False) as f: values={n:f[n] for n in f.files}
        old=str(values['image_ids'][0]); new=old.replace('image_000','other_000'); values['image_ids'][0]=new
        b1.np.savez_compressed(p,**values)
        p=output/'split.csv'; p.write_bytes(p.read_bytes().replace(old.encode(),new.encode()))
        p=output/'e8_metrics.json'; d=json.loads(p.read_bytes()); d['image_ids'][0]=new; p.write_bytes(raw(d))
    rehash(output)
    with pytest.raises((ValueError,RuntimeError)): execution.load_authorized_bundle(output,release=case['release'])


def test_regression_packaged_mapping_must_match_trusted_snapshot_after_rehash(case):
    result=run(case)
    assert result.status=='PASS',result
    output=Path(result.output_dir)
    assert execution.load_authorized_bundle(output,release=case['release'])['metadata']['synthetic'] is True
    release=case['release']
    manifest_before=json.loads((output/'manifest.json').read_bytes())
    preserved={name:(output/name).read_bytes() for name in execution.PAYLOADS
               if name not in {'class_mapping.json','manifest.json'}}
    mapping_path=output/'class_mapping.json'
    mapping=json.loads(mapping_path.read_bytes())
    mapping['0']+=' changed in packaged artifact only'
    mapping_path.write_bytes(raw(mapping))
    rehash(output)
    manifest_after=json.loads((output/'manifest.json').read_bytes())
    assert {k:v for k,v in manifest_before.items() if k!='output_sha256'} == {
        k:v for k,v in manifest_after.items() if k!='output_sha256'}
    assert all((output/name).read_bytes()==value for name,value in preserved.items())
    assert case['release']==release
    with pytest.raises(ValueError,match='published mapping'):
        execution.load_authorized_bundle(output,release=release)


def test_regression_claim_fsync_failure_is_reported_spent_and_never_reused(case,monkeypatch):
    def fail(*args): raise OSError('injected receipt fsync failure after allocation')
    with monkeypatch.context() as patch:
        patch.setattr(os,'fsync',fail)
        result=run(case)
    assert_denied_without_data(case,result)
    assert result.status=='FAIL' and result.code=='CLAIM_WRITE_FAILED'
    claim_path=case['claims']/'fake-grant-01.claim.json'
    assert claim_path.is_file()
    retained=claim_path.read_bytes()
    assert json.loads(retained)['state']=='CLAIMED'
    assert result.claim_consumed is True
    assert not Path(case['request'].output_dir).exists()
    again=run(case)
    assert_denied_without_data(case,again)
    assert again.status=='BLOCKED' and again.code=='AUTHORIZATION_ALREADY_CLAIMED'
    assert again.claim_consumed is True
    assert claim_path.read_bytes()==retained


def test_regression_known_failure_before_claim_allocation_keeps_unconsumed_status(case,monkeypatch):
    original=execution._verify_live
    backup=case['claims'].with_name('original-empty-claims')
    def change_root(*args,**kwargs):
        original(*args,**kwargs)
        case['claims'].rename(backup)
        case['claims'].mkdir()
    monkeypatch.setattr(execution,'_verify_live',change_root)
    result=run(case)
    assert_denied_without_data(case,result)
    assert result.status=='FAIL' and result.code=='CLAIM_ROOT_CHANGED'
    assert result.claim_consumed is False
    assert list(case['claims'].iterdir())==[] and list(backup.iterdir())==[]


def test_regression_uncertain_claim_write_failure_is_conservative_without_state_framework(case,monkeypatch):
    original=os.open
    claim_path=case['claims']/'fake-grant-01.claim.json'
    def fail_open(path,flags,*args,**kwargs):
        if Path(path)==claim_path and flags & os.O_EXCL:
            raise OSError('injected preallocation error indistinguishable in helper result')
        return original(path,flags,*args,**kwargs)
    monkeypatch.setattr(os,'open',fail_open)
    result=run(case)
    assert_denied_without_data(case,result)
    assert result.status=='FAIL' and result.code=='CLAIM_WRITE_FAILED'
    assert not claim_path.exists()
    assert result.claim_consumed is True  # Boolean represents spent OR possibly spent.


def test_official_descriptor_stays_blocked_without_guessing_paths(case):
    case['profile']=dict(kind='official',checkpoint_sha256=auth.CHECKPOINT_SHA256,declaration_sha256=None,n_samples=1934)
    case['descriptor']=replace(case['descriptor'],kind='official',fixture_root='DO_NOT_TOUCH')
    seal(case)
    result=run(case)
    assert_denied_without_data(case,result)
    assert result.status=='BLOCKED' and result.code=='OFFICIAL_DESCRIPTOR_UNCONFIRMED'
    assert not result.claim_consumed


def official_fixture(case):
    """Canonical *fake* CSV and images in the temporary Git checkout only."""
    mapping=json.loads((case['root']/'data/metadata/class_mapping.json').read_bytes())
    text='relative_path,class_id,breed_name,source_split\n'
    image=io.BytesIO(); b1.Image.new('RGB',(9,11),(40,90,160)).save(image,format='PNG')
    for i in range(1934):
        name=f'fixture_breed/image_{i:04d}.png'
        text+=f'{name},{i%30},{mapping[str(i%30)]},official_test\n'
        write(case['root'],b1.IMAGES.as_posix()+'/'+name,image.getvalue())
    write(case['root'],execution.OFFICIAL_CSV,text.encode())
    revision=commit(case['root'])
    case['profile']=dict(kind='official',checkpoint_sha256=auth.CHECKPOINT_SHA256,declaration_sha256=None,n_samples=1934)
    case['approval']['dataset_descriptor'].update(reference=execution.OFFICIAL_CSV,revision=revision)
    case['descriptor']=execution.DatasetDescriptor('official',execution.OFFICIAL_CSV,revision)
    seal(case)


@pytest.mark.parametrize('problem',['schema','source','label','breed','unsafe','duplicate','count','classes'])
def test_official_schema_rejects_malformed_artificial_csv(problem):
    content='relative_path,class_id,breed_name,source_split\na/a.png,0,A,official_test\nb/b.png,1,B,official_test\n'
    if problem=='schema': content=content.replace('source_split','source')
    if problem=='source': content=content.replace('official_test','official_train')
    if problem=='label': content=content.replace(',0,A,',',00,A,')
    if problem=='breed': content=content.replace(',A,',',WRONG,')
    if problem=='unsafe': content=content.replace('a/a.png','../a.png')
    if problem=='duplicate': content=content.replace('b/b.png','a/a.png')
    if problem=='classes': content=content.replace(',1,B,',',0,A,')
    with pytest.raises(ValueError):
        execution._official_rows(content.encode(),{0:'A',1:'B'},3 if problem=='count' else 2)


def test_official_reader_captures_canonical_fake_bytes_and_rejects_csv_drift(case,tmp_path):
    official_fixture(case)
    snapshot=execution._Snapshot(tmp_path/'snapshot')
    mapping={int(k):v for k,v in json.loads((case['root']/'data/metadata/class_mapping.json').read_bytes()).items()}
    csv_path,images,rows,declaration=execution._OfficialReader(case['descriptor']).capture(snapshot,case['profile'],mapping)
    assert len(rows)==1934 and {label for _,label in rows}==set(range(30))
    assert csv_path.read_bytes()==(case['root']/execution.OFFICIAL_CSV).read_bytes()
    assert declaration['synthetic'] is False and declaration['reference']==execution.OFFICIAL_CSV
    assert declaration['canonical_csv_sha256']==sha(csv_path.read_bytes())
    assert sha((images/rows[0][0]).read_bytes())==declaration['images'][rows[0][0]]
    snapshot.verify()
    path=case['root']/execution.OFFICIAL_CSV
    path.write_bytes(path.read_bytes().replace(b'fixture_breed/image_0000',b'fixture_breed/changed_0000'))
    with pytest.raises(ValueError,match='canonical Official split'):
        execution._OfficialReader(case['descriptor']).capture(execution._Snapshot(tmp_path/'drift'),case['profile'],mapping)


def test_official_capture_rejects_link_and_retains_hashed_snapshot_on_source_change(tmp_path,monkeypatch):
    root=tmp_path/'fake-data'; root.mkdir()
    (root/'image.png').write_bytes(b'fake image bytes')
    os.link(root/'image.png',root/'alias.png')
    with pytest.raises(ValueError,match='unaliased'):
        execution._capture_dataset(execution._Snapshot(tmp_path/'hardlink'),root,'image.png','images/one.png')
    (root/'standalone.png').write_bytes(b'old fixture bytes')
    original=execution._dataset_path; calls=[]
    def replace_after_open(base,relative):
        path=original(base,relative); calls.append(relative)
        if len(calls)==2: path.write_bytes(b'changed fixture bytes')
        return path
    monkeypatch.setattr(execution,'_dataset_path',replace_after_open)
    with pytest.raises(ValueError,match='changed during capture'):
        execution._capture_dataset(execution._Snapshot(tmp_path/'race'),root,'standalone.png','images/one.png')


def test_real_e4_adapter_uses_actual_mobilenet_loader_and_all_science_on_fake_images(case,tmp_path,monkeypatch):
    """Real architecture/loader; only the approved-checkpoint hash boundary is a fixture.

    A random temporary checkpoint cannot have the approved trained E4 SHA. This
    test verifies its actual hash with the real verifier, then supplies the trusted
    digest at that one boundary. It does not validate the real trained checkpoint.
    """
    from src.models.model_factory import build_model
    index,inputs,_=execution._controls(case['release'],live=True,runtime=execution.observe_runtime(2))
    snapshot=execution._Snapshot(tmp_path/'real-adapter')
    data,cfg,mapping=execution._settings(snapshot,inputs)
    config=b1.predictor._load_yaml(snapshot.root/b1.predictor.DEFAULT_E4_CONFIG)
    config['model']['weights']='none'
    with b1.torch.random.fork_rng():
        b1.torch.manual_seed(42)
        initial=build_model(config).eval()
    cp=snapshot.root/'fake_e4.pt'
    b1.torch.save(dict(epoch=25,model_state_dict=initial.state_dict()),cp)
    expected=sha(cp.read_bytes()); verifier=b1.predictor.verify_checkpoint_hash
    with pytest.raises(RuntimeError,match='SHA-256 mismatch'):
        execution._load_backend(snapshot,cp,cfg,official=True)
    def fixture_hash(path,*args,**kwargs):
        assert Path(path)==cp
        assert verifier(path,expected)==expected
        return auth.CHECKPOINT_SHA256
    monkeypatch.setattr(b1.predictor,'verify_checkpoint_hash',fixture_hash)
    original_load=b1.torch.load; loads=[]
    def tracked_load(*args,**kwargs):
        loads.append(kwargs.copy()); return original_load(*args,**kwargs)
    monkeypatch.setattr(b1.torch,'load',tracked_load)
    model=execution._load_backend(snapshot,cp,cfg,official=True)
    assert model.classifier[-1].out_features==30 and not model.training
    assert all(not p.requires_grad for p in model.features.parameters())
    assert all(b1.torch.equal(value,model.state_dict()[name]) for name,value in initial.state_dict().items())
    assert loads and all(item['weights_only'] is True for item in loads)
    before={name:value.clone() for name,value in model.state_dict().items()}
    monkeypatch.setattr(execution,'_load_backend',lambda *a,**k:model)
    payload=tmp_path/'real-payload'; payload.mkdir()
    rows=b1.selective._csv_rows(case['data']/'split.csv',mapping,'synthetic')
    # Reader policy has its own canonical tests. Here fake CSV rows drive the
    # actual Official scientific adapter, not Task 04's Validation policy.
    execution._compute(snapshot,payload,cp,data,cfg,mapping,case['data']/'split.csv',case['data']/'images',rows,2,official=True)
    e8=json.loads((payload/'e8_metrics.json').read_bytes())
    with b1.np.load(payload/'clean_predictions.npz',allow_pickle=False) as arrays:
        logits,labels,ids=[arrays[name] for name in ('logits','labels','image_ids')]
    e4=json.loads((payload/'e4_e6_metrics.json').read_bytes())
    e7=json.loads((payload/'selective_metrics.json').read_bytes())
    assert e7['temperature']==e4['calibration']['temperature']==0.80246283961825
    assert e7['threshold']==0.5011672918550012
    assert e8['conditions']==['clean','gaussian_blur','gaussian_noise','occlusion']
    b1._consistency(e4,e8,logits,labels,ids)
    assert all(b1.torch.equal(value,model.state_dict()[name]) for name,value in before.items())
    assert config['model']['weights']=='none'


def test_official_policy_reuses_exact_per_image_corruptions_before_normalization(case,tmp_path):
    _,inputs,_=execution._controls(case['release'],live=True,runtime=execution.observe_runtime(2))
    data,cfg,_=execution._settings(execution._Snapshot(tmp_path/'settings'),inputs)
    base=b1.DogBreedDataset(case['data']/'split.csv',case['data']/'images',b1.robustness._build_pre_norm_transform(data))
    evaluator=b1.robustness.RobustnessEvaluator(b1.robustness._make_synthetic_model(42),cfg)
    for condition in cfg.conditions:
        adapter=execution._OfficialConditionDataset(base,cfg,condition,data)
        for index in (3,0):
            original=base[index]
            expected=evaluator._corrupt(original['image'].unsqueeze(0),[original['relative_path']],condition)[0]
            assert b1.torch.equal(adapter[index]['image'],adapter.normalize(expected))
            assert adapter[index]['relative_path']==original['relative_path']


def test_official_gate_reserved_run_and_consumer_with_fake_1934_images(case,monkeypatch):
    """Full policy/reader/publisher/consumer; checkpoint boundary/inference mocked.

    Actual E4 loader/architecture is tested separately above. No result here is
    evidence of inference on real Official data or of actual leader permission.
    """
    official_fixture(case)
    verifier=b1.predictor.verify_checkpoint_hash
    fake_hash=sha(case['checkpoint'].read_bytes())
    original_read=b1.preflight._read_once
    def fixture_checkpoint_capture(path,*args,**kwargs):
        captured=original_read(path,*args,**kwargs)
        if Path(path)==case['checkpoint'] or Path(path).name=='checkpoint.pt':
            assert Path(path).is_relative_to(Path(execution.tempfile.gettempdir()).absolute())
            assert captured['sha256']==fake_hash
            # A fixture cannot manufacture the trained checkpoint's bytes.
            # Substitute only that hash boundary, including its snapshot receipt;
            # every dataset/source/control/output validator remains real.
            captured=dict(captured,sha256=auth.CHECKPOINT_SHA256)
        return captured
    monkeypatch.setattr(b1.preflight,'_read_once',fixture_checkpoint_capture)
    def fixture_hash(path,expected=auth.CHECKPOINT_SHA256):
        verifier(path,fake_hash)
        assert expected==auth.CHECKPOINT_SHA256
        return auth.CHECKPOINT_SHA256
    monkeypatch.setattr(b1.predictor,'verify_checkpoint_hash',fixture_hash)
    def fake_backend(*args,**kwargs):
        assert kwargs['official'] is True
        assert (Path(case['request'].output_dir)/b1.RUN_MARKER).is_file()
        return b1.robustness._make_synthetic_model(42)
    monkeypatch.setattr(execution,'_load_backend',fake_backend)
    result=run(case)
    assert result.status=='PASS',result
    assert result.code=='OFFICIAL_EXECUTION_COMPLETE' and result.claim_consumed
    assert case['calls']==['reader-after-claim-and-reservation']
    output=Path(result.output_dir)
    bundle=execution.load_authorized_bundle(output,release=case['release'])
    assert bundle['manifest']['format']==execution.OFFICIAL_FORMAT
    assert bundle['manifest']['synthetic'] is False and bundle['metadata']['model_kind']=='e4_mobilenet_v2'
    assert len(bundle['labels'])==1934 and bundle['robustness']['n_samples']==1934
    assert all(bundle['manifest'][name] is True for name in execution.FALSE_FLAGS)
    # All internal hashes can agree while IDs are coordinatedly forged. The
    # independently reviewed canonical Git CSV still rejects the substitution.
    buffers={name:(output/name).read_bytes() for name in execution.PAYLOADS}
    altered=dict(buffers)
    with b1.np.load(io.BytesIO(buffers['clean_predictions.npz']),allow_pickle=False) as stored:
        logits,labels,ids=[stored[name].copy() for name in ('logits','labels','image_ids')]
    old=ids[0]; ids[0]=ids[1].replace('0001','9999')
    stream=io.BytesIO(); b1.np.savez_compressed(stream,logits=logits,labels=labels,image_ids=ids)
    altered['clean_predictions.npz']=stream.getvalue()
    altered['split.csv']=buffers['split.csv'].replace(old.encode(),str(ids[0]).encode())
    e8=json.loads(buffers['e8_metrics.json']); e8['image_ids'][0]=str(ids[0]); altered['e8_metrics.json']=raw(e8)
    declaration=json.loads(buffers['dataset.json']); declaration['images'][str(ids[0])]=declaration['images'].pop(str(old))
    declaration['split_sha256']=sha(altered['split.csv']); altered['dataset.json']=raw(declaration)
    manifest=json.loads(buffers['manifest.json'])
    manifest['inputs']['dataset/images/'+str(ids[0])]=manifest['inputs'].pop('dataset/images/'+str(old))
    for record_name,payload_name in [('dataset/split.csv','split.csv'),('dataset/declaration.json','dataset.json')]:
        manifest['inputs'][record_name]=dict(sha256=sha(altered[payload_name]),size_bytes=len(altered[payload_name]))
    manifest['output_sha256']={name:sha(value) for name,value in altered.items() if name!='manifest.json'}
    altered['manifest.json']=raw(manifest)
    with pytest.raises(ValueError,match='canonical Official split'):
        execution._validate_buffers(altered,case['release'])
    before={name:sha(value) for name,value in buffers.items()}
    again=run(case)
    assert again.status=='BLOCKED' and again.code=='OUTPUT_ALREADY_EXISTS'
    assert before=={name:sha((output/name).read_bytes()) for name in execution.PAYLOADS}


@pytest.mark.parametrize('problem',['draft','pending','conflicting_grant'])
def test_official_gate_denial_has_zero_canonical_blob_or_reader_calls(case,monkeypatch,problem):
    # Canonical reference is metadata only; no fake/real Official CSV is opened.
    case['profile']=dict(kind='official',checkpoint_sha256=auth.CHECKPOINT_SHA256,declaration_sha256=None,n_samples=1934)
    case['approval']['dataset_descriptor']['reference']=execution.OFFICIAL_CSV
    case['descriptor']=execution.DatasetDescriptor('official',execution.OFFICIAL_CSV,case['source_revision'])
    if problem=='pending': case['grant']['status']='PENDING'
    if problem=='conflicting_grant': case['grant']['run_id']='not-the-requested-run'
    seal(case)
    if problem=='draft':
        write(case['root'],execution.LOCK,'**Trạng thái:** BẢN NHÁP — CHƯA KHÓA CẤU HÌNH\n'.encode())
    original=execution._git_buffers
    def no_dataset_blob(revision,names,**kwargs):
        assert execution.OFFICIAL_CSV not in names
        assert kwargs.get('authorized_dataset',False) is False
        return original(revision,names,**kwargs)
    monkeypatch.setattr(execution,'_git_buffers',no_dataset_blob)
    monkeypatch.setattr(execution,'_capture_dataset',lambda *a,**k:pytest.fail('dataset read before gate'))
    result=run(case)
    assert_denied_without_data(case,result)
    assert not result.claim_consumed and not Path(case['request'].output_dir).exists()
