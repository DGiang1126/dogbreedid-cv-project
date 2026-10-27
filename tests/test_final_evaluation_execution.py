"""B1 runs real orchestration/metrics/corruptions on artificial data only.

Git commits below belong exclusively to disposable temporary fixture repositories,
never the user's checkout. Validation fixtures replace only the E4 loader and its
expected checksum; all 300 artificial images, transforms, validators and E8 run.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
import pytest
import torch

from scripts import run_final_evaluation as runner

REAL_ROOT = runner.SOURCE_ROOT


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def _commit(root):
    _git(root, "add", ".")
    _git(root, "-c", "user.name=B1 fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "artificial fixture")


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    for name in (*runner.SOURCE_FILES, *runner.REFERENCES):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((REAL_ROOT / name).read_bytes())
    _git(root, "init", "-q")
    _git(root, "config", "core.autocrlf", "false")
    _commit(root)
    monkeypatch.setattr(runner, "SOURCE_ROOT", root)
    return root


def _run(checkout, tmp_path, **kwargs):
    options = dict(split="synthetic", output_dir=tmp_path / "synthetic_bundle", batch_size=4)
    options.update(kwargs)
    return runner.run_final_evaluation(**options)


def _load_json(path):
    return json.loads(path.read_bytes())


def _change(path, function):
    value = _load_json(path)
    function(value)
    path.write_bytes(runner._json_bytes(value))


def _rebind(output):
    """A coherent but falsified package must still fail semantic validation."""
    manifest = _load_json(output / "manifest.json")
    manifest["output_sha256"] = {name: runner._hash((output / name).read_bytes()) for name in runner.PAYLOADS if name != "manifest.json"}
    (output / "manifest.json").write_bytes(runner._json_bytes(manifest))
    completed = _load_json(output / ".complete")
    completed["output_sha256"] = {name: runner._hash((output / name).read_bytes()) for name in runner.PAYLOADS}
    (output / ".complete").write_bytes(runner._json_bytes(completed))


def _validation_fixture(root, monkeypatch):
    """No real checkpoint/dataset is read; protocol and shape remain production."""
    mapping = _load_json(root / "data/metadata/class_mapping.json")
    csv_path = root / "data/splits/validation.csv"
    csv_path.parent.mkdir(parents=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("relative_path", "class_id", "breed_name", "source_split"))
        writer.writeheader()
        for i in range(300):
            name = f"artificial/image_{i:03d}.png"
            path = root / runner.IMAGES / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (4, 4), (i % 256, i * 2 % 256, i * 3 % 256)).save(path)
            writer.writerow(dict(relative_path=name, class_id=i % 30, breed_name=mapping[str(i % 30)], source_split="official_train"))
    checkpoint = root / runner.predictor.DEFAULT_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"B1 artificial checkpoint fixture - never weights")
    digest = runner._hash(checkpoint.read_bytes())
    monkeypatch.setattr(runner.predictor, "EXPECTED_CHECKPOINT_SHA256", digest)
    _commit(root)
    loaded = []

    def loader(private, path, device):
        assert Path(private) != root
        assert path.read_bytes() == b"B1 artificial checkpoint fixture - never weights"
        runner.predictor.verify_checkpoint_hash(path, runner.predictor.EXPECTED_CHECKPOINT_SHA256)
        model = runner.robustness._make_synthetic_model(42).to(device)
        loaded.append(model)
        return model, digest

    monkeypatch.setattr(runner.predictor, "load_e4_model", loader)
    return loaded


def test_synthetic_happy_path_is_complete_and_honest(checkout, tmp_path):
    before = {p: runner._hash(p.read_bytes()) for p in checkout.rglob('*') if p.is_file()}
    manifest = _run(checkout, tmp_path)
    output = tmp_path / "synthetic_bundle"
    bundle = runner.load_execution_bundle(output)
    assert manifest["source"]["status"] == "PASS"
    assert manifest["synthetic"] is True
    assert manifest["scientific_status"] == "synthetic_fixture_only"
    assert manifest["components"] == dict.fromkeys(runner.COMPONENTS, "PASS")
    assert bundle["logits"].shape == (8, 30)
    assert bundle["logits"].dtype == np.float32
    assert bundle["metadata"]["checkpoint_sha256"] is None
    assert bundle["robustness"]["conditions"] == ["clean", "gaussian_blur", "gaussian_noise", "occlusion"]
    assert bundle["metadata"]["model_weights_unchanged"] is True
    assert all(manifest[key] is False for key in ("official_test_used", "official_final_evaluation", "configuration_lock_approved", "official_evaluation_authorized"))
    assert {p.name for p in output.iterdir()} == {*runner.PAYLOADS, runner.RUN_MARKER, ".complete"}
    assert all(runner._hash(p.read_bytes()) == digest for p, digest in before.items())
    assert not list(tmp_path.glob('.dogbreedid-b1-private-*'))


def test_validation_fixture_runs_all_components_with_same_model(checkout, tmp_path, monkeypatch):
    loaded = _validation_fixture(checkout, monkeypatch)
    output = checkout / "outputs/final_evaluation_b1/validation/fixture_run"
    runner.run_final_evaluation(split="validation", output_dir=output, batch_size=128)
    bundle = runner.load_execution_bundle(output)
    assert len(loaded) == 1
    assert bundle["logits"].shape == (300, 30)
    assert bundle["metadata"]["model_weights_unchanged"] is True
    assert bundle["manifest"]["synthetic"] is False
    assert bundle["manifest"]["official_final_evaluation"] is False
    correct = int((bundle["logits"].argmax(1) == bundle["labels"]).sum())
    assert bundle["robustness"]["clean_accuracy"] == correct / 300


def test_synthetic_never_calls_real_model_or_reads_images(checkout, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic attempted real model/checkpoint/data access")
    monkeypatch.setattr(runner.predictor, "load_e4_model", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    original = runner.preflight._read_once
    def read(path, *args, **kwargs):
        p = Path(path)
        assert not p.is_relative_to(checkout / runner.IMAGES)
        assert p != checkout / runner.predictor.DEFAULT_CHECKPOINT
        return original(path, *args, **kwargs)
    monkeypatch.setattr(runner.preflight, "_read_once", read)
    _run(checkout, tmp_path)


@pytest.mark.parametrize("split", ["test", "official_test", "Official-Test", "validation/../test", "calibration", "train"])
def test_disallowed_split_fails_before_any_input_read(checkout, tmp_path, monkeypatch, split):
    monkeypatch.setattr(runner, "_source_audit", lambda *a: pytest.fail("read inputs"))
    with pytest.raises(PermissionError):
        runner.run_final_evaluation(split=split, output_dir=tmp_path / "bundle")


@pytest.mark.parametrize("flag", ["--official-test", "--allow-official-test", "--bypass", "--root", "--images-dir", "--checkpoint", "--split-csv", "--spl"])
def test_cli_has_no_extra_input_or_bypass_options(flag):
    with pytest.raises(SystemExit) as error:
        runner.main(["--split", "synthetic", "--output-dir", "unused", flag, "unused"])
    assert error.value.code == 2


@pytest.mark.parametrize("component", ["official_test", "Official-Test", "test", "safe/../escape"])
def test_unsafe_output_paths_are_rejected(checkout, tmp_path, component):
    with pytest.raises(PermissionError):
        _run(checkout, tmp_path, output_dir=tmp_path / component)


@pytest.mark.parametrize("split", ["synthetic", "validation"])
def test_existing_even_empty_output_rejected_before_model(checkout, tmp_path, monkeypatch, split):
    output = tmp_path / 'already' if split == "synthetic" else checkout / 'outputs/final_evaluation_b1/validation/already'
    output.mkdir(parents=True)
    monkeypatch.setattr(runner, "_source_audit", lambda *a: pytest.fail("too late output check"))
    with pytest.raises(FileExistsError):
        runner.run_final_evaluation(split=split, output_dir=output)
    assert list(output.iterdir()) == []


def test_foreign_output_wins_reservation_race(checkout, tmp_path, monkeypatch):
    original = runner._source_audit
    def race(split):
        state = original(split)
        output = tmp_path / 'synthetic_bundle'
        output.mkdir()
        (output / 'foreign.txt').write_bytes(b'FOREIGN')
        return state
    monkeypatch.setattr(runner, '_source_audit', race)
    with pytest.raises(FileExistsError):
        _run(checkout, tmp_path)
    assert (tmp_path / 'synthetic_bundle/foreign.txt').read_bytes() == b'FOREIGN'


@pytest.mark.parametrize("failure", ["e4_e6_metrics.json", "risk_coverage_curve.png", "e8_metrics.json"])
def test_private_write_failure_has_no_completed_public_artifacts(checkout, tmp_path, monkeypatch, failure):
    original = runner._write
    def fail(path, data):
        if path.name == failure:
            with path.open('xb') as handle:
                handle.write(b'partial')
            raise OSError('injected write failure')
        original(path, data)
    monkeypatch.setattr(runner, '_write', fail)
    with pytest.raises(OSError, match='injected'):
        _run(checkout, tmp_path)
    output = tmp_path / 'synthetic_bundle'
    assert (output / '.incomplete').exists()
    assert not (output / '.complete').exists()
    assert not (output / 'clean_predictions.npz').exists()
    with pytest.raises(runner.preflight.Blocked):
        runner.load_execution_bundle(output)
    assert not list(tmp_path.glob('.dogbreedid-b1-private-*'))


@pytest.mark.parametrize("victim", ["e4_e6_metrics.json", "risk_coverage_curve.png", ".complete"])
def test_publish_never_clobbers_foreign_file(checkout, tmp_path, monkeypatch, victim):
    original_link = os.link
    original_write = runner._write
    def link(source, destination, *args, **kwargs):
        if destination.name == victim:
            destination.write_bytes(b'FOREIGN')
        return original_link(source, destination, *args, **kwargs)
    def write(path, data):
        if path.name == victim and path.parent.name == 'synthetic_bundle':
            path.write_bytes(b'FOREIGN')
        return original_write(path, data)
    monkeypatch.setattr(os, 'link', link)
    if victim == '.complete':
        monkeypatch.setattr(runner, '_write', write)
    with pytest.raises(FileExistsError):
        _run(checkout, tmp_path)
    output = tmp_path / 'synthetic_bundle'
    assert (output / victim).read_bytes() == b'FOREIGN'
    assert (output / '.incomplete').exists()
    assert not list(tmp_path.glob('.dogbreedid-b1-private-*'))


def test_rollback_never_unlinks_replacement_after_ownership_observation(checkout, tmp_path, monkeypatch):
    original_link = os.link
    def link(source, destination, *args, **kwargs):
        if destination.name == 'e4_e6_metrics.json':
            public = destination.parent / 'clean_predictions.npz'
            assert public.is_file()  # Ownership observation before adversarial replacement.
            foreign = tmp_path / 'foreign'
            foreign.write_bytes(b'FOREIGN REPLACEMENT')
            os.replace(foreign, public)
            raise OSError('failure after replacement')
        return original_link(source, destination, *args, **kwargs)
    monkeypatch.setattr(os, 'link', link)
    with pytest.raises(OSError, match='after replacement'):
        _run(checkout, tmp_path)
    output = tmp_path / 'synthetic_bundle'
    assert (output / 'clean_predictions.npz').read_bytes() == b'FOREIGN REPLACEMENT'
    assert (output / '.incomplete').exists()


@pytest.mark.parametrize("name", ["clean_predictions.npz", "e4_e6_metrics.json", "manifest.json", "risk_coverage_curve.png", ".inference_run", ".complete"])
def test_consumer_rejects_post_publish_tampering(checkout, tmp_path, name):
    _run(checkout, tmp_path)
    output = tmp_path / 'synthetic_bundle'
    path = output / name
    path.write_bytes(path.read_bytes() + b'TAMPER')
    with pytest.raises((ValueError, OSError)):
        runner.load_execution_bundle(output)


def test_failure_marker_always_invalidates_complete_bundle(checkout, tmp_path):
    _run(checkout, tmp_path)
    output = tmp_path / 'synthetic_bundle'
    (output / '.incomplete').write_bytes(b'foreign marker')
    with pytest.raises(runner.preflight.Blocked):
        runner.load_execution_bundle(output)


@pytest.mark.parametrize("problem", ['labels', 'ids', 'schema', 'shape', 'dtype', 'nan'])
def test_bad_prediction_contract_never_completes(checkout, tmp_path, monkeypatch, problem):
    original = runner.predictor.collect_predictions
    def collect(*args, **kwargs):
        logits, labels, ids = original(*args, **kwargs)
        if problem == 'labels': labels[0] = 29
        if problem == 'ids': ids[0] = 'synthetic/wrong.png'
        if problem == 'shape': logits = logits[:, :29]
        if problem == 'dtype': logits = logits.astype(np.int64)
        if problem == 'nan': logits[0, 0] = np.nan
        return logits, labels, ids
    monkeypatch.setattr(runner.predictor, 'collect_predictions', collect)
    if problem == 'schema':
        original_save = runner.predictor.save_raw_predictions
        def save(path, *args, **kwargs):
            original_save(path, *args, **kwargs)
            with np.load(path, allow_pickle=False) as data:
                arrays = {k:data[k] for k in data.files}
            np.savez(path, **arrays, extra=np.zeros(1))
        monkeypatch.setattr(runner.predictor, 'save_raw_predictions', save)
    with pytest.raises(ValueError):
        _run(checkout, tmp_path)
    assert not (tmp_path / 'synthetic_bundle/.complete').exists()


@pytest.mark.parametrize("name,key,value", [
    ('outputs/metrics/E6/temperature.json', 'temperature', 0.8),
    ('outputs/metrics/E7/threshold.json', 'threshold', float(np.nextafter(0.5011672918550012, 1))),
])
def test_changed_frozen_protocol_parameters_are_rejected(checkout, tmp_path, name, key, value):
    _change(checkout / name, lambda data: data.update({key:value}))
    _commit(checkout)  # Committed fixture corruption must not pass protocol validation.
    with pytest.raises(ValueError):
        _run(checkout, tmp_path)


@pytest.mark.parametrize("name", ['src/inference/predictor.py', 'configs/data.yaml'])
def test_dirty_source_or_reference_is_blocked(checkout, tmp_path, monkeypatch, name):
    path = checkout / name
    path.write_bytes(path.read_bytes() + b'\n# dirty fixture\n')
    monkeypatch.setattr(runner.predictor, 'load_e4_model', lambda *a: pytest.fail('loaded model'))
    with pytest.raises(runner.preflight.Blocked):
        _run(checkout, tmp_path)


def test_wrong_checkpoint_hash_fails_before_model_load(checkout, monkeypatch):
    _validation_fixture(checkout, monkeypatch)
    checkpoint = checkout / runner.predictor.DEFAULT_CHECKPOINT
    checkpoint.write_bytes(b'wrong checkpoint')
    monkeypatch.setattr(runner.predictor, 'load_e4_model', lambda *a: pytest.fail('loaded model'))
    with pytest.raises(RuntimeError, match='SHA-256'):
        runner.run_final_evaluation(split='validation', output_dir=checkout / 'outputs/final_evaluation_b1/validation/bad')


def test_changed_canonical_csv_is_blocked_before_images(checkout, monkeypatch):
    _validation_fixture(checkout, monkeypatch)
    csv_path = checkout / 'data/splits/validation.csv'
    csv_path.write_bytes(csv_path.read_bytes() + b'\n')
    monkeypatch.setattr(runner.predictor, 'load_e4_model', lambda *a: pytest.fail('loaded model'))
    with pytest.raises(runner.preflight.Blocked):
        runner.run_final_evaluation(split='validation', output_dir=checkout / 'outputs/final_evaluation_b1/validation/bad')


@pytest.mark.parametrize("key", ['clean_accuracy', 'image_ids', 'labels', 'gaussian_noise_accuracy_drop'])
def test_inconsistent_e8_result_fails(checkout, tmp_path, monkeypatch, key):
    original = runner.robustness._evaluate_loader
    def evaluate(*args, **kwargs):
        value = original(*args, **kwargs)
        if key == 'image_ids': value[key] = ['synthetic/wrong'] * value['n_samples']
        elif key == 'labels': value[key] = [29] * value['n_samples']
        else: value[key] += 0.1
        return value
    if key == 'labels':
        # Label extraction is intentionally in B1; mutate its actual dataset records.
        original_dataset = runner.robustness._PreNormDataset
        def dataset(*args, **kwargs):
            value = original_dataset(*args, **kwargs)
            for row in value.records: row['class_id'] = 29
            return value
        monkeypatch.setattr(runner.robustness, '_PreNormDataset', dataset)
    else:
        monkeypatch.setattr(runner.robustness, '_evaluate_loader', evaluate)
    with pytest.raises(ValueError):
        _run(checkout, tmp_path)


def test_mutating_model_weights_is_detected(checkout, tmp_path, monkeypatch):
    original = runner.robustness._evaluate_loader
    def mutate(model, *args, **kwargs):
        value = original(model, *args, **kwargs)
        with torch.no_grad(): next(model.parameters()).add_(1)
        return value
    monkeypatch.setattr(runner.robustness, '_evaluate_loader', mutate)
    with pytest.raises(ValueError, match='weights/buffers changed'):
        _run(checkout, tmp_path)


def test_no_training_refit_or_threshold_selection_is_called(checkout, tmp_path, monkeypatch):
    from src.calibration import temperature_scaling
    from src.selective import threshold
    def forbidden(*args, **kwargs): pytest.fail('forbidden fit/train call')
    monkeypatch.setattr(temperature_scaling, 'fit_temperature', forbidden)
    # Names discovered from accepted modules rather than assuming a fitting API.
    for name in vars(threshold):
        if name.startswith('select_') and callable(getattr(threshold, name)):
            monkeypatch.setattr(threshold, name, forbidden)
    monkeypatch.setattr(torch.optim.SGD, 'step', forbidden)
    _run(checkout, tmp_path)


@pytest.mark.parametrize("problem", ['e4', 'e7', 'e8', 'status', 'temperature', 'threshold'])
def test_consumer_rejects_semantically_false_rehashed_bundle(checkout, tmp_path, problem):
    _run(checkout, tmp_path)
    output = tmp_path / 'synthetic_bundle'
    if problem == 'e4': _change(output / 'e4_e6_metrics.json', lambda p:p['classification'].update(top1_accuracy=0.9))
    if problem == 'e7': _change(output / 'selective_metrics.json', lambda p:p['selected_metrics'].update(n_accept=999))
    if problem == 'e8': _change(output / 'e8_metrics.json', lambda p:p.update(clean_accuracy=0.9))
    if problem == 'status': _change(output / 'manifest.json', lambda p:p['components'].update(E8='FAIL'))
    if problem == 'temperature': _change(output / 'protocol.json', lambda p:p.update(temperature=0.8))
    if problem == 'threshold': _change(output / 'protocol.json', lambda p:p.update(threshold=float(np.nextafter(p['threshold'], 1))))
    _rebind(output)
    with pytest.raises(ValueError): runner.load_execution_bundle(output)


def test_consumer_parses_exact_verified_bytes_under_aba(checkout, tmp_path, monkeypatch):
    _run(checkout, tmp_path)
    output = tmp_path / 'synthetic_bundle'
    original = runner._validate_bundle
    def validate(buffers):
        path = output / 'metadata.json'
        saved = path.read_bytes()
        path.write_bytes(b'{"official_final_evaluation":true}')
        try:
            result = original(buffers)
            assert result['metadata']['official_final_evaluation'] is False
            return result
        finally: path.write_bytes(saved)
    monkeypatch.setattr(runner, '_validate_bundle', validate)
    runner.load_execution_bundle(output)


def test_input_aba_cannot_change_validated_reference(checkout, tmp_path, monkeypatch):
    original = runner.preflight._protocol
    def protocol(private):
        path = checkout / 'outputs/metrics/E7/threshold.json'
        saved = path.read_bytes()
        _change(path, lambda p:p.update(threshold=0.1))
        try:
            result = original(private)
            assert result['threshold'] == 0.5011672918550012
            return result
        finally: path.write_bytes(saved)
    monkeypatch.setattr(runner.preflight, '_protocol', protocol)
    _run(checkout, tmp_path)


def test_input_aliases_are_rejected_before_read(checkout, tmp_path, monkeypatch):
    target = tmp_path / 'reference.bin'
    target.write_bytes(b'not a dataset')
    alias = checkout / 'configs/data.yaml'
    alias.unlink()
    os.link(target, alias)
    with pytest.raises(PermissionError, match='Hard-linked'):
        _run(checkout, tmp_path)


def test_input_symbolic_link_is_rejected(checkout, tmp_path):
    target = tmp_path / 'reference.bin'
    target.write_bytes(b'not a dataset')
    alias = checkout / 'configs/data.yaml'
    alias.unlink()
    try:
        alias.symlink_to(target)
    except OSError:
        # Windows without symlink privilege: exercise a directory junction.
        directory = tmp_path / 'linked_configs'
        directory.mkdir()
        (directory / 'data.yaml').write_bytes(target.read_bytes())
        assert (checkout / 'configs').resolve().is_relative_to(tmp_path.resolve())
        assert not checkout.resolve().is_relative_to(REAL_ROOT)
        shutil.rmtree(checkout / 'configs')
        result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(checkout / 'configs'), str(directory)], capture_output=True)
        assert result.returncode == 0, result.stderr
    with pytest.raises(PermissionError, match='Symlink/junction'):
        _run(checkout, tmp_path)


def test_synthetic_root_cannot_be_inside_or_above_checkout(checkout, tmp_path):
    for output in (checkout / 'new_run', checkout.parent):
        with pytest.raises((PermissionError, FileExistsError)):
            runner.run_final_evaluation(split='synthetic', output_dir=output)


def test_uncommitted_development_is_synthetic_only(checkout, tmp_path):
    _git(checkout, 'rm', '--cached', *runner.NEW_SOURCE)
    _git(checkout, '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture untracks B1')
    result = _run(checkout, tmp_path)
    assert result['source']['status'] == 'BLOCKED'
    with pytest.raises(runner.preflight.Blocked):
        runner.run_final_evaluation(split='validation', output_dir=checkout / 'outputs/final_evaluation_b1/validation/not_allowed')


@pytest.mark.parametrize('alias', ['name.', 'name ', 'value:stream', 'NUL', 'CON.txt', r'\\?\C:\unused'])
def test_device_and_stream_aliases_are_refused(checkout, tmp_path, alias):
    with pytest.raises(PermissionError):
        runner._safe_path(tmp_path / alias if not alias.startswith('\\\\') else alias)


def test_clean_logits_and_e8_share_normalization_eval_and_no_grad(checkout, tmp_path, monkeypatch):
    original = runner.robustness._make_synthetic_model
    observed = []
    def model_factory(seed):
        model = original(seed)
        def hook(module, args):
            assert not module.training
            assert not torch.is_grad_enabled()
            observed.append(args[0].clone())
        model.register_forward_pre_hook(hook)
        return model
    monkeypatch.setattr(runner.robustness, '_make_synthetic_model', model_factory)
    _run(checkout, tmp_path)
    # Clean collector: batches 0,1. Shared E8 loader evaluates four conditions
    # per batch: 2..5 then 6..9; Clean tensors must be identical to E4 inputs.
    assert len(observed) == 10
    assert torch.equal(observed[0], observed[2])
    assert torch.equal(observed[1], observed[6])
    assert observed[0].min() < 0 and observed[0].max() > 1


def test_per_image_corruption_independent_of_batch_and_order(checkout):
    from src.robustness.robustness_eval import RobustnessEvaluator
    cfg = runner.load_e8_config(checkout / 'configs/experiments/e8_robustness.yaml')
    evaluator = RobustnessEvaluator(runner.robustness._make_synthetic_model(42), cfg)
    images = torch.rand(4, 3, 224, 224, generator=torch.Generator().manual_seed(9))
    ids = [f'synthetic/{i}.png' for i in range(4)]
    order = [3, 1, 0, 2]
    for condition in ('gaussian_noise', 'occlusion'):
        reference = evaluator._corrupt(images, ids, condition)
        reordered = torch.cat([evaluator._corrupt(images[order[i:i+2]], [ids[j] for j in order[i:i+2]], condition) for i in (0, 2)])
        for i, source in enumerate(order):
            assert torch.equal(reference[source], reordered[i])


def test_foreign_marker_survives_rollback(checkout, tmp_path, monkeypatch):
    original = os.link
    def link(source, destination, *args, **kwargs):
        if destination.name == 'e8_metrics.json':
            (destination.parent / '.incomplete').write_bytes(b'FOREIGN MARKER')
            raise OSError('publish failed')
        return original(source, destination, *args, **kwargs)
    monkeypatch.setattr(os, 'link', link)
    with pytest.raises(OSError): _run(checkout, tmp_path)
    assert (tmp_path / 'synthetic_bundle/.incomplete').read_bytes() == b'FOREIGN MARKER'


def test_completion_write_failure_stays_incomplete(checkout, tmp_path, monkeypatch):
    original = runner._write
    def fail(path, data):
        if path.name == '.complete':
            with path.open('xb') as handle: handle.write(b'{partial')
            raise OSError('completion failed')
        original(path, data)
    monkeypatch.setattr(runner, '_write', fail)
    with pytest.raises(OSError, match='completion failed'): _run(checkout, tmp_path)
    assert (tmp_path / 'synthetic_bundle/.incomplete').exists()
    with pytest.raises(runner.preflight.Blocked):
        runner.load_execution_bundle(tmp_path / 'synthetic_bundle')


def test_filesystem_without_hardlinks_fails_closed(checkout, tmp_path, monkeypatch):
    def fail(*args, **kwargs): raise OSError('no hardlinks on this filesystem')
    monkeypatch.setattr(os, 'link', fail)
    with pytest.raises(OSError, match='no hardlinks'): _run(checkout, tmp_path)
    assert not (tmp_path / 'synthetic_bundle/.complete').exists()


@pytest.mark.parametrize('problem', ['preprocessing', 'severity', 'e7_parameter', 'count', 'manifest_hash'])
def test_consumer_detects_changed_protocol_and_internal_hashes(checkout, tmp_path, problem):
    _run(checkout, tmp_path)
    output = tmp_path / 'synthetic_bundle'
    if problem == 'preprocessing': _change(output / 'metadata.json', lambda p:p['preprocessing']['evaluation'].update(resize=255))
    if problem == 'severity': _change(output / 'protocol.json', lambda p:p['corruption_config']['gaussian_noise'].update(std=0.1))
    if problem == 'e7_parameter': _change(output / 'selective_metrics.json', lambda p:p.update(threshold=float(np.nextafter(p['threshold'], 1))))
    if problem == 'count': _change(output / 'e8_metrics.json', lambda p:p['correct_counts'].update(clean=100))
    _rebind(output)
    if problem == 'manifest_hash':
        _change(output / 'manifest.json', lambda p:p['output_sha256'].update({'metadata.json':'0'*64}))
        complete = _load_json(output / '.complete')
        complete['output_sha256']['manifest.json'] = runner._hash((output / 'manifest.json').read_bytes())
        (output / '.complete').write_bytes(runner._json_bytes(complete))
    with pytest.raises(ValueError): runner.load_execution_bundle(output)


def test_path_to_generated_consumer_files_never_follows_package_claim(checkout, tmp_path, monkeypatch):
    _run(checkout, tmp_path)
    output = tmp_path / 'synthetic_bundle'
    _change(output / 'metadata.json', lambda p:p.update(checkpoint_path='blocked/path/not_to_open'))
    _rebind(output)
    original = runner.preflight._read_once
    def read(path, *args, **kwargs):
        assert 'not_to_open' not in str(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(runner.preflight, '_read_once', read)
    # Unrecognized path metadata is inert, never an instruction to open a file.
    runner.load_execution_bundle(output)


def test_legacy_reader_refuses_inflight_and_completed_b1_npz(checkout, tmp_path, monkeypatch):
    original = os.link
    checked = []
    def link(source, destination, *args, **kwargs):
        if destination.name == 'e4_e6_metrics.json':
            public = destination.parent / 'clean_predictions.npz'
            assert public.exists() and not (destination.parent / '.complete').exists()
            with pytest.raises(runner.predictor.InferenceContractError, match='Incomplete'):
                runner.predictor.load_raw_predictions(public)
            checked.append(True)
        return original(source, destination, *args, **kwargs)
    monkeypatch.setattr(os, 'link', link)
    _run(checkout, tmp_path)
    assert checked == [True]
    output = tmp_path / 'synthetic_bundle'
    assert runner.load_execution_bundle(output)['manifest']['execution_status'] == 'complete'
    with pytest.raises(runner.predictor.InferenceContractError):
        runner.predictor.load_raw_predictions(output / 'clean_predictions.npz')


@pytest.mark.parametrize('flag', ['--assume-unchanged', '--skip-worktree'])
def test_p1_regression_hidden_source_mutation_is_rejected(checkout, flag):
    name = 'src/models/mobilenet_v2.py'
    path = checkout / name
    _git(checkout, 'update-index', flag, name)
    path.write_bytes(path.read_bytes() + b'\n# hidden local source change\n')
    assert _git(checkout, 'diff', 'HEAD', '--name-only', '--', name).strip() == b''
    assert path.read_bytes() != _git(checkout, 'show', 'HEAD:' + name)
    with pytest.raises((ValueError, RuntimeError)):
        runner._source_audit('validation')


@pytest.mark.parametrize('rewrite_input_record', [False, True])
def test_p1_regression_rehashed_noncanonical_validation_ids_are_rejected(checkout, monkeypatch, rewrite_input_record):
    _validation_fixture(checkout, monkeypatch)
    output = checkout / 'outputs/final_evaluation_b1/validation/tamper'
    runner.run_final_evaluation(split='validation', output_dir=output, batch_size=128)
    canonical = (checkout / 'data/splits/validation.csv').read_bytes()
    with np.load(output / 'clean_predictions.npz', allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    old_id = str(arrays['image_ids'][0])
    new_id = old_id.replace('image_000', 'other_000')
    arrays['image_ids'][0] = new_id
    np.savez_compressed(output / 'clean_predictions.npz', **arrays)
    (output / 'split.csv').write_bytes((output / 'split.csv').read_bytes().replace(old_id.encode(), new_id.encode()))
    _change(output / 'e8_metrics.json', lambda value: value['image_ids'].__setitem__(0, new_id))
    _change(output / 'metadata.json', lambda value: value.update(
        raw_predictions_sha256=runner._hash((output / 'clean_predictions.npz').read_bytes()),
        csv_sha256=runner._hash((output / 'split.csv').read_bytes())))
    if rewrite_input_record:
        def rewrite(manifest):
            record = next(record for record in manifest['inputs'].values() if record['snapshot_path'] == 'data/splits/validation.csv')
            raw = (output / 'split.csv').read_bytes()
            record.update(sha256=runner._hash(raw), size_bytes=len(raw))
        _change(output / 'manifest.json', rewrite)
    _rebind(output)
    assert (checkout / 'data/splits/validation.csv').read_bytes() == canonical
    assert runner._hash(canonical) != runner._hash((output / 'split.csv').read_bytes())
    with pytest.raises((ValueError, RuntimeError)):
        runner.load_execution_bundle(output)


@pytest.mark.parametrize('problem', ['empty_inputs', 'empty_source_hashes', 'dirty_references', 'untracked_references'])
def test_p1_regression_rehashed_missing_or_contradictory_evidence_is_rejected(checkout, tmp_path, problem):
    _run(checkout, tmp_path)
    output = tmp_path / 'synthetic_bundle'
    def change(manifest):
        if problem == 'empty_inputs': manifest['inputs'] = {}
        if problem == 'empty_source_hashes': manifest['source_sha256'] = {}
        if problem == 'dirty_references': manifest['frozen_references']['modified'] = ['configs/data.yaml']
        if problem == 'untracked_references': manifest['frozen_references']['untracked'] = ['outputs/metrics/E6/temperature.json']
    _change(output / 'manifest.json', change)
    _rebind(output)
    with pytest.raises((ValueError, RuntimeError)):
        runner.load_execution_bundle(output)


@pytest.fixture(scope='module')
def sealed_provenance_bundle(tmp_path_factory):
    """One real synthetic run; negative cases mutate independent public copies."""
    base = tmp_path_factory.mktemp('b1_provenance_seed')
    with pytest.MonkeyPatch.context() as patch:
        root = checkout.__wrapped__(base, patch)
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            _run(root, base)
        finally:
            torch.set_num_threads(old_threads)
        output = base / 'synthetic_bundle'
        return root, {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}


@pytest.fixture
def provenance_case(sealed_provenance_bundle, tmp_path, monkeypatch):
    root, files = sealed_provenance_bundle
    monkeypatch.setattr(runner, 'SOURCE_ROOT', root)
    output = tmp_path / 'copied_bundle'
    output.mkdir()
    for name, raw in files.items():
        (output / name).write_bytes(raw)
    return output


@pytest.mark.parametrize('problem', [
    'missing_source', 'source_hash', 'source_binding', 'source_blob_hash', 'source_modified', 'source_untracked',
    'missing_binding', 'missing_runtime', 'wrong_runtime', 'missing_command', 'wrong_tolerance',
    'missing_csv_record', 'missing_mapping_record', 'record_hash', 'record_size', 'record_type',
    'duplicate_record', 'extra_record', 'reference_coverage', 'missing_source_content',
])
def test_rehashed_bad_provenance_schema_or_content_is_rejected(provenance_case, problem):
    output = provenance_case
    def change(manifest):
        name = 'src/models/mobilenet_v2.py'
        if problem == 'missing_source': manifest['source_sha256'].pop(name)
        if problem == 'source_hash': manifest['source_sha256'][name] = '0' * 64
        if problem == 'source_binding':
            manifest['source_sha256'][name] = '0' * 64
            manifest['source']['content'][name]['sha256'] = '0' * 64
        if problem == 'source_blob_hash': manifest['source']['content'][name]['git_blob_sha256'] = '0' * 64
        if problem == 'source_modified': manifest['source']['modified'] = [name]
        if problem == 'source_untracked': manifest['source']['untracked'] = [name]
        if problem == 'missing_binding': manifest.pop('input_binding')
        if problem == 'missing_runtime': manifest.pop('runtime')
        if problem == 'wrong_runtime': manifest['runtime']['batch_size'] = True
        if problem == 'missing_command': manifest.pop('command')
        if problem == 'wrong_tolerance': manifest['comparison_tolerance']['metrics_relative'] = 0.1
        if problem == 'missing_source_content': manifest['source'].pop('content')
        if problem == 'reference_coverage': manifest['frozen_references']['audited_files'].pop()
        records = manifest['inputs']
        csv_key = next(key for key, record in records.items() if record['snapshot_path'] == 'data/splits/synthetic.csv')
        map_key = next(key for key, record in records.items() if record['snapshot_path'] == 'data/metadata/class_mapping.json')
        if problem == 'missing_csv_record': records.pop(csv_key)
        if problem == 'missing_mapping_record': records.pop(map_key)
        if problem == 'record_hash': records[map_key]['sha256'] = '0' * 64
        if problem == 'record_size': records[map_key]['size_bytes'] += 1
        if problem == 'record_type': records[map_key]['size_bytes'] = True
        if problem == 'duplicate_record': records['another-origin'] = dict(records[map_key])
        if problem == 'extra_record':
            records['extra-origin'] = dict(records[map_key], snapshot_path='images/undeclared.png')
    _change(output / 'manifest.json', change)
    _rebind(output)
    with pytest.raises((ValueError, RuntimeError)):
        runner.load_execution_bundle(output)


@pytest.mark.parametrize('rewrite_record', [False, True])
def test_rehashed_mapping_and_csv_names_must_match_git(provenance_case, rewrite_record):
    output = provenance_case
    mapping = _load_json(output / 'class_mapping.json')
    old_name = mapping['0']
    mapping['0'] = old_name + '_changed'
    (output / 'class_mapping.json').write_bytes(runner._json_bytes(mapping))
    (output / 'split.csv').write_bytes((output / 'split.csv').read_bytes().replace(old_name.encode(), mapping['0'].encode()))
    _change(output / 'e4_e6_metrics.json', lambda value: value['classification']['per_class'][0].update(class_name=mapping['0']))
    _change(output / 'metadata.json', lambda value: value.update(
        csv_sha256=runner._hash((output / 'split.csv').read_bytes()),
        class_mapping_sha256=runner._hash((output / 'class_mapping.json').read_bytes())))
    if rewrite_record:
        def rewrite(manifest):
            for record in manifest['inputs'].values():
                published = {'data/splits/synthetic.csv':'split.csv', 'data/metadata/class_mapping.json':'class_mapping.json'}.get(record['snapshot_path'])
                if published:
                    raw = (output / published).read_bytes()
                    record.update(sha256=runner._hash(raw), size_bytes=len(raw))
        _change(output / 'manifest.json', rewrite)
    _rebind(output)
    with pytest.raises(ValueError):
        runner.load_execution_bundle(output)


def test_consumer_missing_trusted_git_store_is_blocked(provenance_case, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'SOURCE_ROOT', tmp_path / 'missing_git_store')
    with pytest.raises(runner.preflight.Blocked, match='Git blobs'):
        runner.load_execution_bundle(provenance_case)


def test_missing_canonical_git_blob_is_blocked(checkout):
    commit = _git(checkout, 'rev-parse', 'HEAD').decode().strip()
    # Synthetic fixture Git intentionally has no Validation CSV.
    with pytest.raises(runner.preflight.Blocked, match='Missing or invalid Git blob'):
        runner._git_blobs(commit, ['data/splits/validation.csv'])


@pytest.mark.parametrize('newline', ['lf', 'crlf'])
def test_source_audit_verified_newline_variant_records_exact_checked_hash(checkout, newline):
    name = 'src/models/mobilenet_v2.py'
    path = checkout / name
    lf = path.read_bytes().replace(b'\r\n', b'\n')
    path.write_bytes(lf)
    _commit(checkout)
    _git(checkout, 'update-index', '--assume-unchanged', name)
    actual = lf if newline == 'lf' else lf.replace(b'\n', b'\r\n')
    path.write_bytes(actual)
    evidence = runner._source_audit('validation')
    assert evidence['status'] == 'PASS'
    record = evidence['content'][name]
    assert record['sha256'] == runner._hash(actual)
    assert record['git_blob_sha256'] == runner._hash(lf)
    assert record['match_mode'] == ('exact' if newline == 'lf' else 'newline_variant_crlf')


def test_source_hash_uses_checked_buffer_and_detects_later_newline_drift(checkout, tmp_path, monkeypatch):
    name = 'src/models/mobilenet_v2.py'
    path = checkout / name
    lf = path.read_bytes().replace(b'\r\n', b'\n')
    path.write_bytes(lf)
    _commit(checkout)
    _git(checkout, 'update-index', '--assume-unchanged', name)
    original = runner.preflight._read_once
    def capture(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        if Path(value) == path:
            path.write_bytes(lf.replace(b'\n', b'\r\n'))
        return result
    with monkeypatch.context() as patch:
        patch.setattr(runner.preflight, '_read_once', capture)
        source = runner._source_audit('synthetic')
    assert source['content'][name]['sha256'] == runner._hash(lf)
    assert runner._hash(path.read_bytes()) != source['content'][name]['sha256']
    private = tmp_path / 'private'
    private.mkdir()
    inputs = runner.ExecutionInputs(private, 'synthetic', source)
    assert inputs.source_hashes[name] == runner._hash(lf)
    with pytest.raises(runner.preflight.Blocked, match='Source Git state changed'):
        inputs.verify_source()


@pytest.mark.parametrize('position', [3, 4])
def test_publish_failure_at_third_or_fourth_artifact_preserves_foreign_file(tmp_path, monkeypatch, position):
    output = tmp_path / 'partial_bundle'
    original = os.link
    victim = runner.PAYLOADS[position - 1]
    def link(source, destination, *args, **kwargs):
        if destination.name == victim:
            destination.write_bytes(b'FOREIGN WRITER FILE')
        return original(source, destination, *args, **kwargs)
    monkeypatch.setattr(os, 'link', link)
    with pytest.raises(FileExistsError):
        with runner._output_run(output) as staging:
            payload = staging / 'payload'
            payload.mkdir()
            for name in runner.PAYLOADS:
                (payload / name).write_bytes(b'OWNED PRIVATE DATA')
    assert (output / victim).read_bytes() == b'FOREIGN WRITER FILE'
    assert (output / '.incomplete').exists() and not (output / '.complete').exists()
    assert not list(tmp_path.glob('.dogbreedid-b1-private-*'))
    with pytest.raises(runner.preflight.Blocked):
        runner.load_execution_bundle(output)


def test_source_audit_covers_transitive_local_imports_and_package_initializers():
    import ast
    # Resolve imports from actual source, not a duplicated hand-maintained list.
    pending = ['scripts.run_final_evaluation', 'src.models.mobilenet_v2']
    modules, paths = set(), set()
    while pending:
        module = pending.pop()
        if module in modules or not module.startswith(('scripts.', 'src.')):
            continue
        modules.add(module)
        relative = Path(module.replace('.', '/'))
        path = REAL_ROOT / relative.with_suffix('.py')
        package = False
        if not path.is_file():
            path = REAL_ROOT / relative / '__init__.py'
            package = True
        if not path.is_file():
            continue
        paths.add(path.relative_to(REAL_ROOT).as_posix())
        parts = module.split('.')
        pending.extend('.'.join(parts[:i]) for i in range(1, len(parts)))
        scope = parts if package else parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = '.'.join([*scope[:len(scope)-node.level+1], node.module or '']).rstrip('.') if node.level else node.module or ''
                pending.extend([base, *(base + '.' + alias.name for alias in node.names)])
            elif isinstance(node, ast.Call) and getattr(node.func, 'attr', getattr(node.func, 'id', '')) == 'import_module':
                pending.extend(value.value for value in node.args if isinstance(value, ast.Constant) and isinstance(value.value, str))
    assert {'src/calibration/__init__.py', 'src/inference/__init__.py', 'src/robustness/__init__.py'} <= paths
    assert paths <= set(runner.SOURCE_FILES), sorted(paths - set(runner.SOURCE_FILES))


def test_source_binding_uses_pinned_git_object_not_mutable_replace_reference(checkout):
    name = 'src/models/mobilenet_v2.py'
    path = checkout / name
    original_oid = _git(checkout, 'rev-parse', 'HEAD:' + name).decode().strip()
    replacement = path.read_bytes() + b'\n# source hidden behind a mutable Git replace ref\n'
    changed_oid = subprocess.check_output(['git', '-C', str(checkout), 'hash-object', '-w', '--stdin'], input=replacement).decode().strip()
    _git(checkout, 'replace', original_oid, changed_oid)
    _git(checkout, 'update-index', '--assume-unchanged', name)
    path.write_bytes(replacement)
    assert _git(checkout, 'diff', 'HEAD', '--name-only', '--', name).strip() == b''
    with pytest.raises(ValueError, match='canonical Git blob'):
        runner._source_audit('validation')
