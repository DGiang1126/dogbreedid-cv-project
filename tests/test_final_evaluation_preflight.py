"""Phase A audits isolated, explicitly synthetic fixtures; never image inference."""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

from scripts import run_final_evaluation_preflight as runner
from src.inference import predictor


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _fixture(tmp_path, mode="synthetic"):
    root = tmp_path / "fixture"
    for name in runner.REFERENCE_FILES:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((runner.SOURCE_ROOT / name).read_bytes())
    n = 300 if mode == "validation" else 33
    labels = np.arange(n, dtype=np.int64) % 30
    ids = np.array([f'{"validation" if mode == "validation" else "synthetic"}/image_{i:03d}.jpg' for i in range(n)])
    logits = np.random.default_rng(19).normal(size=(n, 30)).astype(np.float32)
    logits[np.arange(n), labels] += np.linspace(1, 7, n, dtype=np.float32)
    prediction = root / (runner.PREDICTIONS if mode == "validation" else "outputs/predictions/synthetic")
    e7_dir = root / (runner.SELECTIVE if mode == "validation" else "outputs/selective/synthetic")
    e8_path = root / (runner.ROBUSTNESS if mode == "validation" else "outputs/robustness/synthetic/e8.json")
    prediction.mkdir(parents=True)
    npz = prediction / "validation_raw_predictions.npz"
    predictor.save_raw_predictions(npz, logits, labels, ids, exclusive=True)
    mapping = json.loads((root / "data/metadata/class_mapping.json").read_text())
    csv_path = root / ("data/splits/validation.csv" if mode == "validation" else "data/splits/synthetic.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("relative_path", "class_id", "breed_name", "source_split"))
        writer.writeheader()
        for image_id, label in zip(ids, labels):
            writer.writerow(dict(relative_path=str(image_id), class_id=int(label), breed_name=mapping[str(label)],
                                 source_split="official_train" if mode == "validation" else "synthetic"))
    checkpoint = root / predictor.DEFAULT_CHECKPOINT
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"synthetic checksum fixture, never model weights")
    data = predictor._load_yaml(root / "configs/data.yaml")
    checkpoint_hash = predictor.EXPECTED_CHECKPOINT_SHA256 if mode == "validation" else predictor.sha256_file(checkpoint)
    metadata = dict(experiment_id="E4_TASK04_FINAL_INFERENCE", split=mode, data_status=mode, synthetic=mode == "synthetic",
                    num_samples=n, num_classes=30, git_sha="a" * 40, class_mapping_path="data/metadata/class_mapping.json",
                    class_mapping_sha256=predictor.sha256_file(root / "data/metadata/class_mapping.json"),
                    checkpoint_path=predictor.DEFAULT_CHECKPOINT.as_posix(), checkpoint_sha256=checkpoint_hash,
                    temperature=predictor.EXPECTED_TEMPERATURE, temperature_artifact_path="outputs/metrics/E6/temperature.json",
                    temperature_source_experiment="E6", temperature_source_split="calibration",
                    preprocessing=dict(config="configs/data.yaml", transform_builder="src.data.transforms.build_transforms",
                                       split_argument="validation" if mode == "validation" else "synthetic",
                                       expected_tensor_shape=[3, 224, 224], evaluation_resize=256, evaluation_center_crop=224,
                                       normalization=data["preprocessing"]["normalization"]),
                    runtime=dict(timestamp_utc="2026-10-25T00:00:00+00:00", python_version="fixture", pytorch_version="fixture",
                                 torchvision_version="fixture", device="cpu", batch_size=32, num_workers=0),
                    safety=dict(official_test_used=False, model_trained_or_updated=False, temperature_refit=False))
    meta_path = prediction / "validation_metadata.json"
    _write_json(meta_path, metadata)
    names = {i: mapping[str(i)] for i in range(30)}
    metrics = predictor.compute_task1_metrics(logits, labels, temperature=predictor.EXPECTED_TEMPERATURE, class_names=names)
    _write_json(prediction / "validation_metrics.json", metrics)
    Image.new("RGB", (2, 2)).save(prediction / "validation_reliability_diagram.png")
    bundle = runner.selective.load_prediction_bundle(npz, meta_path, split=mode, split_csv=csv_path, repo_root=root)
    calibrated = runner.selective.calibrated_predictions(logits, bundle["temperature"])
    correct = calibrated["predictions"] == labels
    accepted = runner.selective.accept_mask(calibrated["confidences"], bundle["threshold"])
    e7 = dict(experiment_id="E4_TASK04_FINAL_SELECTIVE", split=mode, synthetic=mode == "synthetic", official_final_evaluation=False,
              threshold_reselected=False, accept_operator=">=", temperature=bundle["temperature"], threshold=bundle["threshold"],
              source_git_sha="b" * 40, upstream_git_sha=metadata["git_sha"], n_total=n,
              selected_metrics=runner.selective.selective_metrics(correct, accepted),
              aurc=runner.selective.compute_aurc(calibrated["confidences"], correct),
              aurc_convention="grouped right-step over confidence tie groups", input_sha256=bundle["input_sha256"],
              model=dict(experiment="E4", architecture="mobilenet_v2", seed=42, num_classes=30,
                         checkpoint_sha256=predictor.EXPECTED_CHECKPOINT_SHA256 if mode == "validation" else None))
    _write_json(e7_dir / "selective_metrics.json", e7)
    (e7_dir / "risk_coverage.csv").write_text(runner.selective._curve_csv(runner.selective.risk_coverage_curve(calibrated["confidences"], correct)), encoding="utf-8", newline="\n")
    Image.new("RGB", (2, 2)).save(e7_dir / "risk_coverage_curve.png")
    cfg = runner.load_e8_config(root / "configs/experiments/e8_robustness.yaml")
    e8 = dict(experiment_id="E8_TASK04_ROBUSTNESS", split=mode, synthetic=mode == "synthetic", official_evaluation=False,
              num_classes=30, n_samples=n, conditions=cfg.conditions, seed=cfg.reproducibility.seed, image_ids=ids.tolist(),
              model_experiment="E4" if mode == "validation" else "synthetic_dummy", split_csv=str(csv_path),
              checkpoint=str(checkpoint) if mode == "validation" else None,
              checkpoint_sha256=predictor.EXPECTED_CHECKPOINT_SHA256 if mode == "validation" else None,
              preprocessing=data["preprocessing"], corruption_config=json.loads(json.dumps({k: asdict(getattr(cfg, k)) for k in cfg.conditions[1:]})),
              clean_accuracy=float(np.mean(logits.argmax(1) == labels)))
    for name, accuracy in zip(cfg.conditions[1:], (0.4, 0.5, 0.6)):
        e8[name + "_accuracy"] = accuracy
        e8[name + "_accuracy_drop"] = e8["clean_accuracy"] - accuracy
    _write_json(e8_path, e8)
    if mode == "synthetic":
        _write_json(root / runner.FIXTURE_MARKER, dict(format=runner.FIXTURE_FORMAT, synthetic=True,
                    generator="tests/test_final_evaluation_preflight.py"))
    return dict(root=root, prediction=prediction, e7_dir=e7_dir, e8_path=e8_path, checkpoint=checkpoint,
                npz=npz, meta_path=meta_path, csv=csv_path, logits=logits, labels=labels, ids=ids)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    value = _fixture(tmp_path)
    # Only Git evidence is simulated. Validators, metrics and artifact I/O are real.
    monkeypatch.setattr(runner, "_git_state", lambda root, files: dict(
        head="a" * 40, status="PASS", modified=[], untracked=[], audited_files=list(files), evidence="synthetic Git mock"))
    _mock_git_objects(value, monkeypatch)
    return value


def _mock_git_objects(value, monkeypatch):
    # Immutable stand-in for Git's object store, not a mock of snapshot capture,
    # content comparison, the shared validators or the metric calculations.
    root = value["root"]
    names = (*runner.REFERENCE_FILES, value["csv"].relative_to(root).as_posix())
    objects = {name: (root / name).read_bytes() for name in names}
    monkeypatch.setattr(runner, "_head_reference_bytes", lambda origin, commit, relative: objects[relative])


def _run(fixture, **kwargs):
    options = dict(repo_root=fixture["root"], mode="synthetic", output_dir="outputs/preflight/run")
    options.update(kwargs)
    return runner.run_preflight(**options)


def _check(fixture, name):
    rows = json.loads((fixture["root"] / "outputs/preflight/run/preflight_checks.json").read_text())["checks"]
    return next(row for row in rows if row["id"] == name)


def _change_json(path, change):
    payload = json.loads(path.read_text())
    change(payload)
    _write_json(path, payload)


def test_valid_synthetic_fixture_is_read_only_and_completed(fixture):
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in fixture["root"].rglob('*') if p.is_file()}
    result = _run(fixture)
    assert result["overall_status"] == "PASS"
    assert result["official_test_used"] is False
    assert result["configuration_lock_approved"] is False
    assert result["inference_performed"] is False
    output = fixture["root"] / "outputs/preflight/run"
    assert {p.name for p in output.iterdir()} == {*runner.REPORT_FILES, runner.RUN_MARKER, ".complete"}
    assert runner.load_preflight_report(output) == result
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in before.items())


@pytest.mark.parametrize("name", ["npz", "meta_path", "e8_path", "checkpoint"])
def test_missing_required_artifact_blocks_pass(fixture, name):
    fixture[name].unlink()
    result = _run(fixture)
    assert result["overall_status"] == "BLOCKED"
    assert result["check_counts"]["BLOCKED"] > 0


@pytest.mark.parametrize("problem", ["ids", "labels", "schema", "shape", "dtype_logits", "dtype_labels", "dtype_ids", "nan", "inf"])
def test_bad_prediction_contract_is_rejected(fixture, problem):
    arrays = dict(logits=fixture["logits"].copy(), labels=fixture["labels"].copy(), image_ids=fixture["ids"].copy())
    if problem == "ids": arrays["image_ids"][0] = "synthetic/wrong.jpg"
    elif problem == "labels": arrays["labels"][0] = 29
    elif problem == "schema": arrays["extra"] = np.ones(1)
    elif problem == "shape": arrays["logits"] = arrays["logits"][:, :29]
    elif problem == "dtype_logits": arrays["logits"] = arrays["logits"].astype(np.int64)
    elif problem == "dtype_labels": arrays["labels"] = arrays["labels"].astype(np.int32)
    elif problem == "dtype_ids": arrays["image_ids"] = arrays["image_ids"].astype('S64')
    elif problem in {"nan", "inf"}: arrays["logits"][0, 0] = float(problem)
    np.savez(fixture["npz"], **arrays)
    assert _run(fixture)["overall_status"] == "FAIL"
    assert _check(fixture, "prediction_bundle")["status"] == "FAIL"


def test_checkpoint_mismatch_is_rejected(fixture):
    fixture["checkpoint"].write_bytes(b"wrong fixture checkpoint")
    assert _run(fixture)["overall_status"] == "FAIL"
    assert _check(fixture, "checkpoint_sha256")["status"] == "FAIL"


@pytest.mark.parametrize("name,key,value", [
    ("outputs/metrics/E6/temperature.json", "temperature", 0.8),
    ("outputs/metrics/E7/threshold.json", "threshold", 0.5),
])
def test_frozen_temperature_or_threshold_mismatch_fails(fixture, name, key, value):
    _change_json(fixture["root"] / name, lambda payload: payload.update({key: value}))
    assert _run(fixture)["overall_status"] == "FAIL"
    assert _check(fixture, "protocol_references")["status"] == "FAIL"


@pytest.mark.parametrize("problem", ["conditions", "split", "checkpoint", "corruption", "drop", "preprocessing", "ids", "missing_metadata"])
def test_bad_e8_evidence_never_passes(fixture, problem):
    def change(value):
        if problem == "conditions": value["conditions"].pop()
        elif problem == "split": value["split"] = "train"
        elif problem == "checkpoint": value["checkpoint_sha256"] = "f" * 64
        elif problem == "corruption": value["corruption_config"]["gaussian_noise"]["std"] = 0.5
        elif problem == "drop": value["occlusion_accuracy_drop"] += 0.1
        elif problem == "preprocessing": value["preprocessing"]["evaluation"]["resize"] = 128
        elif problem == "ids": value["image_ids"][0] = "synthetic/wrong.jpg"
        else: del value["corruption_config"]
    _change_json(fixture["e8_path"], change)
    result = _run(fixture)
    expected = "BLOCKED" if problem == "missing_metadata" else "FAIL"
    assert result["overall_status"] == expected
    assert _check(fixture, "e8_artifact")["status"] == expected


@pytest.mark.parametrize("problem", ["ece", "top1", "e7_count", "e7_aurc", "e7_hash", "e7_csv"])
def test_corrupted_metrics_are_not_accepted(fixture, problem):
    if problem in {"ece", "top1"}:
        def change(value):
            if problem == "ece": value["calibration"]["after_temperature_scaling"]["ece"] += 0.01
            else: value["classification"]["top1_accuracy"] += 0.01
        _change_json(fixture["prediction"] / "validation_metrics.json", change)
    elif problem == "e7_csv":
        (fixture["e7_dir"] / "risk_coverage.csv").write_text("wrong csv")
    else:
        def change(value):
            if problem == "e7_count": value["selected_metrics"]["n_accept"] += 1
            elif problem == "e7_aurc": value["aurc"] += 0.01
            else: value["input_sha256"]["predictions"] = "f" * 64
        _change_json(fixture["e7_dir"] / "selective_metrics.json", change)
    assert _run(fixture)["overall_status"] == "FAIL"


@pytest.mark.parametrize("marker", ["incomplete", "missing_complete"])
def test_pending_or_incomplete_input_is_blocked(fixture, marker):
    if marker == "incomplete": (fixture["prediction"] / ".incomplete").write_bytes(b"failed")
    else: (fixture["prediction"] / ".inference_run").write_bytes(b"a" * 32 + b"\n")
    assert _run(fixture)["overall_status"] != "PASS"
    assert _check(fixture, "prediction_bundle")["status"] == "BLOCKED"


def test_existing_output_even_empty_is_refused_before_artifact_reads(fixture, monkeypatch):
    output = fixture["root"] / "outputs/preflight/run"
    output.mkdir(parents=True)
    monkeypatch.setattr(runner, "_manifest", lambda *args: pytest.fail("Input opened before output preflight"))
    with pytest.raises(FileExistsError): _run(fixture)
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_write_failure_retains_incomplete_output_and_foreign_file(fixture, monkeypatch, fail_at):
    original = runner.shutil.copyfileobj
    output = fixture["root"] / "outputs/preflight/run"
    calls = 0
    staging = []
    def fail(reader, writer):
        nonlocal calls
        calls += 1
        staging.append(Path(reader.name).parent)
        if calls == fail_at:
            writer.write(b"partial")
            writer.flush()
            writer.close()  # Allow the foreign writer's rename on Windows.
            destination = Path(writer.name)
            destination.rename(fixture["root"] / "owned_partial_backup")
            destination.write_bytes(b"FOREIGN replacement")
            raise OSError("injected publish failure")
        return original(reader, writer)
    monkeypatch.setattr(runner.shutil, "copyfileobj", fail)
    with pytest.raises(OSError, match="injected publish failure"): _run(fixture)
    assert (output / runner.REPORT_FILES[fail_at - 1]).read_bytes() == b"FOREIGN replacement"
    assert (output / ".incomplete").is_file()
    assert not (output / ".complete").exists()
    assert staging and all(not path.exists() for path in staging)
    with pytest.raises(runner.Blocked): runner.load_preflight_report(output)


def test_artifact_created_between_preflight_and_publish_survives(fixture, monkeypatch):
    original = runner._classification
    output = fixture["root"] / "outputs/preflight/run"
    def race(*args):
        (output / runner.REPORT_FILES[1]).write_bytes(b"foreign report")
        return original(*args)
    monkeypatch.setattr(runner, "_classification", race)
    with pytest.raises(FileExistsError): _run(fixture)
    assert (output / runner.REPORT_FILES[1]).read_bytes() == b"foreign report"
    assert (output / ".incomplete").exists()


@pytest.mark.parametrize("mode", ["official_test", "test", "Official Test"])
def test_forbidden_split_is_refused_without_opening_inputs(fixture, monkeypatch, mode):
    monkeypatch.setattr(runner, "_manifest", lambda *args: pytest.fail("Forbidden input opened"))
    with pytest.raises(PermissionError): _run(fixture, mode=mode)


@pytest.mark.parametrize("argument,value", [
    ("predictions_dir", "outputs/predictions/official_test"),
    ("e8_artifact", "outputs/official-test/hidden.json"),
    ("predictions_dir", "outputs/../data/test"),
    ("output_dir", "outputs/metrics/E6/new_run"),
    ("output_dir", "outputs/robustness/new_run"),
])
def test_forbidden_path_is_rejected_lexically(fixture, monkeypatch, argument, value):
    monkeypatch.setattr(runner, "_manifest", lambda *args: pytest.fail("Forbidden input opened"))
    with pytest.raises(PermissionError): _run(fixture, **{argument: value})


def test_link_redirect_is_rejected_before_reading_child(fixture, monkeypatch):
    alias = fixture["prediction"]
    real_is_symlink = Path.is_symlink
    visited = []
    def simulated_link(path):
        visited.append(path)
        if path == alias: return True
        if path.is_relative_to(alias): pytest.fail("Inspected child through a rejected alias")
        return real_is_symlink(path)
    monkeypatch.setattr(Path, "is_symlink", simulated_link)
    with pytest.raises(PermissionError, match="Symlink"): _run(fixture)


@pytest.mark.parametrize("which,status", [("source_integrity", "FAIL"), ("frozen_reference_integrity", "FAIL"), ("source_integrity", "BLOCKED")])
def test_dirty_or_untracked_source_and_refs_block_pass(fixture, monkeypatch, which, status):
    def evidence(root, files):
        is_source = "scripts/run_final_evaluation_preflight.py" in files
        target = "source_integrity" if is_source else "frozen_reference_integrity"
        return dict(head="a" * 40, status=status if target == which else "PASS", modified=["changed.py"] if status == "FAIL" else [], untracked=[])
    monkeypatch.setattr(runner, "_git_state", evidence)
    result = _run(fixture)
    assert result["overall_status"] == status
    assert _check(fixture, which)["status"] == status


@pytest.mark.parametrize("dirty,tracked,expected", [(False, True, "PASS"), (True, True, "FAIL"), (False, False, "BLOCKED")])
def test_git_audit_detects_modified_and_untracked_files(monkeypatch, tmp_path, dirty, tracked, expected):
    requested = ["scripts/fixture.py", "configs/fixture.yaml"]
    def git(args, **kwargs):
        if "rev-parse" in args: return b"a" * 40
        assert args[args.index("--") + 1:] == requested
        if "ls-files" in args: return b"scripts/fixture.py\0configs/fixture.yaml\0" if tracked else b"configs/fixture.yaml\0"
        if "diff" in args: return b"configs/fixture.yaml\0" if dirty else b""
        pytest.fail("Unexpected Git action")
    monkeypatch.setattr(runner.subprocess, "check_output", git)
    assert runner._git_state(tmp_path, requested)["status"] == expected


def test_validation_mode_does_not_call_model_training_or_inference(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, mode="validation")
    _mock_git_objects(fixture, monkeypatch)
    monkeypatch.setattr(runner, "SOURCE_ROOT", fixture["root"])
    monkeypatch.setattr(runner, "_git_state", lambda *args: dict(head="a" * 40, status="PASS"))
    # Only the checksum and historical E8 acceptance binding are synthetic stand-ins.
    monkeypatch.setattr(predictor, "verify_checkpoint_hash", lambda *args: predictor.EXPECTED_CHECKPOINT_SHA256)
    monkeypatch.setattr(runner, "_e8_provenance", lambda *args: {"scope": "synthetic validation-mode test"})
    def forbid(*args, **kwargs): pytest.fail("Preflight must never invoke a model, training or artifact runner")
    for name in ("load_e4_model", "collect_predictions", "run_final_inference"):
        monkeypatch.setattr(predictor, name, forbid)
    monkeypatch.setattr(runner.selective, "run_final_selective", forbid)
    from src.calibration import temperature_scaling
    monkeypatch.setattr(temperature_scaling, "fit_temperature", forbid)
    from src.selective import threshold
    monkeypatch.setattr(threshold, "select_threshold", forbid)
    monkeypatch.setattr(temperature_scaling.TemperatureScaler, "fit", forbid)
    result = runner.run_preflight(mode="validation", repo_root=fixture["root"], output_dir="outputs/preflight/run")
    assert result["overall_status"] == "PASS"
    assert result["mode"] == "validation"
    assert result["inference_performed"] is False


def test_cli_has_no_official_mode_or_bypass(capsys):
    with pytest.raises(SystemExit) as error:
        runner.main(["--mode", "official_test", "--output-dir", "unused"])
    assert error.value.code != 0
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("directory", ["e7_dir", "e8_path"])
def test_incomplete_e7_or_e8_artifacts_are_blocked(fixture, directory):
    parent = fixture[directory] if directory == "e7_dir" else fixture[directory].parent
    (parent / ".incomplete").write_bytes(b"unfinished artifact")
    assert _run(fixture)["overall_status"] == "BLOCKED"


def test_directory_race_never_reuses_a_foreign_output(fixture, monkeypatch):
    original = runner._inputs
    def race(*args):
        root, paths, output = original(*args)
        output.mkdir(parents=True)
        (output / "foreign.txt").write_bytes(b"other writer")
        return root, paths, output
    monkeypatch.setattr(runner, "_inputs", race)
    with pytest.raises(FileExistsError): _run(fixture)
    output = fixture["root"] / "outputs/preflight/run"
    assert {p.name for p in output.iterdir()} == {"foreign.txt"}
    assert (output / "foreign.txt").read_bytes() == b"other writer"


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_staging_write_failure_cleans_private_files_only(fixture, monkeypatch, fail_at):
    original = runner.json.dump
    calls, staging = 0, []
    def fail(payload, handle, *args, **kwargs):
        nonlocal calls
        calls += 1
        staging.append(Path(handle.name).parent)
        if calls == fail_at:
            handle.write('{"partial":')
            raise OSError("injected staging failure")
        return original(payload, handle, *args, **kwargs)
    monkeypatch.setattr(runner.json, "dump", fail)
    with pytest.raises(OSError, match="injected staging failure"): _run(fixture)
    output = fixture["root"] / "outputs/preflight/run"
    assert {p.name for p in output.iterdir()} == {runner.RUN_MARKER, ".incomplete"}
    assert staging and all(not path.exists() for path in staging)
    with pytest.raises(runner.Blocked): runner.load_preflight_report(output)


def test_consumer_rejects_completed_report_modified_after_publication(fixture):
    _run(fixture)
    output = fixture["root"] / "outputs/preflight/run"
    _change_json(output / runner.REPORT_FILES[0], lambda payload: payload.update(overall_status="FAIL"))
    with pytest.raises(runner.Blocked, match="changed|partially written"):
        runner.load_preflight_report(output)


def test_frozen_reference_changed_during_audit_fails_snapshot_check(fixture, monkeypatch):
    original = runner._classification
    def mutate_after_audit(*args):
        result = original(*args)
        reference = fixture["root"] / "docs/final_evaluation/data_contract.md"
        reference.write_bytes(reference.read_bytes() + b"\nconcurrent edit\n")
        return result
    monkeypatch.setattr(runner, "_classification", mutate_after_audit)
    assert _run(fixture)["overall_status"] == "FAIL"
    assert _check(fixture, "input_snapshot_stable")["status"] == "FAIL"


def test_cli_returns_blocked_without_claiming_authorization(fixture, capsys):
    fixture["e8_path"].unlink()
    code = runner.main(["--mode", "synthetic", "--repo-root", str(fixture["root"]), "--output-dir", "outputs/preflight/run"])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall_status"] == "BLOCKED"
    assert payload["official_test_used"] is False
    assert payload["configuration_lock_approved"] is False


@pytest.mark.parametrize("field", ["preprocessing", "temperature", "safety"])
def test_inference_metadata_cannot_hide_protocol_mismatch(fixture, field):
    def change(value):
        if field == "preprocessing": value["preprocessing"]["evaluation_resize"] = 128
        elif field == "temperature": value["temperature"] = 0.8
        else: value["safety"]["temperature_refit"] = True
    _change_json(fixture["meta_path"], change)
    assert _run(fixture)["overall_status"] == "FAIL"
    assert _check(fixture, "prediction_bundle")["status"] == "FAIL"


def test_regression_input_aba_does_not_validate_unhashed_metrics(fixture, monkeypatch):
    path = fixture["prediction"] / "validation_metrics.json"
    good = path.read_bytes()
    wrong = json.loads(good)
    wrong["classification"]["top1_accuracy"] += 0.1
    _write_json(path, wrong)
    bad = path.read_bytes()
    original = runner._classification

    def replace_only_while_validating(*args):
        path.write_bytes(good)
        try:
            return original(*args)
        finally:
            path.write_bytes(bad)

    monkeypatch.setattr(runner, "_classification", replace_only_while_validating)
    result = _run(fixture)
    assert path.read_bytes() == bad
    assert result["overall_status"] == "FAIL"
    assert _check(fixture, "e4_e6_metrics")["status"] == "FAIL"
    manifest = json.loads((fixture["root"] / "outputs/preflight/run/preflight_manifest.json").read_text())
    assert manifest["inputs"][str(path)]["sha256"] == hashlib.sha256(bad).hexdigest()


def test_regression_consumer_aba_never_promotes_blocked_report(fixture, monkeypatch):
    real_git = runner._git_state
    def pending_source(root, files):
        evidence = real_git(root, files)
        if "scripts/run_final_evaluation_preflight.py" in files:
            evidence.update(status="BLOCKED", untracked=["scripts/run_final_evaluation_preflight.py"])
        return evidence
    monkeypatch.setattr(runner, "_git_state", pending_source)
    assert _run(fixture)["overall_status"] == "BLOCKED"
    output = fixture["root"] / "outputs/preflight/run"
    saved = {name: (output / name).read_bytes() for name in (*runner.REPORT_FILES, runner.RUN_MARKER, ".complete")}
    original_json = runner._json
    injections = []

    def transient_json(path):
        if path.parent != output or path.name not in runner.REPORT_FILES:
            return original_json(path)
        value = json.loads(saved[path.name])
        if path.name == runner.REPORT_FILES[0]:
            value.update(overall_status="PASS", check_counts={"PASS": 10, "FAIL": 0, "BLOCKED": 0})
        elif path.name == runner.REPORT_FILES[1]:
            for row in value["checks"]:
                row["status"] = "PASS"
        else:
            value["source"].update(status="PASS", untracked=[])
        path.write_text(json.dumps(value), encoding="utf-8")
        try:
            injections.append(path.name)
            return original_json(path)
        finally:
            path.write_bytes(saved[path.name])

    monkeypatch.setattr(runner, "_json", transient_json)
    assert runner.load_preflight_report(output)["overall_status"] == "BLOCKED"
    assert all((output / name).read_bytes() == raw for name, raw in saved.items())


def test_regression_synthetic_ancestor_cannot_select_checkout_inputs():
    root = runner.SOURCE_ROOT
    with pytest.raises(PermissionError):
        runner._inputs("synthetic", root.parent, root / runner.PREDICTIONS,
                       root / runner.SELECTIVE, root / runner.ROBUSTNESS,
                       root.parent / "outputs/preflight/not_created_by_this_test")


@pytest.mark.parametrize("parameter", ["temperature", "threshold"])
def test_regression_e7_nextafter_protocol_value_is_rejected(fixture, parameter):
    paths = {"npz": fixture["npz"], "metadata": fixture["meta_path"], "csv": fixture["csv"],
             "diagram": fixture["prediction"] / "validation_reliability_diagram.png",
             "e7": fixture["e7_dir"] / "selective_metrics.json", "e7_csv": fixture["e7_dir"] / "risk_coverage.csv",
             "e7_png": fixture["e7_dir"] / "risk_coverage_curve.png"}
    bundle = runner._bundle(paths, fixture["root"], "synthetic")
    _change_json(paths["e7"], lambda data: data.update({parameter: float(np.nextafter(data[parameter], 1.0))}))
    with pytest.raises(ValueError):
        runner._selective(paths, bundle, "synthetic")


@pytest.mark.parametrize("kind", ["npz", "metadata", "csv", "yaml", "checkpoint", "e8", "e8_report"])
def test_snapshot_validates_exact_captured_bytes_not_later_live_replacements(fixture, monkeypatch, kind):
    paths = dict(npz=fixture["npz"], metadata=fixture["meta_path"], csv=fixture["csv"],
                 yaml=fixture["root"] / "configs/data.yaml", checkpoint=fixture["checkpoint"],
                 e8=fixture["e8_path"], e8_report=fixture["root"] / runner.E8_REPORT)
    original_path = paths[kind]
    original = original_path.read_bytes()
    captured = []
    initialize = runner.InputSnapshot.__init__
    observe = runner.InputSnapshot.verify_live_drift

    def capture_then_replace(snapshot, *args, **kwargs):
        initialize(snapshot, *args, **kwargs)
        private = snapshot.copies[original_path]
        captured.append((private, private.read_bytes()))
        assert private != original_path
        assert private.read_bytes() == original
        original_path.write_bytes(b"transient invalid live replacement, not snapshot input")

    def restore_before_observation(snapshot):
        original_path.write_bytes(original)
        return observe(snapshot)

    monkeypatch.setattr(runner.InputSnapshot, "__init__", capture_then_replace)
    monkeypatch.setattr(runner.InputSnapshot, "verify_live_drift", restore_before_observation)
    result = _run(fixture)
    assert result["overall_status"] == "PASS"
    assert captured and captured[0][1] == original
    assert not captured[0][0].exists()  # Private snapshot was cleaned.
    manifest = json.loads((fixture["root"] / "outputs/preflight/run/preflight_manifest.json").read_text())
    assert manifest["inputs"][str(original_path)]["sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest["input_binding"]["private_snapshot_verified"] is True


@pytest.mark.parametrize("kind", ["npz", "metadata", "csv", "yaml", "checkpoint", "e8", "e8_report"])
def test_capture_aba_records_digest_of_copied_stream_and_never_passes_bad_bytes(fixture, monkeypatch, kind):
    paths = dict(npz=fixture["npz"], metadata=fixture["meta_path"], csv=fixture["csv"],
                 yaml=fixture["root"] / "configs/data.yaml", checkpoint=fixture["checkpoint"],
                 e8=fixture["e8_path"], e8_report=fixture["root"] / runner.E8_REPORT)
    target = paths[kind]
    saved = target.read_bytes()
    wrong = b"unusable captured bytes"
    read_once = runner._read_once
    injected = []

    def replace_while_capturing(path, destination=None, limit=None):
        if path == target and destination is not None:
            target.write_bytes(wrong)
            try:
                result = read_once(path, destination, limit)
                injected.append(result["sha256"])
                return result
            finally:
                target.write_bytes(saved)
        return read_once(path, destination, limit)

    monkeypatch.setattr(runner, "_read_once", replace_while_capturing)
    assert _run(fixture)["overall_status"] != "PASS"
    manifest = json.loads((fixture["root"] / "outputs/preflight/run/preflight_manifest.json").read_text())
    assert injected == [hashlib.sha256(wrong).hexdigest()]
    assert manifest["inputs"][str(target)]["sha256"] == injected[0]
    assert target.read_bytes() == saved


@pytest.mark.parametrize("report_name", runner.REPORT_FILES)
def test_consumer_rejects_transient_report_bytes_even_when_original_is_restored(fixture, monkeypatch, report_name):
    fixture["e8_path"].unlink()
    assert _run(fixture)["overall_status"] == "BLOCKED"
    output = fixture["root"] / "outputs/preflight/run"
    path = output / report_name
    saved = path.read_bytes()
    original_read = runner._read_once
    injected = []
    def transient_report(target, destination=None, limit=None):
        if target == path:
            payload = json.loads(saved)
            payload["transient_forgery"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            try:
                result = original_read(target, destination, limit)
                injected.append(result["sha256"])
                return result
            finally:
                path.write_bytes(saved)
        return original_read(target, destination, limit)
    monkeypatch.setattr(runner, "_read_once", transient_report)
    with pytest.raises(runner.Blocked, match="changed|partially"):
        runner.load_preflight_report(output)
    assert injected and injected[0] != hashlib.sha256(saved).hexdigest()
    assert path.read_bytes() == saved


def _rehash_test_reports(output):
    """Simulate a producer publishing internally inconsistent but intact reports."""
    complete = json.loads((output / ".complete").read_text())
    complete["output_sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in runner.REPORT_FILES}
    _write_json(output / ".complete", complete)


@pytest.mark.parametrize("problem", ["overall", "missing", "disabled", "duplicate", "status", "counts", "approval", "official", "mode"])
def test_consumer_rejects_inconsistent_report_even_with_valid_publication_hashes(fixture, problem):
    fixture["e8_path"].unlink()
    assert _run(fixture)["overall_status"] == "BLOCKED"
    output = fixture["root"] / "outputs/preflight/run"
    summary = output / runner.REPORT_FILES[0]
    checks = output / runner.REPORT_FILES[1]
    if problem == "overall": _change_json(summary, lambda data: data.update(overall_status="PASS"))
    elif problem == "counts": _change_json(summary, lambda data: data["check_counts"].update(PASS=999))
    elif problem == "approval": _change_json(summary, lambda data: data.update(configuration_lock_approved=True))
    elif problem == "official": _change_json(output / runner.REPORT_FILES[2], lambda data: data.update(official_test_used=True))
    elif problem == "mode": _change_json(summary, lambda data: data.update(mode="other", synthetic=False))
    else:
        def change(data):
            if problem == "missing": data["checks"].pop()
            elif problem == "disabled": data["checks"][0]["required"] = False
            elif problem == "duplicate": data["checks"].append(dict(data["checks"][0]))
            else: data["checks"][0]["status"] = "IGNORED"
        _change_json(checks, change)
    _rehash_test_reports(output)
    with pytest.raises(ValueError): runner.load_preflight_report(output)


def test_legacy_intact_blocked_report_keeps_its_status(fixture):
    fixture["e8_path"].unlink()
    assert _run(fixture)["overall_status"] == "BLOCKED"
    output = fixture["root"] / "outputs/preflight/run"
    _change_json(output / runner.REPORT_FILES[2], lambda data: data.pop("input_binding"))
    _rehash_test_reports(output)
    assert runner.load_preflight_report(output)["overall_status"] == "BLOCKED"


@pytest.mark.parametrize("root_kind", ["checkout", "inside", "ancestor", "temporary_root"])
def test_synthetic_unsafe_roots_rejected_before_any_content_read(monkeypatch, root_kind):
    roots = dict(checkout=runner.CODE_ROOT, inside=runner.CODE_ROOT / "unused_fixture",
                 ancestor=runner.CODE_ROOT.parent, temporary_root=Path(runner.tempfile.gettempdir()))
    monkeypatch.setattr(runner, "_read_once", lambda *args, **kwargs: pytest.fail("Unsafe root content was read"))
    with pytest.raises(PermissionError):
        runner._inputs("synthetic", roots[root_kind], None, None, None, "outputs/preflight/unused")


@pytest.mark.parametrize("field", ["prediction", "e7", "e8"])
def test_synthetic_each_checkout_artifact_path_is_rejected(fixture, monkeypatch, field):
    read = runner._read_once
    def no_checkout_data(path, *args, **kwargs):
        assert not path.is_relative_to(runner.CODE_ROOT), "Opened checkout data from synthetic mode"
        return read(path, *args, **kwargs)
    monkeypatch.setattr(runner, "_read_once", no_checkout_data)
    arguments = dict(predictions_dir=None, e7_dir=None, e8_artifact=None)
    key, value = {"prediction": ("predictions_dir", runner.CODE_ROOT / runner.PREDICTIONS),
                  "e7": ("e7_dir", runner.CODE_ROOT / runner.SELECTIVE),
                  "e8": ("e8_artifact", runner.CODE_ROOT / runner.ROBUSTNESS)}[field]
    arguments[key] = value
    with pytest.raises(PermissionError):
        _run(fixture, **arguments)


@pytest.mark.parametrize("problem", ["missing", "not_synthetic", "generator"])
def test_synthetic_requires_controlled_fixture_declaration(fixture, problem):
    marker = fixture["root"] / runner.FIXTURE_MARKER
    if problem == "missing": marker.unlink()
    elif problem == "not_synthetic": _change_json(marker, lambda data: data.update(synthetic=False))
    else: _change_json(marker, lambda data: data.update(generator="unapproved"))
    with pytest.raises(PermissionError): _run(fixture)


def test_metric_tolerance_does_not_change_frozen_protocol_values(fixture):
    path = fixture["e7_dir"] / "selective_metrics.json"
    def perturb_metrics(data):
        data["aurc"] += 1e-12
        data["selected_metrics"]["coverage"] += 1e-12
    _change_json(path, perturb_metrics)
    assert _run(fixture)["overall_status"] == "PASS"
    manifest = json.loads((fixture["root"] / "outputs/preflight/run/preflight_manifest.json").read_text())
    assert manifest["comparison_tolerance"]["protocol_parameters"] == "exact"


@pytest.mark.parametrize("problem", ["hash", "source", "report", "commit"])
def test_e8_validation_acceptance_branch_rejects_invalid_bindings(tmp_path, monkeypatch, problem):
    root = tmp_path / "private_acceptance_fixture"
    artifact = root / runner.ROBUSTNESS
    # Unit-test trust anchor, no claim of actual E4 execution or real weights.
    _write_json(artifact, dict(git_sha=runner.E8_COMMIT, unit_test_fixture=True))
    expected_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    report = root / runner.E8_REPORT
    report.parent.mkdir(parents=True)
    report.write_text(f"fixture acceptance: {expected_hash} {runner.E8_COMMIT} {runner.ROBUSTNESS.as_posix()}")
    monkeypatch.setattr(runner, "E8_HASH", expected_hash)
    calls = []
    def git_object(args, **kwargs):
        calls.append(args)
        assert args[-2:] == ["-e", runner.E8_COMMIT + "^{commit}"]
        if problem == "commit":
            raise subprocess.CalledProcessError(1, args)
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(runner.subprocess, "run", git_object)
    if problem == "hash": artifact.write_bytes(artifact.read_bytes() + b"\n")
    elif problem == "source":
        _change_json(artifact, lambda data: data.update(git_sha="b" * 40))
        monkeypatch.setattr(runner, "E8_HASH", hashlib.sha256(artifact.read_bytes()).hexdigest())
    elif problem == "report": report.write_text("fixture without source/hash binding")
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        runner._e8_provenance({"e8": artifact}, root, "validation", runner.CODE_ROOT)
    assert bool(calls) is (problem == "commit")


def test_e8_acceptance_uses_private_captured_artifact_and_report(tmp_path, monkeypatch):
    original_root = tmp_path / "original"
    private_root = tmp_path / "private"
    original_artifact = original_root / runner.ROBUSTNESS
    original_report = original_root / runner.E8_REPORT
    _write_json(original_artifact, dict(git_sha=runner.E8_COMMIT, unit_test_fixture=True))
    digest = hashlib.sha256(original_artifact.read_bytes()).hexdigest()
    original_report.parent.mkdir(parents=True)
    original_report.write_text(f"fixture acceptance: {digest} {runner.E8_COMMIT} {runner.ROBUSTNESS.as_posix()}")
    for relative in (runner.ROBUSTNESS, Path(runner.E8_REPORT)):
        destination = private_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            captured = runner._read_once(original_root / relative, destination=handle)
        assert captured["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    monkeypatch.setattr(runner, "E8_HASH", digest)
    def git_object(args, **kwargs):
        assert args[2] == str(runner.CODE_ROOT)
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(runner.subprocess, "run", git_object)
    # Public evidence changes after capture; validator consumes only private copies.
    original_artifact.write_bytes(b"unrelated live replacement")
    original_report.write_bytes(b"unrelated live evidence")
    proof = runner._e8_provenance({"e8": private_root / runner.ROBUSTNESS}, private_root, "validation", runner.CODE_ROOT)
    assert proof["artifact_sha256"] == digest
    assert proof["source_git_sha"] == runner.E8_COMMIT


def test_snapshot_capture_read_failure_cannot_produce_pass(fixture, monkeypatch):
    read_once = runner._read_once
    def unreadable(path, destination=None, limit=None):
        if path == fixture["npz"] and destination is not None:
            destination.write(b"partial snapshot")
            raise OSError("injected capture read failure")
        return read_once(path, destination, limit)
    monkeypatch.setattr(runner, "_read_once", unreadable)
    result = _run(fixture)
    assert result["overall_status"] == "BLOCKED"
    assert _check(fixture, "input_snapshot_stable")["status"] == "BLOCKED"
    manifest = json.loads((fixture["root"] / "outputs/preflight/run/preflight_manifest.json").read_text())
    assert manifest["input_binding"]["private_snapshot_verified"] is False
    assert manifest["inputs"][str(fixture["npz"])]["state"] == "capture_failed"


def test_synthetic_hard_link_to_external_data_is_rejected(fixture, tmp_path):
    external = tmp_path / "untrusted_payload"
    external.write_bytes(b"untrusted fixture bytes")
    alias = fixture["root"] / "linked_data"
    runner.os.link(external, alias)
    with pytest.raises(PermissionError, match="Hard-linked"):
        runner._safe_path(alias, fixture["root"])


def test_resolved_forbidden_alias_is_rejected_before_target_stat(tmp_path, monkeypatch):
    alias = tmp_path / "alias"
    forbidden = tmp_path / "official_test"
    resolve = Path.resolve
    is_file = Path.is_file
    monkeypatch.setattr(Path, "resolve", lambda path, **kwargs: forbidden / "payload" if path == alias else resolve(path, **kwargs))
    def forbid_target_stat(path):
        assert not path.is_relative_to(forbidden), "Inspected a forbidden resolved target"
        return is_file(path)
    monkeypatch.setattr(Path, "is_file", forbid_target_stat)
    with pytest.raises(PermissionError, match="Resolved"):
        runner._safe_path(alias, tmp_path)


def test_read_handle_changed_during_capture_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "capture_payload"
    path.write_bytes(b"original bytes")
    fstat = runner.os.fstat
    calls = 0
    def change_before_final_stat(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            path.write_bytes(b"replacement of a different size during capture")
        return fstat(descriptor)
    monkeypatch.setattr(runner.os, "fstat", change_before_final_stat)
    with pytest.raises(runner.Blocked, match="changed while being captured"):
        runner._read_once(path)
    assert calls >= 2


def test_consumer_parses_verified_buffers_instead_of_reopening_paths(fixture, monkeypatch):
    fixture["e8_path"].unlink()
    assert _run(fixture)["overall_status"] == "BLOCKED"
    output = fixture["root"] / "outputs/preflight/run"
    path = output / runner.REPORT_FILES[0]
    original = path.read_bytes()
    expected_digest = hashlib.sha256(original).hexdigest()
    parse = runner._parse_json
    called = []
    def change_path_during_parse(raw, label):
        if label == runner.REPORT_FILES[0]:
            fake = json.loads(original)
            fake["overall_status"] = "PASS"
            path.write_text(json.dumps(fake), encoding="utf-8")
            try:
                called.append(hashlib.sha256(raw).hexdigest())
                return parse(raw, label)
            finally:
                path.write_bytes(original)
        return parse(raw, label)
    monkeypatch.setattr(runner, "_parse_json", change_path_during_parse)
    assert runner.load_preflight_report(output)["overall_status"] == "BLOCKED"
    assert called == [expected_digest]
    assert path.read_bytes() == original


@pytest.mark.parametrize("problem", ["legacy", "unverified", "capture_errors", "unbound_reference"])
def test_consumer_requires_bound_input_evidence_before_accepting_pass(fixture, problem):
    assert _run(fixture)["overall_status"] == "PASS"
    output = fixture["root"] / "outputs/preflight/run"
    manifest = output / runner.REPORT_FILES[2]
    if problem == "legacy": _change_json(manifest, lambda data: data.pop("input_binding"))
    elif problem == "unverified": _change_json(manifest, lambda data: data["input_binding"].update(private_snapshot_verified=False))
    elif problem == "capture_errors": _change_json(manifest, lambda data: data["input_binding"].update(capture_errors=["unverified read"]))
    else:
        def unbind(data):
            next(row for row in data["checks"] if row["id"] == "frozen_reference_integrity").pop("snapshot_content_bound_to_git")
        _change_json(output / runner.REPORT_FILES[1], unbind)
    _rehash_test_reports(output)
    with pytest.raises(ValueError, match="PASS requires"):
        runner.load_preflight_report(output)
