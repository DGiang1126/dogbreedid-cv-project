import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.confusion_matrix import (
    compute_confusion_matrix,
    e7_error_breakdown,
    load_class_names,
    load_predictions_npz,
    per_class_metrics,
    save_e9_artifacts,
    select_representative_cases,
    stable_softmax,
    top_confusion_pairs,
    validate_ids_against_split,
    validate_predictions,
)
from scripts.run_error_analysis import load_e9_config, run_analysis


def make_logits(labels, num_classes=30):
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.zeros((len(labels), num_classes), dtype=np.float64)
    for i, label in enumerate(labels):
        logits[i, int(label)] = 3.0
    return logits


def make_ids(n):
    return np.asarray([f"validation/image_{i:04d}.jpg" for i in range(n)], dtype="U64")


def write_mapping(path, count=30):
    path.write_text(json.dumps({"class_to_idx": {f"breed_{i:02d}": i for i in range(count)}}), encoding="utf-8")


def test_confusion_matrix_rows_true_columns_predicted_and_counts_sum():
    y_true = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 2, 2], dtype=np.int64)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1
    assert cm[0, 1] == 1
    assert cm[1, 1] == 1
    assert cm[1, 2] == 1
    assert cm[2, 2] == 1
    assert cm.sum() == len(y_true)
    assert np.trace(cm) == 3


def test_per_class_metrics_hand_calculated():
    cm = np.array([[8, 2, 0], [1, 7, 2], [0, 3, 7]], dtype=np.int64)
    rows = per_class_metrics(cm, ["Chihuahua", "Pug", "Beagle"])
    assert rows[0]["precision"] == pytest.approx(8 / 9)
    assert rows[0]["recall"] == pytest.approx(8 / 10)
    assert rows[0]["f1_score"] == pytest.approx(2 * (8 / 9) * (8 / 10) / ((8 / 9) + (8 / 10)))
    assert rows[0]["support"] == 10
    assert len(rows) == 3


def test_zero_division_metrics_are_zero_not_nan():
    cm = np.zeros((3, 3), dtype=np.int64)
    rows = per_class_metrics(cm, ["a", "b", "c"])
    assert all(r["precision"] == 0 and r["recall"] == 0 and r["f1_score"] == 0 for r in rows)


def test_top_confusion_pairs_directional_and_tie_deterministic():
    cm = np.zeros((4, 4), dtype=np.int64)
    cm[0, 1] = 5
    cm[1, 0] = 5
    cm[2, 0] = 5
    cm[0, 3] = 3
    pairs = top_confusion_pairs(cm, ["A", "B", "C", "D"], top_k=3)
    assert [(p["true_id"], p["predicted_id"]) for p in pairs] == [(0, 1), (0, 3), (1, 0)] or [(p["true_id"], p["predicted_id"]) for p in pairs] == [(0, 1), (1, 0), (2, 0)]
    # Strict tie order is by true id then predicted id; all 5-count pairs sort before the 3-count pair.
    assert [(p["true_id"], p["predicted_id"]) for p in pairs] == [(0, 1), (1, 0), (2, 0)]


def test_validate_predictions_rejects_nan_bad_shape_duplicate_or_empty_ids():
    logits = make_logits([0, 1])
    labels = np.array([0, 1], dtype=np.int64)
    ids = make_ids(2)
    validate_predictions(logits, labels, ids, 30)
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_predictions(np.full((2, 30), np.nan), labels, ids, 30)
    with pytest.raises(ValueError, match="shape"):
        validate_predictions(np.zeros((2, 29)), labels, ids, 30)
    with pytest.raises(ValueError, match="unique"):
        validate_predictions(logits, labels, np.array(["same", "same"]), 30)
    with pytest.raises(ValueError, match="empty"):
        validate_predictions(logits, labels, np.array(["ok", "  "]), 30)


def test_npz_load_requires_schema_and_safe_string_ids(tmp_path):
    path = tmp_path / "valid.npz"
    np.savez_compressed(path, logits=make_logits([0, 1]), labels=np.array([0, 1], dtype=np.int64), image_ids=make_ids(2))
    result = load_predictions_npz(path)
    assert result["logits"].shape == (2, 30)
    assert result["image_ids"].dtype.kind == "U"

    missing = tmp_path / "missing.npz"
    np.savez_compressed(missing, logits=make_logits([0]), labels=np.array([0], dtype=np.int64))
    with pytest.raises(ValueError, match="missing required keys"):
        load_predictions_npz(missing)

    object_ids = tmp_path / "object_ids.npz"
    np.savez_compressed(object_ids, logits=make_logits([0]), labels=np.array([0], dtype=np.int64), image_ids=np.array(["id"], dtype=object))
    with pytest.raises(ValueError):
        load_predictions_npz(object_ids)


def test_class_mapping_supported_layouts(tmp_path):
    path = tmp_path / "mapping.json"
    write_mapping(path)
    names = load_class_names(path, 30)
    assert names[0] == "breed_00"
    assert names[-1] == "breed_29"


def test_stable_softmax_and_temperature_validation():
    logits = np.array([[10000.0, 9999.0, -10000.0]])
    probs = stable_softmax(logits, 1.0)
    assert np.isfinite(probs).all()
    assert probs.sum(axis=1)[0] == pytest.approx(1.0)
    assert np.argmax(probs, axis=1)[0] == 0
    with pytest.raises(ValueError, match="temperature"):
        stable_softmax(logits, 0)
    with pytest.raises(ValueError, match="NaN or Inf"):
        stable_softmax(np.array([[float("nan"), 0.0]]), 1.0)


def test_e7_breakdown_hand_calculated_and_inclusive_threshold():
    # Two samples are confident/accepted, one is deliberately low confidence/rejected.
    logits = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.4, 0.3, 0.2]])
    labels = np.array([0, 0, 0], dtype=np.int64)
    breakdown = e7_error_breakdown(logits, labels, temperature=1.0, threshold=0.90)
    assert breakdown["groups"]["correct_accept"] == 1
    assert breakdown["groups"]["incorrect_accept"] == 1
    assert breakdown["groups"]["correct_reject"] == 1
    assert breakdown["groups"]["incorrect_reject"] == 0
    assert breakdown["num_accept"] == 2
    assert breakdown["num_reject"] == 1
    assert breakdown["coverage"] == pytest.approx(2 / 3)


def test_representative_cases_only_contains_expected_fields():
    labels = np.array([0, 1, 2, 3], dtype=np.int64)
    logits = make_logits(labels)
    logits[1] = 0
    logits[1, 0] = 3.0  # one incorrect prediction
    rows = select_representative_cases(logits, labels, make_ids(4), [f"c{i}" for i in range(30)], 0.8, 0.5, 3)
    assert len(rows) <= 3
    assert all("image_id" in row and "e7_status" in row for row in rows)


def test_save_artifacts_writes_aggregate_and_validation_cases_but_no_official_ids(tmp_path, monkeypatch):
    labels = np.array([0, 0, 1, 2], dtype=np.int64)
    logits = make_logits(labels)
    logits[1, 1] = 4.0
    ids = make_ids(4)
    names = [f"breed_{i:02d}" for i in range(30)]
    # Direct exporter must bind the full Validation reference too. Use a fake
    # repository reference here, never change the project's actual split.
    import src.evaluation.confusion_matrix as analysis
    reference = write_validation_reference(tmp_path, ids, labels)
    monkeypatch.setattr(analysis, "VALIDATION_SPLIT_CSV", reference)

    val_out = tmp_path / "val"
    summary = save_e9_artifacts(logits, labels, ids, names, val_out, "validation", 0.80246283961825, 0.5011672918550012)
    assert summary["num_samples"] == 4
    assert (val_out / "confusion_matrix.csv").exists()
    assert (val_out / "per_class_metrics.csv").exists()
    assert (val_out / "top5_confusion_pairs.csv").exists()
    assert (val_out / "representative_cases.csv").exists()

    off_out = tmp_path / "official"
    save_e9_artifacts(logits, labels, ids, names, off_out, "official", 0.80246283961825, 0.5011672918550012)
    assert not (off_out / "representative_cases.csv").exists()
    summary_text = (off_out / "run_summary.json").read_text(encoding="utf-8")
    assert "validation/image_0000.jpg" not in summary_text


def test_e9_config_and_official_guard(tmp_path):
    config = ROOT / "configs/experiments/e9_error_analysis.yaml"
    loaded = load_e9_config(config)
    assert loaded["e6"]["temperature"] == pytest.approx(0.80246283961825)
    assert loaded["e7"]["threshold"] == pytest.approx(0.5011672918550012)

    input_file = tmp_path / "clean_predictions.npz"
    np.savez_compressed(input_file, logits=make_logits([0]), labels=np.array([0], dtype=np.int64), image_ids=np.array(["official/id"], dtype="U16"))
    mapping = tmp_path / "mapping.json"
    write_mapping(mapping)
    with pytest.raises(PermissionError, match="disabled"):
        run_analysis(config, input_file, "official", tmp_path / "out", mapping, allow_official_analysis=True)


def test_official_gate_requires_both_config_and_cli_flag(tmp_path):
    import yaml
    source_config = ROOT / "configs/experiments/e9_error_analysis.yaml"
    data = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    data["official_analysis"]["enabled"] = True
    config_path = tmp_path / "e9.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    input_file = tmp_path / "clean_predictions.npz"
    np.savez_compressed(input_file, logits=make_logits([0]), labels=np.array([0], dtype=np.int64), image_ids=np.array(["official/id"], dtype="U16"))
    mapping = tmp_path / "mapping.json"
    write_mapping(mapping)
    with pytest.raises(PermissionError, match="explicit"):
        run_analysis(config_path, input_file, "official", tmp_path / "out", mapping, allow_official_analysis=False)


def test_inclusive_threshold_accepts_equal_confidence():
    # Equal logits for 2 classes produce max probability exactly 0.5.
    logits = np.array([[0.0, 0.0]], dtype=np.float64)
    labels = np.array([0], dtype=np.int64)
    breakdown = e7_error_breakdown(logits, labels, temperature=1.0, threshold=0.5)
    assert breakdown["groups"]["correct_accept"] == 1
    assert breakdown["num_reject"] == 0


def test_split_reference_cross_check_uses_aggregate_counts_only(tmp_path):
    import csv
    split_csv = tmp_path / "validation.csv"
    with split_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "label"])
        writer.writeheader()
        writer.writerow({"image_id": "id_a", "label": 0})
        writer.writerow({"image_id": "id_b", "label": 1})
    result = validate_ids_against_split(
        np.array(["id_a", "id_b"]), np.array([0, 1], dtype=np.int64), split_csv
    )
    assert result["match"] is True
    assert result["num_missing_ids"] == 0
    assert result["num_label_mismatches"] == 0
    assert "ids" not in result


def test_config_rejects_changed_locked_temperature_or_threshold(tmp_path):
    import yaml
    source_config = ROOT / "configs/experiments/e9_error_analysis.yaml"
    data = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    data["e6"]["temperature"] = 1.0
    bad_path = tmp_path / "bad_e9.yaml"
    bad_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Temperature is locked"):
        load_e9_config(bad_path)
    data["e6"]["temperature"] = 0.80246283961825
    data["e7"]["threshold"] = 0.5
    bad_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold is locked"):
        load_e9_config(bad_path)


def write_validation_reference(root, ids, labels):
    import csv
    path = root / "data/splits/validation.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "class_id", "breed_name", "source_split"])
        writer.writerows((image_id, int(label), f"breed_{label:02d}", "official_train") for image_id, label in zip(ids, labels))
    return path


def diverse_case_data():
    # High-confidence errors alone exceed max_cases: the old loop starved all
    # later categories. Add low errors, correct high and correct rejected cases.
    labels = np.array([1] * 40 + [0] * 22, dtype=np.int64)
    logits = np.zeros((len(labels), 30), dtype=np.float64)
    logits[:20, 0] = 9.0
    logits[20:40, 0] = 0.1
    logits[40:60, 0] = 9.0
    logits[60:, 0] = 0.1
    ids = np.array([f"synthetic/case_{i:03d}.jpg" for i in range(len(labels))])
    return logits, labels, ids


@pytest.mark.parametrize("max_cases", [1, 3, 5, 15, 100])
def test_representative_categories_balanced_unique_and_deterministic(max_cases):
    logits, labels, ids = diverse_case_data()
    names = [f"c{i}" for i in range(30)]
    original = logits.copy()
    rows = select_representative_cases(logits, labels, ids, names, 0.80246283961825, 0.5011672918550012, max_cases)
    assert rows == select_representative_cases(logits, labels, ids, names, 0.80246283961825, 0.5011672918550012, max_cases)
    permutation = np.random.default_rng(17).permutation(len(labels))
    assert rows == select_representative_cases(logits[permutation], labels[permutation], ids[permutation], names, 0.80246283961825, 0.5011672918550012, max_cases)
    assert 0 < len(rows) <= max_cases
    assert len({row["image_id"] for row in rows}) == len(rows)
    if max_cases >= 5:
        assert {row["case_category"] for row in rows} == {
            "incorrect_high_confidence", "incorrect_low_confidence", "correct_high_confidence", "incorrect_accepted", "rejected",
        }
    confidences = stable_softmax(logits, 0.80246283961825).max(axis=1)
    for row in rows:
        idx = ids.tolist().index(row["image_id"])
        assert row["true_id"] == labels[idx]
        assert row["true_label"] == names[labels[idx]]
        assert row["predicted_id"] == np.argmax(logits[idx])
        assert row["predicted_label"] == names[row["predicted_id"]]
        assert row["calibrated_confidence"] == confidences[idx]
        correct = row["true_id"] == row["predicted_id"]
        accepted = confidences[idx] >= 0.5011672918550012
        assert row["correct"] == correct
        assert row["e7_status"] == ("ACCEPT" if accepted else "REJECT")
        assert {
            "incorrect_high_confidence": not correct and confidences[idx] >= 0.8,
            "incorrect_low_confidence": not correct and not accepted,
            "correct_high_confidence": correct and confidences[idx] >= 0.8,
            "incorrect_accepted": not correct and accepted,
            "rejected": not accepted,
        }[row["case_category"]]
    np.testing.assert_array_equal(logits, original)


def test_representative_overlap_reserves_scarce_categories_without_fake_rows():
    # Both errors can be incorrect_accepted; only the highest is high confidence.
    logits = np.array([[10.0, 0.0], [1.0, 0.0], [0.0, 10.0], [0.0, 0.0], [0.1, 0.0]])
    labels = np.array([1, 1, 1, 1, 0])
    ids = np.array([f"synthetic/{i}" for i in range(5)])
    rows = select_representative_cases(logits, labels, ids, ["a", "b"], 1.0, 0.6, 5)
    assert len(rows) == 5
    assert len({row["case_category"] for row in rows}) == 5
    one = select_representative_cases(logits[:1], labels[:1], ids[:1], ["a", "b"], 1.0, 0.6, 15)
    assert len(one) == 1  # Overlap cannot justify fabricating extra IDs.


@pytest.fixture
def analysis_inputs(tmp_path, monkeypatch):
    import src.evaluation.confusion_matrix as analysis
    labels = np.array([0, 1, 2], dtype=np.int64)
    ids = make_ids(3)
    reference = write_validation_reference(tmp_path, ids, labels)
    monkeypatch.setattr(analysis, "VALIDATION_SPLIT_CSV", reference)
    mapping = tmp_path / "mapping.json"
    write_mapping(mapping)
    return labels, ids, reference, mapping


def test_runner_validation_checks_canonical_before_export_without_optional_reference(tmp_path, analysis_inputs, monkeypatch):
    import src.evaluation.confusion_matrix as analysis
    labels, ids, reference, mapping = analysis_inputs
    path = tmp_path / "predictions.npz"
    np.savez(path, logits=make_logits(labels), labels=labels, image_ids=ids)
    monkeypatch.setattr(analysis, "save_confusion_matrix_figures", lambda *a, **kw: [])
    out = tmp_path / "out"
    result = run_analysis(ROOT / "configs/experiments/e9_error_analysis.yaml", path, "validation", out, mapping)
    assert result["split_reference_check"]["match"] is True
    assert result["split_reference_check"]["order_matches"] is True
    assert result["num_samples"] == 3
    assert result["top1_correct"] == 3
    assert (out / "representative_cases.csv").is_file()


@pytest.mark.parametrize("problem", ["ids", "official_ids", "labels", "missing", "extra", "duplicate", "order", "missing_reference", "external_reference", "clean_name"])
def test_runner_rejects_invalid_validation_before_any_output(tmp_path, analysis_inputs, problem):
    labels, ids, reference, mapping = analysis_inputs
    labels, ids = labels.copy(), ids.copy()
    optional_reference = None
    if problem == "ids": ids[0] = "unknown/image.jpg"
    if problem == "official_ids": ids = np.array([f"n99999999-fake/n99999999_{i}.jpg" for i in range(3)])
    if problem == "labels": labels[0] = 1
    if problem == "missing": labels, ids = labels[:-1], ids[:-1]
    if problem == "extra": labels, ids = np.append(labels, 0), np.append(ids, "extra/image.jpg")
    if problem == "duplicate": ids[1] = ids[0]
    if problem == "order": labels, ids = labels[::-1], ids[::-1]
    if problem == "missing_reference": reference.unlink()  # Fake temp CSV only.
    if problem == "external_reference":
        optional_reference = tmp_path / "forged.csv"
        optional_reference.write_bytes(reference.read_bytes())
    path = tmp_path / ("clean_predictions.npz" if problem == "clean_name" else "predictions.npz")
    np.savez(path, logits=make_logits(labels), labels=labels, image_ids=ids)
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "old.txt"
    sentinel.write_bytes(b"preserve existing output")
    with pytest.raises((ValueError, PermissionError, FileNotFoundError)):
        run_analysis(ROOT / "configs/experiments/e9_error_analysis.yaml", path, "validation", out, mapping, split_reference_csv=optional_reference)
    assert list(out.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"preserve existing output"
    new_out = tmp_path / "new-out"
    with pytest.raises((ValueError, PermissionError, FileNotFoundError)):
        run_analysis(ROOT / "configs/experiments/e9_error_analysis.yaml", path, "validation", new_out, mapping, split_reference_csv=optional_reference)
    assert not new_out.exists()


@pytest.mark.parametrize("ids,name", [
    (["official/id"], "predictions.npz"),
    (["n99999999-fake/n99999999_1.jpg"], "renamed.npz"),
    (["synthetic/n99999999_1.jpg"], "renamed.npz"),
    (["synthetic/../official/id"], "predictions.npz"),
    (["synthetic/fake.jpg"], "clean_predictions.npz"),
])
def test_synthetic_cannot_export_official_like_inputs(tmp_path, ids, name):
    path = tmp_path / name
    np.savez(path, logits=make_logits([0]), labels=np.array([0]), image_ids=np.array(ids))
    out = tmp_path / "out"
    with pytest.raises(PermissionError):
        run_analysis(ROOT / "configs/experiments/e9_error_analysis.yaml", path, "synthetic", out)
    assert not out.exists()


def test_synthetic_fixture_runner_exports_only_explicit_artificial_cases(tmp_path, monkeypatch):
    import src.evaluation.confusion_matrix as analysis
    path = tmp_path / "fixture.npz"
    # A correct high-confidence candidate belongs to the representative pools.
    np.savez(path, logits=make_logits([0]) * 3, labels=np.array([0]), image_ids=np.array(["synthetic/fake.jpg"]))
    mapping = tmp_path / "mapping.json"
    write_mapping(mapping)
    monkeypatch.setattr(analysis, "save_confusion_matrix_figures", lambda *a, **kw: [])
    out = tmp_path / "out"
    result = run_analysis(ROOT / "configs/experiments/e9_error_analysis.yaml", path, "synthetic", out, mapping)
    assert result["dataset_split"] == "synthetic"
    assert "synthetic/fake.jpg" in (out / "representative_cases.csv").read_text()


def test_direct_exporter_rejects_validation_mismatch_without_output(tmp_path, analysis_inputs):
    labels, ids, reference, mapping = analysis_inputs
    out = tmp_path / "direct"
    with pytest.raises(ValueError, match="canonical Validation"):
        save_e9_artifacts(make_logits(labels[:1]), labels[:1], ids[:1], [f"c{i}" for i in range(30)], out, "validation", 0.80246283961825, 0.5011672918550012)
    assert not out.exists()


@pytest.mark.parametrize("kind", ["subset", "superset", "duplicate", "reordered"])
def test_reference_helper_requires_complete_unique_ordered_membership(tmp_path, kind):
    reference = write_validation_reference(tmp_path, ["a", "b"], [0, 1])
    ids, labels = {
        "subset": (["a"], [0]), "superset": (["a", "b", "c"], [0, 1, 2]),
        "duplicate": (["a", "a"], [0, 0]), "reordered": (["b", "a"], [1, 0]),
    }[kind]
    result = validate_ids_against_split(np.array(ids), np.array(labels), reference, "relative_path", "class_id")
    assert result["match"] is False


def test_official_invalid_split_reference_is_rejected_before_npz_read(tmp_path, monkeypatch):
    import yaml
    import scripts.run_error_analysis as runner
    data = yaml.safe_load((ROOT / "configs/experiments/e9_error_analysis.yaml").read_text(encoding="utf-8"))
    data["official_analysis"]["enabled"] = True
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    def forbidden_read(*args, **kwargs):
        pytest.fail("Gate must reject before reading any NPZ")
    monkeypatch.setattr(runner, "load_predictions_npz", forbidden_read)
    out = tmp_path / "out"
    with pytest.raises(PermissionError, match="Official split CSV"):
        runner.run_analysis(config, tmp_path / "clean_predictions.npz", "official", out, allow_official_analysis=True, split_reference_csv=tmp_path / "fake.csv")
    assert not out.exists()


def test_cli_invalid_validation_returns_nonzero_without_output(tmp_path, analysis_inputs, monkeypatch):
    import scripts.run_error_analysis as runner
    labels, ids, reference, mapping = analysis_inputs
    path = tmp_path / "predictions.npz"
    np.savez(path, logits=make_logits(labels[:1]), labels=labels[:1], image_ids=ids[:1])
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["run_error_analysis.py", "--config", str(ROOT / "configs/experiments/e9_error_analysis.yaml"), "--input", str(path), "--split", "validation", "--output-dir", str(out)])
    assert runner.main() != 0
    assert not out.exists()
