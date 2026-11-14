import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.ood import id_selective_metrics, ood_auroc, ood_rejection_metrics

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit_mod = _load("audit_ood_200")
eval_mod = _load("evaluate_ood_200")


# ---------- metrics on synthetic data ----------

def test_id_selective_metrics_synthetic():
    accepted = np.array([True, True, False, True])
    preds = np.array([1, 2, 3, 4])
    labels = np.array([1, 0, 3, 4])
    m = id_selective_metrics(accepted, preds, labels)
    assert m["id_coverage"] == 0.75
    assert m["id_selective_accuracy"] == pytest.approx(2 / 3)
    assert m["id_selective_risk"] == pytest.approx(1 / 3)
    assert m["id_full_accuracy"] == 0.75


def test_id_selective_metrics_with_no_accept():
    m = id_selective_metrics(np.array([False, False]), np.array([0, 1]), np.array([0, 1]))
    assert m["id_coverage"] == 0 and m["id_selective_accuracy"] is None


def test_ood_rates_synthetic():
    m = ood_rejection_metrics(np.array([True, False, False, False]))
    assert m["ood_false_accept_rate"] == 0.25 and m["ood_rejection_rate"] == 0.75
    with pytest.raises(ValueError):
        ood_rejection_metrics(np.array([1, 0]))  # must be boolean


def test_auroc_orientation_ood_positive():
    assert ood_auroc([0.0, 0.1], [0.9, 1.0]) == 1.0  # higher = more OOD
    assert ood_auroc([0.9, 1.0], [0.0, 0.1]) == 0.0
    assert ood_auroc([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_pipeline_accepts_only_when_both_pass():
    big = np.zeros((3, 30)); big[0, 0] = 20.0          # confident ID-like
    big[1, :] = 0.0                                      # flat -> high energy, low conf
    big[2, 0] = 1.2                                      # moderate
    d = eval_mod.method_decisions(big, energy_threshold=-2.0, energy_temperature=1.0)
    assert d["pipeline"].tolist() == (d["energy_detector"] & d["e7_baseline"]).tolist()
    assert d["pipeline"][0] and not d["pipeline"][1]


# ---------- runner guards ----------

def test_runner_refuses_official_test():
    with pytest.raises(eval_mod.RunnerError):
        eval_mod.id_split_rows("official_test")


def test_runner_requires_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_mod, "ROOT", tmp_path)
    (tmp_path / "data/ood_200").mkdir(parents=True)
    with pytest.raises(eval_mod.RunnerError, match="not locked"):
        eval_mod.verify_ood_lock()


def test_runner_detects_manifest_change_after_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_mod, "ROOT", tmp_path)
    root = tmp_path / "data/ood_200"; root.mkdir(parents=True)
    (root / "manifest.csv").write_text("image_id\na\n", encoding="utf-8")
    (root / "LOCK.json").write_text(json.dumps({"manifest_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(eval_mod.RunnerError, match="changed after lock"):
        eval_mod.verify_ood_lock()


def test_threshold_must_come_from_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_mod, "ROOT", tmp_path)
    art = tmp_path / eval_mod.THRESHOLD_ARTIFACT; art.parent.mkdir(parents=True)
    art.write_text(json.dumps({"split": "ood_200", "ood_200_used": True, "threshold": 0}), encoding="utf-8")
    with pytest.raises(eval_mod.RunnerError):
        eval_mod.load_locked_threshold()


def test_select_threshold_refuses_to_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_mod, "ROOT", tmp_path)
    art = tmp_path / eval_mod.THRESHOLD_ARTIFACT; art.parent.mkdir(parents=True); art.write_text("{}")
    with pytest.raises(eval_mod.RunnerError, match="locked"):
        eval_mod.cmd_select_threshold(type("A", (), {"images_dir": "x", "device": "cpu",
                                                     "batch_size": 1, "energy_temperature": 1.0})())


def test_new_run_dir_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_mod, "ROOT", tmp_path)
    first = eval_mod.new_run_dir(Path("runs"), "eval")
    (first / "x").write_text("keep")
    with pytest.raises(FileExistsError):
        first.mkdir(exist_ok=False)
    assert (first / "x").read_text() == "keep"


# ---------- audit ----------

def _make_dataset(tmp_path, specs):
    root = tmp_path / "ood"; (root / "images").mkdir(parents=True)
    rows = []
    for i, (group, color, label, corrupt) in enumerate(specs):
        rel = f"images/img_{i}.jpg"
        if corrupt:
            (root / rel).write_bytes(b"not an image")
            sha, ph, w, h = "x", "0" * 64, 300, 300
        else:
            arr = np.zeros((300, 300, 3), np.uint8); arr[:, : 30 * (i % 10 + 1)] = color
            Image.fromarray(arr).save(root / rel)
            sha = hashlib.sha256((root / rel).read_bytes()).hexdigest()
            ph = audit_mod.phash(Image.open(root / rel)); w = h = 300
        rows.append(dict(image_id=f"id{i}", relative_path=rel, ood_group=group, label=label,
                         source="https://commons.wikimedia.org/wiki/File:x.jpg", license="CC BY-SA 4.0",
                         sha256=sha, phash=ph, width=w, height=h, audit_status="pending",
                         file_title="File:x.jpg", candidate_ref="c#00"))
    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    return root


def test_audit_flags_wrong_counts_corrupt_and_duplicates(tmp_path):
    root = _make_dataset(tmp_path, [
        ("unknown_breed", 200, "Samoyed", False),
        ("unknown_breed", 200, "Samoyed", False),   # same pattern width? differs by i -> fine
        ("cat", 50, "cat", True),                   # corrupt file
    ])
    # duplicate: copy image 0 over image 1 and fix its manifest sha/phash accordingly
    rows = list(csv.DictReader(open(root / "manifest.csv", encoding="utf-8")))
    (root / rows[1]["relative_path"]).write_bytes((root / rows[0]["relative_path"]).read_bytes())
    rows[1]["sha256"], rows[1]["phash"] = rows[0]["sha256"], rows[0]["phash"]
    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    report = audit_mod.audit(root, None)
    text = "\n".join(report["errors"])
    assert report["status"] == "FAIL"
    assert "expected 200 images" in text
    assert "unreadable" in text
    assert "exact duplicate" in text


def test_audit_rejects_id_breed_and_dog_in_non_dog_group(tmp_path):
    root = _make_dataset(tmp_path, [("unknown_breed", 120, "Golden Retriever", False),
                                    ("human", 90, "bench", False)])
    rows = list(csv.DictReader(open(root / "manifest.csv", encoding="utf-8")))
    rows[1]["file_title"] = "File:people with dog on bench.jpg"
    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    text = "\n".join(audit_mod.audit(root, None)["errors"])
    assert "one of the 30 ID classes" in text
    assert "title mentions a dog" in text


def test_audit_license_rule_accepts_ported_cc_and_rejects_nc():
    assert audit_mod.FREE_LICENSE.match("CC BY-SA 3.0 de")
    assert audit_mod.FREE_LICENSE.match("CC0")
    assert not audit_mod.FREE_LICENSE.match("CC BY-NC 2.0")


def test_locked_real_ood_200_is_complete_if_present():
    lock = ROOT / "data/ood_200/LOCK.json"
    if not lock.exists():
        pytest.skip("internal OOD-200 not available on this machine")
    rows, info = eval_mod.verify_ood_lock()
    assert len(rows) == 200
    assert info["group_counts"] == audit_mod.REQUIRED_GROUPS
