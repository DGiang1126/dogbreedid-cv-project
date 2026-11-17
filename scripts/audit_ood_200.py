"""Data Audit for the internal OOD-200 set (Task 07).

Checks: exactly 200 images with the required group counts; every file readable;
label/source/license present and from a free licence; SHA-256 and pHash recomputed
and matching the manifest; no exact or near duplicates (within OOD-200 and, with
--id-splits, against the Validation/Calibration images); Unknown-Breed labels are
not among the 30 project classes; non-dog groups carry a manual "no dog" review.

Usage:
  python scripts/audit_ood_200.py --root data/ood_200 \
      --images-dir data/raw/stanford_dogs/Images --out outputs/research/ood_200
  python scripts/audit_ood_200.py ... --lock   # write LOCK.json after a PASS

The audit never reads Official Test images.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dct

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_GROUPS = {"unknown_breed": 100, "human": 30, "cat": 30, "other_animal": 20, "object_scene": 20}
REQUIRED_COLUMNS = ["image_id", "relative_path", "ood_group", "source", "license", "sha256",
                    "phash", "width", "height", "audit_status"]
# CC BY / BY-SA (optionally with a jurisdiction port such as "3.0 de"), CC0, public domain
FREE_LICENSE = re.compile(r"^(CC0( 1\.0)?|CC BY(-SA)?( \d\.\d( [a-z]{2})?)?|Public domain)$", re.I)
NEAR_DUP_BITS = 6
ID_SPLITS = ("validation", "calibration")  # Official Test intentionally excluded
NON_DOG_GROUPS = {"human", "cat", "other_animal", "object_scene"}
DOG_WORDS = re.compile(r"\b(dog|puppy|canis lupus familiaris|hund|chien|perro)\b", re.I)


def phash(image: Image.Image) -> str:
    """64-bit DCT perceptual hash (same algorithm used when collecting)."""
    g = np.asarray(image.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    c = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8].flatten()
    return "".join("1" if b else "0" for b in c > np.median(c[1:]))


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def _norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def audit(root: Path, images_dir: Path | None) -> dict:
    manifest_path = root / "manifest.csv"
    rows = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in (rows[0] if rows else {})]
    if missing_cols:
        errors.append(f"manifest missing columns: {missing_cols}")
        return {"status": "FAIL", "errors": errors, "warnings": warnings}

    counts = {g: 0 for g in REQUIRED_GROUPS}
    for r in rows:
        counts[r["ood_group"]] = counts.get(r["ood_group"], 0) + 1
    if len(rows) != 200:
        errors.append(f"expected 200 images, found {len(rows)}")
    if counts != REQUIRED_GROUPS:
        errors.append(f"group counts {counts} != {REQUIRED_GROUPS}")
    if len({r["image_id"] for r in rows}) != len(rows):
        errors.append("duplicate image_id values")

    mapping = json.loads((ROOT / "data/metadata/class_mapping.json").read_text(encoding="utf-8"))
    id_names = {_norm(v) for v in mapping.values()}

    hashes = []
    for r in rows:
        rid = r["image_id"]
        path = (root / r["relative_path"]).resolve()
        if root.resolve() not in path.parents:
            errors.append(f"{rid}: path escapes dataset root")
            continue
        for col in ("source", "license"):
            if not r[col].strip():
                errors.append(f"{rid}: empty {col}")
        if not FREE_LICENSE.match(r["license"].strip()):
            errors.append(f"{rid}: license not in allowed set: {r['license']!r}")
        if not r["source"].startswith("https://commons.wikimedia.org/"):
            warnings.append(f"{rid}: source is not a Wikimedia Commons page")
        try:
            data = path.read_bytes()
            with Image.open(path) as im:
                im.load()
                rgb = im.convert("RGB")
        except Exception as exc:  # unreadable / corrupt
            errors.append(f"{rid}: unreadable ({type(exc).__name__})")
            continue
        sha = hashlib.sha256(data).hexdigest()
        if sha != r["sha256"]:
            errors.append(f"{rid}: sha256 mismatch")
        ph = phash(rgb)
        if hamming(ph, r["phash"]) > 2:
            errors.append(f"{rid}: phash mismatch with manifest")
        if (rgb.width, rgb.height) != (int(r["width"]), int(r["height"])):
            errors.append(f"{rid}: size mismatch with manifest")
        if min(rgb.size) < 224:
            warnings.append(f"{rid}: short side {min(rgb.size)} < 224")
        group = r["ood_group"]
        label = r.get("label", "")
        if group == "unknown_breed":
            if not label.strip():
                errors.append(f"{rid}: unknown_breed without breed label")
            elif _norm(label) in id_names:
                errors.append(f"{rid}: breed {label!r} is one of the 30 ID classes")
        elif group in NON_DOG_GROUPS:
            if DOG_WORDS.search(r.get("file_title", "")):
                errors.append(f"{rid}: non-dog group but title mentions a dog: {r.get('file_title')}")
            if "candidate_ref" not in r or not r["candidate_ref"]:
                errors.append(f"{rid}: missing manual review reference (candidate_ref)")
        hashes.append((rid, sha, ph))

    for i, (a_id, a_sha, a_ph) in enumerate(hashes):
        for b_id, b_sha, b_ph in hashes[i + 1:]:
            if a_sha == b_sha:
                errors.append(f"exact duplicate: {a_id} == {b_id}")
            elif hamming(a_ph, b_ph) <= NEAR_DUP_BITS:
                errors.append(f"near duplicate (pHash<={NEAR_DUP_BITS}): {a_id} ~ {b_id}")

    id_checked = 0
    if images_dir is not None:
        for split in ID_SPLITS:
            for r in csv.DictReader(open(ROOT / f"data/splits/{split}.csv", encoding="utf-8")):
                p = images_dir / r["relative_path"]
                with Image.open(p) as im:
                    ph = phash(im)
                id_checked += 1
                for rid, _, oph in hashes:
                    if hamming(ph, oph) <= NEAR_DUP_BITS:
                        errors.append(f"OOD {rid} near-duplicates ID image {split}:{r['relative_path']}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "n_images": len(rows), "group_counts": counts,
        "id_images_cross_checked": id_checked, "near_dup_bits": NEAR_DUP_BITS,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "errors": errors, "warnings": warnings,
        "manual_checks_not_automatable": [
            "non-dog groups contain no dog (visual review of every image, recorded via candidate_ref)",
            "unknown_breed labels are correct breeds (visual review against category/title)",
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="data/ood_200")
    ap.add_argument("--images-dir", default=None, help="Stanford Dogs Images dir for ID cross-check")
    ap.add_argument("--out", default="outputs/research/ood_200")
    ap.add_argument("--lock", action="store_true", help="write LOCK.json if audit PASSES")
    a = ap.parse_args(argv)
    root = (ROOT / a.root).resolve()
    images_dir = (ROOT / a.images_dir).resolve() if a.images_dir else None

    report = audit(root, images_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report.update({"created_at": stamp, "git_sha": _git_sha(), "root": a.root})
    out = (ROOT / a.out / f"audit_{stamp}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"audit {report['status']}: {len(report['errors'])} errors, {len(report['warnings'])} warnings -> {out}")
    for e in report["errors"][:20]:
        print("  ERROR", e)

    if a.lock:
        if report["status"] != "PASS":
            print("refusing to lock: audit did not pass")
            return 1
        lock = root / "LOCK.json"
        if lock.exists():
            print(f"refusing to overwrite existing {lock}")
            return 1
        lock.write_text(json.dumps({
            "locked_at": stamp, "manifest_sha256": report["manifest_sha256"],
            "n_images": report["n_images"], "group_counts": report["group_counts"],
            "audit_report": str(out.relative_to(ROOT).as_posix()), "git_sha": report["git_sha"],
        }, indent=2), encoding="utf-8")
        print(f"locked: {lock}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
