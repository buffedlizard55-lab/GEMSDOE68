"""End-to-end regression suite for GEMSDOE68 S68 O(2) Auto-Context submission & Run Card."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio

from gems.validate_submission import validate, validate_portal_ready

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_CARD = ROOT / "evidence" / "s68_run_card.json"
DOCS_CARD = ROOT / "docs" / "data" / "run_card.json"
CANARY_PATH = ROOT / "evidence" / "s68_leakage_canary.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_run_card_exists_and_matches_docs_copy() -> None:
    assert EVIDENCE_CARD.exists(), "evidence/s68_run_card.json missing"
    assert DOCS_CARD.exists(), "docs/data/run_card.json missing"
    ev = json.loads(EVIDENCE_CARD.read_text())
    dc = json.loads(DOCS_CARD.read_text())
    assert ev == dc, "Canonical Run Card and docs/data/run_card.json diverged"
    assert ev["schema"] == "gems.evidence.run_card.v2"
    assert ev["site"] == "GEMSDOE68"
    assert ev["evidence_class"] == "HOLDOUT-DTI"


def test_protocol_gate_and_submission_metadata() -> None:
    ev = json.loads(EVIDENCE_CARD.read_text())
    w = ev["winner"]
    assert w["gate_gt_2se_all_and_far"] is True
    assert w["ok_to_download_and_submit"] is True
    assert w["delta_vs_stage1_all"] > 0.05
    assert w["delta_vs_stage1_all_z_se"] > 2.0
    assert w["delta_vs_stage1_far"] > 0.02
    assert w["delta_vs_stage1_far_z_se"] > 2.0
    assert w["all_ci95"][0] > 0.18
    assert w["far_gt_1km_ci95"][0] > 0.12

    sub = ev["submission_artifacts"]
    note = sub["submission_note_le_140"]
    assert 10 <= len(note) <= 140
    assert sub["submission_note_length"] == len(note)
    assert sub["unique_submission_name"].startswith("GEMSDOE68-O2AC-MultiPhys-v3-")


def test_leakage_canary_all_59_channels_clean() -> None:
    assert CANARY_PATH.exists(), "evidence/s68_leakage_canary.json missing"
    canary = json.loads(CANARY_PATH.read_text())
    assert canary["n_channels"] == 59
    assert canary["passed"] is True
    assert canary["max_symmetric_auc"] < 0.90
    assert len(canary["flagged_channels_gt_0_90"]) == 0
    for ch_name, ch_val in canary["per_channel"].items():
        s_auc = ch_val["symmetric_auc"] if isinstance(ch_val, dict) else float(ch_val)
        assert s_auc < 0.90, f"Leakage canary trip on {ch_name}: {s_auc}"


def test_uniqueness_audit_against_full_fleet() -> None:
    ev = json.loads(EVIDENCE_CARD.read_text())
    u = ev["uniqueness_audit"]
    assert u["n_unique_rasters_checked"] >= 60
    assert u["passed"] is True
    assert u["max_abs_spearman_surface"] <= 0.90
    assert u["max_3px_dot_overlap"] <= 0.70

    zeros_sha = ev["submission_artifacts"]["recommended_portal_upload_tif"]["sha256"]
    for rec in u["top12_closest_rasters"]:
        assert rec["sha256"] != zeros_sha


def test_emitted_geotiff_bytes_checksums_and_validators() -> None:
    ev = json.loads(EVIDENCE_CARD.read_text())
    sub = ev["submission_artifacts"]

    for key in ("recommended_portal_upload_tif", "sample_submission_nan_companion_tif", "strict_validator_16of16_tif"):
        meta = sub[key]
        p = ROOT / meta["relpath"]
        assert p.exists(), f"Missing artifact {p}"
        assert p.stat().st_size == meta["bytes"]
        assert _sha256(p) == meta["sha256"]

    # Check .zip bundle matches recommended portal TIF byte-for-byte inside archive
    zmeta = sub["recommended_portal_upload_tif"]
    zp = ROOT / zmeta["zip_relpath"]
    assert zp.exists()
    assert zp.stat().st_size == zmeta["zip_bytes"]
    assert _sha256(zp) == zmeta["zip_sha256"]
    with zipfile.ZipFile(zp, "r") as zf:
        assert zmeta["filename"] in zf.namelist()
        raw = zf.read(zmeta["filename"])
        assert hashlib.sha256(raw).hexdigest() == zmeta["sha256"]

    # Check pixel range [0, 1] across all 12,279,160 cells of recommended portal TIF
    p_zeros = ROOT / zmeta["relpath"]
    with rasterio.open(p_zeros) as ds:
        assert ds.count == 1
        assert ds.dtypes[0] == "float32"
        assert str(ds.crs) == "EPSG:32611"
        assert ds.shape == (3730, 3292)
        arr = ds.read(1)
        assert not np.isnan(arr).any(), "Recommended portal TIF must contain zero NaNs anywhere"
        assert not np.isinf(arr).any(), "Recommended portal TIF must contain zero Infs anywhere"
        assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0
        assert int((arr > 0).sum()) == ev["winner"]["emitted_positive_pixels"]

    # Run live validator checks if competition reference data is present in workspace
    ref = ROOT / "data" / "sample_submission.tif"
    feat = ROOT / "data" / "training_features.tif"
    cat = ROOT / "data" / "existing_faults.tif"
    if ref.exists() and feat.exists() and cat.exists():
        res_portal = validate_portal_ready(p_zeros, ref, feat, cat)
        assert res_portal["passed"] is True
        assert res_portal["on_catalogue_positive_pixels"] == 0

        p_strict = ROOT / sub["strict_validator_16of16_tif"]["relpath"]
        res_strict = validate(p_strict, reference_path=ref, features_path=feat)
        assert res_strict["passed"] is True
        assert res_strict["has_failures"] is False


def test_all_run_card_copies_and_github_pages_links_resolve() -> None:
    import re

    canonical_text = EVIDENCE_CARD.read_text()
    # Strict RFC-8259 check: NaN/Infinity forbidden
    ev = json.loads(canonical_text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Non-RFC8259 constant: {c}")))
    assert ev["files"]["primary"]["validator"]["all_checks_passed"] is True
    assert ev["verdict"] == "promote"

    for mirror in (
        ROOT / "evidence" / "run_card.json",
        ROOT / "docs" / "data" / "run_card.json",
        ROOT / "docs" / "runcard.json",
        ROOT / "docs" / "lane68" / "data" / "run-card.json",
        ROOT / "docs" / "lane68" / "runcard.json",
    ):
        assert mirror.exists(), f"Missing mirror {mirror}"
        assert mirror.read_text() == canonical_text, f"Mirror out of sync: {mirror}"

    # Verify every relative href / src across all generated HTML pages resolves to a real file
    html_pages = [
        p for p in (list((ROOT / "docs").glob("*.html")) + list((ROOT / "docs" / "lane68").glob("*.html")))
        if not p.name.endswith(".template.html")
    ]
    assert len(html_pages) >= 15
    attr_re = re.compile(r'(?:href|src)="([^"#?]+)"')
    for page in html_pages:
        text = page.read_text()
        for target in attr_re.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "javascript:")):
                continue
            resolved = (page.parent / target).resolve()
            assert resolved.exists(), f"Broken relative link in {page.relative_to(ROOT)}: {target} -> {resolved}"

