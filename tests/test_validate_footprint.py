"""Shared-validator footprint source (GEMSDOE68 fix, 2026-10-20).

Band-1 of the real organiser stack contains sentinel holes *inside* the
official sample_submission footprint, so a candidate with NaN in those holes
must FAIL against the sample-derived footprint even though it passes the
band-1 fallback.
"""
from __future__ import annotations

import numpy as np
import rasterio
from affine import Affine

from gems.validate_submission import footprint_from_sample, validate


def _write(path, arr, nodata=None):
    profile = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                   dtype="float32", crs="EPSG:32611",
                   transform=Affine(100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0))
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)
    return path


def _status(report, name):
    return next(c["status"] for c in report["checks"] if c["check"] == name)


def test_sample_footprint_catches_nan_inside_band1_hole(tmp_path):
    sample = np.zeros((3730, 3292), np.float32)
    sample[:10, :10] = np.nan            # outside-footprint corner
    cand = np.zeros((3730, 3292), np.float32)
    cand[:10, :10] = np.nan
    cand[100, 100] = np.nan              # hole INSIDE the official footprint
    cand[200, 200] = 1.0
    sp = _write(tmp_path / "sample.tif", sample, nodata=float("nan"))
    cp = _write(tmp_path / "cand.tif", cand, nodata=float("nan"))

    assert footprint_from_sample(sp).sum() == sample.size - 100
    rep = validate(cp, footprint_path=sp)
    assert _status(rep, "no NaN inside footprint") == "FAIL"
    assert not rep["passed"]
    # without any footprint source the check stays SKIPPED (never a pass)
    rep2 = validate(cp)
    assert rep2["skipped_checks_mean_unverified"]
    assert not rep2["passed"]


def test_explicit_footprint_beats_features_footprint(tmp_path):
    # Regression (2026-10-20 build bug): passing BOTH a features raster and an
    # explicit footprint must use the explicit (sample) footprint, not band-1.
    H, W = 3730, 3292
    sample = np.zeros((H, W), np.float32)
    sample[:10, :10] = np.nan
    feats = np.zeros((H, W), np.float32)
    feats[100, 100] = -3.4028235e38      # band-1 sentinel hole inside footprint
    cand = np.zeros((H, W), np.float32)
    cand[:10, :10] = np.nan
    cand[100, 100] = np.nan              # NaN in the hole: must FAIL vs sample fp
    cand[200, 200] = 1.0
    sp = _write(tmp_path / "sample.tif", sample, nodata=float("nan"))
    fp_ = _write(tmp_path / "features.tif", feats)
    cp = _write(tmp_path / "cand.tif", cand, nodata=float("nan"))
    rep = validate(cp, features_path=fp_, footprint_path=sp)
    chk = next(c for c in rep["checks"] if c["check"] == "no NaN inside footprint")
    assert chk["status"] == "FAIL"
    assert "footprint_source=sample/template" in chk["detail"]
