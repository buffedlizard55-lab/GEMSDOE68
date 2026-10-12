"""Tests for the shared tools this lane depends on.

The shared packages (``gems``, ``gems57``) are vendored verbatim from the
template and are *not* forked here.  These tests pin the three behaviours the
auto-context lane relies on:

1. the DTI arithmetic agrees with a literal transcription of the published
   equations (so a stack trained against it is trained against the real metric);
2. the submission writer refuses to emit anything outside [0, 1] and always
   writes NaN exactly outside the footprint;
3. the hide-and-recover holdout is spatially blocked, label-blind, and masks
   visible faults pixel-exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from gems.submission_writer import write_submission
from gems.validate_submission import validate
from gems57 import holdout as H57
from gems57 import metric as M57


# --------------------------------------------------------------------- metric
def test_dti_binary_matches_bruteforce():
    rng = np.random.default_rng(3)
    pred = np.zeros((40, 40), bool)
    pred[rng.random((40, 40)) > 0.93] = True
    truth = np.zeros((40, 40), bool)
    truth[10:14, 10:30] = True
    truth[25:27, 5:20] = True
    a = M57.dti_binary(pred, truth)
    b = M57.dti_bruteforce(pred, truth)
    for k in ("tp", "fp", "fn", "dti"):
        assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-9), k


def test_dti_is_zero_when_nothing_is_emitted():
    truth = np.zeros((30, 30), bool)
    truth[5:9, 5:9] = True
    r = M57.dti_binary(np.zeros((30, 30), bool), truth)
    assert r["dti"] == 0.0
    assert r["fn"] == truth.sum()


def test_known_pixels_are_removed_from_scoring():
    """A dot that sits on a mapped (known) fault must earn no credit.

    The shared evaluator removes ``known`` pixels from the scored domain
    entirely, so such a dot contributes neither TP nor FP and cannot inflate the
    score -- which is the behaviour the holdout relies on when it masks visible
    faults pixel-exactly.
    """
    truth = np.zeros((30, 30), bool)
    truth[5:9, 5:9] = True
    known = np.zeros((30, 30), bool)
    known[5:9, 5:9] = True
    dots = np.zeros((30, 30), bool)
    dots[6, 6] = True
    r = M57.dti_binary(dots, truth, known=known)
    assert r["tp"] == 0.0
    assert r["fp"] == 0.0
    assert r["n_emitted"] == 0
    assert r["dti"] == 0.0
    # without the mask the same dot does earn credit, proving the mask is what
    # removes it
    r2 = M57.dti_binary(dots, truth)
    assert r2["tp"] > 0.0


def test_marginal_accept_is_consistent_with_recomputing_dti():
    rng = np.random.default_rng(11)
    truth = np.zeros((50, 50), bool)
    truth[10:13, 10:40] = True
    dots = np.zeros((50, 50), bool)
    dots[11, 12] = True
    base = M57.dti_binary(dots, truth)
    # add a redundant dot one pixel away from an existing one
    dots2 = dots.copy()
    dots2[11, 13] = True
    after = M57.dti_binary(dots2, truth)
    dT = after["tp"] - base["tp"]
    k_self = 1.0                       # it sits on a truth pixel
    T, M, n, K = base["tp"], float(dots.sum()), float(dots.sum()), float(truth.sum())
    assert M57.marginal_accept(dT, k_self, T, M, n, K) == (after["dti"] > base["dti"])


# ------------------------------------------------------------ submission writer
def _grid(tmp_path):
    """A tiny EPSG:32611 100 m reference grid plus a matching features raster."""
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "ref.tif"
    arr = np.zeros((20, 25), np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=20, width=25, count=1,
                       dtype="float32", crs="EPSG:32611",
                       transform=from_origin(500000, 4000000, 100, 100)) as dst:
        dst.write(arr, 1)
    feats = tmp_path / "features.tif"
    farr = np.full((20, 25), 5.0, np.float32)
    farr[0, 0] = -3.4e38                      # nodata sentinel
    with rasterio.open(feats, "w", driver="GTiff", height=20, width=25, count=1,
                       dtype="float32", crs="EPSG:32611",
                       transform=from_origin(500000, 4000000, 100, 100)) as dst:
        dst.write(farr, 1)
    return path, feats


def test_writer_rejects_out_of_range_values(tmp_path):
    ref, _ = _grid(tmp_path)
    pred = np.zeros((20, 25))
    pred[3, 3] = 1.4
    fp = np.ones((20, 25), bool)
    with pytest.raises(ValueError):
        write_submission(tmp_path / "x.tif", pred, fp, ref, overwrite=True)


def test_writer_rejects_nan_inside_footprint(tmp_path):
    ref, _ = _grid(tmp_path)
    pred = np.zeros((20, 25))
    pred[3, 3] = np.nan
    with pytest.raises(ValueError):
        write_submission(tmp_path / "x.tif", pred, np.ones((20, 25), bool), ref,
                         overwrite=True)


def test_writer_round_trip_passes_the_validator(tmp_path):
    ref, feats = _grid(tmp_path)
    # the footprint the validator derives is "features band 1 is finite and above
    # the nodata sentinel", so the test footprint must match it exactly
    fp = np.ones((20, 25), bool)
    fp[0, 0] = False
    pred = np.zeros((20, 25))
    pred[5:9, 5:20] = 0.7
    out = tmp_path / "sub.tif"
    receipt = write_submission(out, pred, fp, ref, overwrite=True,
                               name="unit-test", note="n" * 140)
    assert receipt["nan_inside"] == 0
    report = validate(out, ref, feats)
    assert report["has_failures"] is False, report
    assert report["passed"], report          # no FAIL and no SKIPPED
    assert all(c["status"] in ("PASS", "INFO") for c in report["checks"])


# ---------------------------------------------------------------------- holdout
def test_quadrant_partition_is_label_blind():
    fp = np.zeros((100, 100), bool)
    fp[10:90, 10:90] = True
    q = H57.quadrant_ids(fp)
    assert set(np.unique(q)) == {-1, 0, 1, 2, 3}
    assert int((q >= 0).sum()) == int(fp.sum())
    # the split depends only on the footprint, never on any label
    q2 = H57.quadrant_ids(fp.copy())
    assert np.array_equal(q, q2)


def test_holdout_masks_visible_faults_and_hides_whole_segments():
    """A tiny synthetic catalogue: withheld pixels must be excluded from both the
    training labels and the scored domain's visible set."""
    from gems57.grid import Grid
    from rasterio.transform import Affine

    h = w = 120
    fp = np.zeros((h, w), bool)
    fp[10:110, 10:110] = True
    cat = np.zeros((h, w), bool)
    # single-pixel-wide strands (the catalogue is a trace network, and the shared
    # segmenter treats thick blobs as junctions), each lying wholly inside the
    # 12 px eroded domain and spread over all four quadrants
    cat[30, 25:55] = True
    cat[30, 65:95] = True
    cat[85, 25:55] = True
    cat[85, 65:95] = True
    cat[45:70, 40] = True
    cat[45:70, 78] = True
    grid = Grid(footprint=fp, catalogue=cat, crs="EPSG:32611",
                transform=Affine(100, 0, 0, 0, -100, 0), height=h, width=w)
    ctx = H57.build_holdout(grid, hide_frac=0.25, seeds=(20,), modes=("all",))
    assert len(ctx.cells) == 4
    seen_hidden = 0
    for cell in ctx.cells:
        hidden = ctx.hidden_by_cell[cell.key]          # full-grid mask
        assert hidden.shape == (h, w)
        assert hidden.any()
        seen_hidden += int(hidden.sum())
        assert (hidden & ~fp).sum() == 0
        visible = grid.catalogue & ~hidden
        # cell.active is cropped to the padded quadrant bbox; compare there
        r0, r1 = cell.bbox[0].start, cell.bbox[0].stop
        c0, c1 = cell.bbox[1].start, cell.bbox[1].stop
        act = np.zeros((h, w), bool)
        act[r0:r1, c0:c1] = cell.active
        assert act.shape == (h, w)
        # truth is hidden and inside the scored domain
        truth = hidden & act
        assert int(truth.sum()) == cell.n_truth > 0
        # the scored domain excludes every visible fault pixel, exactly
        assert (act & visible).sum() == 0
        # nothing withheld may be visible
        assert (hidden & visible).sum() == 0
    # withholding must actually remove a meaningful share of the catalogue
    assert seen_hidden > 0.15 * int(cat.sum())
