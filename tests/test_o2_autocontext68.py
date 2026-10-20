"""Comprehensive mathematical & engineering tests for O(2) Auto-Context,
Differentiable Tversky(0.2, 0.8) backprop, Submodular 300 m kernel thinning,
and DrivenData portal preflight validation (IR-68-PORTAL-01)."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from gems.submission_writer import write_submission
from gems.validate_submission import validate_portal_ready
from gems57.metric import dti_exact
from gems68.o2_autocontext import (
    O2_CONTEXT_NAMES,
    PreparedCatalogueField,
    PreparedStageField,
    TverskyContextNet,
    emit_submodular_300m,
    o2_ridge_orientation,
)


def test_o2_orientation_north_south_and_east_west_exact():
    """Verify the critical PR#4 orientation bug fix:
    A North-South fault ridge MUST have North-South strike tangent (|ty| > 0.99)
    and East-West normal (|nx| > 0.99), and vice versa for East-West."""
    yy, xx = np.mgrid[0:80, 0:80]
    p_ns = np.exp(-((xx - 40.0) ** 2) / 2.0).astype(np.float32)
    ori_ns = o2_ridge_orientation(p_ns)
    assert abs(float(ori_ns["ty"][40, 40])) > 0.995
    assert abs(float(ori_ns["tx"][40, 40])) < 0.05
    assert abs(float(ori_ns["nx"][40, 40])) > 0.995
    assert abs(float(ori_ns["ny"][40, 40])) < 0.05

    p_ew = np.exp(-((yy - 40.0) ** 2) / 2.0).astype(np.float32)
    ori_ew = o2_ridge_orientation(p_ew)
    assert abs(float(ori_ew["tx"][40, 40])) > 0.995
    assert abs(float(ori_ew["ty"][40, 40])) < 0.05
    assert abs(float(ori_ew["ny"][40, 40])) > 0.995


def test_o2_bilateral_gap_bridging_and_non_duplicate_anisotropic_smooths():
    """Verify bilateral MIN bridges a 10-pixel blind buried gap on a N-S fault,
    and along-strike smooths strictly exceed across-strike smooths."""
    yy, xx = np.mgrid[0:90, 0:90]
    fp = np.ones((90, 90), dtype=bool)
    p = np.zeros((90, 90), dtype=np.float32)
    # Two colinear N-S fault segments separated by a blind gap at y in [41..49]
    p[:41, 45] = 0.95
    p[50:, 45] = 0.95

    fp_y, fp_x = np.nonzero(fp)
    sf = PreparedStageField(p[fp], fp, fp_y.astype(np.int32), fp_x.astype(np.int32))
    gap_center_idx = np.array([45 * 90 + 45], dtype=np.int64)
    off_axis_idx = np.array([45 * 90 + 20], dtype=np.int64)

    feats = sf.extract_ac(np.concatenate([gap_center_idx, off_axis_idx]))
    idx_min10 = O2_CONTEXT_NAMES.index("along_min_10")
    idx_ratio11 = O2_CONTEXT_NAMES.index("aniso_ratio_11")

    assert feats[0, idx_min10] > 0.20, f"Gap center must bridge blind gap, got {feats[0, idx_min10]}"
    assert feats[1, idx_min10] < 1e-4
    # Check at a point on the active strand (30, 45): along_11 strictly exceeds across_11
    on_strand = sf.extract_ac(np.array([30 * 90 + 45], dtype=np.int64))
    assert on_strand[0, idx_ratio11] > 0.25


def test_catalogue_context_never_sees_masked_target_fault():
    """Verify leakage invariant: masking target quadrant completely removes target fault."""
    fp = np.ones((60, 60), dtype=bool)
    fp_y, fp_x = np.nonzero(fp)
    full_cat = np.zeros((60, 60), dtype=bool)
    full_cat[10, 10:25] = True   # visible block fault
    full_cat[45, 35:55] = True   # target block fault (MUST BE MASKED)

    target_block = np.zeros((60, 60), dtype=bool)
    target_block[30:, 30:] = True

    vis = full_cat & ~target_block
    cf = PreparedCatalogueField(vis, fp, fp_y.astype(np.int32), fp_x.astype(np.int32))
    sf = PreparedStageField(np.full(fp.sum(), 0.1, np.float32), fp,
                            fp_y.astype(np.int32), fp_x.astype(np.int32))
    idx_on_hidden = np.array([45 * 60 + 45], dtype=np.int64)
    out = cf.extract_cat(idx_on_hidden, sf)
    # Distance to (10, 24) is hypot(35, 21) = 40.8 px => log1p(40.8) > 3.7 (NOT 0!)
    assert out[0, 0] > 3.6


def test_tversky_context_net_analytical_gradients_match_finite_difference():
    """Verify exact analytical gradient of Differentiable Tversky(0.2, 0.8) + BCE."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((48, 9)).astype(np.float64)
    base_logit = rng.standard_normal(48).astype(np.float64)
    target = np.clip(rng.uniform(-0.5, 1.0, size=48), 0.0, 1.0)
    net = TverskyContextNet(in_dim=9, hidden1=6, hidden2=5, seed=11)

    _, grads = net.loss_and_grads(X, base_logit, target)
    eps = 1e-6
    for pname in ("W1", "b1", "W2", "b2", "w3"):
        arr = getattr(net, pname)
        num = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = float(arr[idx])
            arr[idx] = orig + eps
            lp, _ = net.loss_and_grads(X, base_logit, target)
            arr[idx] = orig - eps
            lm, _ = net.loss_and_grads(X, base_logit, target)
            arr[idx] = orig
            num[idx] = (lp - lm) / (2.0 * eps)
            it.iternext()
        rel_err = np.linalg.norm(grads[pname] - num) / (np.linalg.norm(num) + 1e-12)
        assert rel_err < 1e-6, f"Gradient mismatch on {pname}: rel_err={rel_err}"


def test_submodular_300m_beats_topn_on_fault_ribbon_and_respects_catalogue():
    """On a 3px-wide blurred fault ribbon with fixed 12-dot budget, raw top-N clumps
    all dots in one spot, whereas emit_submodular_300m spaces dots along the entire
    ribbon to maximize DTI and places 0 dots on catalogue."""
    truth = np.zeros((60, 60), dtype=bool)
    truth[30, 8:52] = True
    yy, xx = np.mgrid[0:60, 0:60]
    ribbon = np.exp(-((yy - 30.0) ** 2) / 3.0) * (1.0 - 0.002 * np.abs(xx - 30.0))
    ribbon[:, :6] = 0.0
    ribbon[:, 54:] = 0.0
    allowed = np.ones((60, 60), dtype=bool)
    cat = np.zeros((60, 60), dtype=bool)
    cat[10, :] = True
    ribbon[10, :] = 0.99  # high-score mapped fault should be zero-reward & excluded

    topn = np.zeros((60, 60), dtype=np.float32)
    flat_order = np.argsort((ribbon * (~cat)).ravel())[-12:]
    topn.ravel()[flat_order] = 1.0

    subm = emit_submodular_300m(ribbon, allowed, n_dots=12, catalogue_2d=cat)
    assert int((subm > 0).sum()) == 12
    assert int(((subm > 0) & cat).sum()) == 0
    assert dti_exact(subm, truth)["dti"] > dti_exact(topn, truth)["dti"] + 0.25


def test_portal_preflight_catches_3061_nan_mismatch_and_passes_clean_zeros(tmp_path):
    """Regression test for IR-68-PORTAL-01 ('Predicted values must be in range [0, 1]')."""
    tr = from_bounds(500000, 4400000, 502000, 4402000, 20, 20)
    profile = {"driver": "GTiff", "dtype": "float32", "count": 1, "width": 20,
               "height": 20, "crs": "EPSG:32611", "transform": tr}
    sub_arr = np.full((20, 20), np.nan, dtype=np.float32)
    sub_arr[2:18, 2:18] = 0.0  # sample_submission footprint is 16x16 = 256 cells
    feat_arr = np.full((20, 20), -3.4e38, dtype=np.float32)
    feat_arr[2:18, 2:17] = 1.0  # features band 1 missing col 17 (16 cells -> mismatch!)

    sub_p, feat_p = tmp_path / "sample.tif", tmp_path / "feats.tif"
    with rasterio.open(sub_p, "w", **profile) as d:
        d.write(sub_arr, 1)
    with rasterio.open(feat_p, "w", **profile, nodata=-3.4e38) as d:
        d.write(feat_arr, 1)

    fp_feat_only = feat_arr > -1e30
    pred = np.zeros((20, 20), dtype=np.float32)
    pred[5, 5] = 0.9

    # Buggy write using fp_feat_only + outside_fill=nan leaves 16 NaNs inside sample_submission!
    buggy_p = tmp_path / "buggy.tif"
    write_submission(buggy_p, pred, fp_feat_only, sub_p)
    res_buggy = validate_portal_ready(buggy_p, sub_p, feat_p)
    assert not res_buggy["passed"]
    assert res_buggy["nan_in_sample_sub"] == 16

    # Fixed write using union footprint + outside_fill=0.0 passes 100% with 0 NaNs anywhere!
    fp_union = np.isfinite(sub_arr) | fp_feat_only
    clean_p = tmp_path / "clean_zeros.tif"
    write_submission(clean_p, pred, fp_union, sub_p, outside_fill=0.0)
    res_clean = validate_portal_ready(clean_p, sub_p, feat_p)
    assert res_clean["passed"]
    assert res_clean["all_12279160_pixels_finite_in_0_1"]
    assert res_clean["nan_in_sample_sub"] == 0
