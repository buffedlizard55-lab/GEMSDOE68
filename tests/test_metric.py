"""Unit tests for the metric, the grid and the holdout construction."""
from __future__ import annotations

import numpy as np
import pytest

from gems68 import metric
from gems68.autocontext import make_folds, quadrant_blocks


def test_kernel_values():
    assert metric.kernel(0.0) == pytest.approx(1.0)
    assert metric.kernel(3.0) == pytest.approx(0.0)
    assert metric.kernel(1.5) == pytest.approx(0.5)
    assert metric.kernel(5.0) == pytest.approx(0.0)


def test_exact_matches_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(8):
        pred = np.zeros((9, 9))
        truth = np.zeros((9, 9), bool)
        for _ in range(6):
            pred[tuple(rng.integers(0, 9, 2))] = rng.random()
        for _ in range(4):
            truth[tuple(rng.integers(0, 9, 2))] = True
        a = metric.dti_exact(pred, truth.astype(np.int8))
        b = metric.dti_bruteforce(pred, truth)
        assert a["tp"] == pytest.approx(b["tp"], abs=1e-9)
        assert a["fp"] == pytest.approx(b["fp"], abs=1e-9)
        assert a["dti"] == pytest.approx(b["dti"], abs=1e-9)


def test_binary_matches_exact():
    rng = np.random.default_rng(1)
    pred = (rng.random((14, 14)) > 0.85)
    truth = (rng.random((14, 14)) > 0.9)
    a = metric.dti_exact(pred.astype(float), truth.astype(np.int8))
    b = metric.dti_binary(pred, truth)
    assert a["dti"] == pytest.approx(b["dti"], abs=1e-9)


def test_identity_one_over_dti():
    """1/DTI = alpha + alpha*(F/T) + beta*(K/T) with T = TP_w, F = FP_w, K = |G|."""
    rng = np.random.default_rng(2)
    pred = np.where(rng.random((16, 16)) > 0.8, rng.random((16, 16)), 0.0)
    truth = (rng.random((16, 16)) > 0.92)
    r = metric.dti_exact(pred, truth.astype(np.int8))
    T, F, K = r["tp"], r["fp"], float(r["n_truth"])
    lhs = 1.0 / r["dti"]
    rhs = metric.ALPHA + metric.ALPHA * (F / T) + metric.BETA * (K / T)
    assert lhs == pytest.approx(rhs, rel=1e-9)


def test_known_masking_removes_pixels():
    """A prediction sitting entirely on masked known-fault pixels scores zero credit."""
    truth = np.zeros((7, 7), np.int8)
    truth[3, 3] = 1
    pred = np.zeros((7, 7))
    pred[3, 3] = 1.0
    known = np.zeros((7, 7), bool)
    assert metric.dti_exact(pred, truth)["dti"] > 0.99
    known[3, 3] = True
    r = metric.dti_exact(pred, truth, known=known)
    assert r["n_truth"] == 0
    assert r["tp"] == 0.0


def test_marginal_bar_is_alpha_times_dti():
    # adding mass with realised weight k > alpha*DTI must raise DTI
    s = 0.2778
    bar = metric.marginal_bar(s)
    assert bar == pytest.approx(metric.ALPHA * s)
    # the (2,2) diagonal kernel weight on the 100 m lattice sits just above the bar
    assert metric.kernel(np.hypot(2, 2)) > bar


def test_pooled_dti_is_not_the_mean():
    a = dict(tp=1.0, fp=10.0, fn=9.0, n_truth=10)
    b = dict(tp=9.0, fp=1.0, fn=1.0, n_truth=10)
    pooled = metric.pooled_dti([a, b])["dti"]
    mean = 0.5 * (metric.dti_from_components(**{k: a[k] for k in ("tp", "fp", "fn")})
                  + metric.dti_from_components(**{k: b[k] for k in ("tp", "fp", "fn")}))
    assert pooled != pytest.approx(mean)


def test_folds_partition_the_catalogue():
    labels = np.full((40, 40), -1, np.int8)
    labels[5:35, 5:35] = 0
    labels[10, 8:20] = 1
    labels[25:30, 30] = 1
    labels[6, 33] = 1
    fp = labels >= 0
    folds = make_folds(labels, fp, buffer_px=3)
    assert len(folds) == 4
    tot = sum(f.n_hidden for f in folds)
    n_seg = np.sum(np.asarray([f.n_hidden for f in folds]) > 0)
    assert n_seg >= 1
    # every catalogue pixel of >=3 px belongs to exactly one fold
    union = np.zeros_like(fp)
    for f in folds:
        assert not (union & f.hidden).any()
        union |= f.hidden
    pos = (labels == 1) & fp
    from scipy import ndimage
    lab, n = ndimage.label(pos, structure=np.ones((3, 3), bool))
    big = np.isin(lab, [i for i in range(1, n + 1) if np.bincount(lab.ravel())[i] >= 3])
    assert (union == big).all()
    assert tot == int(big.sum())


def test_fold_masks_are_consistent():
    labels = np.full((30, 30), -1, np.int8)
    labels[3:27, 3:27] = 0
    labels[12, 6:18] = 1
    fp = labels >= 0
    f = make_folds(labels, fp, buffer_px=4)[0]
    # the withheld segments and their buffer never appear in the training mask
    from scipy import ndimage
    buf = ndimage.binary_dilation(f.hidden, iterations=4)
    assert not (f.train_mask & buf).any()
    # scoring masks exactly the visible catalogue
    assert f.known.sum() == f.visible.sum()
    assert not (f.known & f.hidden).any()
