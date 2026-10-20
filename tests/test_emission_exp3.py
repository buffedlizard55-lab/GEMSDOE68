"""Exp-3 emission invariants: exact spacing, budget, allowance, crest geometry, determinism."""
import numpy as np
from scipy.spatial import cKDTree

from gems68 import emission as E


def _spacing_ok(dots, min_d):
    y, x = np.nonzero(dots)
    if len(y) < 2:
        return True
    d, _ = cKDTree(np.c_[y, x]).query(np.c_[y, x], k=2)
    return bool((d[:, 1] >= min_d - 1e-9).all())


def test_greedy_spacing_exact_and_budget():
    rng = np.random.default_rng(3)
    s = rng.random((120, 140)).astype(np.float32)
    allowed = np.ones_like(s, bool)
    allowed[:10] = False
    dots = E.greedy_spacing(s, allowed, allowed, 200, 2.8)
    assert dots.sum() == 200
    assert not dots[:10].any()
    assert _spacing_ok(dots, 2.8)
    # deterministic
    assert np.array_equal(dots, E.greedy_spacing(s, allowed, allowed, 200, 2.8))
    # greedy takes the global maximum first
    y, x = np.unravel_index(np.argmax(np.where(allowed, s, -1)), s.shape)
    assert dots[y, x]


def test_greedy_spacing_respects_candidates():
    s = np.ones((50, 50), np.float32)
    cand = np.zeros_like(s, bool)
    cand[::7, ::7] = True
    dots = E.greedy_spacing(s, np.ones_like(s, bool), cand, 10, 2.8)
    assert dots.sum() == 10
    assert (dots <= cand).all()


def test_crest_mask_finds_line_not_blob():
    s = np.zeros((101, 101), np.float32)
    s[50, 10:91] = 1.0           # a 1-px horizontal line
    s[49, 10:91] = s[51, 10:91] = 0.4
    allowed = np.ones_like(s, bool)
    crest = E.crest_mask(s, allowed)
    # crest concentrates near the line row, not spread over the plane
    ys, _ = np.nonzero(crest)
    assert len(ys) > 0
    assert (np.abs(ys - 50) <= 1).mean() > 0.8


def test_greedy_spacing_stratified_quotas_and_spacing():
    rng = np.random.default_rng(5)
    s = rng.random((100, 120)).astype(np.float32)
    allowed = np.ones_like(s, bool)
    zone = np.zeros(s.shape, np.int8)
    zone[:, 60:] = 1
    dots = E.greedy_spacing_stratified(s, allowed, allowed, zone, {0: 30, 1: 70}, 2.8)
    assert int((dots & (zone == 0)).sum()) == 30
    assert int((dots & (zone == 1)).sum()) == 70
    assert _spacing_ok(dots, 2.8)  # spacing holds across the zone boundary too
    assert np.array_equal(dots, E.greedy_spacing_stratified(s, allowed, allowed, zone, {0: 30, 1: 70}, 2.8))


def test_crest_mask_diagonal_line():
    s = np.zeros((121, 121), np.float32)
    for i in range(15, 106):
        s[i, i] = 1.0
    from scipy import ndimage as ndi
    s = ndi.gaussian_filter(s, 0.8)
    crest = E.crest_mask(s, np.ones_like(s, bool))
    ys, xs = np.nonzero(crest)
    assert len(ys) > 0
    assert (np.abs(ys - xs) <= 1).mean() > 0.7  # crest tracks the diagonal, orientation sign correct


def test_rank01_range():
    v = np.arange(100, dtype=np.float32).reshape(10, 10)
    m = np.ones((10, 10), bool)
    r = E.rank01(v, m)
    assert r.min() == 0.0 and abs(r.max() - 1.0) < 1e-6
    assert np.all(np.diff(r.ravel()) >= 0)
