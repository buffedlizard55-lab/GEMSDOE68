"""Official Distance-Weighted Tversky Index (DTI) for the DOE GEMS Prize Challenge.

Transcribed from the official problem description
(https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric)
as re-verified against the shared implementation in ``GEMSDOE32/src/gems32/metric.py``
(blob fetched 2026-10-12 from api.github.com).  The equations are:

    k(d)   = max(1 - d / R, 0)                      R = 300 m = 3 px at 100 m/px
    TP_w   = sum_{g in G} max_{x: d(x,g)<=R} p(x) * k(d(x,g))
    FP_w   = sum_{x: p(x)>0} p(x) * (1 - max_{g in G} k(d(x,g)))
    FN_w   = |G| - TP_w
    DTI    = TP_w / (TP_w + alpha*FP_w + beta*FN_w)      alpha=0.2  beta=0.8

Known-fault masking: the organizer masks pixels that sit on already-mapped faults out of
the scored domain (``active = valid & ~known``); see organizer forum thread 11516 as cited
by the shared implementation.  We apply the same rule.

Two algebraic identities that the emission logic depends on, both proved in
``tests/test_metric.py`` by brute force:

    1/DTI = alpha + alpha*(F/T) + beta*(K/T)
    a unit of extra mass pays for itself  <=>  its realised kernel weight k > alpha*DTI
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

ALPHA: float = 0.2
BETA: float = 0.8
RADIUS_M: float = 300.0
PIXEL_M: float = 100.0
RADIUS_PX: float = RADIUS_M / PIXEL_M
EPS: float = 1e-12


def kernel(d, radius: float = RADIUS_PX) -> np.ndarray:
    """Triangular kernel k(d) = max(1 - d/R, 0); ``d`` in pixels."""
    return np.maximum(1.0 - np.asarray(d, dtype=np.float64) / radius, 0.0)


def kernel_offsets(radius: float = RADIUS_PX) -> list[tuple[int, int, float]]:
    r = int(np.ceil(radius))
    out = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            k = float(kernel(np.hypot(dy, dx), radius))
            if k > 0.0:
                out.append((dy, dx, k))
    return out


OFFSETS = kernel_offsets()


def _masks(pred, truth, valid, known):
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    if pred.ndim != 2 or pred.shape != truth.shape:
        raise ValueError("prediction and truth must be equal-shaped 2-D grids")
    valid = np.ones(pred.shape, bool) if valid is None else np.asarray(valid, bool)
    known = np.zeros(pred.shape, bool) if known is None else np.asarray(known, bool)
    if valid.shape != pred.shape or known.shape != pred.shape:
        raise ValueError("mask grid mismatch")
    return valid & ~known


def dti_exact(pred, truth, valid=None, known=None, alpha: float = ALPHA, beta: float = BETA) -> dict:
    """Exact DTI for soft or binary predictions in [0, 1].

    ``valid``  -- footprint mask (pixels inside the submission footprint).
    ``known``  -- already-mapped fault pixels, masked out of scoring.
    """
    active = _masks(pred, truth, valid, known)
    p = np.where(active & np.isfinite(pred), pred, 0.0).astype(np.float64)
    g = active & (np.asarray(truth) > 0)
    n = int(g.sum())
    if n == 0:
        return dict(tp=0.0, fp=float(p.sum()), fn=0.0, n_truth=0, dti=0.0, coverage=0.0)
    vals = p[active]
    if not np.isfinite(vals).all() or (vals < 0).any() or (vals > 1).any():
        raise ValueError("predictions inside the scored domain must be finite and in [0, 1]")
    # TP_w: for every truth pixel, the best kernel-weighted prediction nearby.
    best = np.zeros(p.shape, dtype=np.float64)
    H, W = p.shape
    for dy, dx, k in OFFSETS:
        ny0, ny1 = max(0, dy), min(H, H + dy)
        nx0, nx1 = max(0, dx), min(W, W + dx)
        sy0, sy1 = max(0, -dy), min(H, H - dy)
        sx0, sx1 = max(0, -dx), min(W, W - dx)
        np.maximum(best[ny0:ny1, nx0:nx1], p[sy0:sy1, sx0:sx1] * k, out=best[ny0:ny1, nx0:nx1])
    tp = float(best[g].sum())
    fn = float(n) - tp
    # FP_w: mass of each predicted pixel not covered by the kernel of any truth pixel.
    dist_to_g = distance_transform_edt(~g)
    fp = float((p * (1.0 - kernel(dist_to_g))).sum())
    dti = tp / (tp + alpha * fp + beta * fn + EPS)
    return dict(tp=tp, fp=fp, fn=fn, n_truth=n, dti=float(dti), coverage=tp / n)


def dti_binary(pred_bool, truth, valid=None, known=None, alpha: float = ALPHA, beta: float = BETA) -> dict:
    """Exact DTI when predictions are binary, computed with two distance transforms."""
    active = _masks(pred_bool, truth, valid, known)
    p = np.asarray(pred_bool, bool) & active
    g = np.asarray(truth, bool) & active
    n = int(g.sum())
    if n == 0:
        return dict(tp=0.0, fp=float(p.sum()), fn=0.0, n_truth=0, dti=0.0, coverage=0.0)
    if not p.any():
        return dict(tp=0.0, fp=0.0, fn=float(n), n_truth=n, dti=0.0, coverage=0.0)
    tp = float(kernel(distance_transform_edt(~p)[g]).sum())
    fn = float(n) - tp
    fp = float((1.0 - kernel(distance_transform_edt(~g)[p])).sum())
    return dict(tp=tp, fp=fp, fn=fn, n_truth=n, dti=tp / (tp + alpha * fp + beta * fn + EPS),
                coverage=tp / n)


def dti_bruteforce(pred, truth, alpha: float = ALPHA, beta: float = BETA, radius: float = RADIUS_PX) -> dict:
    """Literal O(|G|*|P|) transcription of the published equations, for unit tests only."""
    pred = np.asarray(pred, float)
    truth = np.asarray(truth, bool)
    gs = np.argwhere(truth)
    xs = np.argwhere(pred > 0)
    tp = fn = 0.0
    for g in gs:
        best = 0.0
        for x in xs:
            d = float(np.hypot(*(x - g)))
            if d <= radius:
                best = max(best, pred[tuple(x)] * max(1.0 - d / radius, 0.0))
        tp += best
        fn += 1.0 - best
    fp = 0.0
    for x in xs:
        kmax = 0.0
        for g in gs:
            kmax = max(kmax, max(1.0 - float(np.hypot(*(x - g))) / radius, 0.0))
        fp += pred[tuple(x)] * (1.0 - kmax)
    return dict(tp=tp, fp=fp, fn=fn, dti=tp / (tp + alpha * fp + beta * fn + EPS))


def dti_from_components(tp: float, fp: float, fn: float, alpha: float = ALPHA, beta: float = BETA) -> float:
    return float(tp / (tp + alpha * fp + beta * fn + EPS))


def marginal_bar(current_dti: float, alpha: float = ALPHA) -> float:
    """Minimum realised kernel weight for one extra unit of mass to raise DTI.

    Adding mass with realised kernel weight k gives dT = k and dF = 1 - k, so the
    denominator rises by exactly alpha*(k + 1 - k) = alpha.  Hence dDTI > 0 <=> k > alpha*DTI.
    """
    return alpha * current_dti


def pooled_dti(per_fold: list[dict], alpha: float = ALPHA, beta: float = BETA) -> dict:
    """Pool DTI across folds by summing the three components (not by averaging ratios)."""
    tp = sum(float(r["tp"]) for r in per_fold)
    fp = sum(float(r["fp"]) for r in per_fold)
    fn = sum(float(r["fn"]) for r in per_fold)
    n = sum(int(r["n_truth"]) for r in per_fold)
    return dict(tp=tp, fp=fp, fn=fn, n_truth=n, dti=dti_from_components(tp, fp, fn, alpha, beta))
