"""Auto-context stacked refinement (the lane for this repository).

Method: Tu, "Contextual Classification for Semantic Image Interpretation", CVPR 2008,
doi:10.1109/CVPR.2008.4587436 (extended TPAMI version, pages.ucsd.edu/~ztu/publication/
pami_autocontext.pdf).  A second classifier is trained on the *probability map of the first*
classifier together with the original features, so it can learn spatial regularities a
per-pixel model cannot represent: continuity along strike, parallelism between splays,
en echelon spacing.

Protocol implemented here, exactly as specified:

Stage 1  out-of-fold probabilities from spatially blocked, buffered K-fold.  Never in-fold.
Stage 2  LightGBM on four families of input:
         (a) stage-1 probabilities sampled at +-1..15 px offsets along and across the
             local strike (structure-tensor orientation);
         (b) multi-scale anisotropic smooths of that probability map;
         (c) the raw gradient / tilt / scarp channels;
         (d) catalogue-context features computed with the target block's labels masked
             (distance, orientation and spacing to the nearest *visible* mapped fault).
         Iterated twice.
Gate     stage 2 must beat stage 1 on the outer blocks by more than 2 standard errors,
         including on pixels more than 1 km from any mapped fault.

The holdout is hide-and-recover: whole catalogue segments are withheld with a buffer, every
catalogue-derived feature is built only from the faults that remain visible, the visible
faults are masked pixel-exactly at scoring time (the organizer masks mapped faults), and the
pooled DTI uses alpha=0.2, beta=0.8 with a 300 m triangular kernel.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import lightgbm as lgb
from scipy import ndimage

from . import metric
from .data import load_feature_stack, load_footprint, load_labels, rows_of

# -----------------------------------------------------------------------------------------
# spatial blocking
# -----------------------------------------------------------------------------------------


def quadrant_blocks(height: int, width: int, footprint: np.ndarray) -> np.ndarray:
    """Four spatial blocks split at the footprint centroid (a 2x2 quadrant design)."""
    ys, xs = np.nonzero(footprint)
    cy, cx = float(np.median(ys)), float(np.median(xs))
    blk = np.full((height, width), -1, dtype=np.int8)
    yy, xx = np.mgrid[0:height, 0:width]
    blk[(yy < cy) & (xx < cx)] = 0
    blk[(yy < cy) & (xx >= cx)] = 1
    blk[(yy >= cy) & (xx < cx)] = 2
    blk[(yy >= cy) & (xx >= cx)] = 3
    blk[~footprint] = -1
    return blk


def disc(radius: int) -> np.ndarray:
    r = int(radius)
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


# -----------------------------------------------------------------------------------------
# hide-and-recover holdout
# -----------------------------------------------------------------------------------------


@dataclass
class Fold:
    index: int
    hidden: np.ndarray          # withheld catalogue segments (truth)
    visible: np.ndarray         # catalogue pixels that stay visible
    train_mask: np.ndarray      # pixels usable for training (footprint minus hidden buffer)
    known: np.ndarray           # masked pixel-exactly at scoring time (= visible)
    far: np.ndarray             # footprint pixels > 1 km from ANY mapped fault
    n_hidden: int = 0
    n_segments: int = 0

    def __post_init__(self):
        self.n_hidden = int(self.hidden.sum())


def make_folds(labels: np.ndarray, footprint: np.ndarray, buffer_px: int = 12,
               n_blocks: int = 4) -> list[Fold]:
    """Withhold whole catalogue segments, one spatial block at a time, with a buffer."""
    pos = (labels == 1) & footprint
    lab, n = ndimage.label(pos, structure=np.ones((3, 3), bool))
    ys, xs = np.nonzero(footprint)
    blocks = quadrant_blocks(labels.shape[0], labels.shape[1], footprint)
    # one label per connected segment; keep segments with >= 3 px (drop isolated speckles)
    sizes = np.bincount(lab.ravel())
    seg_ids = np.array([i for i in range(1, n + 1) if sizes[i] >= 3], dtype=np.int64)
    cents = ndimage.center_of_mass(pos, lab, seg_ids)
    assign = np.zeros(n + 1, dtype=np.int64)
    for sid, (cy, cx) in zip(seg_ids, cents):
        yi, xi = int(round(cy)), int(round(cx))
        yi = min(max(yi, 0), blocks.shape[0] - 1)
        xi = min(max(xi, 0), blocks.shape[1] - 1)
        assign[sid] = blocks[yi, xi]
    # distance to any mapped fault (full catalogue) for the >1 km stratum
    d_cat = ndimage.distance_transform_edt(~pos)
    folds = []
    for b in range(n_blocks):
        hidden = np.zeros_like(pos)
        ns = 0
        for sid in seg_ids:
            if assign[sid] == b:
                hidden |= (lab == sid)
                ns += 1
        hidden_buf = ndimage.binary_dilation(hidden, structure=disc(buffer_px))
        visible = pos & ~hidden
        train_mask = footprint & ~hidden_buf
        folds.append(Fold(index=b, hidden=hidden, visible=visible, train_mask=train_mask,
                          known=visible.copy(), far=footprint & (d_cat > 10.0),
                          n_segments=ns))
    return folds


# -----------------------------------------------------------------------------------------
# catalogue context, built ONLY from the visible catalogue
# -----------------------------------------------------------------------------------------


@dataclass
class CatalogueContext:
    """Distance / orientation / spacing to the nearest *visible* mapped fault."""
    d_near: np.ndarray      # distance (px) to nearest visible fault
    ny: np.ndarray          # unit vector towards it
    nx: np.ndarray
    strike_cos: np.ndarray  # local strike of the visible fault trace (unit, modulo 180 deg)
    strike_sin: np.ndarray
    dens5: np.ndarray
    dens20: np.ndarray

    @staticmethod
    def build(visible: np.ndarray, strike_theta: np.ndarray, scales=(5.0, 20.0)) -> "CatalogueContext":
        if visible.sum() == 0:
            H, W = visible.shape
            z = np.zeros((H, W), np.float32)
            return CatalogueContext(np.full((H, W), 999.0, np.float32), z, z, z, z, z, z)
        # nearest visible-fault pixel for every pixel, via the EDT with return_indices
        d, (iy, ix) = ndimage.distance_transform_edt(~visible, return_indices=True)
        yy, xx = np.mgrid[0:visible.shape[0], 0:visible.shape[1]]
        dy = (iy - yy).astype(np.float32)
        dx = (ix - xx).astype(np.float32)
        nrm = np.maximum(np.hypot(dy, dx), 1e-6)
        ny = (dy / nrm).astype(np.float32)
        nx = (dx / nrm).astype(np.float32)
        # local strike of the visible trace, sampled at the nearest visible pixel
        sc = np.cos(2 * strike_theta)[iy, ix].astype(np.float32)
        ss = np.sin(2 * strike_theta)[iy, ix].astype(np.float32)
        vis = visible.astype(np.float32)
        dens = [ndimage.uniform_filter(vis, size=int(2 * s + 1)).astype(np.float32) for s in scales]
        return CatalogueContext(d.astype(np.float32), ny, nx, sc, ss, dens[0], dens[-1])

    def rows(self, ys, xs) -> np.ndarray:
        d = self.d_near[ys, xs]
        # alignment between the local strike and the direction to the nearest visible fault:
        # 1 = a collinear continuation (a fault stepping over), 0 = perpendicular (a splay)
        proj = (self.ny[ys, xs] * self.strike_cos[ys, xs]
                + self.nx[ys, xs] * self.strike_sin[ys, xs])
        along = np.abs(proj)
        return np.column_stack([
            np.log1p(d),
            np.exp(-d / 3.0),
            np.exp(-d / 10.0),
            along,
            np.sqrt(np.maximum(1 - along * along, 0)),
            self.dens5[ys, xs],
            self.dens20[ys, xs],
        ]).astype(np.float32)


CTX_NAMES = ["ctx_logd", "ctx_k3", "ctx_k10", "ctx_along", "ctx_across", "ctx_dens5", "ctx_dens20"]


# -----------------------------------------------------------------------------------------
# stage-2 (auto-context) feature families
# -----------------------------------------------------------------------------------------

ALONG_CROSS_OFFSETS = (1, 2, 4, 7, 11, 15)
ANISO_SCALES = (6, 18)


def strike_theta(strike_cos: np.ndarray, strike_sin: np.ndarray) -> np.ndarray:
    """Strike angle in radians from the doubled-angle representation (modulo pi)."""
    return 0.5 * np.arctan2(strike_sin, strike_cos)


class AutoContext:
    """Feature builder for stage 2.  Everything is gathered at pixel indices, so training
    and full-grid prediction share one code path and peak memory stays bounded."""

    def __init__(self, prob: np.ndarray, theta: np.ndarray, ctx: CatalogueContext):
        self.prob = prob.astype(np.float32)
        self.theta = theta.astype(np.float32)
        self.ctx = ctx
        self.H, self.W = prob.shape

    def _sample(self, ys, xs, oy, ox):
        return ndimage.map_coordinates(self.prob, [ys + oy, xs + ox], order=1,
                                       mode="constant", cval=0.0).astype(np.float32)

    def rows(self, ys, xs) -> np.ndarray:
        th = self.theta[ys, xs].astype(np.float64)
        # strike direction (fault trace) and the direction across it
        sx, sy = np.cos(th), np.sin(th)
        ax, ay = -sy, sx
        cols = [self.prob[ys, xs].astype(np.float32)]
        for d in ALONG_CROSS_OFFSETS:
            for sgn in (1.0, -1.0):
                cols.append(self._sample(ys, xs, sgn * d * ay, sgn * d * ax))
        for s in ANISO_SCALES:
            cols.append(0.5 * (self._sample(ys, xs, s * ay, s * ax)
                               + self._sample(ys, xs, -s * ay, -s * ax)))
            cols.append(0.5 * (self._sample(ys, xs, s * ax, s * ay)
                               + self._sample(ys, xs, -s * ax, -s * ay)))
        out = np.column_stack(cols)
        return np.concatenate([out, self.ctx.rows(ys, xs)], axis=1).astype(np.float32)

    @staticmethod
    def n_prob_channels() -> int:
        return 1 + 2 * len(ALONG_CROSS_OFFSETS) + 2 * len(ANISO_SCALES)


# -----------------------------------------------------------------------------------------
# LightGBM helpers
# -----------------------------------------------------------------------------------------

LGB_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_child_samples=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=5.0, verbosity=-1, num_threads=2, seed=20261012)


def fit_lgb(X: np.ndarray, y: np.ndarray, num_boost_round: int = 220,
            scale_pos_weight: float = 4.0) -> lgb.Booster:
    ds = lgb.Dataset(X, label=y, free_raw_data=False)
    p = dict(LGB_PARAMS)
    p["scale_pos_weight"] = scale_pos_weight
    return lgb.train(p, ds, num_boost_round=num_boost_round)


def predict_blocked(booster: lgb.Booster, build_rows, ys: np.ndarray, xs: np.ndarray,
                    block: int = 300_000) -> np.ndarray:
    out = np.empty(ys.size, dtype=np.float32)
    for s in range(0, ys.size, block):
        e = min(ys.size, s + block)
        X = build_rows(ys[s:e], xs[s:e])
        out[s:e] = booster.predict(X, num_iteration=booster.best_iteration).astype(np.float32)
    return out


def scatter_to_grid(vals: np.ndarray, ys: np.ndarray, xs: np.ndarray, shape) -> np.ndarray:
    g = np.zeros(shape, dtype=np.float32)
    g[ys, xs] = vals
    return g


def rank01(v: np.ndarray) -> np.ndarray:
    """Monotone map of an arbitrary score to (0, 1] by average rank."""
    order = np.argsort(v, kind="stable")
    r = np.empty(v.size, dtype=np.float64)
    r[order] = np.arange(1, v.size + 1, dtype=np.float64)
    return (r / (v.size + 1.0)).astype(np.float32)


# -----------------------------------------------------------------------------------------
# sampling
# -----------------------------------------------------------------------------------------


def sample_pixels(mask_pos: np.ndarray, mask_neg: np.ndarray, rng: np.random.Generator,
                  max_pos: int = 60_000, neg_per_pos: float = 4.0):
    py, px = np.nonzero(mask_pos)
    ny, nx = np.nonzero(mask_neg)
    if py.size > max_pos:
        idx = rng.choice(py.size, max_pos, replace=False)
        py, px = py[idx], px[idx]
    n_neg = int(min(ny.size, py.size * neg_per_pos))
    idx = rng.choice(ny.size, n_neg, replace=False)
    ny, nx = ny[idx], nx[idx]
    ys = np.concatenate([py, ny])
    xs = np.concatenate([px, nx])
    y = np.concatenate([np.ones(py.size, np.int8), np.zeros(ny.size, np.int8)])
    return ys, xs, y


def stratified_positives(labels: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                         every: int = 1) -> tuple[np.ndarray, np.ndarray]:
    pos = (labels == 1) & mask
    py, px = np.nonzero(pos)
    if every > 1:
        idx = rng.choice(py.size, max(1, py.size // every), replace=False)
        py, px = py[idx], px[idx]
    return py, px


# -----------------------------------------------------------------------------------------
# one outer fold: stage 1 (inner-OOF) then stage 2 (two auto-context iterations)
# -----------------------------------------------------------------------------------------


@dataclass
class FoldResult:
    fold: int
    n_hidden: int
    n_segments: int
    stage1: dict = field(default_factory=dict)
    stage2: dict = field(default_factory=dict)
    stage1_far: dict = field(default_factory=dict)
    stage2_far: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)
    per_truth_s1: np.ndarray | None = None
    per_truth_s2: np.ndarray | None = None


def per_truth_credit(prob: np.ndarray, hidden: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Best kernel-weighted prediction within 3 px of each hidden truth pixel."""
    active = ~known
    p = np.where(active & np.isfinite(prob), prob, 0.0).astype(np.float64)
    best = np.zeros(p.shape)
    H, W = p.shape
    for dy, dx, k in metric.OFFSETS:
        ny0, ny1 = max(0, dy), min(H, H + dy)
        nx0, nx1 = max(0, dx), min(W, W + dx)
        sy0, sy1 = max(0, -dy), min(H, H - dy)
        sx0, sx1 = max(0, -dx), min(W, W - dx)
        np.maximum(best[ny0:ny1, nx0:nx1], p[sy0:sy1, sx0:sx1] * k, out=best[ny0:ny1, nx0:nx1])
    return best[hidden]


def run_fold(fold: Fold, stack, meta: np.ndarray, labels: np.ndarray, footprint: np.ndarray,
             theta: np.ndarray, raw_idx: list[int], seed: int = 0,
             inner_folds: int = 3, iterations: int = 2,
             max_pos: int = 45_000) -> FoldResult:
    t0 = time.time()
    rng = np.random.default_rng(1234 + seed * 97 + fold.index)
    H, W = footprint.shape
    ys_f, xs_f = np.nonzero(footprint)

    ctx = CatalogueContext.build(fold.visible, theta)

    # ---- stage 1: inner spatial OOF on the training blocks ------------------------------
    # inner blocks are the OTHER quadrants, buffered away from the hidden block.
    inner_blk = quadrant_blocks(H, W, fold.train_mask)
    inner_blk[~fold.train_mask] = -1
    inner_ids = sorted(set(inner_blk[fold.train_mask].tolist()))
    # The visible catalogue IS the training signal: the organizer masks mapped-fault pixels
    # at scoring time, not at training time.  Masking them here would leave no positives.
    train_pos = fold.visible & fold.train_mask
    train_neg = fold.train_mask & (labels == 0)

    prob_tr = np.zeros(ys_f.size, dtype=np.float32)
    flat_idx = np.zeros((H, W), np.int64)
    flat_idx[ys_f, xs_f] = np.arange(ys_f.size)

    def raw_rows(y, x):
        return rows_of(stack, y, x)[np.ix_(np.arange(y.size), np.asarray(raw_idx))]

    t1 = time.time()
    inner_boosters = []
    for i, b in enumerate(inner_ids):
        in_blk = inner_blk == b
        # buffer between inner train and inner valid
        buf = ndimage.binary_dilation(in_blk, structure=disc(6)) & ~in_blk
        tr_mask = fold.train_mask & ~in_blk & ~buf
        va_mask = in_blk & fold.train_mask
        py, px = np.nonzero(fold.visible & tr_mask)
        if py.size == 0:
            continue
        ny, nx = np.nonzero(tr_mask & (labels == 0))
        n_neg = int(min(ny.size, py.size * 4))
        j = rng.choice(ny.size, n_neg, replace=False)
        ny, nx = ny[j], nx[j]
        ty, tx = np.concatenate([py, ny]), np.concatenate([px, nx])
        ty_lab = np.concatenate([np.ones(py.size, np.int8), np.zeros(ny.size, np.int8)])
        b1 = fit_lgb(raw_rows(ty, tx), ty_lab)
        inner_boosters.append((b1, b))
        vy, vx = np.nonzero(va_mask)
        if vy.size:
            sub = np.arange(vy.size)
            rng.shuffle(sub)
            sub = sub[:400_000]
            vy, vx = vy[sub], vx[sub]
            pr = predict_blocked(b1, raw_rows, vy, vx)
            prob_tr[flat_idx[vy, vx]] = pr
    stage1_seconds = time.time() - t1

    prob1_grid = np.zeros((H, W), np.float32)
    prob1_grid[ys_f, xs_f] = prob_tr
    # fill the hidden block (never seen by stage 1) with the inner-mean model
    if inner_boosters:
        need = fold.hidden | (fold.train_mask & (prob_tr[flat_idx] == 0))
        ny2, nx2 = np.nonzero(need & footprint)
        acc = np.zeros(ny2.size, np.float32)
        for b1, _ in inner_boosters:
            acc += predict_blocked(b1, raw_rows, ny2, nx2)
        acc /= len(inner_boosters)
        prob1_grid[ny2, nx2] = acc

    # ---- stage 2: auto-context, iterated ----------------------------------------------
    stage2_seconds = []
    cur = prob1_grid
    models = []
    for it in range(iterations):
        t2 = time.time()
        ac = AutoContext(cur, theta, ctx)

        def ac_rows(y, x, _ac=ac, _ri=raw_idx):
            return np.concatenate([_ac.rows(y, x), raw_rows(y, x)], axis=1)

        # train on the training blocks only; hidden block is never touched
        ty, tx, ty_lab = sample_pixels(train_pos, train_neg, rng, max_pos=max_pos)
        b2 = fit_lgb(ac_rows(ty, tx), ty_lab, num_boost_round=260)
        models.append((b2, ac))
        ny3, nx3 = np.nonzero(footprint)
        vals = predict_blocked(b2, ac_rows, ny3, nx3)
        cur = scatter_to_grid(vals, ny3, nx3, (H, W))
        stage2_seconds.append(time.time() - t2)

    # ---- evaluation ---------------------------------------------------------------------
    def score(prob, extra=None):
        p = np.where(footprint & ~fold.known, prob, 0.0)
        p = np.clip(np.nan_to_num(p.astype(np.float64)), 0.0, 1.0)
        r = metric.dti_exact(p, fold.hidden.astype(np.int8), valid=footprint, known=fold.known)
        if extra is not None:
            m = extra & ~fold.known
            p2 = np.where(m, p, 0.0)
            t2_ = fold.hidden & m
            r = metric.dti_exact(p2, t2_.astype(np.int8), valid=m, known=np.zeros_like(m))
        return r

    s1 = score(prob1_grid)
    s2 = score(cur)
    s1f = score(prob1_grid, fold.far)
    s2f = score(cur, fold.far)

    return FoldResult(
        fold=fold.index, n_hidden=fold.n_hidden, n_segments=fold.n_segments,
        stage1=s1, stage2=s2, stage1_far=s1f, stage2_far=s2f,
        seconds=dict(stage1=stage1_seconds, stage2=stage2_seconds, total=time.time() - t0),
        per_truth_s1=per_truth_credit(prob1_grid, fold.hidden, fold.known),
        per_truth_s2=per_truth_credit(cur, fold.hidden, fold.known),
    )


# -----------------------------------------------------------------------------------------
# sequential full-grid prediction
# -----------------------------------------------------------------------------------------


def predict_grid(booster, ac, stack, raw_idx, footprint, strip: int = 48):
    """Predict over every footprint pixel, reading the feature memmap in row order.

    ``rows_of`` gathers arbitrary pixels, which on a memory-mapped cube becomes random disk
    access; at 5.2 M pixels that dominates the runtime.  Walking horizontal strips keeps the
    reads contiguous.  ``ac=None`` selects the stage-1 (raw-only) feature set.
    """
    H, W = footprint.shape
    out = np.zeros((H, W), np.float32)
    idx = np.asarray(raw_idx)
    for y0 in range(0, H, strip):
        y1 = min(H, y0 + strip)
        sub_fp = footprint[y0:y1]
        if not sub_fp.any():
            continue
        yy, xx = np.nonzero(sub_fp)
        yy = yy + y0
        raw = np.asarray(stack[idx, y0:y1, :], dtype=np.float32)[:, yy - y0, xx].T
        X = np.concatenate([ac.rows(yy, xx), raw], axis=1) if ac is not None else raw
        out[yy, xx] = booster.predict(X).astype(np.float32)
    return out
