# FROZEN PROVENANCE COPY (parallel-session merge 2026-10-20): identical algorithm to autocontext.
# py@80738d0 which produced exp3(seed31,negative)/exp4(seed42,promoted 25k). Canonical pipeline i
# s autocontext.py (PR #11); kept for reproducibility.
"""Auto-context stacked refinement (Tu, CVPR 2008; Tu & Bai, TPAMI 2010).

Stage 1: LightGBM on the raw + physics channels -> probability map p1, always
         produced OUT-OF-FOLD from spatially blocked (quadrant) folds with a
         ``BUFFER_PX`` training exclusion band around the target block.
Stage k: LightGBM on
           (a) p_{k-1} sampled at +-1..15 px offsets ALONG and ACROSS the local
               orientation of p_{k-1} (structure tensor),
           (b) multi-scale isotropic smooths of p_{k-1} (sigma 2, 5, 10 px) and
               anisotropic along/across means (anisotropy index),
           (c) the raw gradient / tilt / scarp channels,
           (d) catalogue-context features computed from the VISIBLE catalogue
               with the target block's labels masked (distance, strike, density).
         Iterated twice (stage 2 and stage 3).

Labels are PU: positives = visible catalogue pixels; "negatives" = everything
else in the footprint (unlabelled), subsampled.  The hide-and-recover truth is
never used for training.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb
from scipy import ndimage as ndi

from . import features as F

BUFFER_PX = 12            # training exclusion band around the target block (1.2 km)
OFFSETS = (1, 3, 7, 15)   # px, sampled along and across local orientation
NEG_PER_POS = 8           # negative subsample ratio for LightGBM
MAX_TRAIN_ROWS = 400_000

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, num_threads=2, verbose=-1, seed=68)
N_ROUNDS = 220
# Model-seed bagging (Exp 3): one booster per seed, probabilities averaged.
# Default (68,) reproduces the Exp 1/2 single-booster behaviour exactly.
MODEL_SEEDS = (68,)


def train_bagged(X, y, feature_names=None, seeds=None):
    """Train one LightGBM booster per model seed on the same rows."""
    boosters = []
    for s in (MODEL_SEEDS if seeds is None else seeds):
        params = {**LGB_PARAMS, "seed": int(s), "bagging_seed": int(s) + 1000,
                  "feature_fraction_seed": int(s) + 2000}
        ds = lgb.Dataset(X, label=y, feature_name=feature_names, free_raw_data=True)
        boosters.append(lgb.train(params, ds, num_boost_round=N_ROUNDS))
    return boosters


def predict_bagged(boosters, X) -> np.ndarray:
    """Mean probability over bagged boosters."""
    out = np.zeros(len(X), np.float64)
    for b in boosters:
        out += b.predict(X, num_threads=2)
    return (out / len(boosters)).astype(np.float32)


def mean_gain(boosters) -> dict:
    """Mean gain importance over bagged boosters (same feature order)."""
    names = boosters[0].feature_name()
    gains = np.mean([b.feature_importance("gain") for b in boosters], axis=0)
    return dict(zip(names, gains.tolist()))


def context_names() -> list[str]:
    names = []
    for d in OFFSETS:
        names += [f"p_along+{d}", f"p_along-{d}", f"p_across+{d}", f"p_across-{d}"]
    names += ["p_self", "p_sm2", "p_sm5", "p_sm10", "p_along_mean", "p_across_mean", "p_aniso",
              "ori_coh"]
    names += F.PHYSICS_NAMES
    names += F.CONTEXT_NAMES
    return names


def _sample(pmap: np.ndarray, yy: np.ndarray, xx: np.ndarray, dy: np.ndarray, dx: np.ndarray):
    h, w = pmap.shape
    y = np.clip(yy + dy, 0, h - 1)
    x = np.clip(xx + dx, 0, w - 1)
    return ndi.map_coordinates(pmap, [y, x], order=1, mode="nearest").astype(np.float32)


def context_matrix(pmap: np.ndarray, footprint: np.ndarray, phys: list[np.ndarray],
                   ctx: list[np.ndarray], out: np.ndarray | None = None) -> np.ndarray:
    """Build the stage-k design matrix (rows = footprint pixels, row-major)."""
    pmap = np.nan_to_num(pmap.astype(np.float32), nan=0.0)
    yy, xx = np.nonzero(footprint)
    yy = yy.astype(np.float32); xx = xx.astype(np.float32)
    _, _, coh, theta = F.orientation_field(pmap, 1.5, 4.0)
    th = theta[footprint]
    ay, ax = np.sin(th), np.cos(th)          # along-line unit vector (row, col)
    cy, cx = -ax, ay                         # across-line unit vector
    cols = []
    along_acc = np.zeros(len(yy), np.float32); across_acc = np.zeros(len(yy), np.float32)
    for d in OFFSETS:
        pa_p = _sample(pmap, yy, xx, d * ay, d * ax)
        pa_m = _sample(pmap, yy, xx, -d * ay, -d * ax)
        pc_p = _sample(pmap, yy, xx, d * cy, d * cx)
        pc_m = _sample(pmap, yy, xx, -d * cy, -d * cx)
        cols += [pa_p, pa_m, pc_p, pc_m]
        along_acc += pa_p + pa_m; across_acc += pc_p + pc_m
    along_mean = along_acc / (2 * len(OFFSETS)); across_mean = across_acc / (2 * len(OFFSETS))
    cols.append(pmap[footprint])
    for s in (2, 5, 10):
        cols.append(ndi.gaussian_filter(pmap, s)[footprint])
    cols += [along_mean, across_mean, (along_mean - across_mean) / (along_mean + across_mean + 1e-3),
             coh[footprint]]
    cols += [c if c.ndim == 1 else c[footprint] for c in phys]
    cols += [c if c.ndim == 1 else c[footprint] for c in ctx]
    if out is None:
        out = np.empty((len(yy), len(cols)), np.float32)
    for j, c in enumerate(cols):
        out[:, j] = c
    return out


def fit_predict(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, pred_idx: np.ndarray,
                rng: np.random.Generator, feature_names=None):
    """Train on a PU-subsampled training set and predict ``pred_idx`` rows.

    Returns (probabilities, boosters): a list with one booster per entry in
    ``MODEL_SEEDS``; probabilities are the bagged mean.  Each bag member gets
    its own negative subsample draw (same positives, same budget).
    """
    pos = train_idx[y[train_idx] == 1]
    neg_all = train_idx[y[train_idx] == 0]
    n_neg = min(len(neg_all), max(NEG_PER_POS * len(pos), 50_000), MAX_TRAIN_ROWS - len(pos))
    boosters = []
    out = np.zeros(len(pred_idx), np.float64)
    for s in MODEL_SEEDS:
        neg = rng.choice(neg_all, size=n_neg, replace=False)
        rows = np.concatenate([pos, neg])
        params = {**LGB_PARAMS, "seed": int(s), "bagging_seed": int(s) + 1000,
                  "feature_fraction_seed": int(s) + 2000}
        ds = lgb.Dataset(X[rows], label=y[rows], feature_name=feature_names, free_raw_data=True)
        booster = lgb.train(params, ds, num_boost_round=N_ROUNDS)
        boosters.append(booster)
        for a in range(0, len(pred_idx), 1_000_000):
            sl = pred_idx[a:a + 1_000_000]
            out[a:a + len(sl)] += booster.predict(X[sl], num_threads=2)
    return (out / len(boosters)).astype(np.float32), boosters


def blocked_oof(X: np.ndarray, y: np.ndarray, block_of_row: np.ndarray, dist_to_block: dict,
                rows_allowed: np.ndarray, blocks: list[int], rng, feature_names=None):
    """Out-of-fold predictions over ``blocks``: for each block b, train on allowed rows
    in other blocks that lie > BUFFER_PX from b, predict rows in b.  Returns a
    full-length vector (NaN where not predicted) and per-block gain tables."""
    oof = np.full(len(y), np.nan, np.float32)
    gains = {}
    for b in blocks:
        pred_idx = np.flatnonzero((block_of_row == b) & rows_allowed)
        tr = np.flatnonzero(rows_allowed & (block_of_row != b) & np.isin(block_of_row, blocks)
                            & (dist_to_block[b] > BUFFER_PX))
        p, boosters = fit_predict(X, y, tr, pred_idx, rng, feature_names)
        oof[pred_idx] = p
        gains[b] = mean_gain(boosters)
    return oof, gains


def to_map(vec: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    m = np.zeros(footprint.shape, np.float32)
    m[footprint] = np.nan_to_num(vec, nan=0.0)
    return m
