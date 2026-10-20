"""Leak-free nested blocked CV for the auto-context lane (Experiment 3).

Why a hide-draw is needed at Stage >= 2 (see IR-68-LEAK-01):
if Stage 2 is trained with ``y = visible`` while ``cat_dist`` is distance to the
*same* ``visible`` mask, then ``cat_dist == 0`` iff ``y == 1`` and the tree
collapses to the trivial rule ``cat_dist == 0``.  Even with ``cat_dist`` dropped,
``cat_dens_1km`` peaks on ``visible`` and the model places every high score inside
the 3 px exclusion collar of the visible catalogue, which is masked out at
evaluation.

Fix (parallel-run protocol rule 2 -- hide-and-recover inside each training fold):
for each training block ``b`` we draw a 25% whole-component sub-withholding
``H_b`` with a 3 px collar, compute ``catalogue_context`` from
``sub_visible = visible & ~collar(H_b)``, and train Stage 2 / Stage 3 on:

* positives = ``H_b`` (genuine fault segments at distance > 3 px from ``sub_visible``),
  with 2.5x up-weighting on far-from-catalogue positives (``cat_dist > 10 px``) so
  Stage 2/3 explicitly learns off-catalogue auto-context strand continuation;
* negatives = footprint pixels inside block ``b`` outside ``visible | collar(H_b) | collar(sub_visible)``
  (so negatives also lie at distance > 3 px from ``sub_visible``, preventing trees from
  learning the artificial 3 px holdout collar as a negative split and preserving high
  Stage-2 probability along visible fault traces for Stage 3).
"""
from __future__ import annotations

import gc
import time
import lightgbm as lgb
import numpy as np
from scipy import ndimage as ndi

from . import autocontext as AC
from . import features as F
from .grid import Grid
from .holdout import FOLD_NAMES, STRUCT3, _pick_segments, quadrant_ids

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def make_hide_draws(
    visible: np.ndarray,
    footprint: np.ndarray,
    quad_full: np.ndarray,
    n_draws: int = 2,
    hide_frac: float = 0.25,
    buffer_px: int = 3,
    seed: int = 900,
) -> list[dict]:
    """Build ``n_draws`` whole-component hide draws inside ``visible`` per quadrant."""
    comp, n_comp = ndi.label(visible, STRUCT3)
    sizes = np.bincount(comp.ravel(), minlength=n_comp + 1)
    draws = []
    for d in range(n_draws):
        hidden = np.zeros(visible.shape, bool)
        for q in range(4):
            reg = (quad_full == q) & footprint
            cand = np.unique(comp[visible & reg])
            cand = cand[cand > 0]
            rng = np.random.default_rng(seed * 1000 + d * 10 + q)
            chosen = _pick_segments(rng, sizes, cand, hide_frac * float((visible & reg).sum()))
            hidden |= np.isin(comp, chosen) & visible
        collar = ndi.distance_transform_edt(~hidden) <= buffer_px
        sub_vis = visible & ~collar
        sub_vis_collar = ndi.distance_transform_edt(~sub_vis) <= buffer_px
        draws.append(
            dict(
                hidden=hidden,
                collar=collar,
                sub_visible=sub_vis,
                sub_vis_collar=sub_vis_collar,
            )
        )
    return draws


def _calibrate_near_far_strata(pred_b: np.ndarray, d_vis_b: np.ndarray, target_far_ratio: float = 0.88) -> np.ndarray:
    """Align the high-quantile tail of far (>1 km) vs near (0.3-1.0 km) pixels using ONLY visible d_vis."""
    out = pred_b.astype(np.float32, copy=True)
    near = (d_vis_b > 3.0) & (d_vis_b <= 10.0)
    far = d_vis_b > 10.0
    if near.sum() > 100 and far.sum() > 100:
        q_near = float(np.quantile(out[near], 0.992))
        q_far = float(np.quantile(out[far], 0.992))
        if q_far > 1e-6 and q_near > 1e-6:
            scale = (target_far_ratio * q_near) / q_far
            if scale > 1.0:
                out[far] = np.clip(out[far] * scale, 0.0, 0.999)
    return out


class Lane:
    """Reusable state for nested CV and final full-data fit (memory-lean row slicing)."""

    def __init__(self, grid: Grid, phys: F.PhysicalFeatures, visible_all: np.ndarray, collar_all: np.ndarray):
        self.g = grid
        self.phys = phys
        self.fp = grid.footprint
        self.fp_rows, self.fp_cols = np.nonzero(self.fp)
        self.fp_rows = self.fp_rows.astype(np.int32)
        self.fp_cols = self.fp_cols.astype(np.int32)
        self.N = grid.n_footprint
        self.q_full = quadrant_ids(self.fp)
        self.q = self.q_full[self.fp]
        self.vis_all = visible_all
        self.y1 = visible_all[self.fp].astype(np.int8)
        self.valid1 = ~collar_all[self.fp]
        self.names2 = AC.stage2_feature_names()

        log("precomputing catalogue_context for outer visible + 2 hide draws...")
        self.eval_ctx = [c[self.fp] for c in F.catalogue_context(visible_all, phys.ay, phys.ax)]
        self.draws = []
        for d in make_hide_draws(visible_all, self.fp, self.q_full, n_draws=2, hide_frac=0.25, seed=900):
            self.draws.append(
                dict(
                    pos=d["hidden"][self.fp],
                    neg=(~visible_all & ~d["collar"] & ~d["sub_vis_collar"])[self.fp],
                    ctx=[c[self.fp] for c in F.catalogue_context(d["sub_visible"], phys.ay, phys.ax)],
                )
            )
        gc.collect()

    def _fit1(self, mask: np.ndarray) -> lgb.Booster:
        pos = np.flatnonzero(mask & (self.y1 == 1))
        neg_pool = np.flatnonzero(mask & (self.y1 == 0))
        rng = np.random.default_rng(int(pos.size) + 17)
        neg = rng.choice(neg_pool, size=min(neg_pool.size, pos.size * AC.NEG_PER_POS), replace=False)
        idx = np.concatenate([pos, neg])
        return lgb.train(
            AC.LGB_PARAMS,
            lgb.Dataset(self.phys.X1[idx], label=self.y1[idx], feature_name=self.phys.names1),
            num_boost_round=AC.N_ROUNDS,
        )

    def stage1(self, outer: int | None):
        """Return OOF Stage-1 probability vector on footprint (and outer predictions)."""
        tq = [q for q in range(4) if q != outer]
        oof = np.full(self.N, np.nan, np.float32)
        for b in tq:
            tr = self.valid1 & np.isin(self.q, [q for q in tq if q != b])
            booster = self._fit1(tr)
            idx = np.flatnonzero(self.q == b)
            oof[idx] = booster.predict(self.phys.X1[idx], num_threads=2)
        imp = None
        if outer is not None:
            booster = self._fit1(self.valid1 & (self.q != outer))
            idx = np.flatnonzero(self.q == outer)
            oof[idx] = booster.predict(self.phys.X1[idx], num_threads=2)
            imp = dict(zip(booster.feature_name(), booster.feature_importance("gain").tolist()))
        return oof, imp

    def _train_rows(self, pmap_prev: np.ndarray, iso_cache: dict[float, np.ndarray], blocks: list[int], exclude_block: int):
        use_blocks = [b for b in blocks if b != exclude_block]
        in_b = np.isin(self.q, use_blocks)
        xs, ys, ws = [], [], []
        for di, d in enumerate(self.draws):
            pos = np.flatnonzero(in_b & d["pos"])
            neg_pool = np.flatnonzero(in_b & d["neg"])
            rng = np.random.default_rng(700 + di * 31 + exclude_block)
            neg = rng.choice(neg_pool, size=min(neg_pool.size, pos.size * AC.NEG_PER_POS), replace=False)
            idx = np.concatenate([pos, neg])
            X_sub = AC.context_matrix_rows(
                pmap_prev, iso_cache, idx, self.fp_rows, self.fp_cols, self.phys, d["ctx"]
            )
            y_arr = np.concatenate([np.ones(pos.size, np.float32), np.zeros(neg.size, np.float32)])
            d_pos = d["ctx"][0][pos]
            w_pos = np.where(d_pos > 10.0, 2.5, 1.0).astype(np.float32)
            w_arr = np.concatenate([w_pos, np.ones(neg.size, np.float32)])
            xs.append(X_sub)
            ys.append(y_arr)
            ws.append(w_arr)
        return np.vstack(xs), np.concatenate(ys), np.concatenate(ws)

    def _predict_block(self, booster: lgb.Booster, pmap_prev: np.ndarray, iso_cache: dict[float, np.ndarray], block: int) -> np.ndarray:
        idx = np.flatnonzero(self.q == block)
        X_blk = AC.context_matrix_rows(
            pmap_prev, iso_cache, idx, self.fp_rows, self.fp_cols, self.phys, self.eval_ctx
        )
        raw_p = booster.predict(X_blk, num_threads=2).astype(np.float32)
        del X_blk
        return _calibrate_near_far_strata(raw_p, self.eval_ctx[0][idx])

    def stage_k(self, pmap_prev: np.ndarray, outer: int | None, prev_vec: np.ndarray | None = None, blend_prev: float = 0.0):
        """Train Stage k on hide-draw rows; predict with visible-catalogue context."""
        tq = [q for q in range(4) if q != outer]
        out = np.full(self.N, np.nan, np.float32)
        iso_cache = AC.precompute_iso_smooths(pmap_prev)
        for b in tq:
            Xtr, ytr, wtr = self._train_rows(pmap_prev, iso_cache, tq, exclude_block=b)
            booster = lgb.train(
                AC.LGB_PARAMS,
                lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=self.names2),
                num_boost_round=AC.N_ROUNDS,
            )
            del Xtr, ytr, wtr
            idx = np.flatnonzero(self.q == b)
            pred_b = self._predict_block(booster, pmap_prev, iso_cache, b)
            if prev_vec is not None and blend_prev > 0.0:
                pred_b = blend_prev * prev_vec[idx] + (1.0 - blend_prev) * pred_b
            out[idx] = pred_b
        imp = None
        if outer is not None:
            Xtr, ytr, wtr = self._train_rows(pmap_prev, iso_cache, tq, exclude_block=outer)
            booster = lgb.train(
                AC.LGB_PARAMS,
                lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=self.names2),
                num_boost_round=AC.N_ROUNDS,
            )
            del Xtr, ytr, wtr
            idx = np.flatnonzero(self.q == outer)
            pred_b = self._predict_block(booster, pmap_prev, iso_cache, outer)
            if prev_vec is not None and blend_prev > 0.0:
                pred_b = blend_prev * prev_vec[idx] + (1.0 - blend_prev) * pred_b
            out[idx] = pred_b
            imp = dict(zip(booster.feature_name(), booster.feature_importance("gain").tolist()))
        del iso_cache
        gc.collect()
        return out, imp

    def nested(self, n_stages: int = 3):
        """Outer LOQO: returns {stage: outer-prediction vector}, importances."""
        p_outer = {k: np.full(self.N, np.nan, np.float32) for k in range(1, n_stages + 1)}
        imps = {k: [] for k in range(1, n_stages + 1)}
        for qo in range(4):
            idx = np.flatnonzero(self.q == qo)
            p_prev, imp = self.stage1(qo)
            p_outer[1][idx] = p_prev[idx]
            imps[1].append(imp)
            log(f"outer {qo} stage1 done")
            for k in range(2, n_stages + 1):
                blend = 0.50 if k >= 3 else 0.0
                pk, imp = self.stage_k(AC.to_map(p_prev, self.fp), qo, prev_vec=p_prev, blend_prev=blend)
                p_outer[k][idx] = pk[idx]
                imps[k].append(imp)
                p_prev = pk
                log(f"outer {qo} stage{k} done")
        return p_outer, imps

    def final(self, n_stages: int = 3) -> dict:
        """Full-data LOQO maps: every pixel predicted out-of-fold at every stage,
        with context from the FULL published catalogue at the final stage."""
        p_prev, _ = self.stage1(None)
        maps = {1: p_prev.copy()}
        full_ctx = [c[self.fp] for c in F.catalogue_context(self.g.catalogue, self.phys.ay, self.phys.ax)]
        for k in range(2, n_stages + 1):
            saved = self.eval_ctx
            self.eval_ctx = full_ctx
            blend = 0.50 if k >= 3 else 0.0
            pk, _ = self.stage_k(AC.to_map(p_prev, self.fp), None, prev_vec=p_prev, blend_prev=blend)
            self.eval_ctx = saved
            maps[k] = pk.copy()
            p_prev = pk
            log(f"final stage{k} done")
        return maps
