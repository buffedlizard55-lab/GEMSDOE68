"""Nested blocked CV driver for the auto-context lane (shared by exp2 and the final build).

Label protocol for stage >= 2 ("hide-draw", fixes IR-68-LEAK-01 from experiment 1):
  * for each training draw seed s, whole catalogue components are withheld with
    the same buffered quadrant draw used for evaluation; the WITHHELD components
    in the training blocks are the positives, and every catalogue-context
    feature for that draw is computed from that draw's VISIBLE catalogue (which
    excludes the withheld components plus a 3 px collar).  A training positive
    therefore never sees itself in cat_dist / cat_density, which is the live
    situation: the truth is unmapped, the context is the published map.
  * stage 1 keeps PU labels = visible catalogue of the evaluation draw and uses
    no catalogue-derived input at all.
  * the outer block is always predicted by a model that never saw its labels,
    with a 12 px training exclusion band, and its context features come from
    the evaluation draw's visible catalogue.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from . import autocontext as AC, features as F
from .holdout import buffered_component_draw


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Lane:
    def __init__(self, grid, X1: np.ndarray, phys: list[np.ndarray], quad_map: np.ndarray,
                 eval_seed: int = 20, train_seeds=(20, 21, 22), rng_seed: int = 68,
                 stage1_names=None):
        self.stage1_names = list(stage1_names) if stage1_names is not None else list(F.STAGE1_NAMES)
        if X1.shape[1] != len(self.stage1_names):
            raise ValueError(f"X1 has {X1.shape[1]} columns but {len(self.stage1_names)} stage-1 names")
        self.g = grid
        self.fp = grid.footprint
        self.X1 = X1
        self.phys = phys
        self.quad_map = quad_map
        self.q = quad_map[self.fp]
        self.N = len(self.q)
        self.rng = np.random.default_rng(rng_seed)
        self.names2 = AC.context_names()
        self.dist_to_block = {b: ndi.distance_transform_edt(quad_map != b)[self.fp] for b in range(4)}
        self.folds, _ = buffered_component_draw(grid, seed=eval_seed)
        self.eval_visible = self.folds[0]["visible"]
        self.y1 = self.eval_visible[self.fp].astype(np.int8)
        self.eval_ctx = [c[self.fp] for c in F.catalogue_context(self.eval_visible)]
        # training draws: positives = withheld components, context = that draw's visible
        self.draws = []
        for s in train_seeds:
            fs, _ = buffered_component_draw(grid, seed=s)
            hidden = fs[0]["hidden_all"]
            vis = fs[0]["visible"]
            self.draws.append(dict(seed=s, y=hidden[self.fp].astype(np.int8),
                                   ctx=[c[self.fp] for c in F.catalogue_context(vis)]))
            log(f"train draw {s}: withheld positives {int(hidden.sum())}")
        self.X2 = np.empty((self.N, len(self.names2)), np.float32)

    # ---- stage 1 ---------------------------------------------------------
    def stage1(self, outer: int | None):
        """OOF p1 on the training blocks + outer prediction. outer=None -> LOQO over all 4."""
        tq = [b for b in range(4) if b != outer]
        oof, _ = AC.blocked_oof(self.X1, self.y1, self.q, self.dist_to_block,
                                np.ones(self.N, bool), tq, self.rng, self.stage1_names)
        imp = None
        if outer is not None:
            pred_idx = np.flatnonzero(self.q == outer)
            tr = np.flatnonzero(np.isin(self.q, tq) & (self.dist_to_block[outer] > AC.BUFFER_PX))
            p, bs = AC.fit_predict(self.X1, self.y1, tr, pred_idx, self.rng, self.stage1_names)
            oof[pred_idx] = p
            imp = AC.mean_gain(bs)
        return oof, imp

    # ---- stage k >= 2 ----------------------------------------------------
    def _train_rows(self, pmap, blocks: list[int], exclude_block: int | None):
        """Assemble training rows (features + labels) across the hide-draws."""
        Xs, ys = [], []
        for d in self.draws:
            AC.context_matrix(pmap, self.fp, self.phys, d["ctx"], out=self.X2)
            ok = np.isin(self.q, blocks)
            if exclude_block is not None:
                ok &= (self.q != exclude_block) & (self.dist_to_block[exclude_block] > AC.BUFFER_PX)
            pos = np.flatnonzero(ok & (d["y"] == 1))
            neg = np.flatnonzero(ok & (d["y"] == 0))
            n_neg = min(len(neg), max(AC.NEG_PER_POS * len(pos), 50_000), AC.MAX_TRAIN_ROWS - len(pos))
            neg = self.rng.choice(neg, size=n_neg, replace=False)
            rows = np.concatenate([pos, neg])
            Xs.append(self.X2[rows].copy()); ys.append(d["y"][rows])
        return np.concatenate(Xs), np.concatenate(ys)

    def stage_k(self, pmap_prev: np.ndarray, outer: int | None):
        import lightgbm as lgb
        tq = [b for b in range(4) if b != outer]
        out = np.full(self.N, np.nan, np.float32)
        # OOF on training blocks (context = evaluation draw's visible, i.e. the published map
        # minus the evaluation hide; used as the next stage's input map)
        for b in tq:
            Xtr, ytr = self._train_rows(pmap_prev, tq, exclude_block=b)
            boosters = AC.train_bagged(Xtr, ytr, self.names2)
            del Xtr, ytr
            AC.context_matrix(pmap_prev, self.fp, self.phys, self.eval_ctx, out=self.X2)
            idx = np.flatnonzero(self.q == b)
            out[idx] = AC.predict_bagged(boosters, self.X2[idx])
        imp = None
        if outer is not None:
            Xtr, ytr = self._train_rows(pmap_prev, tq, exclude_block=outer)
            boosters = AC.train_bagged(Xtr, ytr, self.names2)
            del Xtr, ytr
            AC.context_matrix(pmap_prev, self.fp, self.phys, self.eval_ctx, out=self.X2)
            idx = np.flatnonzero(self.q == outer)
            out[idx] = AC.predict_bagged(boosters, self.X2[idx])
            imp = AC.mean_gain(boosters)
        return out, imp

    def nested(self, n_stages: int = 3):
        """Outer LOQO: returns {stage: outer-prediction vector}, importances."""
        p_outer = {k: np.full(self.N, np.nan, np.float32) for k in range(1, n_stages + 1)}
        imps = {k: [] for k in range(1, n_stages + 1)}
        for qo in range(4):
            idx = np.flatnonzero(self.q == qo)
            p_prev, imp = self.stage1(qo)
            p_outer[1][idx] = p_prev[idx]; imps[1].append(imp)
            log(f"outer {qo} stage1 done")
            for k in range(2, n_stages + 1):
                pk, imp = self.stage_k(AC.to_map(p_prev, self.fp), qo)
                p_outer[k][idx] = pk[idx]; imps[k].append(imp)
                p_prev = pk
                log(f"outer {qo} stage{k} done")
        return p_outer, imps

    def final(self, n_stages: int) -> dict:
        """Full-data LOQO maps: every pixel predicted out-of-fold at every stage,
        with context from the FULL published catalogue at the final stage."""
        p_prev, _ = self.stage1(None)
        maps = {1: p_prev.copy()}
        full_ctx = [c[self.fp] for c in F.catalogue_context(self.g.catalogue)]
        for k in range(2, n_stages + 1):
            # OOF over all four blocks, context for prediction = full catalogue
            saved = self.eval_ctx
            self.eval_ctx = full_ctx
            pk, _ = self.stage_k(AC.to_map(p_prev, self.fp), None)
            self.eval_ctx = saved
            maps[k] = pk.copy(); p_prev = pk
            log(f"final stage{k} done")
        return maps
