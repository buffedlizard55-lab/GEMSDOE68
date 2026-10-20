# PR #13 (session 738) lane implementation, preserved at merge: main's 
# canonical autocontext/experiment modules diverged (parallel sessions), so 
# this session's Lane lives here unchanged apart from the import above.
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

from . import autocontext_738 as AC, features as F
from .holdout import buffered_component_draw


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Lane:
    def __init__(self, grid, X1: np.ndarray, phys: list[np.ndarray], quad_map: np.ndarray,
                 eval_seed: int = 20, train_seeds=(20, 21, 22), rng_seed: int = 68,
                 catalogue_context: bool = True, stage1_names: list[str] | None = None):
        """``catalogue_context=False`` runs the AC68-N1 far-robust ablation:
        stage>=2 receives no catalogue-derived features at all (columns are
        absent from the design matrix, not zeroed).  Stage 1 is unchanged --
        it never used catalogue-derived inputs.  ``stage1_names`` (AC68-N2)
        labels an externally-enriched stage-1 matrix whose width differs
        from the default 26 columns; it must match ``X1.shape[1]``."""
        self.names1 = list(stage1_names) if stage1_names is not None else list(F.STAGE1_NAMES)
        if X1.ndim != 2 or X1.shape[1] != len(self.names1):
            raise ValueError(f"X1 shape {getattr(X1, 'shape', None)} does not match "
                             f"{len(self.names1)} stage-1 feature names")
        self.g = grid
        self.fp = grid.footprint
        self.X1 = X1
        self.phys = phys
        self.quad_map = quad_map
        self.q = quad_map[self.fp]
        self.N = len(self.q)
        self.rng = np.random.default_rng(rng_seed)
        self.catalogue_context = bool(catalogue_context)
        self.names2 = AC.context_names(include_catalogue=self.catalogue_context)
        self.dist_to_block = {b: ndi.distance_transform_edt(quad_map != b)[self.fp] for b in range(4)}
        self.folds, _ = buffered_component_draw(grid, seed=eval_seed)
        self.eval_visible = self.folds[0]["visible"]
        self.y1 = self.eval_visible[self.fp].astype(np.int8)
        self.eval_ctx = ([c[self.fp] for c in F.catalogue_context(self.eval_visible)]
                         if self.catalogue_context else [])
        # training draws: positives = withheld components, context = that draw's visible
        self.draws = []
        for s in train_seeds:
            fs, _ = buffered_component_draw(grid, seed=s)
            hidden = fs[0]["hidden_all"]
            vis = fs[0]["visible"]
            self.draws.append(dict(seed=s, y=hidden[self.fp].astype(np.int8),
                                   ctx=([c[self.fp] for c in F.catalogue_context(vis)]
                                        if self.catalogue_context else [])))
            log(f"train draw {s}: withheld positives {int(hidden.sum())}")
        self.X2 = np.empty((self.N, len(self.names2)), np.float32)

    # ---- stage 1 ---------------------------------------------------------
    def stage1(self, outer: int | None):
        """OOF p1 on the training blocks + outer prediction. outer=None -> LOQO over all 4."""
        tq = [b for b in range(4) if b != outer]
        oof, _ = AC.blocked_oof(self.X1, self.y1, self.q, self.dist_to_block,
                                np.ones(self.N, bool), tq, self.rng, self.names1)
        imp = None
        if outer is not None:
            pred_idx = np.flatnonzero(self.q == outer)
            tr = np.flatnonzero(np.isin(self.q, tq) & (self.dist_to_block[outer] > AC.BUFFER_PX))
            p, b = AC.fit_predict(self.X1, self.y1, tr, pred_idx, self.rng, self.names1)
            oof[pred_idx] = p
            imp = dict(zip(b.feature_name(), b.feature_importance("gain").tolist()))
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
            booster = lgb.train(AC.LGB_PARAMS, lgb.Dataset(Xtr, label=ytr, feature_name=self.names2),
                                num_boost_round=AC.N_ROUNDS)
            del Xtr, ytr
            AC.context_matrix(pmap_prev, self.fp, self.phys, self.eval_ctx, out=self.X2)
            idx = np.flatnonzero(self.q == b)
            out[idx] = booster.predict(self.X2[idx], num_threads=2)
        imp = None
        if outer is not None:
            Xtr, ytr = self._train_rows(pmap_prev, tq, exclude_block=outer)
            booster = lgb.train(AC.LGB_PARAMS, lgb.Dataset(Xtr, label=ytr, feature_name=self.names2),
                                num_boost_round=AC.N_ROUNDS)
            del Xtr, ytr
            AC.context_matrix(pmap_prev, self.fp, self.phys, self.eval_ctx, out=self.X2)
            idx = np.flatnonzero(self.q == outer)
            out[idx] = booster.predict(self.X2[idx], num_threads=2)
            imp = dict(zip(booster.feature_name(), booster.feature_importance("gain").tolist()))
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
        full_ctx = ([c[self.fp] for c in F.catalogue_context(self.g.catalogue)]
                    if self.catalogue_context else [])
        for k in range(2, n_stages + 1):
            # OOF over all four blocks, context for prediction = full catalogue
            saved = self.eval_ctx
            self.eval_ctx = full_ctx
            pk, _ = self.stage_k(AC.to_map(p_prev, self.fp), None)
            self.eval_ctx = saved
            maps[k] = pk.copy(); p_prev = pk
            log(f"final stage{k} done")
        return maps
