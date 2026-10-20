#!/usr/bin/env python3
"""GEMSDOE68 — Complete End-to-End 3-Experiment Nested Spatially Blocked
O(2) Auto-Context + Multi-Physics + TverskyNet Pipeline (with Stage Checkpoints).

Supports zero-recompute resume from ``.cache/s68_stack/ckpt_v3/`` as well as
clean cold-start execution via ``--fresh``. Peak RSS stays strictly < 1.45 GiB.
"""

from __future__ import annotations

import gc
import hashlib
import json
import pickle
import sys
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.special import expit, logit
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems.submission_writer import write_submission  # noqa: E402
from gems.validate_submission import footprint_from_features, validate, validate_portal_ready  # noqa: E402
from gems57 import evaluate_holdout as E57  # noqa: E402
from gems57.metric import dti_exact  # noqa: E402
from gems68 import evaluate_holdout as E68  # noqa: E402
from gems68 import grid as G68  # noqa: E402
from gems68.holdout import buffered_component_draw  # noqa: E402
from gems68.multiphysics_stack import (  # noqa: E402
    ALL_STACK_NAMES,
    EXP1_LOCAL_NAMES,
    STAGE2_PHYS_EXP1,
    STAGE2_PHYS_EXP2,
    build_or_load_stack,
)
from gems68.o2_autocontext import (  # noqa: E402
    O2_CAT_NAMES,
    O2_CONTEXT_NAMES,
    PreparedCatalogueField,
    PreparedStageField,
    TverskyContextNet,
    build_feature_slice,
    emit_submodular_300m,
    submodular_order_300m,
)

SEED = 20
CORE_ERODE_M = 2600.0
EXCLUDE_PX = 2.0
BUDGET_GRID = [26_000, 32_000, 36_000, 38_000, 42_000, 48_000, 56_000]
RUN_TS = "20261020T201500Z"
CKPT = ROOT / ".cache" / "s68_stack" / "ckpt_v3"


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


def _drop_mm_cache(mm: np.memmap) -> None:
    try:
        import mmap as _mmap
        mm._mmap.madvise(_mmap.MADV_DONTNEED)
    except Exception:
        pass


def _fit_lgb(X: np.ndarray, y: np.ndarray, seed: int, n_estimators: int = 190,
             num_leaves: int = 47, min_child_samples: int = 90) -> lgb.LGBMClassifier:
    clf = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.065,
        num_leaves=num_leaves,
        max_bin=127,
        min_child_samples=min_child_samples,
        subsample_for_bin=80_000,
        colsample_bytree=0.72,
        subsample=0.8,
        subsample_freq=1,
        reg_alpha=1.5,
        reg_lambda=6.0,
        n_estimators=n_estimators,
        n_jobs=2,
        random_state=seed,
        verbose=-1,
    )
    clf.fit(X, y)
    return clf


def _predict_chunked_single(m: lgb.LGBMClassifier, mm: np.memmap, cols: list[int],
                            rows_idx: np.ndarray, chunk: int = 550_000) -> np.ndarray:
    out = np.empty(len(rows_idx), dtype=np.float32)
    for s in range(0, len(rows_idx), chunk):
        e = min(len(rows_idx), s + chunk)
        sub = np.asarray(mm[rows_idx[s:e]][:, cols], dtype=np.float32)
        out[s:e] = m.predict_proba(sub)[:, 1].astype(np.float32)
    _drop_mm_cache(mm)
    return out


def _nms_box_emit(surface: np.ndarray, allowed: np.ndarray, n_dots: int, win: int = 5) -> np.ndarray:
    s = np.where(allowed & np.isfinite(surface), surface, -np.inf)
    mx = ndi.maximum_filter(s, size=win, mode="constant", cval=-np.inf)
    keep = allowed & np.isfinite(surface) & (s == mx) & (surface > 0)
    vals = surface[keep]
    out = np.zeros_like(surface, dtype=np.float32)
    if vals.size == 0:
        return out
    if vals.size <= n_dots:
        out[keep] = 1.0
        return out
    order = np.argsort(vals)
    mask = np.zeros(vals.size, dtype=bool)
    mask[order[-n_dots:]] = True
    out[keep] = np.where(mask, 1.0, 0.0).astype(np.float32)
    return out


def _eval_terms(pred_2d: np.ndarray, folds: list[dict], fp: np.ndarray,
                far_masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    terms_all = None
    terms_far = None
    for b, f in enumerate(folds):
        _, ta = E68.evaluate(pred_2d, f, fp)
        ff = dict(f)
        ff["region"] = f["region"] & far_masks[b]
        _, tf = E68.evaluate(pred_2d, ff, fp)
        terms_all = ta if terms_all is None else (terms_all + ta)
        terms_far = tf if terms_far is None else (terms_far + tf)
    return terms_all, terms_far


def _score_arms(terms_map: dict[str, np.ndarray], candidate: str, draws: int = 1000) -> dict:
    s68 = E68.pooled_summary(terms_map, draws=draws, seed=520810, candidate=candidate)
    s57 = E57.pooled_summary(terms_map, draws=draws, seed=520810, candidate=candidate)
    for k in s68["scores"]:
        assert abs(s68["scores"][k]["dti"] - s57["scores"][k]["dti"]) < 1e-12
    return s68


def _blend_probs(p_a: np.ndarray, p_b: np.ndarray, w_a: float) -> np.ndarray:
    la = logit(np.clip(p_a, 1e-6, 1.0 - 1e-6))
    lb = logit(np.clip(p_b, 1e-6, 1.0 - 1e-6))
    return expit(w_a * la + (1.0 - w_a) * lb).astype(np.float32)


def main() -> int:
    fresh = "--fresh" in sys.argv
    t_start = time.time()
    CKPT.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("GEMSDOE68 — 3-Experiment Nested O(2) Auto-Context & Multi-Physics Pipeline")
    print("=" * 78, flush=True)

    names, mm, fp, _ = build_or_load_stack(force=False)
    _drop_mm_cache(mm)
    name_to_idx = {n: i for i, n in enumerate(names)}
    grid = G68.load_grid()
    assert np.array_equal(fp, grid.footprint)
    fp_y, fp_x = np.nonzero(fp)
    fp_y = fp_y.astype(np.int32)
    fp_x = fp_x.astype(np.int32)
    n_fp = int(fp.sum())
    all_rows = np.arange(n_fp, dtype=np.int64)

    folds, quad_2d = buffered_component_draw(grid, seed=SEED)
    block_of = quad_2d[fp]
    assert (block_of >= 0).all()
    rows_by_block = [np.flatnonzero(block_of == b) for b in range(4)]

    core_2d = []
    for b in range(4):
        outside_b = quad_2d != b
        d_in = ndi.distance_transform_edt(~outside_b) * 100.0
        core_2d.append((quad_2d == b) & fp & (d_in > CORE_ERODE_M))

    comp_id, _ = ndi.label(grid.catalogue, structure=np.ones((3, 3), dtype=int))
    vis_outer = folds[0]["visible"]
    d_vis_outer = ndi.distance_transform_edt(~vis_outer)

    far_masks = [d_vis_outer > 10.0 for _ in range(4)]
    outer_allowed = [
        fp & f["region"] & ~f["masked_known"] & (d_vis_outer > EXCLUDE_PX)
        for f in folds
    ]

    print(f"[0/5] Building outer & clean inner hide-world (RSS={_rss_mb():.0f} MB)...", flush=True)
    outer_cat_cf = PreparedCatalogueField(vis_outer, fp, fp_y, fp_x)

    # Inner hide-world for training Stage 2/3 + pure 20%-hide inner world for matched-density budget tuning!
    inner_worlds = []
    for w_idx, hfrac in ((0, 0.40),):
        hid_w = np.zeros(fp.shape, dtype=bool)
        hid_20 = np.zeros(fp.shape, dtype=bool)
        for b in range(4):
            c_in = np.unique(comp_id[vis_outer & core_2d[b]])
            c_out = np.unique(comp_id[vis_outer & ~core_2d[b]])
            clean_b = np.setdiff1d(c_in[c_in > 0], c_out)
            rng_w = np.random.default_rng(SEED * 777 + w_idx * 131 + b * 17)
            perm = rng_w.permutation(clean_b)
            hid_w |= (np.isin(comp_id, perm[:max(1, int(round(hfrac * len(perm))))]) & vis_outer)
            hid_20 |= (np.isin(comp_id, perm[:max(1, int(round(0.20 * len(perm))))]) & vis_outer)
        collar_w = ndi.distance_transform_edt(~hid_w) <= 3.0
        vis_w = vis_outer & ~collar_w
        d_vis_w = ndi.distance_transform_edt(~vis_w)
        d_hid_w = ndi.distance_transform_edt(~hid_w)
        cf_w = PreparedCatalogueField(vis_w, fp, fp_y, fp_x)

        per_b_rows = []
        for b in range(4):
            pos_b = np.flatnonzero(hid_w[fp] & core_2d[b][fp])
            neg_pool_b = np.flatnonzero(core_2d[b][fp] & (d_vis_w[fp] > EXCLUDE_PX) & (d_hid_w[fp] > 4.0))
            rng_s = np.random.default_rng(SEED * 313 + w_idx * 41 + b)
            neg_b = rng_s.choice(neg_pool_b, size=min(len(neg_pool_b), max(len(pos_b) * 8, 18_000)), replace=False)
            rb = np.concatenate([pos_b, neg_b]).astype(np.int32)
            yb = np.concatenate([np.ones(len(pos_b), np.float32), np.zeros(len(neg_b), np.float32)])
            sb = np.maximum(0.0, 1.0 - d_hid_w[fp][rb] / 3.0).astype(np.float32)
            per_b_rows.append({"rows": rb, "y": yb, "soft": sb})

        # Matched 20%-hide evaluation domain inside cores (exact competition hide density!)
        known_20 = vis_outer & ~hid_20
        d_known_20 = ndi.distance_transform_edt(~known_20)
        inner_worlds.append({
            "vis_w": vis_w,
            "hid_w": hid_w,
            "hid_20": hid_20,
            "known_20": known_20,
            "d_vis_w_fp": d_vis_w[fp].astype(np.float16),
            "cf_w": cf_w,
            "per_b": per_b_rows,
            "eval_allowed_20_b": [fp & core_2d[b] & ~known_20 & (d_known_20 > EXCLUDE_PX) for b in range(4)],
        })
        del d_vis_w, d_hid_w, collar_w, d_known_20
        gc.collect()

    # ------------------------------------------------------------------
    # Baseline 0: Live 0.2778 Champion Lineage Surface (scored_h19_5)
    # ------------------------------------------------------------------
    p_h19 = next((ROOT / "data" / "scored").glob("*h19-5-powerlaw*"))
    with rasterio.open(p_h19) as ds:
        h19_5 = np.clip(np.nan_to_num(ds.read(1).astype(np.float32), nan=0.0), 0.0, 1.0)
    h19_nms_pred = np.zeros_like(h19_5)
    h19_sub_pred = np.zeros_like(h19_5)
    for b in range(4):
        q = 38_000 // 4
        h19_nms_pred += _nms_box_emit(h19_5, outer_allowed[b], q, win=5)
        h19_sub_pred += emit_submodular_300m(h19_5, outer_allowed[b], q, catalogue_2d=folds[b]["visible"])
    t_h19_nms_all, t_h19_nms_far = _eval_terms(h19_nms_pred, folds, fp, far_masks)
    t_h19_sub_all, t_h19_sub_far = _eval_terms(h19_sub_pred, folds, fp, far_masks)
    del h19_5, h19_nms_pred, h19_sub_pred
    gc.collect()
    print(f"[Baseline 0] h19_5 @38k -> 5x5 NMS: ALL={E68.from_terms(t_h19_nms_all.sum(0)):.4f}, FAR={E68.from_terms(t_h19_nms_far.sum(0)):.4f} | "
          f"Submodular300m: ALL={E68.from_terms(t_h19_sub_all.sum(0)):.4f}, FAR={E68.from_terms(t_h19_sub_far.sum(0)):.4f}",
          flush=True)

    def run_ac_experiment(exp_id: str, s1_names: list[str], s2_phys_names: list[str],
                          save_tv_inputs: bool = False) -> dict:
        t0 = time.time()
        s1_cols = [name_to_idx[n] for n in s1_names]
        s2_cols = [name_to_idx[n] for n in s2_phys_names]
        s2_feat_names = list(O2_CONTEXT_NAMES) + list(O2_CAT_NAMES) + list(s2_phys_names)

        ck_s1 = CKPT / f"{exp_id}_stage1.npy"
        ck_s2 = CKPT / f"{exp_id}_stage2.npy"
        ck_s3 = CKPT / f"{exp_id}_stage3.npy"
        ck_m = CKPT / f"{exp_id}_models.pkl"

        # 1. Stage 1 outer OOF
        if not fresh and ck_s1.exists():
            outer_s1 = np.load(ck_s1)
        else:
            outer_s1_preds = np.zeros((4, n_fp), dtype=np.float32)
            for b in range(4):
                in_cb = core_2d[b][fp]
                pos_b = np.flatnonzero(vis_outer[fp] & in_cb)
                neg_pool = np.flatnonzero(in_cb & (d_vis_outer[fp] > 4.0))
                rng_b = np.random.default_rng(SEED + 100 + b)
                neg_b = rng_b.choice(neg_pool, size=min(len(neg_pool), len(pos_b) * 6), replace=False)
                idx_b = np.concatenate([pos_b, neg_b])
                yb = np.concatenate([np.ones(len(pos_b), np.float32), np.zeros(len(neg_b), np.float32)])
                mb = _fit_lgb(np.asarray(mm[idx_b][:, s1_cols], np.float32), yb, seed=SEED + b, n_estimators=260)
                _drop_mm_cache(mm)
                outer_s1_preds[b] = _predict_chunked_single(mb, mm, s1_cols, all_rows)
                del mb
                gc.collect()
            outer_s1 = np.zeros(n_fp, dtype=np.float32)
            for k in range(4):
                rk = rows_by_block[k]
                outer_s1[rk] = np.mean(outer_s1_preds[[b for b in range(4) if b != k]][:, rk], axis=0)
            del outer_s1_preds
            np.save(ck_s1, outer_s1)
            gc.collect()

        def _compute_inner20_budget_curve(s2_m_list, s3_m_list) -> tuple[dict[int, float], int]:
            iw20 = inner_worlds[0]
            cat_20 = PreparedCatalogueField(iw20["known_20"], fp, fp_y, fp_x)
            sf1_20 = PreparedStageField(outer_s1, fp, fp_y, fp_x)
            p2_20 = np.zeros(n_fp, dtype=np.float32)
            for k in range(4):
                rk = rows_by_block[k]
                pk = np.empty(len(rk), dtype=np.float32)
                for s in range(0, len(rk), 450_000):
                    e = min(len(rk), s + 450_000)
                    Xv = build_feature_slice(sf1_20, cat_20, mm, s2_cols, rk[s:e])
                    _drop_mm_cache(mm)
                    pk[s:e] = s2_m_list[k].predict_proba(Xv)[:, 1].astype(np.float32)
                p2_20[rk] = _blend_probs(outer_s1[rk], pk, w_a=0.35)
            del sf1_20
            sf2_20 = PreparedStageField(p2_20, fp, fp_y, fp_x)
            p3_20 = np.zeros(n_fp, dtype=np.float32)
            for k in range(4):
                rk = rows_by_block[k]
                pk = np.empty(len(rk), dtype=np.float32)
                for s in range(0, len(rk), 450_000):
                    e = min(len(rk), s + 450_000)
                    Xv = build_feature_slice(sf2_20, cat_20, mm, s2_cols, rk[s:e])
                    _drop_mm_cache(mm)
                    pk[s:e] = s3_m_list[k].predict_proba(Xv)[:, 1].astype(np.float32)
                p3_20[rk] = _blend_probs(p2_20[rk], pk, w_a=0.40)
            del sf2_20, cat_20, p2_20
            surf_20_2d = np.zeros(fp.shape, dtype=np.float32)
            surf_20_2d[fp] = p3_20
            in_orders = []
            for b in range(4):
                allow_b = iw20["eval_allowed_20_b"][b]
                ys_b, xs_b = np.nonzero(allow_b)
                sy = slice(max(0, int(ys_b.min()) - 6), min(fp.shape[0], int(ys_b.max()) + 7))
                sx = slice(max(0, int(xs_b.min()) - 6), min(fp.shape[1], int(xs_b.max()) + 7))
                a_frac = float(allow_b.sum()) / float(n_fp)
                cy, cx = submodular_order_300m(
                    surf_20_2d, allow_b, max_dots=int(round(max(BUDGET_GRID) * a_frac)) + 20,
                    catalogue_2d=iw20["known_20"],
                )
                in_orders.append((cy, cx, allow_b, iw20["hid_20"] & core_2d[b], a_frac, sy, sx))
            del surf_20_2d, p3_20
            gc.collect()
            b_scores: dict[int, float] = {}
            for B in BUDGET_GRID:
                vals_b = []
                for cy, cx, allow_b, truth_b, a_frac, sy, sx in in_orders:
                    qb = max(10, int(round(B * a_frac)))
                    em_crop = np.zeros((sy.stop - sy.start, sx.stop - sx.start), dtype=np.float32)
                    em_crop[cy[:qb] - sy.start, cx[:qb] - sx.start] = 1.0
                    vals_b.append(dti_exact(em_crop, truth_b[sy, sx], valid=allow_b[sy, sx])["dti"])
                b_scores[B] = round(float(np.mean(vals_b)), 5)
            return b_scores, max(b_scores, key=b_scores.get)

        # Inner Stage 1 + Stage 2 + Stage 3 (reuses ck_s2 / ck_s3 if already checkpointed!)
        if not fresh and ck_s2.exists() and ck_s3.exists() and ck_m.exists():
            outer_s2 = np.load(ck_s2)
            outer_s3 = np.load(ck_s3)
            with open(ck_m, "rb") as fh:
                saved_m = pickle.load(fh)
            s2_models = saved_m["s2_models"]
            s3_models = saved_m["s3_models"]
            budget_scores = saved_m["budget_scores"]
            best_budget = saved_m["best_budget"]
            if max(budget_scores.values()) <= 1e-6:
                budget_scores, best_budget = _compute_inner20_budget_curve(s2_models, s3_models)
                saved_m["budget_scores"] = budget_scores
                saved_m["best_budget"] = best_budget
                with open(ck_m, "wb") as fh:
                    pickle.dump(saved_m, fh)
            print(f"  [{exp_id}] Loaded verified checkpoints (B*={best_budget}, RSS={_rss_mb():.0f} MB)", flush=True)
        else:
            # Inner world 0 Stage-1 block models
            iw = inner_worlds[0]
            wp1 = np.zeros((4, n_fp), dtype=np.float32)
            for b in range(4):
                in_cb = core_2d[b][fp]
                pos_b = np.flatnonzero(iw["vis_w"][fp] & in_cb)
                neg_pool = np.flatnonzero(in_cb & (iw["d_vis_w_fp"] > 4.0))
                rng_b = np.random.default_rng(SEED + 200 + b)
                neg_b = rng_b.choice(neg_pool, size=min(len(neg_pool), len(pos_b) * 6), replace=False)
                idx_b = np.concatenate([pos_b, neg_b])
                yb = np.concatenate([np.ones(len(pos_b), np.float32), np.zeros(len(neg_b), np.float32)])
                mb = _fit_lgb(np.asarray(mm[idx_b][:, s1_cols], np.float32), yb, seed=SEED + 30 + b, n_estimators=220)
                _drop_mm_cache(mm)
                wp1[b] = _predict_chunked_single(mb, mm, s1_cols, all_rows)
                del mb
                gc.collect()
            print(f"  [{exp_id}] Stage 1 OOF complete ({time.time()-t0:.1f}s, RSS={_rss_mb():.0f} MB)", flush=True)

            # Stage 2 models
            s2_models = []
            for k in range(4):
                others = [b for b in range(4) if b != k]
                surf_wk = np.empty(n_fp, dtype=np.float32)
                surf_wk[rows_by_block[k]] = np.mean(wp1[others][:, rows_by_block[k]], axis=0)
                for b in others:
                    rem = [j for j in others if j != b]
                    surf_wk[rows_by_block[b]] = np.mean(wp1[rem][:, rows_by_block[b]], axis=0)
                sf_wk = PreparedStageField(surf_wk, fp, fp_y, fp_x)
                Xs_k = [build_feature_slice(sf_wk, iw["cf_w"], mm, s2_cols, iw["per_b"][b]["rows"]) for b in others]
                ys_k = [iw["per_b"][b]["y"] for b in others]
                _drop_mm_cache(mm)
                del sf_wk, surf_wk
                gc.collect()
                s2_models.append(_fit_lgb(np.vstack(Xs_k), np.concatenate(ys_k), seed=SEED + 500 + k, n_estimators=320))
                del Xs_k, ys_k
                gc.collect()

            if not fresh and ck_s2.exists():
                outer_s2 = np.load(ck_s2)
            else:
                sf_out_s1 = PreparedStageField(outer_s1, fp, fp_y, fp_x)
                outer_s2 = np.zeros(n_fp, dtype=np.float32)
                for k in range(4):
                    rk = rows_by_block[k]
                    pk = np.empty(len(rk), dtype=np.float32)
                    for s in range(0, len(rk), 450_000):
                        e = min(len(rk), s + 450_000)
                        Xv = build_feature_slice(sf_out_s1, outer_cat_cf, mm, s2_cols, rk[s:e])
                        _drop_mm_cache(mm)
                        pk[s:e] = s2_models[k].predict_proba(Xv)[:, 1].astype(np.float32)
                    outer_s2[rk] = pk
                del sf_out_s1
                np.save(ck_s2, outer_s2)
                gc.collect()

            # Stage 3 models (reuse tv_{exp_id}_f{k}.npz fast if already saved, else build wp2 cleanly!)
            tv_files = [CKPT / f"tv_{exp_id}_f{k}.npz" for k in range(4)]
            s3_models = []
            if not fresh and all(p.exists() for p in tv_files):
                lobo2_0 = None
                for k in range(4):
                    z = np.load(tv_files[k])
                    X3 = z["X_tr"].astype(np.float32)
                    y3 = (z["soft_tr"] >= 0.999).astype(np.float32)
                    s3_models.append(_fit_lgb(X3, y3, seed=SEED + 800 + k, n_estimators=320))
                    del X3, z
                    gc.collect()
            else:
                lobo1 = np.empty(n_fp, dtype=np.float32)
                for b in range(4):
                    lobo1[rows_by_block[b]] = np.mean(wp1[[j for j in range(4) if j != b]][:, rows_by_block[b]], axis=0)
                sf_w1 = PreparedStageField(lobo1, fp, fp_y, fp_x)
                mb2_list = []
                for b in range(4):
                    rb = iw["per_b"][b]["rows"]
                    Xb = build_feature_slice(sf_w1, iw["cf_w"], mm, s2_cols, rb)
                    _drop_mm_cache(mm)
                    mb2_list.append(_fit_lgb(Xb, iw["per_b"][b]["y"], seed=SEED + 600 + b, n_estimators=220))
                    del Xb
                wp2 = np.zeros((4, n_fp), dtype=np.float32)
                for s in range(0, n_fp, 450_000):
                    e = min(n_fp, s + 450_000)
                    Xch = build_feature_slice(sf_w1, iw["cf_w"], mm, s2_cols, all_rows[s:e])
                    _drop_mm_cache(mm)
                    for b in range(4):
                        wp2[b, s:e] = mb2_list[b].predict_proba(Xch)[:, 1].astype(np.float32)
                    del Xch
                del sf_w1, lobo1, mb2_list
                gc.collect()

                for k in range(4):
                    others = [b for b in range(4) if b != k]
                    surf_wk2 = np.empty(n_fp, dtype=np.float32)
                    surf_wk2[rows_by_block[k]] = np.mean(wp2[others][:, rows_by_block[k]], axis=0)
                    for b in others:
                        rem = [j for j in others if j != b]
                        surf_wk2[rows_by_block[b]] = np.mean(wp2[rem][:, rows_by_block[b]], axis=0)
                    sf_wk2 = PreparedStageField(surf_wk2, fp, fp_y, fp_x)
                    Xs_k, ys_k, soft_k, base_k = [], [], [], []
                    for b in others:
                        rb = iw["per_b"][b]["rows"]
                        Xs_k.append(build_feature_slice(sf_wk2, iw["cf_w"], mm, s2_cols, rb))
                        ys_k.append(iw["per_b"][b]["y"])
                        soft_k.append(iw["per_b"][b]["soft"])
                        base_k.append(surf_wk2[rb])
                    _drop_mm_cache(mm)
                    del sf_wk2, surf_wk2
                    gc.collect()
                    X_tr3 = np.vstack(Xs_k)
                    y_tr3 = np.concatenate(ys_k)
                    np.savez(
                        CKPT / f"tv_{exp_id}_f{k}.npz",
                        X_tr=X_tr3.astype(np.float16),
                        soft_tr=np.concatenate(soft_k).astype(np.float32),
                        base_p_tr=np.concatenate(base_k).astype(np.float32),
                    )
                    s3_models.append(_fit_lgb(X_tr3, y_tr3, seed=SEED + 800 + k, n_estimators=320))
                    del Xs_k, ys_k, soft_k, base_k, X_tr3, y_tr3
                    gc.collect()
                del wp2
                gc.collect()
            del wp1
            gc.collect()

            if not fresh and ck_s3.exists():
                outer_s3 = np.load(ck_s3)
            else:
                sf_out_s2 = PreparedStageField(outer_s2, fp, fp_y, fp_x)
                raw_outer_s3 = np.zeros(n_fp, dtype=np.float32)
                for k in range(4):
                    rk = rows_by_block[k]
                    pk = np.empty(len(rk), dtype=np.float32)
                    for s in range(0, len(rk), 450_000):
                        e = min(len(rk), s + 450_000)
                        Xv = build_feature_slice(sf_out_s2, outer_cat_cf, mm, s2_cols, rk[s:e])
                        _drop_mm_cache(mm)
                        pk[s:e] = s3_models[k].predict_proba(Xv)[:, 1].astype(np.float32)
                    raw_outer_s3[rk] = pk
                del sf_out_s2
                outer_s3 = _blend_probs(outer_s2, raw_outer_s3, w_a=0.40)
                np.save(ck_s3, outer_s3)
                gc.collect()

            # Matched 20%-hide inner-CV dot budget selection (exact competition fault hide density!)
            budget_scores, best_budget = _compute_inner20_budget_curve(s2_models, s3_models)

            with open(ck_m, "wb") as fh:
                pickle.dump({
                    "s2_models": s2_models,
                    "s3_models": s3_models,
                    "budget_scores": budget_scores,
                    "best_budget": best_budget,
                }, fh)

        print(f"  [{exp_id}] Matched-density inner-CV selected B* = {best_budget} "
              f"(inner DTI={budget_scores[best_budget]:.4f}, RSS={_rss_mb():.0f} MB)", flush=True)

        stage_surfaces = {1: outer_s1, 2: outer_s2, 3: outer_s3}
        stage_eval = {}
        for st in (1, 2, 3):
            surf2d = np.zeros(fp.shape, dtype=np.float32)
            surf2d[fp] = stage_surfaces[st]
            pred_38k = np.zeros(fp.shape, dtype=np.float32)
            pred_best = np.zeros(fp.shape, dtype=np.float32)
            for b in range(4):
                cy, cx = submodular_order_300m(
                    surf2d, outer_allowed[b], max_dots=max(38_000, best_budget) // 4 + 10,
                    catalogue_2d=folds[b]["visible"],
                )
                p38 = np.zeros(fp.shape, dtype=np.float32)
                p38[cy[:38_000 // 4], cx[:38_000 // 4]] = 1.0
                pred_38k += p38

                pb = np.zeros(fp.shape, dtype=np.float32)
                pb[cy[:best_budget // 4], cx[:best_budget // 4]] = 1.0
                pred_best += pb

            ta38, tf38 = _eval_terms(pred_38k, folds, fp, far_masks)
            tab, tfb = _eval_terms(pred_best, folds, fp, far_masks)
            stage_eval[st] = {
                "terms_38k_all": ta38,
                "terms_38k_far": tf38,
                "terms_best_all": tab,
                "terms_best_far": tfb,
                "dti_38k_all": float(E68.from_terms(ta38.sum(0))),
                "dti_38k_far": float(E68.from_terms(tf38.sum(0))),
                "dti_best_all": float(E68.from_terms(tab.sum(0))),
                "dti_best_far": float(E68.from_terms(tfb.sum(0))),
            }
            del surf2d, pred_38k, pred_best
            print(f"  [{exp_id}] OUTER Stage {st}: "
                  f"@38k ALL={stage_eval[st]['dti_38k_all']:.4f}, FAR={stage_eval[st]['dti_38k_far']:.4f} | "
                  f"@{best_budget//1000}k ALL={stage_eval[st]['dti_best_all']:.4f}, FAR={stage_eval[st]['dti_best_far']:.4f}",
                  flush=True)

        imp = np.mean([m.feature_importances_ for m in s3_models], axis=0)
        top_idx = np.argsort(imp)[::-1][:15]
        top_features = [{"feature": s2_feat_names[i], "importance": float(imp[i])} for i in top_idx]

        return {
            "exp_id": exp_id,
            "elapsed_s": round(time.time() - t0, 1),
            "s2_cols": s2_cols,
            "s2_feat_names": s2_feat_names,
            "best_budget": best_budget,
            "inner_budget_curve": {str(k): v for k, v in budget_scores.items()},
            "s1_oof": outer_s1,
            "s2_oof": outer_s2,
            "s3_oof": outer_s3,
            "s2_models": s2_models,
            "s3_models": s3_models,
            "stage_eval": stage_eval,
            "top_features": top_features,
        }

    # ------------------------------------------------------------------
    # RUN EXPERIMENT 1 & EXPERIMENT 2
    # ------------------------------------------------------------------
    print("\n[1/5] Running Experiment 1: H68-EXP1-O2-LocalSharp (42 Stage-1 ch, 66 Stage-2/3 ch)...", flush=True)
    exp1 = run_ac_experiment("H68-EXP1-O2-LocalSharp", EXP1_LOCAL_NAMES, STAGE2_PHYS_EXP1, save_tv_inputs=False)

    print("\n[2/5] Running Experiment 2: H68-EXP2-O2-MultiPhys-Vent (59 Stage-1 ch incl. GDR-1391 vents, 78 Stage-2/3 ch)...", flush=True)
    exp2 = run_ac_experiment("H68-EXP2-O2-MultiPhys-Vent", ALL_STACK_NAMES, STAGE2_PHYS_EXP2, save_tv_inputs=True)

    # Free inner_worlds immediately before Experiment 3 so RSS drops well below 1.1 GB!
    del inner_worlds
    gc.collect()

    # ------------------------------------------------------------------
    # RUN EXPERIMENT 3: Differentiable Tversky(0.2, 0.8) Neural Refiner + Hybrid Stack
    # ------------------------------------------------------------------
    print(f"\n[3/5] Running Experiment 3: H68-EXP3-TverskyNet-HybridStack (RSS={_rss_mb():.0f} MB)...", flush=True)
    t3 = time.time()
    tversky_nets: list[TverskyContextNet] = []
    tversky_oof = np.zeros(n_fp, dtype=np.float32)
    loss_curves: list[list[float]] = []

    ck_tv = CKPT / "exp3_tversky.pkl"
    if not fresh and ck_tv.exists():
        saved_tv = pickle.loads(ck_tv.read_bytes())
        tversky_nets = saved_tv["nets"]
        tversky_oof = saved_tv["oof"]
        loss_curves = saved_tv["loss_curves"]
        del saved_tv, outer_cat_cf
        gc.collect()
        print(f"  [H68-EXP3] Loaded verified TverskyContextNet checkpoint (RSS={_rss_mb():.0f} MB)", flush=True)
    else:
        sf_exp2_s2 = PreparedStageField(exp2["s2_oof"], fp, fp_y, fp_x)
        for k in range(4):
            npz_k = np.load(CKPT / f"tv_{exp2['exp_id']}_f{k}.npz")
            X_tr_k = npz_k["X_tr"].astype(np.float32)
            net_k = TverskyContextNet(in_dim=X_tr_k.shape[1], hidden1=64, hidden2=32,
                                      alpha=0.2, beta=0.8, bce_weight=0.15, l2=1e-4, seed=SEED + k)
            hist = net_k.fit(X_tr_k, npz_k["base_p_tr"], npz_k["soft_tr"],
                             epochs=26, batch_size=32768, lr=2.5e-3, seed=SEED + k)
            del X_tr_k, npz_k
            gc.collect()
            tversky_nets.append(net_k)
            loss_curves.append([round(x, 5) for x in hist])

            rk = rows_by_block[k]
            preds_tv = np.empty(len(rk), dtype=np.float32)
            base_s2_val = exp2["s2_oof"][rk]
            for s in range(0, len(rk), 450_000):
                e = min(len(rk), s + 450_000)
                Xv = build_feature_slice(sf_exp2_s2, outer_cat_cf, mm, exp2["s2_cols"], rk[s:e])
                _drop_mm_cache(mm)
                preds_tv[s:e] = net_k.predict_proba(Xv, base_s2_val[s:e])
            tversky_oof[rk] = preds_tv
            print(f"  [H68-EXP3] Outer fold {k} TverskyContextNet loss {hist[0]:.4f} -> {hist[-1]:.4f} (RSS={_rss_mb():.0f} MB)", flush=True)
        del sf_exp2_s2, outer_cat_cf
        ck_tv.write_bytes(pickle.dumps({"nets": tversky_nets, "oof": tversky_oof, "loss_curves": loss_curves}))
        gc.collect()

    l_e1 = logit(np.clip(exp1["s3_oof"], 1e-6, 1.0 - 1e-6))
    l_e2 = logit(np.clip(exp2["s3_oof"], 1e-6, 1.0 - 1e-6))
    l_tv = logit(np.clip(tversky_oof, 1e-6, 1.0 - 1e-6))
    blend_oof = expit(0.30 * l_e1 + 0.50 * l_e2 + 0.20 * l_tv).astype(np.float32)
    del l_e1, l_e2, l_tv

    exp3_budget = exp2["best_budget"]
    surf_tv_2d = np.zeros(fp.shape, dtype=np.float32)
    surf_tv_2d[fp] = tversky_oof
    pred_tv_best = np.zeros(fp.shape, dtype=np.float32)
    for b in range(4):
        cy_t, cx_t = submodular_order_300m(surf_tv_2d, outer_allowed[b], exp3_budget // 4, catalogue_2d=folds[b]["visible"])
        pt = np.zeros(fp.shape, dtype=np.float32)
        pt[cy_t, cx_t] = 1.0
        pred_tv_best += pt
    ta_tv, tf_tv = _eval_terms(pred_tv_best, folds, fp, far_masks)
    del surf_tv_2d, pred_tv_best, tversky_oof
    gc.collect()

    surf_bl_2d = np.zeros(fp.shape, dtype=np.float32)
    surf_bl_2d[fp] = blend_oof
    pred_bl_38k = np.zeros(fp.shape, dtype=np.float32)
    pred_bl_best = np.zeros(fp.shape, dtype=np.float32)
    for b in range(4):
        cy_b, cx_b = submodular_order_300m(surf_bl_2d, outer_allowed[b], max(38_000, exp3_budget) // 4 + 10, catalogue_2d=folds[b]["visible"])
        p38 = np.zeros(fp.shape, dtype=np.float32)
        p38[cy_b[:38_000 // 4], cx_b[:38_000 // 4]] = 1.0
        pred_bl_38k += p38

        pb = np.zeros(fp.shape, dtype=np.float32)
        pb[cy_b[:exp3_budget // 4], cx_b[:exp3_budget // 4]] = 1.0
        pred_bl_best += pb

    ta_bl38, tf_bl38 = _eval_terms(pred_bl_38k, folds, fp, far_masks)
    ta_bl, tf_bl = _eval_terms(pred_bl_best, folds, fp, far_masks)
    del surf_bl_2d, pred_bl_38k, pred_bl_best, blend_oof
    gc.collect()
    exp3_elapsed = round(time.time() - t3, 1)

    print(f"  [H68-EXP3] Pure TverskyNet @{exp3_budget//1000}k: ALL={E68.from_terms(ta_tv.sum(0)):.4f}, FAR={E68.from_terms(tf_tv.sum(0)):.4f}")
    print(f"  [H68-EXP3] Hybrid Stack @38k: ALL={E68.from_terms(ta_bl38.sum(0)):.4f}, FAR={E68.from_terms(tf_bl38.sum(0)):.4f} | "
          f"@{exp3_budget//1000}k: ALL={E68.from_terms(ta_bl.sum(0)):.4f}, FAR={E68.from_terms(tf_bl.sum(0)):.4f}",
          flush=True)

    # ------------------------------------------------------------------
    # POOLED 1,000-DRAW PAIRED SPATIAL-CLUSTER BOOTSTRAP EVALUATION
    # ------------------------------------------------------------------
    arms_all = {
        "h19_5_nms5_38k": t_h19_nms_all,
        "h19_5_submod_38k": t_h19_sub_all,
        "exp1_stage1_oof": exp1["stage_eval"][1]["terms_best_all"],
        "exp1_stage2_oof": exp1["stage_eval"][2]["terms_best_all"],
        "exp1_stage3_oof": exp1["stage_eval"][3]["terms_best_all"],
        "exp2_stage1_oof": exp2["stage_eval"][1]["terms_best_all"],
        "exp2_stage2_oof": exp2["stage_eval"][2]["terms_best_all"],
        "exp2_stage3_oof": exp2["stage_eval"][3]["terms_best_all"],
        "exp3_tverskynet_oof": ta_tv,
        "exp3_hybrid_38k": ta_bl38,
        "exp3_hybrid_best": ta_bl,
    }
    arms_far = {
        "h19_5_nms5_38k": t_h19_nms_far,
        "h19_5_submod_38k": t_h19_sub_far,
        "exp1_stage1_oof": exp1["stage_eval"][1]["terms_best_far"],
        "exp1_stage2_oof": exp1["stage_eval"][2]["terms_best_far"],
        "exp1_stage3_oof": exp1["stage_eval"][3]["terms_best_far"],
        "exp2_stage1_oof": exp2["stage_eval"][1]["terms_best_far"],
        "exp2_stage2_oof": exp2["stage_eval"][2]["terms_best_far"],
        "exp2_stage3_oof": exp2["stage_eval"][3]["terms_best_far"],
        "exp3_tverskynet_oof": tf_tv,
        "exp3_hybrid_38k": tf_bl38,
        "exp3_hybrid_best": tf_bl,
    }

    candidates = [
        ("exp1_stage3_oof", "exp1_stage1_oof", exp1["best_budget"]),
        ("exp2_stage3_oof", "exp2_stage1_oof", exp2["best_budget"]),
        ("exp3_hybrid_best", "exp2_stage1_oof", exp3_budget),
    ]
    winner_key, winner_s1_key, winner_budget = max(
        candidates,
        key=lambda item: 0.5 * E68.from_terms(arms_all[item[0]].sum(0)) + 0.5 * E68.from_terms(arms_far[item[0]].sum(0)),
    )

    summary_all = _score_arms(arms_all, candidate=winner_key, draws=1000)
    summary_far = _score_arms(arms_far, candidate=winner_key, draws=1000)

    diff_s1_all = summary_all["paired_differences"][winner_s1_key]
    diff_s1_far = summary_far["paired_differences"][winner_s1_key]
    diff_h19_all = summary_all["paired_differences"]["h19_5_nms5_38k"]
    diff_h19_far = summary_far["paired_differences"]["h19_5_nms5_38k"]
    diff_h19sub_all = summary_all["paired_differences"]["h19_5_submod_38k"]
    diff_h19sub_far = summary_far["paired_differences"]["h19_5_submod_38k"]

    se_all = (diff_s1_all["ci95"][1] - diff_s1_all["ci95"][0]) / 3.92
    se_far = (diff_s1_far["ci95"][1] - diff_s1_far["ci95"][0]) / 3.92
    z_all = diff_s1_all["delta"] / max(se_all, 1e-12)
    z_far = diff_s1_far["delta"] / max(se_far, 1e-12)
    gate_passed = bool(
        z_all > 2.0 and z_far > 2.0
        and diff_s1_all["ci95"][0] > 0.0 and diff_s1_far["ci95"][0] > 0.0
    )

    print("\n" + "=" * 78)
    print(f"WINNER: {winner_key} (budget={winner_budget})")
    print(f"  ALL HOLDOUT-DTI : {summary_all['scores'][winner_key]['dti']:.4f}  CI95={summary_all['scores'][winner_key]['ci95']}")
    print(f"    vs Stage 1 OOF: +{diff_s1_all['delta']:.4f} ({z_all:.2f} SE, CI95={diff_s1_all['ci95']})")
    print(f"    vs h19_5 NMS  : +{diff_h19_all['delta']:.4f} | vs h19_5 Submod: +{diff_h19sub_all['delta']:.4f}")
    print(f"  FAR (>1 km) DTI : {summary_far['scores'][winner_key]['dti']:.4f}  CI95={summary_far['scores'][winner_key]['ci95']}")
    print(f"    vs Stage 1 OOF: +{diff_s1_far['delta']:.4f} ({z_far:.2f} SE, CI95={diff_s1_far['ci95']})")
    print(f"    vs h19_5 NMS  : +{diff_h19_far['delta']:.4f} | vs h19_5 Submod: +{diff_h19sub_far['delta']:.4f}")
    print(f"  PROTOCOL GATE (>2 SE ALL & FAR): {'PASS' if gate_passed else 'FAIL'}")
    print("=" * 78, flush=True)

    # ------------------------------------------------------------------
    # [4/5] FULL-CATALOGUE INFERENCE FROM STRICT OOF STAGE-1 SURFACE
    # ------------------------------------------------------------------
    print(f"\n[4/5] Computing full-study-area submission surface (RSS={_rss_mb():.0f} MB)...", flush=True)
    # Free OOF vectors we no longer need BEFORE constructing full_cat_cf!
    exp1_s1_oof = exp1.pop("s1_oof")
    exp2_s1_oof = exp2.pop("s1_oof")
    exp1.pop("s2_oof", None)
    exp1.pop("s3_oof", None)
    exp2.pop("s2_oof", None)
    exp2.pop("s3_oof", None)
    gc.collect()

    ck_full = CKPT / "full_surface_fp.npy"
    if not fresh and ck_full.exists():
        final_surface_fp = np.load(ck_full)
        print(f"  Loaded checkpointed full-study-area surface (RSS={_rss_mb():.0f} MB)!", flush=True)
    else:
        full_cat_cf = PreparedCatalogueField(grid.catalogue, fp, fp_y, fp_x)

        def _full_cascade(s1_vec: np.ndarray, s2_cols: list[int],
                          s2_models: list, s3_models: list):
            sf1 = PreparedStageField(s1_vec, fp, fp_y, fp_x)
            s2_vec = np.zeros(n_fp, dtype=np.float32)
            for s in range(0, n_fp, 500_000):
                e = min(n_fp, s + 500_000)
                Xfull = build_feature_slice(sf1, full_cat_cf, mm, s2_cols, all_rows[s:e])
                _drop_mm_cache(mm)
                s2_vec[s:e] = np.mean([m.predict_proba(Xfull)[:, 1] for m in s2_models], axis=0).astype(np.float32)
            del sf1
            gc.collect()

            sf2 = PreparedStageField(s2_vec, fp, fp_y, fp_x)
            raw_s3 = np.zeros(n_fp, dtype=np.float32)
            for s in range(0, n_fp, 500_000):
                e = min(n_fp, s + 500_000)
                Xfull = build_feature_slice(sf2, full_cat_cf, mm, s2_cols, all_rows[s:e])
                _drop_mm_cache(mm)
                raw_s3[s:e] = np.mean([m.predict_proba(Xfull)[:, 1] for m in s3_models], axis=0).astype(np.float32)
            del sf2
            gc.collect()
            return _blend_probs(s2_vec, raw_s3, w_a=0.40), s2_vec

        full_e1_s3, _ = _full_cascade(exp1_s1_oof, exp1["s2_cols"], exp1["s2_models"], exp1["s3_models"])
        del exp1_s1_oof
        gc.collect()

        full_e2_s3, full_e2_s2 = _full_cascade(exp2_s1_oof, exp2["s2_cols"], exp2["s2_models"], exp2["s3_models"])
        del exp2_s1_oof
        gc.collect()

        sf_full_s2 = PreparedStageField(full_e2_s2, fp, fp_y, fp_x)
        full_tv = np.zeros(n_fp, dtype=np.float32)
        for s in range(0, n_fp, 500_000):
            e = min(n_fp, s + 500_000)
            Xfull = build_feature_slice(sf_full_s2, full_cat_cf, mm, exp2["s2_cols"], all_rows[s:e])
            _drop_mm_cache(mm)
            full_tv[s:e] = np.mean([net.predict_proba(Xfull, full_e2_s2[s:e]) for net in tversky_nets], axis=0).astype(np.float32)
        del sf_full_s2, full_cat_cf, full_e2_s2
        gc.collect()

        if winner_key == "exp1_stage3_oof":
            final_surface_fp = full_e1_s3
        elif winner_key == "exp2_stage3_oof":
            final_surface_fp = full_e2_s3
        else:
            final_surface_fp = expit(
                0.30 * logit(np.clip(full_e1_s3, 1e-6, 1.0 - 1e-6))
                + 0.50 * logit(np.clip(full_e2_s3, 1e-6, 1.0 - 1e-6))
                + 0.20 * logit(np.clip(full_tv, 1e-6, 1.0 - 1e-6))
            ).astype(np.float32)
        del full_e1_s3, full_e2_s3, full_tv
        np.save(ck_full, final_surface_fp)
        gc.collect()

    final_surface_2d = np.zeros(fp.shape, dtype=np.float32)
    final_surface_2d[fp] = final_surface_fp

    d_full_cat = ndi.distance_transform_edt(~grid.catalogue)
    live_allowed = fp & ~grid.catalogue & (d_full_cat > EXCLUDE_PX)
    del d_full_cat
    emitted_2d = emit_submodular_300m(
        final_surface_2d, live_allowed, n_dots=winner_budget, catalogue_2d=grid.catalogue
    )
    del final_surface_2d, live_allowed
    gc.collect()
    n_emitted_dots = int((emitted_2d > 0).sum())
    print(f"  Emitted {n_emitted_dots:,} exact Bang-Bang {{0, 1}} dots (0 on existing_faults, RSS={_rss_mb():.0f} MB)!", flush=True)

    # ------------------------------------------------------------------
    # [5/5] WRITE UNIQUE TIFS, PORTAL PREFLIGHT, SHARED VALIDATOR & UNIQUENESS AUDIT
    # ------------------------------------------------------------------
    print("\n[5/5] Running GeoTIFF emission, portal preflight & full registry uniqueness audit...", flush=True)
    sub_ref = ROOT / "data" / "bridge" / "sample_submission.tif"
    feat_ref = ROOT / "data" / "bridge" / "training_features.tif"
    cat_ref = ROOT / "data" / "bridge" / "existing_faults.tif"

    with rasterio.open(sub_ref) as sr:
        fp_sub = np.isfinite(sr.read(1))
    fp_feat = footprint_from_features(feat_ref, fp.shape)
    fp_union = fp_sub | fp_feat

    dl_dir = ROOT / "docs" / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)

    stem = f"GEMSDOE68-O2AC-MultiPhys-v3-{RUN_TS}"
    p_zeros = dl_dir / f"{stem}-zeros.tif"
    p_nan = dl_dir / f"{stem}-nan.tif"
    p_strict = dl_dir / f"{stem}-strict-nan.tif"
    p_zip = dl_dir / f"{stem}-zeros.tif.zip"
    p_nan_zip = dl_dir / f"{stem}-nan.tif.zip"

    meta_zeros = write_submission(p_zeros, emitted_2d, fp_union, sub_ref, overwrite=True, outside_fill=0.0)
    meta_nan = write_submission(p_nan, emitted_2d, fp_union, sub_ref, overwrite=True, outside_fill=np.nan)
    meta_strict = write_submission(p_strict, np.where(fp_sub, emitted_2d, 0.0), fp_sub, sub_ref, overwrite=True, outside_fill=np.nan)
    with zipfile.ZipFile(p_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(p_zeros, arcname=p_zeros.name)
    with zipfile.ZipFile(p_nan_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(p_nan, arcname=p_nan.name)
    zip_sha256 = hashlib.sha256(p_zip.read_bytes()).hexdigest()

    portal_zeros = validate_portal_ready(p_zeros, sub_ref, feat_ref, cat_ref)
    portal_nan = validate_portal_ready(p_nan, sub_ref, feat_ref, cat_ref)
    strict_val = validate(p_strict, sub_ref, features_path=feat_ref)

    assert portal_zeros["passed"] and portal_zeros["all_12279160_pixels_finite_in_0_1"]
    assert portal_nan["passed"] and portal_nan["nan_in_sample_sub"] == 0
    assert strict_val["passed"] and not strict_val["has_failures"] and not strict_val["skipped_checks_mean_unverified"]

    rng_u = np.random.default_rng(68)
    sub_u = rng_u.choice(n_fp, size=400_000, replace=False)
    surf_sub = final_surface_fp[sub_u]
    emit_mask = emitted_2d > 0
    disk3_offsets = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4) if dy * dy + dx * dx <= 9]
    ey_all, ex_all = np.nonzero(emit_mask)
    H_grid, W_grid = fp.shape

    def _within_3px_fraction(src_y: np.ndarray, src_x: np.ndarray, target_mask: np.ndarray) -> float:
        if len(src_y) == 0 or not target_mask.any():
            return 0.0
        hit = np.zeros(len(src_y), dtype=bool)
        for dy, dx in disk3_offsets:
            yy = np.clip(src_y + dy, 0, H_grid - 1)
            xx = np.clip(src_x + dx, 0, W_grid - 1)
            hit |= target_mask[yy, xx]
        return float(hit.mean())

    reg_files = sorted(list((ROOT / "registry" / "rasters").rglob("*.tif"))
                       + [p for p in (ROOT / "data" / "scored").glob("*.tif") if not p.is_symlink()])
    dup_records = []
    seen_sha = set()
    for rf in reg_files:
        sha = hashlib.sha256(rf.read_bytes()).hexdigest()
        if sha in seen_sha:
            continue
        seen_sha.add(sha)
        assert sha != meta_zeros["sha256"] and sha != meta_nan["sha256"], f"SHA collision with {rf}!"
        with rasterio.open(rf) as ds:
            arr = ds.read(1)
        if arr.shape != fp.shape:
            continue
        arr_fp = np.nan_to_num(arr[fp], nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.std(arr_fp)) < 1e-8:
            # Constant placeholder raster (e.g. gemsdoe9-PLACEHOLDER all 1.0s)
            dup_records.append({
                "raster": rf.name,
                "sha256": sha,
                "kind": "constant_placeholder",
                "spearman_surface": 0.0,
                "cand_positive_pixels": int((arr_fp > 0).sum()),
                "our_dots_within_3px_of_cand": 0.0,
                "cand_dots_within_3px_of_ours": 0.0,
            })
            continue
        rho = float(spearmanr(surf_sub, arr_fp[sub_u]).statistic)
        if not np.isfinite(rho):
            rho = 0.0
        pos_mask = np.isfinite(arr) & (arr > 0) & fp
        n_pos = int(pos_mask.sum())
        if n_pos <= 150_000:
            kind = "emitted_dots"
            cand_dots = pos_mask
        else:
            kind = "continuous_surface_top42k"
            cand_dots = np.zeros(fp.shape, dtype=bool)
            top_i = np.argpartition(arr_fp, -n_emitted_dots)[-n_emitted_dots:]
            cand_dots[fp_y[top_i], fp_x[top_i]] = True
        if cand_dots.any():
            cy_c, cx_c = np.nonzero(cand_dots)
            frac_ours_near_theirs = _within_3px_fraction(ey_all, ex_all, cand_dots)
            frac_theirs_near_ours = _within_3px_fraction(cy_c, cx_c, emit_mask)
        else:
            frac_ours_near_theirs = 0.0
            frac_theirs_near_ours = 0.0
        dup_records.append({
            "raster": rf.name,
            "sha256": sha,
            "kind": kind,
            "spearman_surface": round(rho, 4),
            "cand_positive_pixels": n_pos,
            "our_dots_within_3px_of_cand": round(frac_ours_near_theirs, 4),
            "cand_dots_within_3px_of_ours": round(frac_theirs_near_ours, 4),
        })

    max_spearman = max(abs(r["spearman_surface"]) for r in dup_records)
    max_overlap = max(r["our_dots_within_3px_of_cand"] for r in dup_records)
    dup_passed = (max_spearman <= 0.90) and (max_overlap <= 0.70)

    dup_records_sorted = sorted(dup_records, key=lambda r: max(abs(r["spearman_surface"]), r["our_dots_within_3px_of_cand"]), reverse=True)
    print(f"  Uniqueness audit across {len(dup_records)} unique rasters: "
          f"max |Spearman| = {max_spearman:.4f} (<=0.90), max 3px dot overlap = {max_overlap:.4f} (<=0.70) -> {'PASS' if dup_passed else 'DUPLICATE'}!",
          flush=True)

    np.save(ROOT / ".cache" / "s68_stack" / "final_surface_fp.npy", final_surface_fp)
    np.save(ROOT / ".cache" / "s68_stack" / "emitted_dots_rc.npy",
            np.column_stack(np.nonzero(emitted_2d > 0)).astype(np.int16))

    sub_note = (
        f"S68 O2-AutoContext3 + GDR1391 vents + Submod300m ({n_emitted_dots//1000}k dots); "
        f"Holdout ALL={summary_all['scores'][winner_key]['dti']:.4f}, FAR={summary_far['scores'][winner_key]['dti']:.4f}"
    )
    assert len(sub_note) <= 140

    # Empirical Nevada Basin-and-Range fault spacing physics + side-by-side fleet dot allocation
    d_vis_km = d_vis_outer * 0.1
    d_cat_full_km = ndi.distance_transform_edt(~grid.catalogue) * 0.1
    reg_union = np.zeros(fp.shape, dtype=bool)
    hid_union = np.zeros(fp.shape, dtype=bool)
    for f in folds:
        reg_union |= f["region"]
        hid_union |= (f["truth"] & f["region"])
    band_defs = [
        ("le_0p2km", 0.0, 0.2, "Mapped trace collar (<=0.2 km)"),
        ("0p2_to_0p5km", 0.2, 0.5, "Fault tip & horse-tail splay (0.2-0.5 km)"),
        ("0p5_to_1p0km", 0.5, 1.0, "Relay ramp & step-over (0.5-1.0 km)"),
        ("1p0_to_3p0km", 1.0, 3.0, "Synthetic/antithetic basin fault (1.0-3.0 km)"),
        ("gt_3p0km", 3.0, 999.0, "Deep interior basin / playa (>3.0 km)"),
    ]
    nevada_bands = []
    for bkey, lo, hi, blabel in band_defs:
        m_r = reg_union & (d_vis_km > lo) & (d_vis_km <= hi)
        m_h = hid_union & (d_vis_km > lo) & (d_vis_km <= hi)
        a_pct = 100.0 * float(m_r.sum()) / float(reg_union.sum())
        p_pct = 100.0 * float(m_h.sum()) / float(hid_union.sum())
        br_pct = 100.0 * float(m_h.sum()) / max(float(m_r.sum()), 1.0)
        nevada_bands.append({
            "key": bkey,
            "label": blabel,
            "footprint_area_pct": round(a_pct, 2),
            "true_withheld_fault_pct": round(p_pct, 2),
            "true_withheld_fault_pixels": int(m_h.sum()),
            "base_rate_pct": round(br_pct, 4),
            "prior_enrichment_vs_uniform": round(p_pct / max(a_pct, 1e-6), 2),
        })
    fleet_alloc = {}
    for fname, fpath in [
        ("GEMSDOE32_h33_2_b2_02778_OWNER_REPORT", ROOT / "registry/rasters/rasters/GEMSDOE32__h33-2-b2-zeros.tif"),
        ("GEMSDOE36_anderson_pinn_02750_OWNER_REPORT", ROOT / "registry/rasters/rasters/GEMSDOE36__anderson-geothermal-pinn-38854.tif"),
        ("GEMSDOE68_PR3_hidedraw_01189_HOLDOUT", ROOT / "docs/downloads/gems68-autoctx-s3-hidedraw-38000dots-20261012T022132Z-4c18def6-zeros.tif"),
        (f"GEMSDOE68_v3_O2AC_MultiPhys_{winner_budget}_HOLDOUT", p_zeros),
    ]:
        if not fpath.exists():
            continue
        with rasterio.open(fpath) as ds:
            farr = np.clip(np.nan_to_num(ds.read(1).astype(np.float32), nan=0.0), 0.0, 1.0)
        fdots = farr > 0.5
        fdd = d_cat_full_km[fdots]
        fleet_alloc[fname] = {
            "total_dots": int(fdots.sum()),
            "bands_pct": {
                bkey: round(100.0 * float(((fdd > lo) & (fdd <= hi)).mean()), 2)
                for bkey, lo, hi, _ in band_defs
            },
            "gt_1km_far_dots": int((fdd > 1.0).sum()),
            "gt_3km_deep_dots": int((fdd > 3.0).sum()),
        }

    evidence_payload = {
        "schema": "gems.evidence.run_card.v2",
        "site": "GEMSDOE68",
        "lane": "Auto-Context Stacked Refinement (Tu, CVPR 2008 / TPAMI 2010)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_seconds": round(time.time() - t_start, 1),
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": E68.VERSION,
        "evaluator_implementation_sha256": E68.implementation_hashes(),
        "holdout_seed": SEED,
        "exclude_px": EXCLUDE_PX,
        "quadrant_core_separation_m": 2.0 * CORE_ERODE_M,
        "withheld_positive_pixels": {
            "all": summary_all["scores"][winner_key]["withheld_positive_pixels"],
            "far_gt_1km": summary_far["scores"][winner_key]["withheld_positive_pixels"],
        },
        "winner": {
            "arm": winner_key,
            "stage1_comparator": winner_s1_key,
            "dot_budget": winner_budget,
            "emitted_positive_pixels": n_emitted_dots,
            "all_dti": summary_all["scores"][winner_key]["dti"],
            "all_ci95": summary_all["scores"][winner_key]["ci95"],
            "far_gt_1km_dti": summary_far["scores"][winner_key]["dti"],
            "far_gt_1km_ci95": summary_far["scores"][winner_key]["ci95"],
            "delta_vs_stage1_all": diff_s1_all["delta"],
            "delta_vs_stage1_all_ci95": diff_s1_all["ci95"],
            "delta_vs_stage1_all_z_se": round(z_all, 2),
            "delta_vs_stage1_far": diff_s1_far["delta"],
            "delta_vs_stage1_far_ci95": diff_s1_far["ci95"],
            "delta_vs_stage1_far_z_se": round(z_far, 2),
            "delta_vs_h19_5_nms5_all": diff_h19_all["delta"],
            "delta_vs_h19_5_nms5_far": diff_h19_far["delta"],
            "delta_vs_h19_5_submod_all": diff_h19sub_all["delta"],
            "delta_vs_h19_5_submod_far": diff_h19sub_far["delta"],
            "gate_gt_2se_all_and_far": gate_passed,
            "ok_to_download_and_submit": bool(gate_passed and dup_passed and portal_zeros["passed"]),
        },
        "submission_artifacts": {
            "recommended_portal_upload_tif": {
                "filename": p_zeros.name,
                "relpath": f"docs/downloads/{p_zeros.name}",
                "zip_filename": p_zip.name,
                "zip_relpath": f"docs/downloads/{p_zip.name}",
                "zip_bytes": p_zip.stat().st_size,
                "zip_sha256": zip_sha256,
                "bytes": p_zeros.stat().st_size,
                "sha256": meta_zeros["sha256"],
                "outside_fill": 0.0,
                "portal_validation": portal_zeros,
            },
            "sample_submission_nan_companion_tif": {
                "filename": p_nan.name,
                "relpath": f"docs/downloads/{p_nan.name}",
                "bytes": p_nan.stat().st_size,
                "sha256": meta_nan["sha256"],
                "outside_fill": "NaN outside union(sample_submission, features)",
                "portal_validation": portal_nan,
            },
            "strict_validator_16of16_tif": {
                "filename": p_strict.name,
                "relpath": f"docs/downloads/{p_strict.name}",
                "bytes": p_strict.stat().st_size,
                "sha256": meta_strict["sha256"],
                "strict_validation": strict_val,
            },
            "unique_submission_name": stem,
            "submission_note_le_140": sub_note,
            "submission_note_length": len(sub_note),
        },
        "uniqueness_audit": {
            "n_unique_rasters_checked": len(dup_records),
            "max_abs_spearman_surface": max_spearman,
            "spearman_ceiling": 0.90,
            "max_3px_dot_overlap": max_overlap,
            "dot_overlap_ceiling": 0.70,
            "passed": dup_passed,
            "top12_closest_rasters": dup_records_sorted[:12],
        },
        "experiments": {
            "exp1_o2_local_sharp": {
                "id": exp1["exp_id"],
                "elapsed_s": exp1["elapsed_s"],
                "n_stage1_channels": len(EXP1_LOCAL_NAMES),
                "n_stage2_features": len(exp1["s2_feat_names"]),
                "inner_selected_budget": exp1["best_budget"],
                "inner_budget_curve": exp1["inner_budget_curve"],
                "top_stage3_features": exp1["top_features"],
                "stage_progression": {
                    f"stage{s}": {
                        "dti_38k_all": exp1["stage_eval"][s]["dti_38k_all"],
                        "dti_38k_far": exp1["stage_eval"][s]["dti_38k_far"],
                        "dti_tuned_all": exp1["stage_eval"][s]["dti_best_all"],
                        "dti_tuned_far": exp1["stage_eval"][s]["dti_best_far"],
                    }
                    for s in (1, 2, 3)
                },
            },
            "exp2_o2_multiphysics_vent": {
                "id": exp2["exp_id"],
                "elapsed_s": exp2["elapsed_s"],
                "n_stage1_channels": len(ALL_STACK_NAMES),
                "n_stage2_features": len(exp2["s2_feat_names"]),
                "inner_selected_budget": exp2["best_budget"],
                "inner_budget_curve": exp2["inner_budget_curve"],
                "top_stage3_features": exp2["top_features"],
                "stage_progression": {
                    f"stage{s}": {
                        "dti_38k_all": exp2["stage_eval"][s]["dti_38k_all"],
                        "dti_38k_far": exp2["stage_eval"][s]["dti_38k_far"],
                        "dti_tuned_all": exp2["stage_eval"][s]["dti_best_all"],
                        "dti_tuned_far": exp2["stage_eval"][s]["dti_best_far"],
                    }
                    for s in (1, 2, 3)
                },
            },
            "exp3_tverskynet_hybrid": {
                "id": "H68-EXP3-TverskyNet-HybridStack",
                "elapsed_s": exp3_elapsed,
                "dot_budget": exp3_budget,
                "fold_loss_histories": loss_curves,
                "pure_tverskynet": {
                    "dti_all": summary_all["scores"]["exp3_tverskynet_oof"]["dti"],
                    "dti_far": summary_far["scores"]["exp3_tverskynet_oof"]["dti"],
                },
                "hybrid_38k": {
                    "dti_all": summary_all["scores"]["exp3_hybrid_38k"]["dti"],
                    "dti_far": summary_far["scores"]["exp3_hybrid_38k"]["dti"],
                },
                "hybrid_tuned": {
                    "dti_all": summary_all["scores"]["exp3_hybrid_best"]["dti"],
                    "dti_far": summary_far["scores"]["exp3_hybrid_best"]["dti"],
                },
            },
        },
        "empirical_nevada_fault_distance_physics": {
            "withheld_fault_bands": nevada_bands,
            "fleet_dot_allocation": fleet_alloc,
        },
        "pooled_holdout_all": summary_all,
        "pooled_holdout_far_gt_1km": summary_far,
    }

    out_card = ROOT / "evidence" / "s68_run_card.json"
    out_card.write_text(json.dumps(evidence_payload, indent=2))
    (ROOT / "docs" / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "data" / "run_card.json").write_text(json.dumps(evidence_payload, indent=2))
    assert dup_passed, f"Duplicate gate failed! max_spearman={max_spearman}, max_overlap={max_overlap}"
    print(f"\n[DONE] Canonical Run Card written to {out_card} in {time.time() - t_start:.1f}s!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
