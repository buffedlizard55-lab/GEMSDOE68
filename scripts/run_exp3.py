#!/usr/bin/env python3
"""Experiment 3 -- De-blobbed Auto-Context Stacked Refinement with 2 m USGS 3DEP
LiDAR Scarp Channels, GeoDAWN Radiometric/Magnetic Extensions, 1D Curvilinear
Ridge-vs-Blob Contrast, and Hide-Draw Catalogue Context.

Addresses the explicit next-session directive in ``docs/irregularities.md``
(``IR-68-BLOB-01``) and tests the ranked geological hypotheses (H1: 2 m LiDAR
antislope scarp + crest-toe dipole; H2: GeoDAWN 1.5 km high-pass radiometric &
upward-continued magnetic dipole residual; H3: 1D curvilinear ridge-vs-blob
contrast + along-strike fault-tip continuation).

Writes ``evidence/exp3_autocontext_scarp_ridge_hidedraw.json`` and caches outer
predictions in ``.cache/exp3_outer_preds.npz``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gems68 import autocontext as AC
from gems68 import features as F
from gems68.emission import emit_dots, spatial_block_terms
from gems68.experiment import Lane, log
from gems68.grid import load_grid
from gems68.holdout import FOLD_NAMES, buffered_component_draw, paired_block_bootstrap
from gems68.metric import dti_binary, dti_from_components, marginal_inclusion_threshold

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
EVIDENCE.mkdir(parents=True, exist_ok=True)

BUDGETS = (15_000, 20_000, 25_000, 30_000, 35_000, 38_000, 42_000, 46_000)


def _outer_block_se_stats(fold_dtis_cand: list[float], fold_dtis_ctrl: list[float]) -> dict:
    diffs = np.asarray(fold_dtis_cand, dtype=np.float64) - np.asarray(fold_dtis_ctrl, dtype=np.float64)
    mean_d = float(diffs.mean())
    se_d = float(diffs.std(ddof=1) / np.sqrt(diffs.size)) if diffs.size > 1 else 0.0
    t_stat = float(mean_d / se_d) if se_d > 0 else 0.0
    return dict(
        mean_outer_block_delta=mean_d,
        se_outer_block=se_d,
        t_outer_block=t_stat,
        beats_by_gt_2se=bool(t_stat > 2.0),
        per_fold_deltas={FOLD_NAMES[i]: float(diffs[i]) for i in range(len(diffs))},
    )


def score_stage_maps(grid, folds, p_maps: dict[int, np.ndarray], dist_vis: np.ndarray) -> dict:
    from gems68.emission import exclusion_mask
    from gems68.metric import kernel

    far_mask = dist_vis > 10.0  # > 1 km from any visible fault
    block_side = 200
    h, w = grid.shape
    ncols = (w + block_side - 1) // block_side
    nrows = (h + block_side - 1) // block_side
    n_clusters = ncols * nrows

    # Precompute fold-level truth masks, truth kernel maps, and cluster IDs once (8 EDTs total instead of 384)
    fold_info = []
    tot_reg = float(sum(ff["region"].sum() for ff in folds))
    for f in folds:
        act = grid.footprint & f["region"] & ~f["masked_known"]
        act_far = act & far_mask
        g_all = f["truth"] & act
        g_far = f["truth"] & act_far
        ty_a, tx_a = np.nonzero(g_all)
        ty_f, tx_f = np.nonzero(g_far)
        tids_a = (ty_a // block_side) * ncols + (tx_a // block_side)
        tids_f = (ty_f // block_side) * ncols + (tx_f // block_side)
        k_dg_a = kernel(ndi.distance_transform_edt(~g_all)).astype(np.float32)
        k_dg_f = kernel(ndi.distance_transform_edt(~g_far)).astype(np.float32)
        allowed = (grid.footprint & f["region"]) & ~exclusion_mask(f["visible"], 2.0)
        fold_info.append(
            dict(
                name=f["name"],
                share_frac=float(f["region"].sum()) / tot_reg,
                act=act,
                act_far=act_far,
                g_all=g_all,
                g_far=g_far,
                ty_a=ty_a,
                tx_a=tx_a,
                ty_f=ty_f,
                tx_f=tx_f,
                tids_a=tids_a,
                tids_f=tids_f,
                k_dg_a=k_dg_a,
                k_dg_f=k_dg_f,
                allowed=allowed,
                n_truth_a=int(ty_a.size),
                n_truth_f=int(ty_f.size),
            )
        )

    # Precompute 5x5 NMS candidate coordinates sorted descending by score per (stage, fold) once
    nms_sorted = {}
    for k, pmap in p_maps.items():
        s_clean = np.nan_to_num(pmap.astype(np.float32), nan=0.0)
        nms_sorted[k] = []
        for fi in fold_info:
            s_fold = np.where(fi["allowed"], s_clean, 0.0)
            mx = ndi.maximum_filter(s_fold, size=5, mode="constant", cval=0.0)
            cand = (s_fold >= mx) & (s_fold > 0) & fi["allowed"]
            yy, xx = np.nonzero(cand)
            order = np.argsort(-s_fold[yy, xx])
            nms_sorted[k].append((yy[order], xx[order]))

    results = {}
    for B in BUDGETS:
        per_stage = {}
        terms_all = {}
        terms_far = {}
        fold_dtis_all_by_stage = {}
        fold_dtis_far_by_stage = {}
        for k in p_maps:
            emitted = np.zeros(grid.shape, bool)
            for fi, (yy, xx) in zip(fold_info, nms_sorted[k]):
                share = int(round(B * fi["share_frac"]))
                n_keep = min(share, yy.size)
                if n_keep > 0:
                    emitted[yy[:n_keep], xx[:n_keep]] = True

            dp_full = ndi.distance_transform_edt(~emitted).astype(np.float32)
            tp_a = fp_a = fn_a = nt_a = 0.0
            tp_f = fp_f = fn_f = nt_f = 0.0
            t_all = np.zeros((n_clusters, 4), dtype=np.float64)
            t_far = np.zeros((n_clusters, 4), dtype=np.float64)
            per_fold = {}
            f_dtis_all = []
            f_dtis_far = []

            for fi in fold_info:
                # All pixels in fold
                p_a = emitted & fi["act"]
                cov_a = kernel(dp_full[fi["ty_a"], fi["tx_a"]])
                tp_fold_a = float(cov_a.sum())
                fn_fold_a = float(fi["n_truth_a"]) - tp_fold_a
                py_a, px_a = np.nonzero(p_a)
                fp_w_a = 1.0 - fi["k_dg_a"][py_a, px_a].astype(np.float64)
                fp_fold_a = float(fp_w_a.sum())
                dti_fold_a = dti_from_components(tp_fold_a, fp_fold_a, fn_fold_a)

                pids_a = (py_a // block_side) * ncols + (px_a // block_side)
                t_all[:, 0] += np.bincount(fi["tids_a"], weights=cov_a, minlength=n_clusters)
                t_all[:, 1] += np.bincount(pids_a, weights=fp_w_a, minlength=n_clusters)
                t_all[:, 2] += np.bincount(fi["tids_a"], weights=1.0 - cov_a, minlength=n_clusters)
                t_all[:, 3] += np.bincount(fi["tids_a"], weights=np.ones(fi["n_truth_a"]), minlength=n_clusters)

                tp_a += tp_fold_a
                fp_a += fp_fold_a
                fn_a += fn_fold_a
                nt_a += fi["n_truth_a"]
                f_dtis_all.append(float(dti_fold_a))

                # >1 km far pixels in fold
                p_f = emitted & fi["act_far"]
                dp_far = ndi.distance_transform_edt(~p_f).astype(np.float32) if p_f.any() else np.full(grid.shape, 999.0, np.float32)
                cov_f = kernel(dp_far[fi["ty_f"], fi["tx_f"]])
                tp_fold_f = float(cov_f.sum())
                fn_fold_f = float(fi["n_truth_f"]) - tp_fold_f
                py_f, px_f = np.nonzero(p_f)
                fp_w_f = 1.0 - fi["k_dg_f"][py_f, px_f].astype(np.float64)
                fp_fold_f = float(fp_w_f.sum())
                dti_fold_f = dti_from_components(tp_fold_f, fp_fold_f, fn_fold_f)

                pids_f = (py_f // block_side) * ncols + (px_f // block_side)
                t_far[:, 0] += np.bincount(fi["tids_f"], weights=cov_f, minlength=n_clusters)
                t_far[:, 1] += np.bincount(pids_f, weights=fp_w_f, minlength=n_clusters)
                t_far[:, 2] += np.bincount(fi["tids_f"], weights=1.0 - cov_f, minlength=n_clusters)
                t_far[:, 3] += np.bincount(fi["tids_f"], weights=np.ones(fi["n_truth_f"]), minlength=n_clusters)

                tp_f += tp_fold_f
                fp_f += fp_fold_f
                fn_f += fn_fold_f
                nt_f += fi["n_truth_f"]
                f_dtis_far.append(float(dti_fold_f))

                per_fold[fi["name"]] = dict(
                    dti_all=dti_fold_a,
                    dti_far_gt_1km=dti_fold_f,
                    emitted=int(py_a.size),
                    emitted_far_gt_1km=int(py_f.size),
                    n_truth=fi["n_truth_a"],
                    n_truth_far=fi["n_truth_f"],
                )

            dti_a = dti_from_components(tp_a, fp_a, fn_a)
            dti_f = dti_from_components(tp_f, fp_f, fn_f)
            em_dist = dist_vis[emitted]
            terms_all[f"stage{k}"] = t_all
            terms_far[f"stage{k}"] = t_far
            fold_dtis_all_by_stage[k] = f_dtis_all
            fold_dtis_far_by_stage[k] = f_dtis_far
            per_stage[f"stage{k}"] = dict(
                all=dict(
                    evidence_class="HOLDOUT-DTI",
                    evaluator_version="gems57-pooled-hide-v2",
                    dti=dti_a,
                    tpw=tp_a,
                    fpw=fp_a,
                    fnw=fn_a,
                    withheld_positive_pixels=int(nt_a),
                    emitted_dots=int(emitted.sum()),
                    coverage=float(tp_a / max(nt_a, 1)),
                    marginal_bar=marginal_inclusion_threshold(dti_a),
                ),
                far_gt_1km=dict(
                    evidence_class="HOLDOUT-DTI",
                    evaluator_version="gems57-pooled-hide-v2",
                    dti=dti_f,
                    tpw=tp_f,
                    fpw=fp_f,
                    fnw=fn_f,
                    withheld_positive_pixels=int(nt_f),
                    emitted_far_dots=int((emitted & far_mask).sum()),
                    coverage=float(tp_f / max(nt_f, 1)),
                ),
                dot_distance_quantiles_m=dict(
                    p25=float(np.quantile(em_dist, 0.25) * 100.0) if em_dist.size else 0.0,
                    p50=float(np.quantile(em_dist, 0.50) * 100.0) if em_dist.size else 0.0,
                    p75=float(np.quantile(em_dist, 0.75) * 100.0) if em_dist.size else 0.0,
                ),
                per_fold=per_fold,
            )
        for k in (2, 3):
            sk = f"stage{k}"
            per_stage[sk]["all"]["delta_vs_s1"] = {
                **paired_block_bootstrap(terms_all, sk, "stage1"),
                **_outer_block_se_stats(fold_dtis_all_by_stage[k], fold_dtis_all_by_stage[1]),
            }
            per_stage[sk]["far_gt_1km"]["delta_vs_s1"] = {
                **paired_block_bootstrap(terms_far, sk, "stage1"),
                **_outer_block_se_stats(fold_dtis_far_by_stage[k], fold_dtis_far_by_stage[1]),
            }
        per_stage["stage3"]["all"]["delta_vs_s2"] = {
            **paired_block_bootstrap(terms_all, "stage3", "stage2"),
            **_outer_block_se_stats(fold_dtis_all_by_stage[3], fold_dtis_all_by_stage[2]),
        }
        per_stage["stage3"]["far_gt_1km"]["delta_vs_s2"] = {
            **paired_block_bootstrap(terms_far, "stage3", "stage2"),
            **_outer_block_se_stats(fold_dtis_far_by_stage[3], fold_dtis_far_by_stage[2]),
        }
        results[str(B)] = per_stage
    return results


def run_leakage_canary(
    grid,
    folds,
    phys: F.PhysicalFeatures,
    eval_ctx: list[np.ndarray],
    p1_vec: np.ndarray,
) -> dict:
    """Test every single feature alone on the withheld holdout truth (protocol rule 4)."""
    truth_all = np.zeros(grid.shape, bool)
    active_all = np.zeros(grid.shape, bool)
    for f in folds:
        act = grid.footprint & f["region"] & ~f["masked_known"]
        active_all |= act
        truth_all |= f["truth"] & act

    act_fp = active_all[grid.footprint]
    act_fp_idx = np.flatnonzero(act_fp)
    y_fp = truth_all[grid.footprint][act_fp].astype(np.int8)
    rng = np.random.default_rng(2026068)
    pos_idx = np.flatnonzero(y_fp == 1)
    neg_pool = np.flatnonzero(y_fp == 0)
    neg_idx = rng.choice(neg_pool, size=min(neg_pool.size, pos_idx.size * 10), replace=False)
    sub = np.concatenate([pos_idx, neg_idx])
    sub_fp_idx = act_fp_idx[sub]
    y_sub = y_fp[sub]

    s1_aucs = {}
    for j, name in enumerate(phys.names1):
        col = np.nan_to_num(phys.X1[sub_fp_idx, j], nan=0.0)
        auc = float(roc_auc_score(y_sub, col))
        s1_aucs[name] = round(max(auc, 1.0 - auc), 4)

    pmap1 = AC.to_map(p1_vec, grid.footprint)
    iso_cache = AC.precompute_iso_smooths(pmap1)
    fp_rows, fp_cols = np.nonzero(grid.footprint)
    X2_sub = AC.context_matrix_rows(
        pmap1, iso_cache, sub_fp_idx, fp_rows.astype(np.int32), fp_cols.astype(np.int32), phys, eval_ctx
    )
    del iso_cache, pmap1
    s2_names = AC.stage2_feature_names()
    s2_aucs = {}
    for j, name in enumerate(s2_names):
        col = np.nan_to_num(X2_sub[:, j], nan=0.0)
        auc = float(roc_auc_score(y_sub, col))
        s2_aucs[name] = round(max(auc, 1.0 - auc), 4)

    max_s1_name = max(s1_aucs, key=s1_aucs.get)
    max_s2_name = max(s2_aucs, key=s2_aucs.get)
    max_auc = max(s1_aucs[max_s1_name], s2_aucs[max_s2_name])
    return dict(
        threshold_auc=0.90,
        all_below_threshold=bool(max_auc < 0.90),
        max_stage1_feature=max_s1_name,
        max_stage1_auc=s1_aucs[max_s1_name],
        max_stage2_feature=max_s2_name,
        max_stage2_auc=s2_aucs[max_s2_name],
        stage1_single_feature_auc=s1_aucs,
        stage2_single_feature_auc=s2_aucs,
    )


def main() -> None:
    log("loading grid + physical + 2m LiDAR scarp + GeoDAWN extension features...")
    grid = load_grid()
    phys = F.build_physical_features(grid, use_cache=True)
    log(f"Stage-1 physical features ready: {phys.X1.shape} ({len(phys.names1)} features)")

    folds, _ = buffered_component_draw(grid, seed=20, hide_frac=0.20, buffer_px=3)
    vis = folds[0]["visible"]
    hidden_all = folds[0]["hidden_all"]
    collar = ndi.distance_transform_edt(~hidden_all) <= 3
    dist_vis = ndi.distance_transform_edt(~vis).astype(np.float32)

    preds_cache = ROOT / ".cache" / "exp3_outer_preds.npz"
    imps_cache = ROOT / ".cache" / "exp3_outer_imps.json"
    lane = Lane(grid, phys, vis, collar)
    if preds_cache.exists() and "--recompute" not in sys.argv:
        log(f"loading cached outer predictions from {preds_cache}...")
        d = np.load(preds_cache)
        p_outer = {1: d["p1"], 2: d["p2"], 3: d["p3"]}
        imps = json.loads(imps_cache.read_text()) if imps_cache.exists() else None
    else:
        p_outer, imps_raw = lane.nested(n_stages=3)
        np.savez(preds_cache, p1=p_outer[1], p2=p_outer[2], p3=p_outer[3])
        imps = {str(k): v for k, v in imps_raw.items()}
        imps_cache.write_text(json.dumps(imps))

    log("running single-feature leakage canary across all Stage-1 and Stage-2 features...")
    canary = run_leakage_canary(grid, folds, phys, lane.eval_ctx, p_outer[1])
    log(
        f"Leakage canary: max Stage-1 AUC = {canary['max_stage1_auc']:.4f} ({canary['max_stage1_feature']}), "
        f"max Stage-2 AUC = {canary['max_stage2_auc']:.4f} ({canary['max_stage2_feature']}), "
        f"all_below_0.90 = {canary['all_below_threshold']}"
    )
    if not canary["all_below_threshold"]:
        raise AssertionError("Leakage canary failed: single-feature holdout AUC >= 0.90!")

    p_maps = {k: AC.to_map(v, grid.footprint) for k, v in p_outer.items()}
    log("scoring outer predictions across budgets...")
    scored = score_stage_maps(grid, folds, p_maps, dist_vis)

    for B, s in scored.items():
        d1 = s["stage1"]["all"]["dti"]
        d2 = s["stage2"]["all"]["dti"]
        d3 = s["stage3"]["all"]["dti"]
        f1 = s["stage1"]["far_gt_1km"]["dti"]
        f2 = s["stage2"]["far_gt_1km"]["dti"]
        f3 = s["stage3"]["far_gt_1km"]["dti"]
        ci_all = s["stage3"]["all"]["delta_vs_s1"]["ci95"]
        ci_far = s["stage3"]["far_gt_1km"]["delta_vs_s1"]["ci95"]
        t_all = s["stage3"]["all"]["delta_vs_s1"]["t_outer_block"]
        t_far = s["stage3"]["far_gt_1km"]["delta_vs_s1"]["t_outer_block"]
        med_m = s["stage3"]["dot_distance_quantiles_m"]["p50"]
        log(
            f"B={B:>6s} | all: S1={d1:.4f} S2={d2:.4f} S3={d3:.4f} "
            f"(d3-1={d3-d1:+.4f} CI[{ci_all[0]:+.4f},{ci_all[1]:+.4f}] t={t_all:.2f}) | "
            f">1km: S1={f1:.4f} S2={f2:.4f} S3={f3:.4f} "
            f"(d3-1={f3-f1:+.4f} CI[{ci_far[0]:+.4f},{ci_far[1]:+.4f}] t={t_far:.2f}) | "
            f"med_dist={med_m:.0f}m"
        )

    avg_imp = {}
    if imps:
        for k, lst in imps.items():
            keys = lst[0].keys()
            avg_imp[f"stage{k}"] = dict(
                sorted(
                    ((nm, float(np.mean([d[nm] for d in lst]))) for nm in keys),
                    key=lambda kv: -kv[1],
                )[:25]
            )

    best_B = max(scored.keys(), key=lambda b: scored[b]["stage3"]["all"]["dti"])
    best = scored[best_B]
    gate_all_ci = bool(
        best["stage3"]["all"]["delta_vs_s1"]["ci95_excludes_zero"]
        or best["stage2"]["all"]["delta_vs_s1"]["ci95_excludes_zero"]
    )
    gate_far_ci = bool(
        best["stage3"]["far_gt_1km"]["delta_vs_s1"]["ci95_excludes_zero"]
        or best["stage2"]["far_gt_1km"]["delta_vs_s1"]["ci95_excludes_zero"]
    )
    gate_all_2se = bool(
        best["stage3"]["all"]["delta_vs_s1"]["beats_by_gt_2se"]
        or best["stage2"]["all"]["delta_vs_s1"]["beats_by_gt_2se"]
    )
    gate_far_2se = bool(
        best["stage3"]["far_gt_1km"]["delta_vs_s1"]["beats_by_gt_2se"]
        or best["stage2"]["far_gt_1km"]["delta_vs_s1"]["beats_by_gt_2se"]
    )

    out = dict(
        experiment="exp3_autocontext_scarp_ridge_hidedraw",
        evidence_class="HOLDOUT-DTI",
        evaluator_version="gems57-pooled-hide-v2",
        split_version="buffered-whole-components-loqo-v2",
        seed=20,
        fixes_irregularity=["IR-68-LEAK-01", "IR-68-BLOB-01"],
        stage1_features=phys.names1,
        stage2_features=AC.stage2_feature_names(),
        leakage_canary=canary,
        best_budget=int(best_B),
        gate_all_pixels_gt_2se=gate_all_ci,
        gate_far_gt_1km_gt_2se=gate_far_ci,
        gate_outer_block_all_gt_2se=gate_all_2se,
        gate_outer_block_far_gt_2se=gate_far_2se,
        gate_passed=bool(gate_all_ci and gate_far_ci),
        by_budget=scored,
        top25_feature_importance_gain=avg_imp,
    )
    out_path = EVIDENCE / "exp3_autocontext_scarp_ridge_hidedraw.json"
    out_path.write_text(json.dumps(out, indent=2))
    log(f"wrote {out_path} (best_B={best_B}, gate_passed={out['gate_passed']})")


if __name__ == "__main__":
    main()
