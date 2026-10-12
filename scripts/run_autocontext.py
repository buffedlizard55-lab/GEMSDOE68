#!/usr/bin/env python3
"""Experiment 1: nested blocked CV of auto-context stacking on the hide-and-recover holdout.

Outer loop  : leave-one-quadrant-out (4 folds, buffered-whole-components-loqo-v2 draw).
Inner loop  : stage-k OOF probabilities on the three training quadrants are
              produced leave-one-training-quadrant-out with a 12 px buffer; the
              outer quadrant is predicted by a model fit on all three.
Stages      : 1 (raw+physics) -> 2 (auto-context) -> 3 (auto-context iterated).
Emission    : identical rule for every stage (exclude <=2 px from the visible
              catalogue, 5x5 local maxima, top-N dots) at several global budgets.
Gate        : stage 2 (and 3) must beat stage 1 on the pooled outer blocks by
              > 2 SE (paired 20 km-block bootstrap), including on pixels > 1 km
              from any visible mapped fault.
Canary      : single-feature AUC of every stage-2 input on the outer truth.

Writes evidence/exp1_autocontext.json and .cache/exp1/p{1,2,3}_outer.npy.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems68 import autocontext as AC, emission as E, evaluate_holdout as EH, features as F, grid as G  # noqa: E402
from gems68.holdout import FOLD_NAMES, buffered_component_draw  # noqa: E402

BUDGETS = (15_000, 25_000, 38_000, 50_000, 70_000)
SEED = 20
CACHE = ROOT / ".cache" / "exp1"
FAR_PX = 10  # "> 1 km from any mapped fault"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main() -> int:
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    g = G.load_grid()
    fp = g.footprint
    folds, quad_map = buffered_component_draw(g, seed=SEED)
    visible = folds[0]["visible"]
    masked_known = folds[0]["masked_known"]
    y = visible[fp].astype(np.int8)                # PU labels: visible catalogue only
    q = np.load(ROOT / ".cache/features/quad.npy")
    X1 = np.ascontiguousarray(np.load(ROOT / ".cache/features/X1.npy", mmap_mode="r"))
    phys = list(np.load(ROOT / ".cache/features/phys.npy"))
    N = len(y)
    log("loaded", X1.shape, "positives(visible)", int(y.sum()))

    dist_to_block = {b: ndi.distance_transform_edt(quad_map != b)[fp] for b in range(4)}
    ctx_maps = F.catalogue_context(visible)        # visible-only context (rule 2)
    ctx_vecs = [c[fp] for c in ctx_maps]
    del ctx_maps
    allowed = ~E.exclusion_mask(visible) & fp
    far = ndi.distance_transform_edt(~masked_known) > FAR_PX
    rows_all = np.ones(N, bool)
    rng = np.random.default_rng(SEED)
    names2 = AC.context_names()

    p_outer = {k: np.full(N, np.nan, np.float32) for k in (1, 2, 3)}
    importances = {k: [] for k in (1, 2, 3)}
    X2 = np.empty((N, len(names2)), np.float32)
    cached = all((CACHE / f"p{k}_outer.npy").is_file() for k in (1, 2, 3)) and (CACHE / "importances.json").is_file()
    if cached:
        p_outer = {k: np.load(CACHE / f"p{k}_outer.npy") for k in (1, 2, 3)}
        importances = {int(k): v for k, v in json.loads((CACHE / "importances.json").read_text()).items()}
        log("resumed outer predictions from cache")
    for qo in (() if cached else range(4)):
        tq = [b for b in range(4) if b != qo]
        pred_idx = np.flatnonzero(q == qo)
        # ---- stage 1
        oof1, _ = AC.blocked_oof(X1, y, q, dist_to_block, rows_all, tq, rng, F.STAGE1_NAMES)
        tr = np.flatnonzero(np.isin(q, tq) & (dist_to_block[qo] > AC.BUFFER_PX))
        p1o, b1 = AC.fit_predict(X1, y, tr, pred_idx, rng, F.STAGE1_NAMES)
        importances[1].append(dict(zip(b1.feature_name(), b1.feature_importance("gain").tolist())))
        p_outer[1][pred_idx] = p1o
        p_prev = oof1.copy(); p_prev[pred_idx] = p1o
        log(f"outer {FOLD_NAMES[qo]} stage1 done")
        # ---- stages 2, 3
        for k in (2, 3):
            AC.context_matrix(AC.to_map(p_prev, fp), fp, phys, ctx_vecs, out=X2)
            oofk, _ = AC.blocked_oof(X2, y, q, dist_to_block, rows_all, tq, rng, names2)
            pko, bk = AC.fit_predict(X2, y, tr, pred_idx, rng, names2)
            importances[k].append(dict(zip(bk.feature_name(), bk.feature_importance("gain").tolist())))
            p_outer[k][pred_idx] = pko
            p_prev = oofk.copy(); p_prev[pred_idx] = pko
            log(f"outer {FOLD_NAMES[qo]} stage{k} done")
    if not cached:
        for k in (1, 2, 3):
            np.save(CACHE / f"p{k}_outer.npy", p_outer[k])
        (CACHE / "importances.json").write_text(json.dumps(importances))
    log("all stages done", round(time.time() - t0), "s")

    # ---- canary: single-feature AUC of stage-2 inputs on outer truth (stage-2 inputs
    # rebuilt from the OUTER stage-1 map, i.e. never in-fold)
    AC.context_matrix(AC.to_map(p_outer[1], fp), fp, phys, ctx_vecs, out=X2)
    truth_all = np.zeros(fp.shape, bool)
    for f in folds:
        truth_all |= f["truth"]
    active = np.zeros(fp.shape, bool)
    for f in folds:
        active |= f["region"]
    active &= fp & ~masked_known
    rows_act = np.flatnonzero(active[fp])
    t_vec = truth_all[fp][rows_act]
    sub = np.random.default_rng(1).choice(rows_act, size=min(600_000, len(rows_act)), replace=False)
    sub = np.union1d(sub, rows_act[t_vec])  # keep all positives
    t_sub = truth_all[fp][sub]
    canary = {}
    for j, nm in enumerate(names2):
        auc = roc_auc_score(t_sub, X2[sub, j])
        canary[nm] = dict(auc=float(auc), flag=bool(max(auc, 1 - auc) > 0.90))
    for k in (1, 2, 3):
        auc = roc_auc_score(t_sub, p_outer[k][sub])
        canary[f"stage{k}_prob"] = dict(auc=float(auc), flag=bool(auc > 0.90))
    del X2

    # ---- emission + evaluation at matched budgets
    maps = {k: AC.to_map(p_outer[k], fp) for k in (1, 2, 3)}
    results = {}
    for n_total in BUDGETS:
        terms = {"all": {k: [] for k in (1, 2, 3)}, "far": {k: [] for k in (1, 2, 3)}}
        per_fold = {}
        for f in folds:
            n_q = E.quota(n_total, fp, f["region"])
            for k in (1, 2, 3):
                dots = E.emit_dots(maps[k], allowed & f["region"], n_q)
                pred = dots.astype(np.float32)
                r_all, t_all = EH.evaluate(pred, f, fp, block_side=200, global_shape=fp.shape)
                r_far, t_far = EH.evaluate(pred, f, fp & far, block_side=200, global_shape=fp.shape)
                terms["all"][k].append(t_all); terms["far"][k].append(t_far)
                per_fold[f"{f['name']}_stage{k}"] = dict(
                    dti=r_all["dti"], tpw=r_all["tpw"], fpw=r_all["fpw"], n_truth=r_all["n_truth"],
                    n_emitted=r_all["n_emitted"], far_dti=r_far["dti"], far_n_truth=r_far["n_truth"])
        out = {"per_fold": per_fold}
        for scope in ("all", "far"):
            arms = {f"stage{k}": np.sum(terms[scope][k], axis=0) for k in (1, 2, 3)}
            out[scope] = {
                "cand_stage2": EH.pooled_summary(arms, candidate="stage2"),
                "cand_stage3": EH.pooled_summary(arms, candidate="stage3"),
            }
        results[str(n_total)] = out
        s2 = out["all"]["cand_stage2"]
        log(f"budget {n_total}: " + " ".join(f"{a}={s2['scores'][a]['dti']:.4f}" for a in s2["scores"])
            + f" | d(2-1)={s2['paired_differences']['stage1']['delta']:+.4f} CI {s2['paired_differences']['stage1']['ci95']}"
            + f" | far: " + " ".join(f"{a}={out['far']['cand_stage2']['scores'][a]['dti']:.4f}" for a in s2["scores"]))

    def mean_imp(lst):
        keys = lst[0].keys()
        return {k: float(np.mean([d[k] for d in lst])) for k in keys}

    # ---- gate decision
    gate = {}
    for n_total in BUDGETS:
        r = results[str(n_total)]
        for cand in ("cand_stage2", "cand_stage3"):
            d_all = r["all"][cand]["paired_differences"]["stage1"]
            d_far = r["far"][cand]["paired_differences"]["stage1"]
            gate[f"{n_total}_{cand}"] = dict(
                delta_all=d_all["delta"], ci_all=d_all["ci95"], delta_far=d_far["delta"], ci_far=d_far["ci95"],
                passes=bool(d_all["ci95"][0] > 0 and d_far["ci95"][0] > 0))
    evidence = dict(
        experiment="exp1_autocontext_nested_blocked_cv", seed=SEED, evaluator_version=EH.VERSION,
        holdout=dict(split=folds[0]["receipt"]["split_version"], receipts=[f["receipt"] for f in folds],
                     withheld_positive_pixels=int(sum(f["receipt"]["withheld_positive_pixels"] for f in folds))),
        model=dict(lgb_params=AC.LGB_PARAMS, rounds=AC.N_ROUNDS, buffer_px=AC.BUFFER_PX,
                   offsets_px=AC.OFFSETS, neg_per_pos=AC.NEG_PER_POS, max_train_rows=AC.MAX_TRAIN_ROWS),
        emission=dict(exclude_px=E.EXCLUDE_PX, nms=5, budgets=BUDGETS, far_px=FAR_PX),
        canary=canary, feature_importance_gain={k: mean_imp(v) for k, v in importances.items()},
        results=results, gate=gate, seconds=round(time.time() - t0, 1),
        evidence_class="HOLDOUT-DTI",
    )
    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / "evidence" / "exp1_autocontext.json").write_text(json.dumps(evidence, indent=2, default=float))
    log("wrote evidence/exp1_autocontext.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
