#!/usr/bin/env python3
"""Experiment 2: auto-context with hide-draw labels (fix for IR-68-LEAK-01).

Same outer LOQO hide-and-recover evaluation, same emission rule and budgets as
experiment 1; only the stage>=2 training protocol changed (see experiment.py).
Writes evidence/exp2_autocontext_hidedraw.json and caches outer maps.
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
from gems68 import autocontext as AC, emission as E, evaluate_holdout as EH, grid as G  # noqa: E402
from gems68.experiment import Lane, log  # noqa: E402
from gems68.holdout import quadrant_ids  # noqa: E402

BUDGETS = (15_000, 25_000, 38_000, 50_000, 70_000)
FAR_PX = 10
CACHE = ROOT / ".cache" / "exp2"
N_STAGES = 3


def evaluate_maps(lane: Lane, p_outer: dict, budgets=BUDGETS, far_px=FAR_PX) -> tuple[dict, dict]:
    fp = lane.fp
    folds = lane.folds
    masked_known = folds[0]["masked_known"]
    allowed = ~E.exclusion_mask(lane.eval_visible) & fp
    far = ndi.distance_transform_edt(~masked_known) > far_px
    maps = {k: AC.to_map(v, fp) for k, v in p_outer.items()}
    stages = sorted(maps)
    results, gate = {}, {}
    for n_total in budgets:
        terms = {"all": {k: [] for k in stages}, "far": {k: [] for k in stages}}
        per_fold = {}
        for f in folds:
            n_q = E.quota(n_total, fp, f["region"])
            for k in stages:
                pred = E.emit_dots(maps[k], allowed & f["region"], n_q).astype(np.float32)
                r_all, t_all = EH.evaluate(pred, f, fp, block_side=200, global_shape=fp.shape)
                r_far, t_far = EH.evaluate(pred, f, fp & far, block_side=200, global_shape=fp.shape)
                terms["all"][k].append(t_all); terms["far"][k].append(t_far)
                per_fold[f"{f['name']}_stage{k}"] = dict(
                    dti=r_all["dti"], tpw=r_all["tpw"], fpw=r_all["fpw"], n_truth=r_all["n_truth"],
                    n_emitted=r_all["n_emitted"], far_dti=r_far["dti"], far_n_truth=r_far["n_truth"])
        out = {"per_fold": per_fold}
        for scope in ("all", "far"):
            arms = {f"stage{k}": np.sum(terms[scope][k], axis=0) for k in stages}
            out[scope] = {f"cand_stage{k}": EH.pooled_summary(arms, candidate=f"stage{k}") for k in stages if k > 1}
        results[str(n_total)] = out
        for k in stages:
            if k == 1:
                continue
            d_all = out["all"][f"cand_stage{k}"]["paired_differences"]["stage1"]
            d_far = out["far"][f"cand_stage{k}"]["paired_differences"]["stage1"]
            gate[f"{n_total}_stage{k}"] = dict(
                dti_stage1=out["all"][f"cand_stage{k}"]["scores"]["stage1"]["dti"],
                dti_cand=out["all"][f"cand_stage{k}"]["scores"][f"stage{k}"]["dti"],
                delta_all=d_all["delta"], ci_all=d_all["ci95"],
                far_dti_stage1=out["far"][f"cand_stage{k}"]["scores"]["stage1"]["dti"],
                far_dti_cand=out["far"][f"cand_stage{k}"]["scores"][f"stage{k}"]["dti"],
                delta_far=d_far["delta"], ci_far=d_far["ci95"],
                passes=bool(d_all["ci95"][0] > 0 and d_far["ci95"][0] > 0))
        s = out["all"]["cand_stage2"]["scores"]
        log(f"budget {n_total}: " + " ".join(f"{a}={s[a]['dti']:.4f}" for a in s)
            + f" | far: " + " ".join(f"{a}={out['far']['cand_stage2']['scores'][a]['dti']:.4f}" for a in s))
    return results, gate


def canary(lane: Lane, p_outer: dict) -> dict:
    fp = lane.fp
    truth_all = np.zeros(fp.shape, bool); active = np.zeros(fp.shape, bool)
    for f in lane.folds:
        truth_all |= f["truth"]; active |= f["region"]
    active &= fp & ~lane.folds[0]["masked_known"]
    rows_act = np.flatnonzero(active[fp])
    t_vec = truth_all[fp][rows_act]
    sub = np.random.default_rng(1).choice(rows_act, size=min(600_000, len(rows_act)), replace=False)
    sub = np.union1d(sub, rows_act[t_vec]); t_sub = truth_all[fp][sub]
    AC.context_matrix(AC.to_map(p_outer[1], fp), fp, lane.phys, lane.eval_ctx, out=lane.X2)
    out = {}
    for j, nm in enumerate(lane.names2):
        auc = roc_auc_score(t_sub, lane.X2[sub, j])
        out[nm] = dict(auc=float(auc), flag=bool(max(auc, 1 - auc) > 0.90))
    for k, v in p_outer.items():
        auc = roc_auc_score(t_sub, v[sub]); out[f"stage{k}_prob"] = dict(auc=float(auc), flag=bool(auc > 0.90))
    return out


def main() -> int:
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    g = G.load_grid()
    X1 = np.ascontiguousarray(np.load(ROOT / ".cache/features/X1.npy", mmap_mode="r"))
    phys = list(np.load(ROOT / ".cache/features/phys.npy"))
    lane = Lane(g, X1, phys, quadrant_ids(g.footprint))
    log("lane ready")
    if all((CACHE / f"p{k}_outer.npy").is_file() for k in range(1, N_STAGES + 1)) and (CACHE / "imps.json").is_file():
        p_outer = {k: np.load(CACHE / f"p{k}_outer.npy") for k in range(1, N_STAGES + 1)}
        imps = {int(k): v for k, v in json.loads((CACHE / "imps.json").read_text()).items()}
        log("resumed from cache")
    else:
        p_outer, imps = lane.nested(N_STAGES)
        for k, v in p_outer.items():
            np.save(CACHE / f"p{k}_outer.npy", v)
        (CACHE / "imps.json").write_text(json.dumps(imps))
    can = canary(lane, p_outer)
    results, gate = evaluate_maps(lane, p_outer)

    def mean_imp(lst):
        return {k: float(np.mean([d[k] for d in lst])) for k in lst[0]}

    ev = dict(
        experiment="exp2_autocontext_hidedraw_nested_blocked_cv", eval_seed=20, train_seeds=[d["seed"] for d in lane.draws],
        evaluator_version=EH.VERSION, evidence_class="HOLDOUT-DTI",
        holdout=dict(split=lane.folds[0]["receipt"]["split_version"], receipts=[f["receipt"] for f in lane.folds],
                     withheld_positive_pixels=int(sum(f["receipt"]["withheld_positive_pixels"] for f in lane.folds))),
        model=dict(lgb_params=AC.LGB_PARAMS, rounds=AC.N_ROUNDS, buffer_px=AC.BUFFER_PX, offsets_px=AC.OFFSETS,
                   neg_per_pos=AC.NEG_PER_POS, max_train_rows=AC.MAX_TRAIN_ROWS, n_stages=N_STAGES),
        emission=dict(exclude_px=E.EXCLUDE_PX, nms=5, budgets=BUDGETS, far_px=FAR_PX),
        canary=can, feature_importance_gain={k: mean_imp(v) for k, v in imps.items()},
        results=results, gate=gate, seconds=round(time.time() - t0, 1))
    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / "evidence" / "exp2_autocontext_hidedraw.json").write_text(json.dumps(ev, indent=2, default=float))
    log("wrote evidence/exp2_autocontext_hidedraw.json")
    for k, v in gate.items():
        log(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
