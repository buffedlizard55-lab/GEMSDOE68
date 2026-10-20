#!/usr/bin/env python3
"""Experiment 4 (AC68-N2): stage-1 external-channel enrichment, catalogue
context restored.  Preregistered in evidence/exp4_preregistration.json.

Same instrument/evaluation as exp2/exp3 (imported from scripts/run_exp2.py);
X1 is the memory-mapped 46-column matrix from scripts/build_features_ext.py.
Adds a stage-1 single-column leakage canary (protocol rule 4) on top of the
shared stage>=2 canary.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_exp2 as R2  # noqa: E402
from gems68 import autocontext as AC, emission as E, evaluate_holdout as EH, grid as G  # noqa: E402
from gems68.experiment import Lane, log  # noqa: E402
from gems68.holdout import quadrant_ids  # noqa: E402

CACHE = ROOT / ".cache" / "exp4_ext"
N_STAGES = 3


def canary_stage1(lane: Lane, X1: np.ndarray, names: list[str]) -> dict:
    """Single-column AUC of each stage-1 feature: withheld positives in the
    evaluation draw vs a seeded subsample of active negatives."""
    fp = lane.fp
    truth_all = np.zeros(fp.shape, bool)
    active = np.zeros(fp.shape, bool)
    for f in lane.folds:
        truth_all |= f["truth"]
        active |= f["region"]
    active &= fp & ~lane.folds[0]["masked_known"]
    rows_act = np.flatnonzero(active[fp])
    t_vec = truth_all[fp][rows_act]
    sub = np.random.default_rng(1).choice(rows_act, size=min(600_000, len(rows_act)), replace=False)
    sub = np.union1d(sub, rows_act[t_vec])
    t_sub = truth_all[fp][sub]
    out = {}
    sub_sorted = np.sort(sub)
    pos_in_sorted = np.searchsorted(sub_sorted, sub)  # keep X1 memmap reads sequential-ish
    Xsub = np.asarray(X1[sub_sorted])[pos_in_sorted]
    for j, nm in enumerate(names):
        auc = roc_auc_score(t_sub, Xsub[:, j])
        out[nm] = dict(auc=float(auc), flag=bool(max(auc, 1 - auc) > 0.90))
    return out


def main() -> int:
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    g = G.load_grid()
    names1 = json.loads((ROOT / ".cache/features/stage1_ext_names.json").read_text())
    X1 = np.load(ROOT / ".cache/features/X1_ext.npy", mmap_mode="r")
    phys = list(np.load(ROOT / ".cache/features/phys.npy"))
    lane = Lane(g, X1, phys, quadrant_ids(g.footprint), catalogue_context=True,
                stage1_names=names1)
    log(f"lane ready (AC68-N2, stage1 {len(names1)} cols incl. 20 external, catalogue_context=True)")
    if all((CACHE / f"p{k}_outer.npy").is_file() for k in range(1, N_STAGES + 1)) and (CACHE / "imps.json").is_file():
        p_outer = {k: np.load(CACHE / f"p{k}_outer.npy") for k in range(1, N_STAGES + 1)}
        imps = {int(k): v for k, v in json.loads((CACHE / "imps.json").read_text()).items()}
        log("resumed outer maps from cache")
    else:
        p_outer, imps = lane.nested(N_STAGES)
        for k, v in p_outer.items():
            np.save(CACHE / f"p{k}_outer.npy", v)
        (CACHE / "imps.json").write_text(json.dumps(imps))
    can1 = canary_stage1(lane, X1, names1)
    can = R2.canary(lane, p_outer)
    results, gate = R2.evaluate_maps(lane, p_outer)

    def mean_imp(lst):
        return {k: float(np.mean([d[k] for d in lst])) for k in lst[0]} if lst and lst[0] else {}

    import lightgbm, scipy, sklearn
    ev = dict(
        experiment="exp4_autocontext_stage1_external_AC68-N2", eval_seed=20,
        train_seeds=[d["seed"] for d in lane.draws],
        evaluator_version=EH.VERSION, evidence_class="HOLDOUT-DTI",
        preregistration="evidence/exp4_preregistration.json",
        stage1_feature_names=names1,
        external_feature_names=names1[len(names1) - 20:],
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         scipy=scipy.__version__, sklearn=sklearn.__version__,
                         lightgbm=lightgbm.__version__),
        holdout=dict(split=lane.folds[0]["receipt"]["split_version"],
                     receipts=[f["receipt"] for f in lane.folds],
                     withheld_positive_pixels=int(sum(f["receipt"]["withheld_positive_pixels"]
                                                       for f in lane.folds))),
        model=dict(lgb_params=AC.LGB_PARAMS, rounds=AC.N_ROUNDS, buffer_px=AC.BUFFER_PX,
                   offsets_px=AC.OFFSETS, neg_per_pos=AC.NEG_PER_POS,
                   max_train_rows=AC.MAX_TRAIN_ROWS, n_stages=N_STAGES),
        emission=dict(exclude_px=E.EXCLUDE_PX, nms=5, budgets=R2.BUDGETS, far_px=R2.FAR_PX),
        canary_stage1=can1, canary=can,
        feature_importance_gain={k: mean_imp(v) for k, v in imps.items()},
        results=results, gate=gate, seconds=round(time.time() - t0, 1))
    (ROOT / "evidence" / "exp4_autocontext_ext.json").write_text(json.dumps(ev, indent=2, default=float))
    log("wrote evidence/exp4_autocontext_ext.json")
    for k, v in gate.items():
        log(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
