#!/usr/bin/env python3
"""Experiment 3 (AC68-N1): catalogue-context-free auto-context, preregistered
in evidence/exp3_preregistration.json BEFORE this run.

Identical instrument to experiment 2 (whole-component hide-and-recover,
quota budgets, valid-masked >1 km stratum, paired spatial-block bootstrap);
the ONLY change is ``Lane(catalogue_context=False)`` -- stage>=2 receives no
catalogue-derived features.  Evaluation and canary code are imported from
scripts/run_exp2.py so the instrument cannot drift between experiments.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_exp2 as R2  # noqa: E402  (evaluate_maps, canary, BUDGETS, FAR_PX)
from gems68 import autocontext_738 as AC, emission as E, evaluate_holdout as EH, grid as G  # noqa: E402
from gems68.experiment_738 import Lane, log  # noqa: E402
from gems68.holdout import quadrant_ids  # noqa: E402

CACHE = ROOT / ".cache" / "exp3_ccfree"
N_STAGES = 3


def main() -> int:
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    g = G.load_grid()
    X1 = np.ascontiguousarray(np.load(ROOT / ".cache/features/X1.npy", mmap_mode="r"))
    phys = list(np.load(ROOT / ".cache/features/phys.npy"))
    lane = Lane(g, X1, phys, quadrant_ids(g.footprint), catalogue_context=False)
    assert not any(n.startswith("cat_") for n in lane.names2), lane.names2
    log(f"lane ready (AC68-N1, catalogue_context=False, {len(lane.names2)} stage>=2 features)")
    if all((CACHE / f"p{k}_outer.npy").is_file() for k in range(1, N_STAGES + 1)) and (CACHE / "imps.json").is_file():
        p_outer = {k: np.load(CACHE / f"p{k}_outer.npy") for k in range(1, N_STAGES + 1)}
        imps = {int(k): v for k, v in json.loads((CACHE / "imps.json").read_text()).items()}
        log("resumed outer maps from cache")
    else:
        p_outer, imps = lane.nested(N_STAGES)
        for k, v in p_outer.items():
            np.save(CACHE / f"p{k}_outer.npy", v)
        (CACHE / "imps.json").write_text(json.dumps(imps))
    can = R2.canary(lane, p_outer)
    results, gate = R2.evaluate_maps(lane, p_outer)

    def mean_imp(lst):
        return {k: float(np.mean([d[k] for d in lst])) for k in lst[0]} if lst and lst[0] else {}

    import lightgbm, scipy, sklearn
    ev = dict(
        experiment="exp3_autocontext_ccfree_AC68-N1", eval_seed=20,
        train_seeds=[d["seed"] for d in lane.draws],
        evaluator_version=EH.VERSION, evidence_class="HOLDOUT-DTI",
        preregistration="evidence/exp3_preregistration.json",
        ablation=dict(catalogue_context=False,
                      dropped_features=["cat_dist", "cat_strike_cos2", "cat_strike_sin2",
                                        "cat_dens_1km", "cat_dens_3km"],
                      stage_ge2_feature_names=lane.names2),
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
        canary=can, feature_importance_gain={k: mean_imp(v) for k, v in imps.items()},
        results=results, gate=gate, seconds=round(time.time() - t0, 1))
    (ROOT / "evidence" / "exp3_autocontext_ccfree.json").write_text(json.dumps(ev, indent=2, default=float))
    log("wrote evidence/exp3_autocontext_ccfree.json")
    for k, v in gate.items():
        log(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
