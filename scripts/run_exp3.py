#!/usr/bin/env python3
"""Experiment 3 (pre-registered in evidence/exp3_preregistration.json).

Confirmatory auto-context run on a FRESH evaluation hide-draw (seed 31):
stage-1 = 44 channels (X1ext), offsets (1,2,3,5,8,12,15), 2-seed bagging,
180 rounds.  Candidate fixed in advance: stage 3 @ 38,000 dots.  Evaluation
machinery is imported unchanged from scripts/run_exp2.py.
Outputs: evidence/exp3_autocontext_ext_bagged.json, .cache/exp3/p{1,2,3}_outer.npy.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems68 import autocontext as AC, emission as E, evaluate_holdout as EH, grid as G  # noqa: E402
from gems68.experiment import Lane, log  # noqa: E402
from gems68.holdout import quadrant_ids  # noqa: E402

spec = importlib.util.spec_from_file_location("run_exp2", ROOT / "scripts" / "run_exp2.py")
exp2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp2)

# ---- Exp3 recipe (must match evidence/exp3_preregistration.json) ----
AC.OFFSETS = (1, 2, 3, 5, 8, 12, 15)
AC.MODEL_SEEDS = (68, 71)
AC.N_ROUNDS = 180
EVAL_SEED, TRAIN_SEEDS = 31, (31, 32, 33)
N_STAGES = 3
CACHE = ROOT / ".cache" / "exp3"
CAND_STAGE, CAND_BUDGET = 3, 38000
EXP_NAME, OUT_NAME, PREREG = ("exp3_autocontext_ext_bagged_nested_blocked_cv",
                              "exp3_autocontext_ext_bagged.json",
                              "evidence/exp3_preregistration.json")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    ap.add_argument("--train-seeds", default=",".join(map(str, TRAIN_SEEDS)))
    ap.add_argument("--candidate-stage", type=int, default=CAND_STAGE)
    ap.add_argument("--candidate-budget", type=int, default=CAND_BUDGET)
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--out", default=OUT_NAME)
    ap.add_argument("--experiment", default=EXP_NAME)
    ap.add_argument("--prereg", default=PREREG)
    a = ap.parse_args()
    eval_seed = a.eval_seed
    train_seeds = tuple(int(s) for s in a.train_seeds.split(","))
    cache = Path(a.cache)
    t0 = time.time()
    cache.mkdir(parents=True, exist_ok=True)
    g = G.load_grid()
    X1 = np.load(ROOT / ".cache/features/X1ext.npy", mmap_mode="r")
    names1 = json.loads((ROOT / ".cache/features/X1ext_names.json").read_text())["names"]
    phys = list(np.load(ROOT / ".cache/features/phys.npy"))
    lane = Lane(g, np.ascontiguousarray(X1), phys, quadrant_ids(g.footprint),
                eval_seed=eval_seed, train_seeds=train_seeds, stage1_names=names1)
    log(f"lane ready: X1 {X1.shape}, stage2 features {len(lane.names2)}")
    if all((cache / f"p{k}_outer.npy").is_file() for k in range(1, N_STAGES + 1)) and (cache / "imps.json").is_file():
        p_outer = {k: np.load(cache / f"p{k}_outer.npy") for k in range(1, N_STAGES + 1)}
        imps = {int(k): v for k, v in json.loads((cache / "imps.json").read_text()).items()}
        log("resumed from cache")
    else:
        p_outer, imps = lane.nested(N_STAGES)
        for k, v in p_outer.items():
            np.save(cache / f"p{k}_outer.npy", v)
        (cache / "imps.json").write_text(json.dumps(imps))
    can = exp2.canary(lane, p_outer)
    results, gate = exp2.evaluate_maps(lane, p_outer)

    def mean_imp(lst):
        return {k: float(np.mean([d[k] for d in lst])) for k in lst[0]}

    cand = gate[f"{a.candidate_budget}_stage{a.candidate_stage}"]
    ev = dict(
        experiment=a.experiment,
        preregistration=a.prereg,
        eval_seed=eval_seed, train_seeds=list(train_seeds),
        evaluator_version=EH.VERSION, evidence_class="HOLDOUT-DTI",
        holdout=dict(split=lane.folds[0]["receipt"]["split_version"],
                     receipts=[f["receipt"] for f in lane.folds],
                     withheld_positive_pixels=int(sum(f["receipt"]["withheld_positive_pixels"] for f in lane.folds))),
        model=dict(lgb_params=AC.LGB_PARAMS, rounds=AC.N_ROUNDS, buffer_px=AC.BUFFER_PX,
                   offsets_px=list(AC.OFFSETS), model_seeds=list(AC.MODEL_SEEDS),
                   neg_per_pos=AC.NEG_PER_POS, max_train_rows=AC.MAX_TRAIN_ROWS,
                   n_stages=N_STAGES, stage1_names=names1),
        emission=dict(exclude_px=E.EXCLUDE_PX, nms=5, budgets=list(exp2.BUDGETS), far_px=exp2.FAR_PX),
        preregistered_candidate=dict(stage=a.candidate_stage, budget=a.candidate_budget, gate=cand),
        canary=can, feature_importance_gain={str(k): mean_imp(v) for k, v in imps.items()},
        results=results, gate=gate, seconds=round(time.time() - t0, 1))
    (ROOT / "evidence" / a.out).write_text(
        json.dumps(ev, indent=2, default=float))
    log(f"wrote evidence/{a.out}")
    log(f"PREREGISTERED stage{a.candidate_stage}@{a.candidate_budget} passes={cand['passes']} "
        f"dti {cand['dti_stage1']:.4f}->{cand['dti_cand']:.4f} "
        f"far {cand['far_dti_stage1']:.4f}->{cand['far_dti_cand']:.4f}")
    flagged = [k for k, v in can.items() if v["flag"]]
    log("canary flagged:", flagged)
    return 0


if __name__ == "__main__":
    sys.exit(main())
