#!/usr/bin/env python3
"""Apply the preregistered Exp3 decision rule (evidence/exp3_preregistration.json)
to evidence/exp3_autocontext_ccfree.json and write evidence/exp3_decision.json.

Computes, per budget/stage: bootstrap gate (from the experiment file),
per-fold paired mean>2SE gate (n=4, all and far strata), canary status, and a
comparison with the historical exp2 catalogue-context-full arm at matched
budget.  Chooses clean-promote > promote-with-caveats > negative exactly as
preregistered.  No threshold is invented here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_BUDGETS = (25_000, 38_000, 50_000)
FOLDS = ("NW", "NE", "SW", "SE")


def per_fold_gate(results: dict, budget: int, stage: int) -> dict:
    pf = results[str(budget)]["per_fold"]
    out = {}
    for key, field in (("all", "dti"), ("far", "far_dti")):
        deltas = np.array([pf[f"{f}_stage{stage}"][field] - pf[f"{f}_stage1"][field]
                           for f in FOLDS], dtype=float)
        se = float(deltas.std(ddof=1) / np.sqrt(deltas.size)) if deltas.size > 1 else float("nan")
        out[key] = dict(per_fold_deltas=[float(x) for x in deltas],
                        mean_delta=float(deltas.mean()), se=se,
                        t_stat=(float(deltas.mean() / se) if se and np.isfinite(se) and se > 0 else None),
                        passes_gt_2se=bool(np.isfinite(se) and deltas.mean() > 2 * se),
                        n_folds=int(deltas.size))
    return out


def main() -> int:
    exp3 = json.loads((ROOT / "evidence/exp3_autocontext_ccfree.json").read_text())
    exp2 = json.loads((ROOT / "evidence/exp2_autocontext_hidedraw.json").read_text())
    canary_flags = [k for k, v in exp3["canary"].items() if v.get("flag")]
    max_canary_auc = max(max(v["auc"], 1 - v["auc"]) for v in exp3["canary"].values())

    arms = {}
    for budget in CANDIDATE_BUDGETS:
        for stage in (2, 3):
            g = exp3["gate"][f"{budget}_stage{stage}"]
            pf = per_fold_gate(exp3["results"], budget, stage)
            hist = exp2["gate"].get(f"{budget}_stage{stage}")
            bootstrap_pass = bool(g["ci_all"][0] > 0 and g["ci_far"][0] > 0)
            t_pass = bool(pf["all"]["passes_gt_2se"] and pf["far"]["passes_gt_2se"])
            arms[f"{budget}_stage{stage}"] = dict(
                budget=budget, stage=stage,
                evidence_class="HOLDOUT-DTI", evaluator_version=exp3["evaluator_version"],
                withheld_positive_pixels=exp3["holdout"]["withheld_positive_pixels"],
                dti_stage1=g["dti_stage1"], dti_candidate=g["dti_cand"],
                delta_all=g["delta_all"], ci95_all=g["ci_all"],
                far_dti_stage1=g["far_dti_stage1"], far_dti_candidate=g["far_dti_cand"],
                delta_far=g["delta_far"], ci95_far=g["ci_far"],
                bootstrap_gate_both_strata=bootstrap_pass,
                per_fold_gate=pf, per_fold_t_gate_both_strata=t_pass,
                exp2_ccfull_same_instrument=dict(
                    dti_candidate=hist["dti_cand"], far_dti_candidate=hist["far_dti_cand"],
                    delta_all=hist["delta_all"], delta_far=hist["delta_far"],
                    bootstrap_passes=hist["passes"]) if hist else None,
                ccfree_minus_ccfull_all=(g["dti_cand"] - hist["dti_cand"]) if hist else None,
                ccfree_minus_ccfull_far=(g["far_dti_cand"] - hist["far_dti_cand"]) if hist else None,
            )

    def rank(item):
        _k, a = item
        return (a["dti_candidate"], -a["budget"], -a["stage"])

    clean = [kv for kv in arms.items()
             if kv[1]["bootstrap_gate_both_strata"] and kv[1]["per_fold_t_gate_both_strata"]
             and not canary_flags]
    caveat = [kv for kv in arms.items()
              if kv[1]["bootstrap_gate_both_strata"] and not canary_flags]
    if clean:
        tier, pool = "promote", clean
    elif caveat:
        tier, pool = "promote-with-caveats", caveat
    else:
        tier, pool = "negative", list(arms.items())
    chosen_key, chosen = max(pool, key=rank)

    decision = dict(
        preregistration="evidence/exp3_preregistration.json",
        experiment="evidence/exp3_autocontext_ccfree.json",
        canary=dict(flags=canary_flags, max_auc_distance_from_0p5=float(max_canary_auc),
                    limit=0.90, clean=not canary_flags),
        arms=arms, tier_before_uniqueness_and_format=tier,
        chosen=dict(key=chosen_key, stage=chosen["stage"], budget=chosen["budget"]),
        note="Uniqueness vs registry and format validation are applied at build time on the actual bytes; "
             "this tier can only be downgraded there, never upgraded.",
    )
    (ROOT / "evidence" / "exp3_decision.json").write_text(json.dumps(decision, indent=1))
    print(json.dumps({k: decision[k] for k in ("canary", "tier_before_uniqueness_and_format", "chosen")}, indent=1))
    for k, a in arms.items():
        print(k, f"dti {a['dti_stage1']:.4f}->{a['dti_candidate']:.4f} (CI {a['ci95_all']})",
              f"far {a['far_dti_stage1']:.4f}->{a['far_dti_candidate']:.4f} (CI {a['ci95_far']})",
              f"boot={a['bootstrap_gate_both_strata']} tfold={a['per_fold_t_gate_both_strata']}",
              f"vsCCfull all={a['ccfree_minus_ccfull_all']:+.4f} far={a['ccfree_minus_ccfull_far']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
