#!/usr/bin/env python3
"""Apply the Exp4 preregistered rule (evidence/exp4_preregistration.json):
bootstrap + per-fold gates vs own stage 1 AND beating the Exp2 holdout best
at the same budget/stage.  Writes evidence/exp4_decision.json.  Also records
the final Exp3 disposition (promote-tier by its own preregistration, but
NEGATIVE for promotion because it does not beat the holdout best)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_decision import per_fold_gate  # noqa: E402  (shared gate arithmetic)

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (25_000, 38_000, 50_000)


def main() -> int:
    exp4 = json.loads((ROOT / "evidence/exp4_autocontext_ext.json").read_text())
    exp2 = json.loads((ROOT / "evidence/exp2_autocontext_hidedraw.json").read_text())
    exp3dec = json.loads((ROOT / "evidence/exp3_decision.json").read_text())
    flags = [f"stage1:{k}" for k, v in exp4["canary_stage1"].items() if v["flag"]] + \
            [f"stage>=2:{k}" for k, v in exp4["canary"].items() if v["flag"]]
    max_auc = max([max(v["auc"], 1 - v["auc"]) for v in exp4["canary_stage1"].values()] +
                  [max(v["auc"], 1 - v["auc"]) for v in exp4["canary"].values()])
    arms = {}
    for budget in BUDGETS:
        for stage in (2, 3):
            g = exp4["gate"][f"{budget}_stage{stage}"]
            h = exp2["gate"][f"{budget}_stage{stage}"]
            pf = per_fold_gate(exp4["results"], budget, stage)
            arms[f"{budget}_stage{stage}"] = dict(
                budget=budget, stage=stage, evidence_class="HOLDOUT-DTI",
                evaluator_version=exp4["evaluator_version"],
                withheld_positive_pixels=exp4["holdout"]["withheld_positive_pixels"],
                dti_stage1=g["dti_stage1"], dti_candidate=g["dti_cand"],
                delta_all=g["delta_all"], ci95_all=g["ci_all"],
                far_dti_stage1=g["far_dti_stage1"], far_dti_candidate=g["far_dti_cand"],
                delta_far=g["delta_far"], ci95_far=g["ci_far"],
                bootstrap_gate_both_strata=bool(g["ci_all"][0] > 0 and g["ci_far"][0] > 0),
                per_fold_gate=pf,
                per_fold_t_gate_both_strata=bool(pf["all"]["passes_gt_2se"] and pf["far"]["passes_gt_2se"]),
                exp2_holdout_best_comparator=dict(dti_candidate=h["dti_cand"],
                                                  far_dti_candidate=h["far_dti_cand"]),
                beats_exp2_all=bool(g["dti_cand"] > h["dti_cand"]),
                far_vs_exp2=g["far_dti_cand"] - h["far_dti_cand"],
                far_ok_vs_exp2=bool(g["far_dti_cand"] - h["far_dti_cand"] >= -0.005))
    clean = [kv for kv in arms.items() if kv[1]["bootstrap_gate_both_strata"]
             and kv[1]["per_fold_t_gate_both_strata"] and kv[1]["beats_exp2_all"]
             and kv[1]["far_ok_vs_exp2"] and not flags]
    caveat = [kv for kv in arms.items() if kv[1]["bootstrap_gate_both_strata"]
              and kv[1]["beats_exp2_all"] and kv[1]["far_ok_vs_exp2"] and not flags]
    tier, pool = (("promote", clean) if clean else
                  ("promote-with-caveats", caveat) if caveat else
                  ("negative", list(arms.items())))

    def rank(item):
        return (item[1]["dti_candidate"], -item[1]["budget"], -item[1]["stage"])

    key, chosen = max(pool, key=rank)
    decision = dict(
        preregistration="evidence/exp4_preregistration.json",
        experiment="evidence/exp4_autocontext_ext.json",
        canary=dict(flags=flags, max_auc_distance_from_0p5=float(max_auc), limit=0.90,
                    clean=not flags),
        arms=arms, tier_before_uniqueness_and_format=tier,
        chosen=dict(key=key, stage=chosen["stage"], budget=chosen["budget"]),
        exp3_final_disposition=dict(
            tier_under_exp3_preregistration=exp3dec["tier_before_uniqueness_and_format"],
            chosen=exp3dec["chosen"],
            promotion_verdict="negative",
            reason="AC68-N1 does not beat the current holdout best (Exp2) at any budget "
                   "(best CC-free candidate 0.0770 vs 0.1181 at 25k); per the standing brief a "
                   "slot must not be spent on it. Retained as a scientific ablation result."),
        note="Uniqueness/format applied at build time can only downgrade this tier.")
    (ROOT / "evidence" / "exp4_decision.json").write_text(json.dumps(decision, indent=1))
    print(json.dumps({k: decision[k] for k in ("canary", "tier_before_uniqueness_and_format", "chosen")}, indent=1))
    for k, a in arms.items():
        print(k, f"s1 {a['dti_stage1']:.4f} cand {a['dti_candidate']:.4f} farCand {a['far_dti_candidate']:.4f}",
              f"boot={a['bootstrap_gate_both_strata']} tfold={a['per_fold_t_gate_both_strata']}",
              f"beatsExp2={a['beats_exp2_all']} ({a['exp2_holdout_best_comparator']['dti_candidate']:.4f})",
              f"farVsExp2={a['far_vs_exp2']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
