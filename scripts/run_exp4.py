#!/usr/bin/env python3
"""Experiment 4 (preregistered, knowledge/03_exp4_preregistration.md): stratified spacing.

Loads the Exp-3 cached outer maps (.cache/exp3/p*_outer.npy) — identical surfaces,
identical folds, identical evaluator — and scores the H68-G stratified arms
(s2/s3 x V6 all-candidates / V7 crest-candidates) plus the s1_V0 / s3_V0 / s3_V2
controls needed for paired terms.  Writes evidence/exp4_stratified.json with the
mechanical decision (same gates as Exp-3).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems68 import autocontext as AC, emission as E, evaluate_holdout as EH, grid as G  # noqa: E402
from gems68.experiment import Lane, log  # noqa: E402
from gems68.holdout import quadrant_ids  # noqa: E402

BUDGETS = (25_000, 38_000)
FAR_PX = 10
ARMS = ("s1_V0", "s3_V0", "s3_V2", "s2_V6", "s3_V6", "s2_V7", "s3_V7")
CANDIDATE_ARMS = ("s3_V6", "s2_V6", "s3_V7", "s2_V7")


def main() -> int:
    t0 = time.time()
    g = G.load_grid()
    X1 = np.ascontiguousarray(np.load(ROOT / ".cache/features/X1.npy", mmap_mode="r"))
    phys = list(np.load(ROOT / ".cache/features/phys.npy"))
    lane = Lane(g, X1, phys, quadrant_ids(g.footprint))
    fp = lane.fp
    folds = lane.folds
    masked_known = folds[0]["masked_known"]
    allowed = ~E.exclusion_mask(lane.eval_visible) & fp
    far_score = ndi.distance_transform_edt(~masked_known) > FAR_PX
    zone_far = ndi.distance_transform_edt(~lane.eval_visible) > FAR_PX
    zone_id = zone_far.astype(np.int8)
    maps = {k: AC.to_map(np.load(ROOT / f".cache/exp3/p{k}_outer.npy"), fp) for k in (1, 2, 3)}
    log("computing crest masks")
    crests = {k: E.crest_mask(maps[k], allowed) for k in (2, 3)}

    def emit(arm, region, n_q):
        stage, variant = int(arm[1]), arm[3:]
        ar = allowed & region
        if variant == "V0":
            return E.emit_dots(maps[stage], ar, n_q)
        if variant == "V2":
            return E.greedy_spacing(maps[stage], ar, ar, n_q)
        area_far = int((ar & zone_far).sum())
        area_near = int((ar & ~zone_far).sum())
        n_far = int(round(n_q * area_far / max(area_far + area_near, 1)))
        quotas = {0: n_q - n_far, 1: n_far}
        cand = ar if variant == "V6" else crests[stage]
        return E.greedy_spacing_stratified(maps[stage], ar, cand, zone_id, quotas)

    results = {}
    for n_total in BUDGETS:
        terms = {scope: {a: [] for a in ARMS} for scope in ("all", "far", "farfull")}
        emitted = {a: 0 for a in ARMS}
        far_dots = {a: 0 for a in ARMS}
        for f in folds:
            n_q = E.quota(n_total, fp, f["region"])
            far_fold = dict(f, truth=f["truth"] & far_score)
            for arm in ARMS:
                dots = emit(arm, f["region"], n_q)
                pred = dots.astype(np.float32)
                emitted[arm] += int(dots.sum())
                far_dots[arm] += int((dots & zone_far).sum())
                _, t_all = EH.evaluate(pred, f, fp, block_side=200, global_shape=fp.shape)
                _, t_far = EH.evaluate(pred, f, fp & far_score, block_side=200, global_shape=fp.shape)
                _, t_ff = EH.evaluate(pred, far_fold, fp, block_side=200, global_shape=fp.shape)
                terms["all"][arm].append(t_all)
                terms["far"][arm].append(t_far)
                terms["farfull"][arm].append(t_ff)
            log(f"budget {n_total} fold {f['name']} done")
        pooled = {scope: {a: np.sum(terms[scope][a], axis=0) for a in ARMS} for scope in terms}
        block = {"emitted_total": emitted, "far_zone_dots": far_dots, "scopes": {}}
        for scope in ("all", "far", "farfull"):
            summary = EH.pooled_summary(pooled[scope], candidate="s3_V0")
            entry = {"scores": summary["scores"], "paired_vs": {}}
            for cand in CANDIDATE_ARMS + ("s3_V2",):
                s_c = EH.pooled_summary(pooled[scope], candidate=cand)
                entry["paired_vs"][cand] = {ref: s_c["paired_differences"][ref] for ref in ("s1_V0", "s3_V0")}
            block["scopes"][scope] = entry
        results[str(n_total)] = block
        sc = block["scopes"]["all"]["scores"]
        log(f"budget {n_total}: " + " ".join(f"{a}={sc[a]['dti']:.4f}(far{far_dots[a]})" for a in ARMS))

    exp3 = json.loads((ROOT / "evidence/exp3_ridgecrest.json").read_text())
    b = results["38000"]
    per_arm = {}
    for arm in CANDIDATE_ARMS:
        d_all = b["scopes"]["all"]["paired_vs"][arm]["s1_V0"]
        d_far = b["scopes"]["far"]["paired_vs"][arm]["s1_V0"]
        d_best = b["scopes"]["all"]["paired_vs"][arm]["s3_V0"]
        d_ff = b["scopes"]["farfull"]["paired_vs"][arm]["s1_V0"]
        per_arm[arm] = dict(
            dti_all=b["scopes"]["all"]["scores"][arm]["dti"], emitted=b["emitted_total"][arm],
            far_zone_dots=b["far_zone_dots"][arm],
            gate1_all_vs_s1_ci=d_all["ci95"], gate1_far_vs_s1_ci=d_far["ci95"],
            gate2_all_vs_s3V0_ci=d_best["ci95"], gate3_farfull_delta_vs_s1=d_ff["delta"],
            passes=bool(d_all["ci95"][0] > 0 and d_far["ci95"][0] > 0 and d_best["ci95"][0] > 0 and d_ff["delta"] > 0))
    eligible = [a for a in CANDIDATE_ARMS if per_arm[a]["passes"]]
    chosen = max(eligible, key=lambda a: per_arm[a]["dti_all"]) if eligible else None
    decision = dict(replication_ok=exp3["replication"]["replicated"],
                    canary_clean=exp3["decision"]["canary_clean"], per_arm=per_arm,
                    chosen_arm=chosen, verdict="promote" if chosen else "negative")
    out = dict(experiment="exp4_stratified_spacing_on_exp3_maps", evidence_class="HOLDOUT-DTI",
               evaluator_version=EH.VERSION, eval_seed=20,
               preregistration="knowledge/03_exp4_preregistration.md",
               withheld_positive_pixels=exp3["withheld_positive_pixels"],
               budgets=list(BUDGETS), arms=list(ARMS),
               zone_rule="far zone = >10 px from visible (eval-draw) catalogue; quota proportional to zone allowed-area per fold region",
               results=results, decision=decision, seconds=round(time.time() - t0, 1))
    (ROOT / "evidence/exp4_stratified.json").write_text(json.dumps(out, indent=1, default=float))
    log(f"decision: {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
