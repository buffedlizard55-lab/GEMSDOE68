#!/usr/bin/env python3
"""Experiment 3 (preregistered, knowledge/02_exp3_preregistration_and_hypotheses.md):
ridge-crest / spacing / concordance emission variants on exp2-identical outer maps.

Training is byte-for-byte the exp2 protocol (same Lane, seeds, LightGBM params);
outer maps are cached in .cache/exp3/.  Only the EMISSION arms are new:

  V0  exp2 rule (5x5 local maxima, top-k)                  -- control / current best
  V2  greedy >=2.8 px spacing over all allowed pixels      -- spacing-only control
  V1  across-strike crest candidates + greedy spacing      -- H68-A (primary)
  V3  crest candidates ranked by rank(surface)xrank(ridge) -- H68-D (stage 3 only)

Budgets 25,000 / 38,000 only.  Scopes: all / far (exp2 stratum) / farfull (far truth,
all emitted mass counts as FP).  Writes evidence/exp3_ridgecrest.json including the
V0-vs-committed-exp2 replication check and the mechanical decision-rule output.
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
sys.path.insert(0, str(ROOT / "scripts"))
from gems68 import autocontext as AC, emission as E, evaluate_holdout as EH, grid as G  # noqa: E402
from gems68 import features as F  # noqa: E402
from gems68.experiment import Lane, log  # noqa: E402
from gems68.holdout import quadrant_ids  # noqa: E402
from run_exp2 import canary  # noqa: E402

BUDGETS = (25_000, 38_000)
FAR_PX = 10
N_STAGES = 3
CACHE = ROOT / ".cache" / "exp3"
ARMS = ("s1_V0", "s2_V0", "s3_V0", "s2_V2", "s3_V2", "s2_V1", "s3_V1", "s3_V3")
CANDIDATE_ARMS = ("s3_V1", "s2_V1", "s3_V3", "s3_V2")


def emit_arm(arm, maps, crests, score_v3, allowed_region, n_q):
    stage = int(arm[1])
    variant = arm[3:]
    if variant == "V0":
        return E.emit_dots(maps[stage], allowed_region, n_q)
    if variant == "V2":
        return E.greedy_spacing(maps[stage], allowed_region, allowed_region, n_q)
    if variant == "V1":
        return E.greedy_spacing(maps[stage], allowed_region, crests[stage], n_q)
    if variant == "V3":
        return E.greedy_spacing(score_v3, allowed_region, crests[stage], n_q)
    raise ValueError(arm)


def evaluate_exp3(lane: Lane, p_outer: dict) -> dict:
    fp = lane.fp
    folds = lane.folds
    masked_known = folds[0]["masked_known"]
    allowed = ~E.exclusion_mask(lane.eval_visible) & fp
    far = ndi.distance_transform_edt(~masked_known) > FAR_PX
    maps = {k: AC.to_map(v, fp) for k, v in p_outer.items()}
    log("computing crest masks (stage 2/3, global allowed)")
    crests = {k: E.crest_mask(maps[k], allowed) for k in (2, 3)}
    # H68-D concordance support: max percentile rank of |Hessian ridge| physics maps
    ridge_idx = [F.PHYSICS_NAMES.index(n) for n in ("elev_ridge_s2", "rtp_ridge_s2", "grav_ridge_s2")]
    support = np.zeros(fp.shape, np.float32)
    for j in ridge_idx:
        support = np.maximum(support, E.rank01(np.abs(AC.to_map(lane.phys[j], fp)), fp))
    score_v3 = np.sqrt(E.rank01(maps[3], fp) * support).astype(np.float32)
    crest_counts = {int(k): int(v.sum()) for k, v in crests.items()}

    results = {}
    for n_total in BUDGETS:
        terms = {scope: {a: [] for a in ARMS} for scope in ("all", "far", "farfull")}
        emitted = {a: 0 for a in ARMS}
        for f in folds:
            n_q = E.quota(n_total, fp, f["region"])
            far_fold = dict(f, truth=f["truth"] & far)
            for arm in ARMS:
                pred = emit_arm(arm, maps, crests, score_v3, allowed & f["region"], n_q).astype(np.float32)
                emitted[arm] += int(pred.sum())
                _, t_all = EH.evaluate(pred, f, fp, block_side=200, global_shape=fp.shape)
                _, t_far = EH.evaluate(pred, f, fp & far, block_side=200, global_shape=fp.shape)
                _, t_ff = EH.evaluate(pred, far_fold, fp, block_side=200, global_shape=fp.shape)
                terms["all"][arm].append(t_all)
                terms["far"][arm].append(t_far)
                terms["farfull"][arm].append(t_ff)
            log(f"budget {n_total} fold {f['name']} done")
        pooled = {scope: {a: np.sum(terms[scope][a], axis=0) for a in ARMS} for scope in terms}
        block = {"emitted_total": emitted, "scopes": {}}
        for scope in ("all", "far", "farfull"):
            summary = EH.pooled_summary(pooled[scope], candidate="s3_V0")
            entry = {"scores": summary["scores"], "paired_vs": {}}
            for cand in CANDIDATE_ARMS + ("s2_V0", "s3_V0", "s2_V2"):
                s_c = EH.pooled_summary(pooled[scope], candidate=cand)
                entry["paired_vs"][cand] = {
                    ref: s_c["paired_differences"][ref] for ref in ("s1_V0", "s3_V0") if ref != cand
                }
            block["scopes"][scope] = entry
        results[str(n_total)] = block
        sc = block["scopes"]["all"]["scores"]
        log(f"budget {n_total}: " + " ".join(f"{a}={sc[a]['dti']:.4f}({emitted[a]})" for a in ARMS))
    return {"results": results, "crest_counts": crest_counts}


def replication_check(results: dict) -> dict:
    committed = json.loads((ROOT / "evidence/exp2_autocontext_hidedraw.json").read_text())["gate"]
    out = {}
    ok_all = True
    for budget in BUDGETS:
        sc = results[str(budget)]["scopes"]["all"]["scores"]
        for stage in (2, 3):
            ref = committed[f"{budget}_stage{stage}"]
            got_s1, got_c = sc["s1_V0"]["dti"], sc[f"s{stage}_V0"]["dti"]
            d = max(abs(got_s1 - ref["dti_stage1"]), abs(got_c - ref["dti_cand"]))
            out[f"{budget}_stage{stage}"] = dict(
                committed_s1=ref["dti_stage1"], got_s1=got_s1,
                committed_cand=ref["dti_cand"], got_cand=got_c, max_abs_diff=d, within_0p005=bool(d <= 0.005))
            ok_all &= bool(d <= 0.005)
    out["replicated"] = ok_all
    return out


def decide(results: dict, replication: dict, can: dict) -> dict:
    b = results["38000"]
    canary_clean = not any(v["flag"] for v in can.values())
    per_arm = {}
    for arm in CANDIDATE_ARMS:
        d_all_s1 = b["scopes"]["all"]["paired_vs"][arm]["s1_V0"]
        d_far_s1 = b["scopes"]["far"]["paired_vs"][arm]["s1_V0"]
        d_all_best = b["scopes"]["all"]["paired_vs"][arm].get("s3_V0")
        d_ff_s1 = b["scopes"]["farfull"]["paired_vs"][arm]["s1_V0"]
        per_arm[arm] = dict(
            dti_all=b["scopes"]["all"]["scores"][arm]["dti"],
            emitted=b["emitted_total"][arm],
            gate1_all_vs_s1_ci=d_all_s1["ci95"], gate1_far_vs_s1_ci=d_far_s1["ci95"],
            gate2_all_vs_s3V0_ci=(d_all_best["ci95"] if d_all_best else None),
            gate3_farfull_delta_vs_s1=d_ff_s1["delta"],
            passes=bool(d_all_s1["ci95"][0] > 0 and d_far_s1["ci95"][0] > 0
                        and (d_all_best is None or d_all_best["ci95"][0] > 0)
                        and d_ff_s1["delta"] > 0))
    eligible = [a for a in CANDIDATE_ARMS if per_arm[a]["passes"]]
    chosen = max(eligible, key=lambda a: per_arm[a]["dti_all"]) if eligible else None
    return dict(replication_ok=replication["replicated"], canary_clean=canary_clean,
                per_arm=per_arm, chosen_arm=chosen,
                verdict=("promote" if (chosen and replication["replicated"] and canary_clean) else "negative"))


def main() -> int:
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    g = G.load_grid()
    X1 = np.ascontiguousarray(np.load(ROOT / ".cache/features/X1.npy", mmap_mode="r"))
    phys = list(np.load(ROOT / ".cache/features/phys.npy"))
    lane = Lane(g, X1, phys, quadrant_ids(g.footprint))
    log("lane ready")
    if all((CACHE / f"p{k}_outer.npy").is_file() for k in range(1, N_STAGES + 1)):
        p_outer = {k: np.load(CACHE / f"p{k}_outer.npy") for k in range(1, N_STAGES + 1)}
        log("resumed outer maps from .cache/exp3")
    else:
        p_outer, _imps = lane.nested(N_STAGES)
        for k, v in p_outer.items():
            np.save(CACHE / f"p{k}_outer.npy", v)
        log("outer maps cached")
    can = canary(lane, p_outer)
    ev = evaluate_exp3(lane, p_outer)
    replication = replication_check(ev["results"])
    decision = decide(ev["results"], replication, can)
    out = dict(
        experiment="exp3_ridgecrest_emission_on_exp2_protocol", evidence_class="HOLDOUT-DTI",
        evaluator_version=EH.VERSION, eval_seed=20, train_seeds=[d["seed"] for d in lane.draws],
        preregistration="knowledge/02_exp3_preregistration_and_hypotheses.md",
        withheld_positive_pixels=int(sum(f["receipt"]["withheld_positive_pixels"] for f in lane.folds)),
        budgets=list(BUDGETS), arms=list(ARMS), crest_counts=ev["crest_counts"],
        canary=can, results=ev["results"], replication=replication, decision=decision,
        seconds=round(time.time() - t0, 1))
    (ROOT / "evidence/exp3_ridgecrest.json").write_text(json.dumps(out, indent=1, default=float))
    log("wrote evidence/exp3_ridgecrest.json")
    log(f"decision: {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
