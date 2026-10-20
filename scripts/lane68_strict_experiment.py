#!/usr/bin/env python3
"""Preregistered H68-M1: magnetic-edge-tangent auto-context, nested hide/recover.

ONE experiment: four *outer* label-blind quadrants of gems57.holdout's
buffered whole-component draw; four buffered inner spatial blocks per outer
fit. Every target quadrant's catalogue is masked from the feature geometry
and all model fits, and stage-1/2 training inputs are OOF, never in-fold.

This is a local catalogue recovery proxy, NOT a DrivenData score. Run only with
hash-verified, locally bridged data. No raster is published by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gems57 import grid as G57, holdout as H57                     # noqa: E402
from gems57 import evaluate_holdout as EH                          # noqa: E402
from gems68 import blocks as B, catalogue_context as CC           # noqa: E402
from gems68 import channel_stack as F, emission as E              # noqa: E402
from lane68_autocontext import (INNER_BUFFER_PX, ROW_BLOCK,       # noqa: E402
    bbox_of, context_at_rows, make_model, own_block_catalogue_context,
    predict_box, quadrant_partition, stage1_row_matrix, stage2_row_matrix,
    strip_features_stage1, strip_features_stage2, paired_gate)

SEED = 20
BUDGET = 38000  # predeclared; no holdout sweep or leaderboard tuning
EXCLUDE_PX = 2.0
OUTER_BUFFER_PX = 12
FEATURE_BUFFER_PX = 3
NEG_PER_POS = 3
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_BLOCK_PX = 200
FAR_PX = 10


def log(*args):
    print(time.strftime("[%H:%M:%S]"), *args, flush=True)


def orientation(stack, index, footprint):
    """Tangent to a magnetic contrast edge, not the dominant gradient itself."""
    field = np.array(stack[index["mag_anom"]], dtype=np.float32)
    field[~np.isfinite(field) | ~footprint] = 0.0
    c, s, coh = B.structure_tensor_orientation(field, sigma=3.0)
    return -c, -s, coh


def outer_training_domain(fp: np.ndarray, quad: np.ndarray, fold: dict) -> np.ndarray:
    """Training is outside the entire test quadrant, with a 12 px exclusion band."""
    outer = quad == fold["fold"]
    domain = fp & ~outer & (ndi.distance_transform_edt(~outer) > OUTER_BUFFER_PX)
    domain &= ndi.distance_transform_edt(~fold["hidden_all"]) > FEATURE_BUFFER_PX
    if np.any(domain & outer) or np.any(domain & fold["hidden_all"]):
        raise AssertionError("outer truth can enter a training block")
    return domain


def sample_rows(grid, visible, domain, block):
    rng = np.random.default_rng(20261020)
    py, px = np.nonzero(visible & domain)
    ny, nx = np.nonzero(domain & ~grid.catalogue)
    if len(py) < 50 or len(ny) < 50:
        raise ValueError("too few training rows")
    take = rng.choice(len(ny), size=min(len(ny), NEG_PER_POS * len(py)), replace=False)
    ry = np.r_[py, ny[take]].astype(np.int64)
    rx = np.r_[px, nx[take]].astype(np.int64)
    y = np.r_[np.ones(len(py), np.uint8), np.zeros(len(take), np.uint8)]
    rb = block[ry, rx]
    if (rb < 0).any():
        raise AssertionError("a sample fell outside the inner spatial blocks")
    return dict(ry=ry, rx=rx, y=y, rb=rb)


def feature_canary_auc(labels: np.ndarray, values: np.ndarray) -> tuple[float | None, bool]:
    """Constant scores have AUC=0.5 (ties), not an undefined/leaky channel."""
    vals = np.asarray(values)
    if not np.isfinite(vals).all():
        return None, True
    if np.ptp(vals) == 0:
        return 0.5, False
    auc = float(roc_auc_score(labels, vals))
    return auc, bool(max(auc, 1 - auc) > .90)


def audit_canary(cache, stack, idx, cc, p1, p2, cs, region, truth, known, c, s, fold_name):
    """Single-feature AUC on withheld positives vs unlabelled outer-block pixels.

    These are NOT training labels.  The catalogue context used here has had the
    *entire target quadrant* removed. AUC>0.90 in either direction flags leakage.
    """
    pos = np.flatnonzero((truth & region & ~known).ravel())
    neg = np.flatnonzero((region & ~truth & ~known).ravel())
    if len(pos) < 50 or len(neg) < 50:
        raise ValueError("canary fold needs both classes")
    take = np.random.default_rng(101).choice(neg, min(len(neg), 5 * len(pos)), replace=False)
    chosen = np.r_[pos, take]
    ry, rx = np.unravel_index(chosen, truth.shape)
    labels = np.r_[np.ones(len(pos), np.uint8), np.zeros(len(take), np.uint8)]
    cat_features = np.stack([cc[name][ry, rx] for name in B.CATALOGUE_CONTEXT], axis=1)
    matrices = (
        ("stage1", B.STAGE1_CHANNELS + B.CATALOGUE_CONTEXT,
         np.hstack([stage1_row_matrix(stack, idx, ry, rx), cat_features])),
        ("stage2", B.stage2_feature_names("p1"),
         stage2_row_matrix(stack, idx, p1, dict(ry=ry, rx=rx), cat_features, c, s)),
        ("stage3", B.stage2_feature_names("p2"),
         stage2_row_matrix(stack, idx, p2, dict(ry=ry, rx=rx), cat_features, c, s)),
    )
    for stage, names, x in matrices:
        if x.shape[1] != len(names):
            raise AssertionError("canary feature name alignment")
        for j, name in enumerate(names):
            auc, flagged = feature_canary_auc(labels, x[:, j])
            cache.append(dict(fold=fold_name, feature=f"{stage}/{name}",
                              auc=auc, flagged=flagged,
                              constant=bool(auc == 0.5 and np.ptp(x[:, j]) == 0)
                                       if auc is not None else False,
                              reason="non-finite values; not cleared" if auc is None else None))
        del x


def run_fold(fold, quad, grid, stack, idx, footprint, c, s, canary):
    start = time.monotonic()
    h, w = grid.shape
    outer = quad == fold["fold"]
    # The shared v2 holdout hides whole 8-connected components globally and
    # removes a 3px collar from *feature* context; the evaluation known mask
    # remains pixel-exact (no expanded exclusion of visible known pixels).
    hidden = fold["hidden_all"]
    visible = fold["visible"] & ~outer
    known = fold["masked_known"]
    region = fold["region"] & footprint & ~known
    truth = fold["truth"] & region
    if (visible & outer).any() or (visible & hidden).any():
        raise AssertionError("test labels leaked into catalogue features")
    train = outer_training_domain(footprint, quad, fold)
    inner_id = quadrant_partition(train)
    inner = [inner_id == b for b in range(4)]
    if any(not m.any() for m in inner):
        raise ValueError("empty inner fold")
    distance_to_inner = [ndi.distance_transform_edt(~m) for m in inner]
    cs = sample_rows(grid, visible, train, inner_id)
    own_cat = own_block_catalogue_context(
        dict(**cs, inner=inner), visible, footprint, c, s)
    a1 = np.hstack([stage1_row_matrix(stack, idx, cs["ry"], cs["rx"]), own_cat])
    far = [(cs["rb"] != b) & (d[cs["ry"], cs["rx"]] > INNER_BUFFER_PX)
           for b, d in enumerate(distance_to_inner)]
    box_inner = [bbox_of(m, h, w) for m in inner]
    box_outer = bbox_of(region, h, w)
    if box_outer is None:
        raise ValueError("empty holdout fold")
    log(fold["name"], "truth", int(truth.sum()), "train pos", int(cs["y"].sum()))

    p1_oof = np.zeros((h, w), np.float32)
    for b in range(4):
        if int(cs["y"][far[b]].sum()) < 50:
            raise ValueError(f"inner fold {b} lacks positives")
        model = make_model(); model.fit(a1[far[b]], cs["y"][far[b]])
        cc = CC.catalogue_context(visible & ~inner[b], footprint, c, s)
        p = predict_box(model, lambda rows, cols, m=model, fc=cc:
                        strip_features_stage1(stack, idx, rows, fc, cols),
                        h, w, box_inner[b])
        p1_oof[inner[b]] = p[inner[b]]
        del model, cc, p
    model1 = make_model(); model1.fit(a1, cs["y"])
    cc_target = CC.catalogue_context(visible, footprint, c, s)
    p1_outer = predict_box(model1, lambda rows, cols:
        strip_features_stage1(stack, idx, rows, cc_target, cols), h, w, box_outer)
    p1 = np.where(train, p1_oof, p1_outer).astype(np.float32)
    del model1, a1, p1_oof
    log(fold["name"], "stage1", round(time.monotonic()-start), "s")

    a2 = stage2_row_matrix(stack, idx, p1, cs, own_cat, c, s)
    p2_oof = np.zeros((h, w), np.float32)
    for b in range(4):
        model = make_model(); model.fit(a2[far[b]], cs["y"][far[b]])
        cc = CC.catalogue_context(visible & ~inner[b], footprint, c, s)
        p = predict_box(model, lambda rows, cols, m=model, fc=cc:
            strip_features_stage2(stack, idx, rows, p1, fc, c, s, cols),
            h, w, box_inner[b])
        p2_oof[inner[b]] = p[inner[b]]
        del model, cc, p
    model2 = make_model(); model2.fit(a2, cs["y"])
    p2_outer = predict_box(model2, lambda rows, cols:
        strip_features_stage2(stack, idx, rows, p1, cc_target, c, s, cols),
        h, w, box_outer)
    p2 = np.where(train, p2_oof, p2_outer).astype(np.float32)
    del model2, a2, p2_oof
    log(fold["name"], "stage2", round(time.monotonic()-start), "s")

    a3 = stage2_row_matrix(stack, idx, p2, cs, own_cat, c, s)
    model3 = make_model(); model3.fit(a3, cs["y"])
    p3_outer = predict_box(model3, lambda rows, cols:
        strip_features_stage2(stack, idx, rows, p2, cc_target, c, s, cols),
        h, w, box_outer)
    del a3, model3
    log(fold["name"], "stage3", round(time.monotonic()-start), "s")

    audit_canary(canary, stack, idx, cc_target, p1, p2, cs, region,
                 truth, known, c, s, fold["name"])
    n_train_rows = int(len(cs["y"]))
    del cc_target, p1, p2, cs, own_cat, inner, distance_to_inner

    eligible = region & (ndi.distance_transform_edt(~known) > EXCLUDE_PX)
    quota = E.quota(BUDGET, footprint, region)
    # All three arms share the exact same candidate grid and emission budget.
    fields = dict(stage1=p1_outer, stage2=p2_outer, stage3=p3_outer)
    box = bbox_of(region, h, w, pad=6)
    sl = (slice(*box[0]), slice(*box[1]))
    crop = {k: v[sl] for k, v in fold.items() if k in ("region", "truth", "masked_known")}
    crop["region"] = region[sl]
    crop["truth"] = truth[sl]
    far_mask = ndi.distance_transform_edt(~known) > FAR_PX
    results = {"fold": fold["name"], "n_truth": int(truth.sum()),
               "n_truth_far": int((truth & far_mask).sum()),
               "n_train_pos": int((visible & train).sum()),
               "n_train_rows": n_train_rows,
               "quota": quota, "dti": {}, "dti_far": {}, "terms": {}, "terms_far": {}}
    for stage, field in fields.items():
        dots = E.emit_dots(field, eligible, quota, nms=5)
        if int(dots.sum()) != quota:
            raise ValueError(f"{fold['name']} {stage}: 5x5 peak field cannot fill "
                             f"the fixed matched quota ({int(dots.sum())} of {quota}); "
                             "stop rather than tune the emission rule")
        # Evaluator crops remain globally aligned.  The 6px crop pad covers
        # the complete 300m kernel around every scored truth and dot.
        r, terms = EH.evaluate(dots[sl].astype(np.float32), crop, footprint[sl],
                 block_side=BOOTSTRAP_BLOCK_PX,
                 origin=(sl[0].start, sl[1].start), global_shape=grid.shape)
        rf, terms_far = EH.evaluate(dots[sl].astype(np.float32), crop,
                 footprint[sl] & far_mask[sl], block_side=BOOTSTRAP_BLOCK_PX,
                 origin=(sl[0].start, sl[1].start), global_shape=grid.shape)
        results["dti"][stage] = {k: r[k] for k in
            ("dti", "tp", "fp", "fn", "n_truth", "n_emitted")}
        results["dti_far"][stage] = {k: rf[k] for k in
            ("dti", "tp", "fp", "fn", "n_truth", "n_emitted")}
        results["terms"][stage] = terms
        results["terms_far"][stage] = terms_far
    log(fold["name"], "scored", round(time.monotonic()-start), "s")
    return results


def summarize(folds, canary):
    """Pool sufficient statistics, bootstrap physical blocks, then gate paired folds."""
    def pool(key):
        return EH.pooled_summary({stage: sum((fold[key][stage] for fold in folds))
            for stage in ("stage1", "stage2", "stage3")},
            draws=BOOTSTRAP_DRAWS, candidate="stage2")
    all_ = pool("terms")
    far_ = pool("terms_far")
    gate_all = paired_gate(folds, "dti")
    gate_far = paired_gate(folds, "dti_far")
    issues = [r for r in canary if r["auc"] is None or r.get("flagged")]
    return {
        "pooled_all": all_, "pooled_far": far_,
        "paired_outer_all": gate_all, "paired_outer_far": gate_far,
        "canary": {"features_tested": len(canary), "not_cleared": issues,
                   "max_directional_auc": max((max(r["auc"], 1-r["auc"]) for r in canary
                                               if r["auc"] is not None), default=None),
                   "pass": not issues},
        "gate": {
            "all_gt_2se": gate_all["stage2_vs_stage1"]["passes_gt_2se"],
            "far_gt_1km_gt_2se": gate_far["stage2_vs_stage1"]["passes_gt_2se"],
            "clean_canary": not issues,
            "meets_protocol": (gate_all["stage2_vs_stage1"]["passes_gt_2se"]
                               and gate_far["stage2_vs_stage1"]["passes_gt_2se"]
                               and not issues),
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evidence/ac68_m1_holdout.json")
    args = ap.parse_args(argv)
    t0 = time.monotonic()
    grid = G57.load_grid()
    names, stack, fp = F.load_channels()
    idx = F.channel_index()
    c, s, coh = orientation(stack, idx, fp)
    folds, quad = H57.buffered_component_draw(grid, seed=SEED,
                      buffer_px=FEATURE_BUFFER_PX, boundary_px=OUTER_BUFFER_PX)
    canary = []
    results = [run_fold(fold, quad, grid, stack, idx, fp, c, s, canary)
               for fold in folds]
    scored = summarize(results, canary)
    for result in results:
        result.pop("terms")
        result.pop("terms_far")
    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hypothesis": "H68-M1 magnetic-edge-tangent auto-context (preregistered)",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator_version": EH.VERSION,
        "evaluator_hashes": EH.implementation_hashes(),
        "metric": dict(alpha=.2, beta=.8, triangular_radius_m=300,
                       spatial_bootstrap_cluster_m=BOOTSTRAP_BLOCK_PX*100),
        "emitter_version": E.EMITTER_VERSION,
        "emitter_sha256": hashlib.sha256((ROOT / "src/gems68/emission.py").read_bytes()).hexdigest(),
        "split": "gems57.holdout.buffered_component_draw v2, outer quadrant removed from fits and context, 12px outer/6px inner buffers, whole components + 3px feature collar",
        "seed": SEED, "budget": BUDGET, "emission": dict(nms=5, exclude_px=EXCLUDE_PX),
        "features": dict(stack_channels=len(names), stage1_channels=len(B.STAGE1_CHANNELS),
                         stage2_channels=len(B.stage2_feature_names('p1')),
                         orientation="tangent to structure-tensor gradient of mag_anom, sigma=3 px"),
        "n_withheld": sum(f["n_truth"] for f in results),
        "n_withheld_far": sum(f["n_truth_far"] for f in results),
        "folds": results, "summary": scored,
        "canary_results": canary,  # per-feature AUCs for a fresh rerun; old run may only save summary
        "elapsed_seconds": round(time.monotonic()-t0, 1),
        "caveat": "Withheld catalogue components are NOT the unknown panel truth; this is not an organizer score. CI is conditional on these folds, trained models and emission budget.",
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    log("H68-M1 HOLDOUT-DTI", {n: v["dti"] for n, v in scored["pooled_all"]["scores"].items()},
        "gate", scored["gate"], "elapsed", report["elapsed_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
