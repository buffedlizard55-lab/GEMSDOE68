#!/usr/bin/env python3
"""Hide-and-recover holdout for the auto-context lane.

Withholds whole catalogue segments with a buffer, derives every catalogue-based feature only
from the visible faults, masks visible faults pixel-exactly, and scores pooled DTI
(alpha 0.2, beta 0.8, 300 m triangular kernel).  Runs the stage-1 / stage-2 gate and the
per-feature leakage canary, then writes one JSON evidence file.

Usage:
    PYTHONPATH=src python scripts/run_holdout.py --folds 4 --max-pos 45000 --out evidence/holdout.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gems68 import metric
from gems68.autocontext import (AutoContext, CTX_NAMES, CatalogueContext, Fold, make_folds,
                                run_fold, sample_pixels, predict_blocked, fit_lgb, quadrant_blocks)
from gems68.data import (load_feature_stack, load_footprint, load_labels, rows_of, STACK_META,
                         build_feature_stack)


def paired_bootstrap_se(d: np.ndarray, rng, n_boot: int = 2000) -> float:
    """SE of the mean of a paired per-pixel difference."""
    if d.size == 0:
        return float("nan")
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return float(means.std(ddof=1))


def leakage_canary(fold: Fold, stack, raw_idx: list[int], names: list[str], labels,
                   rng, n_pos: int = 20000, n_neg: int = 60000) -> list[dict]:
    """AUC of each single feature at separating hidden truth from background.

    Protocol: AUC above 0.90 is leakage until proven otherwise.  The canary is run on the
    *hidden* block with catalogue features built only from the visible catalogue, so a high
    value means the feature has somehow seen the withheld labels.
    """
    py, px = np.nonzero(fold.hidden)
    if py.size > n_pos:
        i = rng.choice(py.size, n_pos, replace=False)
        py, px = py[i], px[i]
    by, bx = np.nonzero(fold.train_mask & (labels == 0) & ~fold.known)
    i = rng.choice(by.size, min(n_neg, by.size), replace=False)
    by, bx = by[i], bx[i]
    ys = np.concatenate([py, by])
    xs = np.concatenate([px, bx])
    y = np.concatenate([np.ones(py.size, np.int8), np.zeros(by.size, np.int8)])
    X = rows_of(stack, ys, xs)[np.ix_(np.arange(ys.size), np.asarray(raw_idx))]
    out = []
    for j, nm in enumerate(names):
        v = X[:, j].astype(np.float64)
        if np.nanstd(v) < 1e-12:
            out.append(dict(feature=nm, auc=0.5))
            continue
        a = float(roc_auc_score(y, v))
        out.append(dict(feature=nm, auc=round(max(a, 1 - a), 4), signed_auc=round(a, 4)))
    out.sort(key=lambda r: -r["auc"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--max-pos", type=int, default=45000)
    ap.add_argument("--buffer-px", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("evidence/holdout.json"))
    ap.add_argument("--no-canary", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    if not STACK_META.exists():
        build_feature_stack()
    stack, meta = load_feature_stack()
    labels = load_labels()
    footprint = load_footprint()
    names = meta["channels"]
    raw_idx = list(range(len(names)))
    print(f"[stack] {meta['n_channels']} channels {stack.shape} footprint_px={meta['footprint_px']}")

    # local strike from the structure-tensor channels
    ci = {n: i for i, n in enumerate(names)}
    c2 = np.asarray(stack[ci["st_cos2"]], np.float32)
    s2 = np.asarray(stack[ci["st_sin2"]], np.float32)
    theta = (0.5 * np.arctan2(s2, c2)).astype(np.float32)
    del c2, s2

    folds = make_folds(labels, footprint, buffer_px=args.buffer_px)
    folds = folds[: args.folds]
    rng = np.random.default_rng(args.seed)

    canary = None
    if not args.no_canary:
        t = time.time()
        canary = leakage_canary(folds[0], stack, raw_idx, names, labels, rng)
        print(f"[canary] top5 {[(c['feature'], c['auc']) for c in canary[:5]]}  ({time.time()-t:.0f}s)")

    results = []
    for f in folds:
        t = time.time()
        r = run_fold(f, stack, meta, labels, footprint, theta, raw_idx,
                     seed=args.seed, iterations=args.iterations, max_pos=args.max_pos)
        print(f"[fold {r.fold}] n_hidden={r.n_hidden} segs={r.n_segments} "
              f"s1={r.stage1['dti']:.5f} s2={r.stage2['dti']:.5f} "
              f"far s1={r.stage1_far['dti']:.5f} s2={r.stage2_far['dti']:.5f}  "
              f"({time.time()-t:.0f}s)")
        results.append(r)

    pooled_s1 = metric.pooled_dti([r.stage1 for r in results])
    pooled_s2 = metric.pooled_dti([r.stage2 for r in results])
    pooled_s1_far = metric.pooled_dti([r.stage1_far for r in results])
    pooled_s2_far = metric.pooled_dti([r.stage2_far for r in results])

    d = np.concatenate([r.per_truth_s2 - r.per_truth_s1 for r in results])
    d_far = np.concatenate([(r.per_truth_s2 - r.per_truth_s1)[r.per_truth_s1 >= 0] for r in results])
    se = paired_bootstrap_se(d, np.random.default_rng(args.seed + 1))
    diff = float(d.mean())
    z = diff / se if se > 0 else float("nan")

    gate = dict(delta=round(diff, 6), se=round(se, 6), z=round(z, 3),
                passes=bool(z > 2.0),
                delta_far=round(float(d_far.mean()), 6),
                s2_beats_s1_folds=int(sum(r.stage2["dti"] > r.stage1["dti"] for r in results)),
                s2_beats_s1_far_folds=int(sum(r.stage2_far["dti"] > r.stage1_far["dti"]
                                              for r in results)))

    payload = dict(
        schema=1,
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        evaluator="gems68.hide_and_recover.v1",
        protocol="hide whole catalogue segments with a 12 px buffer; catalogue features from "
                 "visible faults only; visible faults masked pixel-exactly; pooled DTI "
                 "alpha=0.2 beta=0.8 R=300 m triangular kernel",
        metric=dict(alpha=metric.ALPHA, beta=metric.BETA, radius_m=metric.RADIUS_M,
                    pixel_m=metric.PIXEL_M),
        data=dict(feature_stack_sha256=meta["source_sha256"], n_channels=meta["n_channels"],
                  footprint_px=meta["footprint_px"], grid=meta["grid"]),
        n_folds=len(results),
        withheld_positives_total=int(sum(r.n_hidden for r in results)),
        withheld_segments_total=int(sum(r.n_segments for r in results)),
        stage1_pooled=pooled_s1, stage2_pooled=pooled_s2,
        stage1_pooled_far1km=pooled_s1_far, stage2_pooled_far1km=pooled_s2_far,
        gate=gate,
        per_fold=[dict(fold=r.fold, n_hidden=r.n_hidden, n_segments=r.n_segments,
                       stage1=round(r.stage1["dti"], 6), stage2=round(r.stage2["dti"], 6),
                       stage1_far=round(r.stage1_far["dti"], 6),
                       stage2_far=round(r.stage2_far["dti"], 6),
                       seconds={k: round(v, 1) if not isinstance(v, list) else [round(x, 1) for x in v]
                                for k, v in r.seconds.items()}) for r in results],
        canary_top20=(canary or [])[:20],
        canary_flags=[c for c in (canary or []) if c["auc"] > 0.90],
        seconds=round(time.time() - t_start, 1),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(dict(stage1=pooled_s1["dti"], stage2=pooled_s2["dti"], gate=gate), indent=1))
    print(f"[write] {args.out}  ({payload['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
