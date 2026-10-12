#!/usr/bin/env python3
"""Build the GEMSDOE68 auto-context submission end to end.

1. Stage 1 -- out-of-fold probabilities from spatially blocked, buffered K-fold over the
   full catalogue.  Never in-fold.
2. Stage 2 -- auto-context stacked refinement (Tu, CVPR 2008), iterated twice, on
   (a) stage-1 probabilities sampled at +-1..15 px along and across the local strike,
   (b) multi-scale anisotropic smooths of that map,
   (c) the raw gradient / tilt / scarp channels,
   (d) catalogue-context features (distance / orientation / spacing to the nearest mapped
       fault, with the target block's labels masked).
3. Emission -- a budgeted, minimum-separation dot set, restricted to pixels more than 200 m
   from the mapped catalogue, because the organizer masks mapped-fault pixels out of scoring
   and every unit of emitted mass raises the DTI denominator by exactly alpha.
4. Write both twins (``-zeros`` all-finite and ``-nan`` footprint-masked), validate them
   against the organizer template, and run the registry duplicate check.

Usage:
    PYTHONPATH=src python scripts/build_submission.py --budget 40000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gems68 import metric
from gems68.autocontext import (AutoContext, CTX_NAMES, CatalogueContext, disc, fit_lgb,
                                predict_blocked, predict_grid, quadrant_blocks, rank01, sample_pixels,
                                scatter_to_grid, stratified_positives)
from gems68.data import (build_feature_stack, load_feature_stack, load_footprint, load_labels,
                         rows_of, STACK_META)
from gems68.submission import (dot_overlap, spearman, validate_submission, write_submission,
                               zip_submission)

REG_DIR = Path("registry/rasters")


def train_full(stack, names, raw_idx, labels, footprint, theta, inner_folds=4,
               iterations=2, max_pos=60000, buffer_px=12, seed=0, log=print):
    """Stage 1 (blocked OOF over the whole catalogue) then stage 2 (auto-context x N)."""
    rng = np.random.default_rng(seed)
    H, W = footprint.shape
    ys_f, xs_f = np.nonzero(footprint)
    blocks = quadrant_blocks(H, W, footprint)

    def raw_rows(y, x):
        return rows_of(stack, y, x)[np.ix_(np.arange(y.size), np.asarray(raw_idx))]

    # ---- stage 1 ------------------------------------------------------------------------
    t0 = time.time()
    prob_oof = np.zeros(H * W, dtype=np.float32)
    flat = np.zeros((H, W), np.int64)
    flat[ys_f, xs_f] = np.arange(ys_f.size)
    for b in range(inner_folds):
        va = blocks == b
        buf = ndimage.binary_dilation(va, structure=disc(buffer_px)) & ~va
        tr = footprint & ~va & ~buf
        py, px = stratified_positives(labels, tr & (labels == 1), rng)
        ny, nx = np.nonzero(tr & (labels == 0))
        j = rng.choice(ny.size, int(min(ny.size, py.size * 4)), replace=False)
        ny, nx = ny[j], nx[j]
        ty, tx = np.concatenate([py, ny]), np.concatenate([px, nx])
        y = np.concatenate([np.ones(py.size, np.int8), np.zeros(ny.size, np.int8)])
        bst = fit_lgb(raw_rows(ty, tx), y)
        g = predict_grid(bst, None, stack, raw_idx, va)
        prob_oof[flat[ys_f, xs_f]] = g[ys_f, xs_f]
        log(f"  [s1] block {b}: train {ty.size} px, predict {int(va.sum())} px", flush=True)
    stage1 = scatter_to_grid(prob_oof, ys_f, xs_f, (H, W))
    t1 = time.time() - t0

    # ---- stage 2 ------------------------------------------------------------------------
    ctx = CatalogueContext.build((labels == 1) & footprint, theta)
    train_pos = (labels == 1) & footprint
    train_neg = footprint & (labels == 0)
    cur = stage1
    out = {"stage1_seconds": round(t1, 1), "stage2_seconds": [], "iterations": iterations}
    for it in range(iterations):
        t = time.time()
        ac = AutoContext(cur, theta, ctx)

        def ac_rows(y, x, _ac=ac):
            return np.concatenate([_ac.rows(y, x), raw_rows(y, x)], axis=1)

        ty, tx, y = sample_pixels(train_pos, train_neg, rng, max_pos=max_pos)
        bst = fit_lgb(ac_rows(ty, tx), y, num_boost_round=300)
        cur = predict_grid(bst, ac, stack, raw_idx, footprint)
        out["stage2_seconds"].append(round(time.time() - t, 1))
        log(f"  [s2] iter {it}: train {ty.size} px, predict {ys_f.size} px "
            f"({time.time()-t:.0f}s)")
    return stage1, cur, out


def emit(prob: np.ndarray, footprint: np.ndarray, catalogue: np.ndarray,
         exclude_px: float = 2.0, r_min: float = 2.4, budget: int = 40000) -> tuple[np.ndarray, dict]:
    """Budgeted minimum-separation dot set, off-catalogue only.

    The organizer masks mapped-fault pixels, so mass there can never earn credit while still
    raising the denominator by alpha -- emitting on the catalogue is pure loss.  ``r_min``
    keeps dots far enough apart that each one carries its own 300 m kernel rather than
    sharing one with a neighbour.
    """
    d_cat = ndimage.distance_transform_edt(~catalogue)
    cand = footprint & (d_cat > exclude_px)
    # local maxima only: a dot that is not a local peak is dominated by its neighbour
    peak = prob >= ndimage.maximum_filter(prob, size=5, mode="constant", cval=-1.0)
    cand &= peak
    cy, cx = np.nonzero(cand)
    order = np.argsort(-prob[cy, cx], kind="stable")
    cy, cx = cy[order], cx[order]
    r = int(np.ceil(r_min))
    blocked = np.zeros(prob.shape, bool)
    keep_y, keep_x = [], []
    for y, x in zip(cy, cx):
        if blocked[y, x]:
            continue
        keep_y.append(y)
        keep_x.append(x)
        y0, y1 = max(0, y - r), min(prob.shape[0], y + r + 1)
        x0, x1 = max(0, x - r), min(prob.shape[1], x + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        blocked[y0:y1, x0:x1] |= ((yy - y) ** 2 + (xx - x) ** 2) <= r_min * r_min
        if len(keep_y) >= budget:
            break
    out = np.zeros(prob.shape, np.float32)
    out[np.asarray(keep_y, int), np.asarray(keep_x, int)] = 1.0
    info = dict(candidates=int(cand.sum()), emitted=len(keep_y), budget=budget,
                r_min_px=r_min, exclude_catalogue_px=exclude_px,
                on_catalogue=int((out > 0)[d_cat <= exclude_px].sum()))
    return out, info


def registry_check(raster: np.ndarray, footprint: np.ndarray, surface: np.ndarray | None = None,
                   limit=40) -> list[dict]:
    """Lane-drift check.  Spearman on the continuous surface (rank correlation is
    meaningless on a sparse binary dot field -- two similar dot sets score negatively
    because ties dominate); 3 px dot overlap on the final dots.  The protocol asks for
    both, on the surface before placement and on the dots after."""
    out = []
    if not REG_DIR.exists():
        return out
    files = sorted(REG_DIR.glob("*.tif"))[:limit]
    for f in files:
        try:
            with rasterio.open(f) as s:
                if s.count != 1 or (s.height, s.width) != raster.shape:
                    out.append(dict(file=f.name, skip=f"grid mismatch {s.height}x{s.width}"))
                    continue
                a = s.read(1).astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            out.append(dict(file=f.name, skip=f"unreadable: {exc}"))
            continue
        surf = surface if surface is not None else raster
        out.append(dict(file=f.name,
                        spearman=round(spearman(np.where(footprint, surf, np.nan),
                                                 np.where(footprint, a, np.nan)), 4),
                        overlap3px=round(dot_overlap(raster, a, footprint)["within"], 4),
                        sha256_12=""))
        out[-1]["sha256_12"] = __import__("hashlib").sha256(f.read_bytes()).hexdigest()[:12]
    out.sort(key=lambda r: -abs(r.get("spearman") or -9))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=40000)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--inner-folds", type=int, default=4)
    ap.add_argument("--max-pos", type=int, default=60000)
    ap.add_argument("--r-min", type=float, default=2.4)
    ap.add_argument("--exclude-px", type=float, default=2.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--outdir", type=Path, default=Path("docs/downloads"))
    args = ap.parse_args()

    t_start = time.time()
    if not STACK_META.exists():
        build_feature_stack()
    stack, meta = load_feature_stack()
    labels = load_labels()
    footprint = load_footprint()
    names = meta["channels"]
    raw_idx = list(range(len(names)))
    ci = {n: i for i, n in enumerate(names)}
    theta = (0.5 * np.arctan2(np.asarray(stack[ci["st_sin2"]], np.float32),
                              np.asarray(stack[ci["st_cos2"]], np.float32))).astype(np.float32)

    log = lambda m, **kw: print(m, flush=True)
    log(f"[stack] {meta['n_channels']} channels, footprint {meta['footprint_px']} px")
    stage1, stage2, timing = train_full(stack, names, raw_idx, labels, footprint, theta,
                                        inner_folds=args.inner_folds,
                                        iterations=args.iterations, max_pos=args.max_pos,
                                        seed=0, log=log)

    catalogue = (labels == 1) & footprint
    dots, emit_info = emit(stage2, footprint, catalogue, exclude_px=args.exclude_px,
                           r_min=args.r_min, budget=args.budget)
    log(f"[emit] {emit_info}")

    # monotone normalisation of the continuous surface, for the diagnostic twin
    surface = np.zeros_like(stage2)
    surface[footprint] = rank01(stage2[footprint])

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = args.name or f"gems68-autoctx-{args.budget}-{stamp}"
    results = {}
    for mode in ("zeros", "nan"):
        p = args.outdir / f"{base}-{mode}.tif"
        results[mode] = write_submission(dots, footprint, p, outside=mode)
        results[mode]["format_check"] = validate_submission(p).as_json()
        if mode == "zeros":
            results[mode + "_zip"] = zip_submission(p)
    surf_info = write_submission(surface, footprint,
                                 args.outdir / f"{base}-surface-zeros.tif", outside="zeros")
    results["surface"] = surf_info
    results["surface"]["format_check"] = validate_submission(surf_info["path"]).as_json()

    reg = registry_check(dots, footprint, surface=surface)
    verdict = dict(duplicate=False, worst_spearman=None, worst_overlap=None)
    if reg:
        sp = [r.get("spearman") for r in reg if isinstance(r.get("spearman"), float)]
        ov = [r.get("overlap3px") for r in reg if isinstance(r.get("overlap3px"), float)]
        if sp:
            verdict["worst_spearman"] = max(sp)
            verdict["duplicate"] = verdict["duplicate"] or (max(sp) > 0.90)
        if ov:
            verdict["worst_overlap"] = max(ov)
            verdict["duplicate"] = verdict["duplicate"] or (max(ov) > 0.70)
    verdict["lane_ok"] = not verdict["duplicate"]

    payload = dict(
        schema=1, generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        lane="auto-context stacked refinement (Tu, CVPR 2008, doi:10.1109/CVPR.2008.4587436)",
        base_name=base,
        note=(f"GEMSDOE68 autoctx stage2 x{args.iterations} budget {args.budget} "
              f"rmin {args.r_min} offcat>{args.exclude_px}px"[:200]),
        data=dict(feature_stack_sha256=meta["source_sha256"], n_channels=meta["n_channels"],
                  footprint_px=meta["footprint_px"], grid=meta["grid"]),
        timing=timing, emission=emit_info,
        marginal_bar_at_0p2778=round(metric.marginal_bar(0.2778), 5),
        outputs=results, registry=reg[:25], registry_verdict=verdict,
        seconds=round(time.time() - t_start, 1),
    )
    out_json = Path("evidence/submission_build.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=1))
    log(json.dumps(dict(emission=emit_info, verdict=verdict,
                        zeros_sha256=results["zeros"]["sha256"],
                        zeros_ok=results["zeros"]["format_check"]["ok"],
                        nan_ok=results["nan"]["format_check"]["ok"]), indent=1))
    log(f"[write] {out_json}  ({payload['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
