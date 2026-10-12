#!/usr/bin/env python3
"""GEMSDOE68 -- auto-context stacked-refinement experiment (HOLDOUT-DTI).

Lane (single method paragraph from the standing brief)
-----------------------------------------------------
Auto-context (Tu, CVPR 2008, doi:10.1109/CVPR.2008.4587436; extended TPAMI
version at pages.ucsd.edu/~ztu/publication/pami_autocontext.pdf) trains a second
classifier on the first classifier's probability maps plus the original
features, so it can learn spatial regularities a per-pixel model misses:
continuity, parallelism, en-echelon spacing.

* **Stage 1** -- LightGBM on the cached 45-channel stack plus catalogue-context
  features derived from the *visible* faults only.  Out-of-fold probabilities
  come from spatially blocked, buffered K-fold -- never in-fold predictions.
* **Stage 2** -- LightGBM on four kinds of input: stage-1 probabilities sampled
  at +/-1..15 px along and across the local orientation; multi-scale anisotropic
  smooths of those probabilities; the raw gradient/tilt/scarp channels; and
  catalogue-context features computed with the target block's labels masked.
* **Iterate twice** -- stage 3 repeats the refinement on stage-2 probabilities.
* **Nested blocked CV** -- 4 outer spatial quadrant folds (the shared
  hide-and-recover holdout) x 4 inner spatial blocks inside each outer training
  domain, with a 600 m inner buffer and a 1.2 km outer buffer.
* **Gate** -- stage 2 / stage 3 must beat stage 1 on outer blocks by more than
  two standard errors, including on withheld pixels more than 1 km from any
  mapped fault.

Shared tools are reused, never forked: ``gems57.holdout`` (hide-and-recover
spatially blocked holdout), ``gems57.metric`` (DTI, alpha 0.2 / beta 0.8 /
300 m triangular kernel), ``gems.submission_writer`` / ``gems.validate_submission``.

Usage::

    PYTHONPATH=src python scripts/run_autocontext.py --seeds 20 \\
        --out evidence/ac68_seed20.json --dot-budget 32000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems.submission_writer import sha256_file                      # noqa: E402
from gems57 import grid as G57                                      # noqa: E402
from gems57 import holdout as H57                                   # noqa: E402
from gems57 import metric as M57                                    # noqa: E402
from gems68 import blocks as B                                      # noqa: E402
from gems68 import catalogue_context as CC                          # noqa: E402
from gems68 import channel_stack as F                                    # noqa: E402

INNER_BLOCKS = 4
INNER_BUFFER_PX = 6            # 600 m spatial separation inside a training domain
OUTER_BUFFER_PX = H57.DOMAIN_ERODE   # 12 px = 1.2 km, the shared holdout erosion
FAR_PX = 10                    # "more than 1 km from any mapped fault"
ROW_BLOCK = 250                # prediction strip height
SWEEP_RADII = (0, 3)           # min distance (px) from the visible catalogue
SWEEP_BUDGETS = (20000, 32000, 44090)
NEG_RATIO = 3                   # negatives sampled per positive, both runs
SEED_ROW_SAMPLE = 20261012
CAT_N = len(B.CATALOGUE_CONTEXT)
S1_N = len(B.STAGE1_CHANNELS)
RAW_N = len(B.STAGE2_RAW)
CTX_N = len(B.context_feature_names("p"))

LGBM_PARAMS = dict(
    objective="binary", n_estimators=220, learning_rate=0.09, num_leaves=63,
    min_child_samples=40, feature_fraction=0.9, bagging_fraction=0.9,
    bagging_freq=1, lambda_l2=1.0, verbose=-1, n_jobs=2, seed=7,
)


def _jsonable(o):
    """numpy scalars/arrays are not JSON serialisable; make them plain Python."""
    import numpy as _np

    if isinstance(o, _np.generic):
        return o.item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_model():
    import lightgbm as lgb

    return lgb.LGBMClassifier(**LGBM_PARAMS)


def quadrant_partition(mask: np.ndarray) -> np.ndarray:
    """Label-blind 4-way spatial partition of ``mask`` (ids 0..3, -1 outside)."""
    ys, xs = np.nonzero(mask)
    out = np.full(mask.shape, -1, np.int8)
    if ys.size == 0:
        return out
    ym, xm = int(np.median(ys)), int(np.median(xs))
    out[ys, xs] = ((ys >= ym).astype(np.int8) * 2 + (xs >= xm).astype(np.int8))
    return out


def row_slices(h: int, step: int = ROW_BLOCK):
    for r0 in range(0, h, step):
        yield slice(r0, min(h, r0 + step))


def bbox_of(mask: np.ndarray, h: int, w: int, pad: int = 20):
    """Row/column window covering ``mask`` plus ``pad`` pixels of halo."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return ((max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)),
            (max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)))


def predict_box(model, feat_fn, h: int, w: int, box) -> np.ndarray:
    """Dense prediction restricted to a row/column window.

    ``feat_fn(rows, cols)`` returns the feature matrix for that window.  Scoring
    only the window keeps the cost proportional to the region of interest.
    """
    out = np.zeros((h, w), np.float32)
    if box is None:
        return out
    (r0, r1), (c0, c1) = box
    for a in range(r0, r1, ROW_BLOCK):
        rows = slice(a, min(r1, a + ROW_BLOCK))
        got = feat_fn(rows, (c0, c1))
        X = got[0] if isinstance(got, tuple) else got
        if X.shape[0] == 0:
            continue
        p = model.predict_proba(X)[:, 1].astype(np.float32)
        out[rows, c0:c1] = p.reshape(rows.stop - rows.start, c1 - c0)
    return out


def topn_dots(score: np.ndarray, eligible: np.ndarray, n: int) -> np.ndarray:
    """Flat indices of the ``n`` highest-scoring eligible cells."""
    s = np.where(eligible, score, -np.inf).ravel()
    n_avail = int(np.isfinite(s).sum() and (s > -np.inf).sum())
    n = min(n, n_avail)
    if n <= 0:
        return np.empty(0, np.int64)
    idx = np.argpartition(-s, n - 1)[:n]
    return idx[s[idx] > -np.inf]


def dti_of_dots(dots_idx: np.ndarray, truth: np.ndarray, known: np.ndarray,
                shape) -> dict:
    dots = np.zeros(shape, bool)
    if dots_idx.size:
        dots.ravel()[dots_idx] = True
    return M57.dti_binary(dots, truth, valid=None, known=known)


# ---------------------------------------------------------------------------
def cell_setup(cell, ctx, grid):
    """Geometry of one holdout cell: training domain, inner blocks, row sample."""
    h, w = grid.height, grid.width
    key = cell.key
    hidden = ctx.hidden_by_cell[key]
    visible = grid.catalogue & ~hidden
    quad = ctx.quad == cell.fold
    domain = ndi.binary_erosion(quad, iterations=OUTER_BUFFER_PX) & grid.footprint
    train_domain = grid.footprint & (ndi.distance_transform_edt(~hidden) > OUTER_BUFFER_PX)
    region = domain & ~visible                    # scored domain for this cell
    truth = hidden & domain

    blk = quadrant_partition(train_domain)
    inner = [blk == b for b in range(INNER_BLOCKS)]
    dist_to_blk = [ndi.distance_transform_edt(~m) for m in inner]

    rng = np.random.default_rng(SEED_ROW_SAMPLE)
    pos_yy, pos_xx = np.nonzero(visible & train_domain)
    neg_yy, neg_xx = np.nonzero(train_domain & ~grid.catalogue)
    n_neg = min(neg_yy.size, NEG_RATIO * pos_yy.size)
    sel = rng.choice(neg_yy.size, size=n_neg, replace=False)
    ry = np.concatenate([pos_yy, neg_yy[sel]]).astype(np.int64)
    rx = np.concatenate([pos_xx, neg_xx[sel]]).astype(np.int64)
    y = np.concatenate([np.ones(pos_yy.size, np.float32), np.zeros(n_neg, np.float32)])
    rb = blk[ry, rx]
    keep = rb >= 0
    ry, rx, y, rb = ry[keep], rx[keep], y[keep], rb[keep]
    far = [d[ry, rx] > INNER_BUFFER_PX for d in dist_to_blk]
    return dict(key=key, hidden=hidden, visible=visible, domain=domain,
                train_domain=train_domain, region=region, truth=truth, blk=blk,
                inner=inner, dist_to_blk=dist_to_blk, ry=ry, rx=rx, y=y, rb=rb,
                far=far, n_truth=int(truth.sum()), n_region=int(region.sum()),
                n_rows=int(ry.size), n_pos=int(y.sum()))


def stage1_row_matrix(stack, idx, ry, rx):
    X = np.empty((ry.size, S1_N), np.float32)
    for j, name in enumerate(B.STAGE1_CHANNELS):
        X[:, j] = np.asarray(stack[idx[name]])[ry, rx]
    return X


def own_block_catalogue_context(cs, visible, footprint, local_cos2, local_sin2):
    """Catalogue context for each training row with its own inner block masked."""
    Xc = np.empty((cs["ry"].size, CAT_N), np.float32)
    for b in range(INNER_BLOCKS):
        m = cs["rb"] == b
        if not m.any():
            continue
        cc = CC.catalogue_context(visible & ~cs["inner"][b], footprint,
                                  local_cos2, local_sin2)
        for j, name in enumerate(B.CATALOGUE_CONTEXT):
            Xc[m, j] = cc[name][cs["ry"][m], cs["rx"][m]]
        del cc
    return Xc


def context_at_rows(prev, ry, rx, local_cos2, local_sin2, chunk: int = 128):
    """Stage-2 context channels evaluated only at the sampled training rows.

    Rows are processed in contiguous chunks so the strip machinery is reused
    instead of being invoked once per distinct raster row.
    """
    out = np.empty((ry.size, CTX_N), np.float32)
    order = np.argsort(ry, kind="stable")
    ys, xs, o = ry[order], rx[order], order
    i = 0
    n = ys.size
    while i < n:
        r_lo = int(ys[i])
        j = i
        while j < n and ys[j] < r_lo + chunk:
            j += 1
        r_hi = min(int(ys[j - 1]) + 1, r_lo + chunk)
        blk = B.build_context_block(prev, slice(r_lo, r_hi), local_cos2, local_sin2)
        rr = ys[i:j] - r_lo
        out[o[i:j]] = blk[:, rr, xs[i:j]].T
        i = j
    return out


def stage2_row_matrix(stack, idx, prev, cs, own_cat, local_cos2, local_sin2):
    ctx = context_at_rows(prev, cs["ry"], cs["rx"], local_cos2, local_sin2)
    raw = np.empty((cs["ry"].size, RAW_N), np.float32)
    for j, name in enumerate(B.STAGE2_RAW):
        raw[:, j] = np.asarray(stack[idx[name]])[cs["ry"], cs["rx"]]
    return np.hstack([ctx, raw, own_cat])


def strip_features_stage1(stack, idx, rows, cc, cols=None):
    ch = B.read_rows(stack, idx, rows, 0, B.STAGE1_CHANNELS, cols)
    Xc = np.empty((CAT_N, ch.shape[1], ch.shape[2]), np.float32)
    for j, name in enumerate(B.CATALOGUE_CONTEXT):
        Xc[j] = cc[name][rows][:, cols[0]:cols[1]] if cols else cc[name][rows]
    return np.concatenate([ch.reshape(S1_N, -1).T, Xc.reshape(CAT_N, -1).T], axis=1)


def strip_features_stage2(stack, idx, rows, prev, cc, local_cos2, local_sin2,
                          cols=None):
    ctx = B.build_context_block(prev, rows, local_cos2, local_sin2, cols)
    raw = B.core_rows(B.read_rows(stack, idx, rows, B.PAD_STAGE2, B.STAGE2_RAW, cols),
                      B.PAD_STAGE2, rows.stop - rows.start)
    Xc = np.empty((CAT_N, raw.shape[1], raw.shape[2]), np.float32)
    for j, name in enumerate(B.CATALOGUE_CONTEXT):
        Xc[j] = cc[name][rows][:, cols[0]:cols[1]] if cols else cc[name][rows]
    return np.concatenate([ctx.reshape(CTX_N, -1).T, raw.reshape(RAW_N, -1).T,
                           Xc.reshape(CAT_N, -1).T], axis=1)


def run_cell(cell, ctx, grid, stack, idx, footprint, local_cos2, local_sin2,
             neg_ratio, dot_budget, canary):
    h, w = grid.height, grid.width
    cs = cell_setup(cell, ctx, grid)
    log(f"  {cs['key']}: rows={cs['n_rows']} pos={cs['n_pos']} "
        f"region={cs['n_region']} truth={cs['n_truth']} "
        f"blocks={[int(m.sum()) for m in cs['inner']]}")
    visible, train_domain = cs["visible"], cs["train_domain"]

    X1 = stage1_row_matrix(stack, idx, cs["ry"], cs["rx"])
    own_cat = own_block_catalogue_context(cs, visible, footprint,
                                          local_cos2, local_sin2)
    A1 = np.hstack([X1, own_cat])

    # ---------------- leakage canary: single-channel AUC, no model ----------
    if canary is not None:
        neg_pool = train_domain & ~grid.catalogue
        ny, nx = np.nonzero(neg_pool)
        rng = np.random.default_rng(99)
        sel = rng.choice(ny.size, size=min(ny.size, 5 * cs["n_pos"]), replace=False)
        neg_vals = {}
        for name in B.STAGE1_CHANNELS:
            a = np.asarray(stack[idx[name]])
            neg_vals[name] = a[ny[sel], nx[sel]]
        from sklearn.metrics import roc_auc_score
        for name in B.STAGE1_CHANNELS:
            a = np.asarray(stack[idx[name]])
            pos = a[cs["ry"][cs["y"] > 0], cs["rx"][cs["y"] > 0]]
            neg = neg_vals[name]
            gp = np.isfinite(pos)
            gn = np.isfinite(neg)
            if gp.sum() < 50 or gn.sum() < 50 or np.ptp(pos[gp]) == 0:
                continue
            ytrue = np.concatenate([np.ones(int(gp.sum())), np.zeros(int(gn.sum()))])
            yscore = np.concatenate([pos[gp], neg[gn]])
            try:
                auc = float(roc_auc_score(ytrue, yscore))
            except ValueError:
                continue
            canary.append({"cell": cs["key"], "channel": name, "auc": auc})

    # ---- catalogue contexts are built on demand and released immediately:
    # keeping several resident at once exceeds the sandbox memory budget.
    def cc_for(mask):
        return CC.catalogue_context(mask, footprint, local_cos2, local_sin2)

    box_blk = [bbox_of(m, h, w) for m in cs["inner"]]
    box_region = bbox_of(cs["region"], h, w)
    box_train = bbox_of(train_domain, h, w)

    # ============================== stage 1 ================================
    _t = time.time()
    p1_oof = np.zeros((h, w), np.float32)
    for b in range(INNER_BLOCKS):
        tr = cs["far"][b] & (cs["rb"] != b)
        if tr.sum() < 50 or cs["y"][tr].sum() < 10:
            continue
        m = make_model()
        m.fit(A1[tr], cs["y"][tr])
        cc = cc_for(visible & ~cs["inner"][b])
        p = predict_box(m, lambda rows, cols, m=m, cc=cc:
                        strip_features_stage1(stack, idx, rows, cc, cols),
                        h, w, box_blk[b])
        p1_oof[cs["inner"][b]] = p[cs["inner"][b]]
        del p, cc
    m1_full = make_model()
    m1_full.fit(A1, cs["y"])
    cc_full = cc_for(visible)
    p1_region = predict_box(m1_full, lambda rows, cols:
                            strip_features_stage1(stack, idx, rows, cc_full, cols),
                            h, w, box_region)
    p1_comb = np.where(train_domain, p1_oof, p1_region).astype(np.float32)
    imp1 = sorted(zip(list(B.STAGE1_CHANNELS) + list(B.CATALOGUE_CONTEXT),
                      m1_full.feature_importances_), key=lambda t: -t[1])[:12]

    log(f"    stage 1 done in {time.time()-_t:.0f}s")
    _t = time.time()
    # ============================== stage 2 ================================
    A2 = stage2_row_matrix(stack, idx, p1_comb, cs, own_cat, local_cos2, local_sin2)
    p2_oof = np.zeros((h, w), np.float32)
    for b in range(INNER_BLOCKS):
        tr = cs["far"][b] & (cs["rb"] != b)
        if tr.sum() < 50 or cs["y"][tr].sum() < 10:
            continue
        m = make_model()
        m.fit(A2[tr], cs["y"][tr])
        cc = cc_for(visible & ~cs["inner"][b])
        p = predict_box(m, lambda rows, cols, m=m, cc=cc:
                        strip_features_stage2(stack, idx, rows, p1_comb, cc,
                                              local_cos2, local_sin2, cols),
                        h, w, box_blk[b])
        p2_oof[cs["inner"][b]] = p[cs["inner"][b]]
        del p, cc
    m2_full = make_model()
    m2_full.fit(A2, cs["y"])
    p2_region = predict_box(m2_full, lambda rows, cols:
                            strip_features_stage2(stack, idx, rows, p1_comb, cc_full,
                                                  local_cos2, local_sin2, cols),
                            h, w, box_region)
    p2_comb = np.where(train_domain, p2_oof, p2_region).astype(np.float32)
    imp2 = sorted(zip(B.stage2_feature_names("p1"), m2_full.feature_importances_),
                  key=lambda t: -t[1])[:12]

    log(f"    stage 2 done in {time.time()-_t:.0f}s")
    _t = time.time()
    # ====================== stage 3 (iterate twice) ========================
    A3 = stage2_row_matrix(stack, idx, p2_comb, cs, own_cat, local_cos2, local_sin2)
    m3 = make_model()
    m3.fit(A3, cs["y"])
    p3_region = predict_box(m3, lambda rows, cols:
                            strip_features_stage2(stack, idx, rows, p2_comb, cc_full,
                                                  local_cos2, local_sin2, cols),
                            h, w, box_region)
    log(f"    stage 3 done in {time.time()-_t:.0f}s")
    _t = time.time()
    # ============================ emission + DTI ===========================
    d_vis = ndi.distance_transform_edt(~visible)
    far_truth = cs["truth"] & (d_vis > FAR_PX)
    el0 = cs["region"] & ~visible
    res = {"cell": cs["key"], "fold": cell.fold, "mode": cell.mode,
           "n_truth": cs["n_truth"], "n_truth_far": int(far_truth.sum()),
           "n_region": cs["n_region"], "n_train_rows": cs["n_rows"],
           "n_pos": cs["n_pos"], "n_blocks": [int(m.sum()) for m in cs["inner"]],
           "dot_budget": int(dot_budget), "dti": {}, "dti_far": {}, "dots": {},
           "sweep": [], "stage1_top_features": imp1, "stage2_top_features": imp2}
    fields = (("stage1", p1_region), ("stage2", p2_region), ("stage3", p3_region))
    for stage, field in fields:
        dots = topn_dots(field, el0, dot_budget)
        r = dti_of_dots(dots, cs["truth"], visible, (h, w))
        rf = dti_of_dots(dots, far_truth, visible, (h, w))
        res["dots"][stage] = int(dots.size)
        res["dti"][stage] = {k: r[k] for k in ("dti", "tp", "fp", "fn", "n_truth",
                                               "n_emitted", "coverage")}
        res["dti_far"][stage] = {k: rf[k] for k in ("dti", "tp", "fp", "fn",
                                                    "n_truth", "n_emitted",
                                                    "coverage")}
    # ---- emission-parameter sweep (catalogue exclusion radius x dot budget) --
    for rad in SWEEP_RADII:
        el = el0 & (d_vis > rad)
        for stage, field in fields:
            srt = np.where(el, field, -np.inf).ravel()
            order = np.argsort(-srt)[:max(SWEEP_BUDGETS)]
            order = order[srt[order] > -np.inf]
            for n in SWEEP_BUDGETS:
                if n > order.size:
                    continue
                dots = order[:n]
                r = dti_of_dots(dots, cs["truth"], visible, (h, w))
                rf = dti_of_dots(dots, far_truth, visible, (h, w))
                res["sweep"].append({
                    "stage": stage, "radius_px": int(rad), "budget": int(n),
                    "dti": r["dti"], "coverage": r["coverage"],
                    "dti_far": rf["dti"], "n_emitted": int(dots.size)})
    log(f"    emission + scoring done in {time.time()-_t:.0f}s")
    return res


def paired_gate(results, key="dti"):
    """Paired per-outer-block comparison of stages 2 and 3 against stage 1."""
    s1 = np.array([r[key]["stage1"]["dti"] for r in results])
    s2 = np.array([r[key]["stage2"]["dti"] for r in results])
    s3 = np.array([r[key]["stage3"]["dti"] for r in results])
    out = {}
    for name, arr in (("stage2_vs_stage1", s2 - s1), ("stage3_vs_stage1", s3 - s1),
                      ("stage3_vs_stage2", s3 - s2)):
        d = arr
        se = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else float("nan")
        out[name] = {
            "mean_delta": float(d.mean()),
            "se": se,
            "n_blocks": int(d.size),
            "t_stat": float(d.mean() / se) if se and np.isfinite(se) and se > 0 else None,
            "passes_gt_2se": bool(d.mean() > 2 * se) if np.isfinite(se) else False,
            "per_block": [float(x) for x in d],
        }
    out["stage_means"] = {"stage1": float(s1.mean()), "stage2": float(s2.mean()),
                          "stage3": float(s3.mean())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[20])
    ap.add_argument("--modes", nargs="+", default=["all"])
    ap.add_argument("--out", default="evidence/ac68.json")
    ap.add_argument("--dot-budget", type=int, default=32000)
    ap.add_argument("--neg-ratio", type=int, default=NEG_RATIO)
    ap.add_argument("--no-canary", action="store_true")
    args = ap.parse_args(argv)

    NEG_RATIO = args.neg_ratio
    t0 = time.time()
    log("loading grid + cached channel stack")
    grid = G57.load_grid(ROOT / "data" / "bridge")
    names, stack, footprint = F.load_channels()
    idx = F.channel_index()
    h, w = grid.height, grid.width

    log("local orientation of the detrended-elevation gradient (fold invariant)")
    gmag = np.array(stack[idx["det_elev_gmag"]], dtype=np.float32)  # writable copy
    gmag[~footprint] = 0.0
    local_cos2, local_sin2, _ = B.structure_tensor_orientation(gmag, sigma=3.0)
    del gmag

    log("building the shared hide-and-recover holdout")
    ctx = H57.build_holdout(grid, seeds=tuple(args.seeds), modes=tuple(args.modes))
    log(f"{len(ctx.cells)} holdout cells")

    canary = [] if not args.no_canary else None
    results = []
    for cell in ctx.cells:
        t = time.time()
        results.append(run_cell(cell, ctx, grid, stack, idx, footprint,
                                local_cos2, local_sin2, NEG_RATIO,
                                args.dot_budget, canary))
        r = results[-1]
        log(f"  -> {r['cell']} dti s1={r['dti']['stage1']['dti']:.4f} "
            f"s2={r['dti']['stage2']['dti']:.4f} s3={r['dti']['stage3']['dti']:.4f} "
            f"({time.time()-t:.0f}s)")

    gate = paired_gate(results)
    gate_far = paired_gate(results, key="dti_far")

    # pooled DTI over all cells
    pooled = {}
    for stage in ("stage1", "stage2", "stage3"):
        tp = sum(r["dti"][stage]["tp"] for r in results)
        fp = sum(r["dti"][stage]["fp"] for r in results)
        fn = sum(r["dti"][stage]["fn"] for r in results)
        pooled[stage] = M57.dti_from_components(tp, fp, fn)

    canary_summary = {}
    if canary:
        import collections
        by_ch = collections.defaultdict(list)
        for row in canary:
            by_ch[row["channel"]].append(row["auc"])
        canary_summary = {k: {"mean_auc": float(np.mean(v)), "max_auc": float(np.max(v)),
                              "n": len(v)} for k, v in sorted(by_ch.items())}
        flagged = {k: v for k, v in canary_summary.items() if v["max_auc"] > 0.90}

    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lane": "auto-context stacked refinement (Tu, CVPR 2008, doi:10.1109/CVPR.2008.4587436)",
        "evidence_class": "HOLDOUT-DTI",
        "evaluator": "gems57.metric.dti_binary / gems57.holdout (pooled hide-and-recover)",
        "alpha": M57.ALPHA, "beta": M57.BETA,
        "triangular_radius_m": M57.R_M,
        "outer_buffer_px": OUTER_BUFFER_PX, "inner_buffer_px": INNER_BUFFER_PX,
        "inner_blocks": INNER_BLOCKS, "neg_ratio": NEG_RATIO,
        "dot_budget": args.dot_budget, "seeds": args.seeds, "modes": args.modes,
        "cells": results,
        "pooled_dti": pooled,
        "gate_all_pixels": gate,
        "gate_far_gt_1km": gate_far,
        "leakage_canary": {"per_channel": canary_summary,
                           "flagged_auc_gt_0.90": flagged if canary else None},
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=_jsonable))
    log(f"wrote {args.out} in {time.time()-t0:.0f}s")
    log("pooled DTI: " + json.dumps({k: round(v, 4) for k, v in pooled.items()}))
    log("gate (>2 SE): " + json.dumps(
        {k: v["passes_gt_2se"] for k, v in gate.items() if isinstance(v, dict)
         and "passes_gt_2se" in v}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
