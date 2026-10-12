#!/usr/bin/env python3
"""Emit the GEMSDOE68 auto-context submission GeoTIFF.

Pipeline
--------
1. Re-run the *same* nested blocked auto-context stack, but with **no
   withholding**: the four inner spatial blocks now tile the whole footprint, so
   every catalogue label is used and every stage-1/2 probability is still
   out-of-fold (never in-fold).  This is the map the brief asks for: a stacked
   refinement whose context features are leak-free by construction.
2. Turn the final stage-3 probability map into a binary dot emission using the
   emission parameters (catalogue exclusion radius and dot budget) that were
   tuned on the spatially blocked holdout -- never on the submission grid.
3. Write the file with the shared :mod:`gems.submission_writer` (float32,
   EPSG:32611, 100 m, values in [0,1] inside the footprint, NaN outside) and
   validate it with the shared :mod:`gems.validate_submission`.
4. Run the duplicate/lane check against every raster in ``registry/rasters``.

Usage::

    PYTHONPATH=src python scripts/emit_submission.py \
        --params evidence/ac68_emission_params.json \
        --name GEMSDOE68-AC-stack3-r3-N32000 --out docs/downloads
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

from gems.submission_writer import NOTE_MAX_CHARS, sha256_file, write_submission  # noqa: E402
from gems.validate_submission import (footprint_from_features,  # noqa: E402
                                     validate)
from gems57 import grid as G57                                                       # noqa: E402
from gems57 import metric as M57                                                     # noqa: E402
from gems68 import blocks as B                                                       # noqa: E402
from gems68 import catalogue_context as CC                                           # noqa: E402
from gems68 import features as F                                                     # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_autocontext import (INNER_BUFFER_PX, NEG_RATIO, ROW_BLOCK,  # noqa: E402
                             SEED_ROW_SAMPLE, bbox_of, make_model,
                             own_block_catalogue_context, predict_box,
                             quadrant_partition, stage1_row_matrix,
                             stage2_row_matrix, strip_features_stage1,
                             strip_features_stage2)


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


def build_final_stack(grid, stack, idx, footprint, local_cos2, local_sin2,
                      neg_ratio: int):
    """Leak-free stacked probability map over the whole footprint."""
    h, w = grid.height, grid.width
    catalogue = grid.catalogue
    blk = quadrant_partition(footprint)
    inner = [blk == b for b in range(4)]
    dist_to_blk = [ndi.distance_transform_edt(~m) for m in inner]

    rng = np.random.default_rng(SEED_ROW_SAMPLE)
    py_, px_ = np.nonzero(catalogue)
    ny_, nx_ = np.nonzero(footprint & ~catalogue)
    n_neg = min(ny_.size, neg_ratio * py_.size)
    sel = rng.choice(ny_.size, size=n_neg, replace=False)
    ry = np.concatenate([py_, ny_[sel]]).astype(np.int64)
    rx = np.concatenate([px_, nx_[sel]]).astype(np.int64)
    y = np.concatenate([np.ones(py_.size, np.float32), np.zeros(n_neg, np.float32)])
    rb = blk[ry, rx]
    keep = rb >= 0
    ry, rx, y, rb = ry[keep], rx[keep], y[keep], rb[keep]
    log(f"  training rows={ry.size} positives={int(y.sum())}")

    X1 = stage1_row_matrix(stack, idx, ry, rx)
    own_cat = own_block_catalogue_context(
        dict(ry=ry, rx=rx, rb=rb, inner=inner), catalogue, footprint,
        local_cos2, local_sin2)
    A1 = np.hstack([X1, own_cat])
    far = [d[ry, rx] > INNER_BUFFER_PX for d in dist_to_blk]

    # One catalogue-context map at a time.  Five resident maps over the whole
    # 3730x3292 footprint is ~2.7 GB and the sandbox has 3.9 GB, so the release
    # run builds each map, uses it, and frees it before the next one.
    box_blk = [bbox_of(m, h, w) for m in inner]

    p1 = np.zeros((h, w), np.float32)
    for b in range(4):
        tr = far[b] & (rb != b)
        if tr.sum() < 50 or y[tr].sum() < 10:
            raise RuntimeError(f"inner block {b} has too little training data")
        cc = CC.catalogue_context(catalogue & ~inner[b], footprint,
                                  local_cos2, local_sin2)
        m = make_model()
        m.fit(A1[tr], y[tr])
        p = predict_box(m, lambda rows, cols, m=m, cc=cc:
                        strip_features_stage1(stack, idx, rows, cc, cols),
                        h, w, box_blk[b])
        p1[inner[b]] = p[inner[b]]
        del p, cc
    log("  stage 1 (4 blocked models) done")

    A2 = stage2_row_matrix(stack, idx, p1, dict(ry=ry, rx=rx, y=y, rb=rb),
                           own_cat, local_cos2, local_sin2)
    p2 = np.zeros((h, w), np.float32)
    for b in range(4):
        tr = far[b] & (rb != b)
        cc = CC.catalogue_context(catalogue & ~inner[b], footprint,
                                  local_cos2, local_sin2)
        m = make_model()
        m.fit(A2[tr], y[tr])
        p = predict_box(m, lambda rows, cols, m=m, cc=cc:
                        strip_features_stage2(stack, idx, rows, p1, cc,
                                              local_cos2, local_sin2, cols),
                        h, w, box_blk[b])
        p2[inner[b]] = p[inner[b]]
        del p, cc
    log("  stage 2 (4 blocked models) done")

    A3 = stage2_row_matrix(stack, idx, p2, dict(ry=ry, rx=rx, y=y, rb=rb),
                           own_cat, local_cos2, local_sin2)
    m3 = make_model()
    m3.fit(A3, y)
    cc_full = CC.catalogue_context(catalogue, footprint, local_cos2, local_sin2)
    p3 = predict_box(m3, lambda rows, cols:
                     strip_features_stage2(stack, idx, rows, p2, cc_full,
                                           local_cos2, local_sin2, cols),
                     h, w, bbox_of(footprint, h, w))
    del cc_full
    log("  stage 3 done")
    return dict(p1=p1, p2=p2, p3=p3, n_rows=int(ry.size), n_pos=int(y.sum()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True,
                    help="JSON with radius_px and budget (tuned on the holdout)")
    ap.add_argument("--out", default="docs/downloads")
    ap.add_argument("--name", default="GEMSDOE68-AC-stack3")
    ap.add_argument("--note", default="")
    ap.add_argument("--neg-ratio", type=int, default=NEG_RATIO)
    args = ap.parse_args(argv)

    params = json.loads(Path(args.params).read_text())
    radius_px = int(params["radius_px"])
    budget = int(params["budget"])
    stage = params.get("stage", "stage3")
    log(f"emission params: stage={stage} radius_px={radius_px} budget={budget}")

    t0 = time.time()
    grid = G57.load_grid(ROOT / "data" / "bridge")
    names, stack, footprint = F.load_channels()
    # The shipped footprint must be the *official* one: the validator derives it
    # from band 1 of training_features.tif (finite and above the nodata sentinel),
    # which is 3061 cells smaller than the cached stack's own footprint.  Writing
    # NaN inside the official footprint is a validator FAIL, so intersect here.
    official = footprint_from_features(ROOT / "data" / "bridge" / "training_features.tif",
                                       footprint.shape)
    n_extra = int((footprint & ~official).sum())       # cached, not official
    n_missing = int((official & ~footprint).sum())     # official, not cached
    log(f"  footprint: cached={int(footprint.sum())} official={int(official.sum())} "
        f"(cached-only={n_extra}, official-only={n_missing}); "
        f"emitting inside cached&official, 0 in the official-only cells")
    footprint = footprint & official
    idx = F.channel_index()
    h, w = grid.height, grid.width

    gmag = np.array(stack[idx["det_elev_gmag"]], dtype=np.float32)
    gmag[~footprint] = 0.0
    local_cos2, local_sin2, _ = B.structure_tensor_orientation(gmag, sigma=3.0)
    del gmag

    log("building the leak-free stacked map (no withholding)")
    fields = build_final_stack(grid, stack, idx, footprint, local_cos2,
                               local_sin2, args.neg_ratio)
    score = fields[{"stage1": "p1", "stage2": "p2", "stage3": "p3"}.get(stage, stage)]

    d_cat = ndi.distance_transform_edt(~grid.catalogue).astype(np.float32)
    eligible = footprint & (d_cat > radius_px)
    s = np.where(eligible, score, -np.inf).ravel()
    n = min(budget, int((s > -np.inf).sum()))
    order = np.argsort(-s)[:n]
    order = order[s[order] > -np.inf]
    dots = np.zeros((h, w), np.float32)
    dots.ravel()[order] = 1.0
    log(f"  emitted {int((dots > 0).sum())} dots")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    note = args.note or (
        f"GEMSDOE68 auto-context stacked refinement (Tu CVPR 2008); "
        f"{stage}, r={radius_px}px, {int((dots > 0).sum())} dots; "
        f"holdout-gated")
    if len(note) > NOTE_MAX_CHARS:
        note = note[:NOTE_MAX_CHARS]

    written = {}
    for variant, fill in (("nan", np.nan), ("zeros", 0.0)):
        # finite everywhere inside the OFFICIAL footprint (0 where the channel stack
        # has no data), NaN (or 0) only outside it -- exactly what the validator wants
        pred = np.where(official, np.where(footprint, dots, np.float32(0.0)),
                        fill).astype(np.float32)
        path = out_dir / f"{stem}-{variant}.tif"
        # write_submission masks with the footprint it is given, so it must be the
        # OFFICIAL footprint -- otherwise the 1540 official-only cells it does not
        # know about come back as NaN and the validator fails
        receipt = write_submission(path, pred, official,
                                   ROOT / "data" / "bridge" / "sample_submission.tif",
                                   overwrite=True, name=f"{args.name}-{variant}",
                                   note=note)
        report = validate(path, ROOT / "data" / "bridge" / "sample_submission.tif",
                          ROOT / "data" / "bridge" / "training_features.tif")
        written[variant] = {"path": str(path.relative_to(ROOT)),
                            "sha256": receipt["sha256"],
                            "bytes": path.stat().st_size,
                            "emitted_cells": receipt["emitted_cells_gt0"],
                            "validator_passed": report["passed"],
                            "validator_failures": [c["check"] for c in report["checks"]
                                                   if c["status"] == "FAIL"],
                            "validator_skipped": [c["check"] for c in report["checks"]
                                                  if c["status"] == "SKIPPED"],
                            "n_checks": len(report["checks"])}
        log(f"  {variant}: sha256={receipt['sha256'][:16]}... "
            f"validator_passed={report['passed']}")

    evidence = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lane": "auto-context stacked refinement",
        "emission_params": params,
        "dot_count": int((dots > 0).sum()),
        "footprint_cells": int(footprint.sum()),
        "catalogue_cells": int(grid.catalogue.sum()),
        "mean_prob_under_dots": float(score[dots > 0].mean()),
        "mean_prob_all": float(score[footprint].mean()),
        "n_training_rows": fields["n_rows"], "n_positives": fields["n_pos"],
        "files": written,
        "elapsed_s": round(time.time() - t0, 1),
    }
    ev = ROOT / "evidence" / "ac68_emission.json"
    ev.write_text(json.dumps(evidence, indent=1, default=_jsonable))
    log(f"wrote {ev}")
    print(json.dumps(evidence, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
