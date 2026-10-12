#!/usr/bin/env python3
"""Build and cache the stage-1 matrix and the physics channels (footprint rows).

Outputs (git-ignored, .cache/features/):
  X1.npy        float32 [N_footprint, 26]  19 raw bands + 7 physics channels
  phys.npy      float32 [7, N_footprint]   physics channels (re-used by stage 2)
  quad.npy      int8   [N_footprint]       quadrant id per row
  feature_receipt.json
Memory-lean: bands are read one at a time; only det_elev / rtp / iso_grav_anom
are retained as full maps for the physics channels.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems68 import features as F, grid as G  # noqa: E402
from gems68.holdout import quadrant_ids  # noqa: E402

OUT = ROOT / ".cache" / "features"


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    g = G.load_grid()
    fp = g.footprint
    n = int(fp.sum())
    X = np.empty((n, len(F.STAGE1_NAMES)), np.float32)
    keep = {}
    stats = []
    for i, name in enumerate(G.BAND_NAMES):
        b = G.read_band(i + 1)
        n_nan = int((~np.isfinite(b[fp])).sum())
        b = F._fill(b, fp)
        X[:, i] = b[fp]
        stats.append(dict(band=name, index=i + 1, nan_in_footprint=n_nan,
                          p05=float(np.percentile(X[:, i], 5)), median=float(np.median(X[:, i])),
                          p95=float(np.percentile(X[:, i], 95))))
        if name in ("det_elev", "rtp", "iso_grav_anom"):
            keep[name] = b
        del b
        print(f"band {i+1:2d} {name:22s} nan_in_fp={n_nan}", flush=True)
    bands = [None] * 19
    for k, v in keep.items():
        bands[G.BAND_NAMES.index(k)] = v
    phys = F.physics_stack(bands)
    P = np.empty((len(phys), n), np.float32)
    for j, c in enumerate(phys):
        X[:, 19 + j] = c[fp]
        P[j] = c[fp]
    np.save(OUT / "X1.npy", X)
    np.save(OUT / "phys.npy", P)
    np.save(OUT / "quad.npy", quadrant_ids(fp)[fp])
    receipt = dict(n_footprint=n, n_catalogue=int(g.catalogue.sum()), stage1_names=F.STAGE1_NAMES,
                   physics_names=F.PHYSICS_NAMES, band_stats=stats, seconds=round(time.time() - t0, 1),
                   source_sha256={f: G.sha256_file(G.DATA / f) for f in
                                  ("training_features.tif", "existing_faults.tif", "sample_submission.tif")})
    (ROOT / "registry" / "feature_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if k != "band_stats"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
