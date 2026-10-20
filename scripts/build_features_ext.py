#!/usr/bin/env python3
"""Exp 3 stage-1 matrix: cached X1 (26 ch) + mirrored external channels (18 ch).

External rasters (registry/data_manifest.json, group "external", SHA-256 pinned,
restored to .cache/gems_data/external/):
  * lidar_scarp_features_u8.tif  bands 1-10 (ex_max, ex_mean, step_max,
    lapneg_max, lappos_max, downface_max, upface_max, cross_max, relief,
    coh100).  Band 11 (strike, unknown circular u8 encoding) and band 12
    (valid/coverage) are EXCLUDED: strike's encoding is undocumented in this
    checkout and coverage can encode mapping effort rather than geology.
  * geodawn_rad_u8.tif           K, Th, U, TC
  * geodawn_extensions_u8.tif    ThK, UK, UTh, TMI_up150
  * derived_sgmc_faults_100m_u8.tif is EXCLUDED: it is a fault catalogue
    raster, i.e. label-derived, not a geophysical observation.

All rasters must match the pinned grid (EPSG:32611, 3730x3292, transform);
values are u8 scaled to [0,1] float32 (nodata 0 stays 0 = no coverage).
Output: .cache/features/X1ext.npy [N_footprint, 44] + names in
.cache/features/X1ext_names.json.  Row order = footprint row-major, same as X1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems68 import features as F, grid as G  # noqa: E402

EXT = ROOT / ".cache" / "gems_data" / "external"
OUT = ROOT / ".cache" / "features"

PLAN = [
    ("lidar_scarp_features_u8.tif", list(range(1, 11)), "lidar_"),
    ("geodawn_rad_u8.tif", [1, 2, 3, 4], "rad_"),
    ("geodawn_extensions_u8.tif", [1, 2, 3, 4], "ext_"),
]


def main() -> int:
    g = G.load_grid()
    fp = g.footprint
    X1 = np.load(OUT / "X1.npy")
    assert X1.shape == (int(fp.sum()), len(F.STAGE1_NAMES)), X1.shape
    cols, names, stats = [], [], []
    for fname, bands, prefix in PLAN:
        with rasterio.open(EXT / fname) as src:
            G._check(src, fname)
            for b in bands:
                desc = src.descriptions[b - 1] or f"band{b}"
                a = src.read(b).astype(np.float32) / 255.0
                v = a[fp]
                cols.append(v)
                names.append(prefix + desc)
                stats.append(dict(name=prefix + desc, source=fname, band=b,
                                  frac_zero_in_footprint=float((v == 0).mean()),
                                  mean=float(v.mean())))
                print(f"{prefix+desc:24s} frac0={stats[-1]['frac_zero_in_footprint']:.3f}", flush=True)
    X = np.hstack([X1] + [c[:, None] for c in cols]).astype(np.float32)
    all_names = list(F.STAGE1_NAMES) + names
    np.save(OUT / "X1ext.npy", X)
    (OUT / "X1ext_names.json").write_text(json.dumps(
        dict(names=all_names, external=stats,
             excluded=["lidar strike (undocumented circular encoding)",
                       "lidar valid (coverage/mapping-effort proxy)",
                       "derived_sgmc_faults_100m_u8 (catalogue-derived, label-like)"]), indent=2))
    print(json.dumps(dict(shape=list(X.shape), n_names=len(all_names))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
