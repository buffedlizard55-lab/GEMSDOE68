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
    del X, P, phys, bands, keep
    build_exp3_features(g)
    return 0


def build_exp3_features(g=None) -> Path:
    """Build .cache/features/X1_exp3.npy (37 local scarp/ridge/gradient channels, zero regional blobs)."""
    import gc
    import rasterio
    from scipy import ndimage as ndi

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if g is None:
        g = G.load_grid()
    fp = g.footprint
    n = int(fp.sum())
    h, w = fp.shape
    yy, xx = np.nonzero(fp)
    yy_f = yy.astype(np.float32)
    xx_f = xx.astype(np.float32)

    X1_old = np.load(OUT / "X1.npy", mmap_mode="r")
    old_names = F.STAGE1_NAMES

    def get_old_map(name: str) -> np.ndarray:
        m = np.zeros((h, w), np.float32)
        m[fp] = X1_old[:, old_names.index(name)]
        return m

    keep_old = [
        "tmi_hg", "tmi_vg", "tc", "iso_grav_anom_slope", "iso_grav_anom_hg", "iso_grav_anom_vg",
        "det_elev_slope", "elev_grad", "elev_lap", "elev_ridge_s1", "elev_ridge_s2", "elev_tilt",
        "rtp_ridge_s2", "grav_ridge_s2",
    ]
    new_phys_names = ["elev_asym", "elev_hp15", "tc_grad", "db_grad", "db_hp15", "cs_grad", "cs_hp15"]
    spec_l = json.loads((ROOT / "data" / "external" / "lidar_scarp_features.json").read_text())
    lidar_names = [f"lidar_{nm}" for nm in spec_l["bands"] if nm != "strike"] + ["lidar_scarp_contrast"]
    ext_names = [f"ext_{nm}_hp" for nm in ("ThK", "UK", "UTh", "TMI_up150")]
    s1_names = keep_old + new_phys_names + lidar_names + ext_names

    mm = np.empty((n, len(s1_names)), dtype=np.float32)
    col = 0
    for k in keep_old:
        mm[:, col] = X1_old[:, old_names.index(k)]
        col += 1

    det_elev = get_old_map("det_elev")
    elev_grad = get_old_map("elev_grad")
    c2_e, s2_e, _, th_e = F.orientation_field(det_elev, 1.5, 5.0)
    ay_e = np.sin(th_e[fp]).astype(np.float32)
    ax_e = np.cos(th_e[fp]).astype(np.float32)
    cy_e, cx_e = -ax_e, ay_e
    np.savez(OUT / "orient_e.npz", c2=c2_e[fp], s2=s2_e[fp], ay=ay_e, ax=ax_e)
    del c2_e, s2_e, th_e
    gc.collect()

    y_p5 = np.clip(np.rint(yy_f + 5.0 * cy_e).astype(np.int32), 0, h - 1)
    x_p5 = np.clip(np.rint(xx_f + 5.0 * cx_e).astype(np.int32), 0, w - 1)
    y_m5 = np.clip(np.rint(yy_f - 5.0 * cy_e).astype(np.int32), 0, h - 1)
    x_m5 = np.clip(np.rint(xx_f - 5.0 * cx_e).astype(np.int32), 0, w - 1)
    elev_grad_sm = ndi.gaussian_filter(elev_grad, 2.0)
    mm[:, col] = np.abs(elev_grad_sm[y_p5, x_p5] - elev_grad_sm[y_m5, x_m5]).astype(np.float32)
    col += 1
    del elev_grad, elev_grad_sm, y_p5, x_p5, y_m5, x_m5
    gc.collect()

    mm[:, col] = (det_elev - ndi.gaussian_filter(det_elev, 15.0))[fp].astype(np.float32)
    col += 1
    del det_elev
    gc.collect()

    tc = get_old_map("tc")
    mm[:, col] = np.hypot(ndi.sobel(tc, axis=1) / 8.0, ndi.sobel(tc, axis=0) / 8.0)[fp].astype(np.float32)
    col += 1
    del tc
    gc.collect()

    depth_base = get_old_map("depth_to_base_surf")
    mm[:, col] = np.hypot(ndi.sobel(depth_base, axis=1) / 8.0, ndi.sobel(depth_base, axis=0) / 8.0)[fp].astype(np.float32)
    col += 1
    mm[:, col] = (depth_base - ndi.gaussian_filter(depth_base, 15.0))[fp].astype(np.float32)
    col += 1
    del depth_base
    gc.collect()

    cond_surf = get_old_map("cond_surf")
    mm[:, col] = np.hypot(ndi.sobel(cond_surf, axis=1) / 8.0, ndi.sobel(cond_surf, axis=0) / 8.0)[fp].astype(np.float32)
    col += 1
    mm[:, col] = (cond_surf - ndi.gaussian_filter(cond_surf, 15.0))[fp].astype(np.float32)
    col += 1
    del cond_surf
    gc.collect()

    down_v = up_v = cross_v = None
    with rasterio.open(ROOT / "data" / "external" / "lidar_scarp_features_u8.tif") as src:
        for i, nm in enumerate(spec_l["bands"], 1):
            if nm == "strike":
                continue
            q = src.read(i)[fp].astype(np.float32)
            xmax, tr = spec_l["quantisation"][nm]
            if nm == "valid":
                v = (q > 0).astype(np.float32)
            else:
                t = np.clip((q - 1.0) / 254.0, 0.0, 1.0)
                v = float(xmax) * (t * t if tr == "sqrt" else t)
                v[q <= 0] = np.nanmedian(v[q > 0])
            mm[:, col] = v
            col += 1
            if nm == "downface_max":
                down_v = v
            elif nm == "upface_max":
                up_v = v
            elif nm == "cross_max":
                cross_v = v
    mm[:, col] = np.maximum(down_v, up_v) - cross_v
    col += 1
    del down_v, up_v, cross_v
    gc.collect()

    with rasterio.open(ROOT / "data" / "external" / "geodawn_extensions_u8.tif") as src:
        for i, nm in enumerate(("ThK", "UK", "UTh", "TMI_up150"), 1):
            q = src.read(i).astype(np.float32)
            val = np.clip((q - 1.0) / 254.0, 0.0, 1.0)
            val[q <= 0] = float(np.nanmedian(val[q > 0]))
            mm[:, col] = (val - ndi.gaussian_filter(val, 12.0))[fp].astype(np.float32)
            col += 1
            del q, val
            gc.collect()

    assert col == len(s1_names)
    np.save(OUT / "X1_exp3.npy", mm)
    (OUT / "X1_exp3_names.json").write_text(json.dumps(s1_names, indent=2))
    print(f"Saved X1_exp3.npy ({mm.shape}) in {time.time()-t0:.1f}s", flush=True)
    return OUT / "X1_exp3.npy"


if __name__ == "__main__":
    sys.exit(main())
