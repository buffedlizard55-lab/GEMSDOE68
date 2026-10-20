#!/usr/bin/env python3
"""Independent Pass-3 verification of the Exp-4 release raster.

Trusts nothing in the run card: re-reads the written GeoTIFF and the pinned
competition files, and recomputes sha256, grid/range/footprint checks, dot
count, catalogue exclusion, min spacing, near/far allocation, binary-raster
Spearman vs every registry raster (fixed 400k footprint sample), 3 px dot
overlap vs the champion and the exp2 candidate, and the portal-range
invariant (global min/max over ALL cells incl. outside footprint).
Writes evidence/release_verification_exp4.json; exit 1 on any hard failure.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".cache" / "gems_data"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    card = json.loads((ROOT / "evidence/run_card_exp4.json").read_text())
    if not card.get("download_cleared"):
        print("run card says not cleared; nothing to verify")
        return 1
    name = card["raster"]["primary"]["name"]
    path = ROOT / "docs/downloads" / name
    with rasterio.open(path) as s:
        a = s.read(1)
        meta = dict(count=s.count, dtype=str(s.dtypes[0]), crs=str(s.crs),
                    shape=(s.height, s.width), transform=tuple(float(v) for v in tuple(s.transform)[:6]),
                    nodata=s.nodata)
    with rasterio.open(DATA / "sample_submission.tif") as s:
        sub = s.read(1)
    with rasterio.open(DATA / "existing_faults.tif") as s:
        cat_raw = s.read(1)
    fp = np.isfinite(sub)
    cat = (cat_raw == 1) & fp
    fin = np.isfinite(a)
    dots = fin & (a > 0)
    dy, dx = np.nonzero(dots)
    dcat = ndi.distance_transform_edt(~cat)
    dd = dcat[dots]
    tree = cKDTree(np.c_[dy, dx])
    nn, _ = tree.query(np.c_[dy, dx], k=2)

    yy, xx = np.nonzero(fp)
    pick = np.random.default_rng(68).choice(len(yy), size=400_000, replace=False)
    ys, xs = yy[pick], xx[pick]
    mine = a[ys, xs]
    worst_rho, overlaps = (None, -1.0), {}
    for rp in sorted((ROOT / "registry/rasters/rasters").glob("*.tif")):
        with rasterio.open(rp) as s:
            r = s.read(1).astype(np.float32)
        r[~np.isfinite(r)] = 0.0
        rho = float(spearmanr(mine, r[ys, xs]).correlation) if np.std(r[ys, xs]) > 0 else 0.0
        if abs(rho) > worst_rho[1]:
            worst_rho = (rp.name, abs(rho))
        if "h33-h33-2-b2" in rp.name and rp.name.endswith("-zeros.tif") or "s3-hidedraw" in rp.name:
            near = ndi.distance_transform_edt(~(r > 0)) <= 3.0
            overlaps[rp.name] = float((dots & near).sum() / max(len(dy), 1))
        del r
    with rasterio.open(ROOT / "docs/downloads/gems68-autoctx-s3-hidedraw-38000dots-20261012T022132Z-4c18def6-zeros.tif") as s:
        e2 = s.read(1)
    near_e2 = ndi.distance_transform_edt(~(np.isfinite(e2) & (e2 > 0))) <= 3.0
    overlaps["exp2_candidate_s3_hidedraw"] = float((dots & near_e2).sum() / max(len(dy), 1))

    checks = dict(
        sha256_matches_card=sha256(path) == card["raster"]["primary"]["sha256"],
        single_band=meta["count"] == 1, float32=meta["dtype"] == "float32",
        crs_epsg32611=meta["crs"] == "EPSG:32611", shape_3730x3292=meta["shape"] == (3730, 3292),
        transform_ok=meta["transform"] == (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0),
        no_nodata_tag=meta["nodata"] is None,
        all_cells_finite=bool(fin.all()), no_nan_anywhere=int(np.isnan(a).sum()) == 0,
        global_min=float(a.min()), global_max=float(a.max()),
        values_subset_01=bool(np.all(np.isin(a, [0.0, 1.0]))),
        nan_inside_sample_footprint=int((fp & ~fin).sum()),
        dots_outside_footprint=int((dots & ~fp).sum()),
        n_dots=int(dots.sum()), dots_on_catalogue=int((dots & cat).sum()),
        min_dist_catalogue_px=float(dd.min()), frac_dots_gt10px=float((dd > 10).mean()),
        median_dist_catalogue_px=float(np.median(dd)),
        min_nn_spacing_px=float(nn[:, 1].min()), median_nn_spacing_px=float(np.median(nn[:, 1])),
        worst_abs_spearman_vs_registry={"name": worst_rho[0], "abs_rho": worst_rho[1]},
        dot_overlap_3px=overlaps,
    )
    hard_fail = [
        k for k in ("sha256_matches_card", "single_band", "float32", "crs_epsg32611",
                    "shape_3730x3292", "transform_ok", "all_cells_finite", "values_subset_01")
        if not checks[k]
    ] + (["nan_inside"] if checks["nan_inside_sample_footprint"] else []) \
        + (["dots_outside"] if checks["dots_outside_footprint"] else []) \
        + (["on_catalogue"] if checks["dots_on_catalogue"] else []) \
        + (["range"] if not (0.0 <= checks["global_min"] and checks["global_max"] <= 1.0) else []) \
        + (["spearman"] if worst_rho[1] > 0.90 else [])
    out = dict(file=name, checks=checks, hard_failures=hard_fail, verdict="PASS" if not hard_fail else "FAIL")
    (ROOT / "evidence/release_verification_exp4.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0 if not hard_fail else 1


if __name__ == "__main__":
    sys.exit(main())
