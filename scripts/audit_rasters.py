#!/usr/bin/env python3
"""Audit every candidate/champion GeoTIFF from its bytes (this session, measured here).

For each raster: sha256, dtype/CRS/shape/transform vs the pinned grid, value range,
NaN inside/outside the official footprint (sample_submission.tif finite cells),
dot count (>0; and ==1), distance of dots to the published catalogue
(existing_faults.tif==1), nearest-neighbour dot spacing summary, and overlap with
the 0.2778 champion's dots.  Writes evidence/raster_audit_20261020.json.
Nothing here reads any prior run card: every number is recomputed from disk bytes.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".cache" / "gems_data"
PIN = dict(crs="EPSG:32611", shape=(3730, 3292), transform=(100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load(p: Path):
    with rasterio.open(p) as s:
        a = s.read(1)
        meta = dict(crs=str(s.crs), shape=(s.height, s.width), dtype=str(s.dtypes[0]),
                    transform=tuple(float(v) for v in tuple(s.transform)[:6]),
                    count=s.count, nodata=(None if s.nodata is None else float(s.nodata)))
    return a, meta


def main() -> int:
    sub, _ = load(DATA / "sample_submission.tif")
    fp = np.isfinite(sub)
    cat_raw, _ = load(DATA / "existing_faults.tif")
    cat = (cat_raw == 1) & fp
    dist_cat = ndi.distance_transform_edt(~cat)
    champ_path = ROOT / "registry/rasters/rasters/gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif"
    champ, _ = load(champ_path)
    champ_dots = np.isfinite(champ) & (champ > 0)
    champ_near = ndi.distance_transform_edt(~champ_dots) <= 3.0

    targets = [champ_path] + sorted((ROOT / "docs/downloads").glob("*.tif"))
    rows = []
    for p in targets:
        a, m = load(p)
        fin = np.isfinite(a)
        inside = fin & fp
        dots = inside & (a > 0)
        dy, dx = np.nonzero(dots)
        nn_med = None
        if len(dy) > 2:
            tree = cKDTree(np.c_[dy, dx])
            d, _ = tree.query(np.c_[dy, dx], k=2)
            nn_med = float(np.median(d[:, 1]))
        vals = a[inside]
        dc = dist_cat[dots]
        rows.append(dict(
            file=p.name, sha256=sha256(p), bytes=p.stat().st_size,
            crs_ok=m["crs"] == PIN["crs"], shape_ok=m["shape"] == PIN["shape"],
            transform_ok=m["transform"] == PIN["transform"], dtype=m["dtype"], count=m["count"], nodata=m["nodata"],
            nan_inside_footprint=int((fp & ~fin).sum()), finite_outside_footprint=int((~fp & fin).sum()),
            min_inside=float(vals.min()) if vals.size else None, max_inside=float(vals.max()) if vals.size else None,
            values_in_0_1=bool(vals.size and vals.min() >= 0 and vals.max() <= 1),
            n_gt0=int(dots.sum()), n_eq1=int((inside & (a == 1)).sum()),
            binary_01=bool(np.all(np.isin(vals, [0.0, 1.0]))),
            min_dist_catalogue_px=float(dc.min()) if dc.size else None,
            frac_dots_le2px_catalogue=float((dc <= 2).mean()) if dc.size else None,
            frac_dots_on_catalogue=float((dc == 0).mean()) if dc.size else None,
            median_dist_catalogue_px=float(np.median(dc)) if dc.size else None,
            median_nn_spacing_px=nn_med,
            frac_dots_within3px_of_champion=float(champ_near[dots].mean()) if dc.size else None,
        ))
        r = rows[-1]
        print(f"{r['file'][:52]:54s} dots={r['n_gt0']:6d} bin={r['binary_01']} range=[{r['min_inside']},{r['max_inside']}] "
              f"nanIn={r['nan_inside_footprint']} mindcat={r['min_dist_catalogue_px']} medNN={r['median_nn_spacing_px']}", flush=True)
    out = dict(grid=PIN, footprint_cells=int(fp.sum()), catalogue_cells=int(cat.sum()),
               champion=champ_path.name, rasters=rows)
    (ROOT / "evidence/raster_audit_20261020.json").write_text(json.dumps(out, indent=1))
    print("wrote evidence/raster_audit_20261020.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
