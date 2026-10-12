"""Submission writer, format validator and registry duplicate check.

The organizer accepts "a single-band GeoTIFF (.tif) file, or a .zip file containing a single
GeoTIFF ... It must match the submission format's CRS, shape, and geotransform."
(https://www.drivendata.org/competitions/306/competition-doe-gems/data/)

The mirrored ``sample_submission.tif`` is float32, single band, EPSG:32611,
3730 x 3292, transform (100, 0, 243350, 0, -100, 4508550), nodata = NaN, finite exactly on
the 5,167,373-pixel footprint.  Two twins are written:

* ``-zeros.tif`` -- all-finite, 0.0 outside the footprint.  Cannot trip the portal's
  "Predicted values must be in range [0, 1]" check, because no NaN exists anywhere.
* ``-nan.tif``   -- NaN outside the footprint, byte-for-byte the sample_submission masking.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from .data import SAMPLE_TIF, load_grid

PROFILE_TEMPLATE = dict(driver="GTiff", dtype="float32", count=1, compress="lzw",
                        tiled=False, interleave="band")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_submission(values: np.ndarray, footprint: np.ndarray, out: Path,
                     outside: str = "zeros", template: Path = SAMPLE_TIF) -> dict:
    """Write a submission GeoTIFF whose grid is copied from the organizer template.

    ``values`` is a full-grid float array; only the footprint is written.  ``outside`` is
    either ``"zeros"`` (all-finite) or ``"nan"`` (matches sample_submission masking).
    """
    grid = load_grid(template)
    if values.shape != (grid.height, grid.width):
        raise ValueError(f"values shape {values.shape} != grid {(grid.height, grid.width)}")
    if outside not in ("zeros", "nan"):
        raise ValueError("outside must be 'zeros' or 'nan'")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(values, dtype=np.float32).copy()
    a[~np.isfinite(a)] = 0.0
    a = np.clip(a, 0.0, 1.0).astype(np.float32)
    a[~footprint] = 0.0 if outside == "zeros" else np.float32(np.nan)
    tr = from_origin(grid.transform[2], grid.transform[5],
                     grid.transform[0], -grid.transform[4])
    prof = dict(PROFILE_TEMPLATE)
    prof.update(height=grid.height, width=grid.width, crs=grid.crs, transform=tr)
    prof["nodata"] = 0.0 if outside == "zeros" else float("nan")
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(a, 1)
        dst.update_tags(**{"SOFTWARE": "GEMSDOE68 auto-context lane",
                           "AREA_OR_POINT": "Area"})
    return dict(path=str(out), bytes=int(out.stat().st_size), sha256=sha256_file(out),
                outside=outside)


def zip_submission(tif: Path) -> dict:
    tif = Path(tif)
    z = tif.with_suffix(".zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(tif, arcname=tif.name)
    return dict(path=str(z), bytes=int(z.stat().st_size), sha256=sha256_file(z))


@dataclass
class FormatCheck:
    ok: bool
    checks: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    sha256: str = ""
    bytes: int = 0

    def as_json(self) -> dict:
        return dict(ok=self.ok, checks=self.checks, failures=self.failures,
                    sha256=self.sha256, bytes=self.bytes)


def validate_submission(path: Path, template: Path = SAMPLE_TIF) -> FormatCheck:
    """The range/CRS/shape/transform/NaN checks the portal applies, run locally first."""
    path = Path(path)
    checks: dict = {}
    failures: list = []

    def want(name: str, ok: bool, detail="") -> None:
        checks[name] = bool(ok) if detail == "" else {"ok": bool(ok), "detail": detail}
        if not ok:
            failures.append(f"{name}: {detail}" if detail else name)

    with rasterio.open(template) as t:
        want("single_band", True)
        ref = dict(crs=str(t.crs), shape=(t.height, t.width),
                   transform=tuple(round(v, 6) for v in t.transform)[:6])
        tmpl = t.read(1)
    tmpl_foot = np.isfinite(tmpl)
    del tmpl

    with rasterio.open(path) as s:
        want("driver_geotiff", s.driver == "GTiff", s.driver)
        want("count_1", s.count == 1, str(s.count))
        want("dtype_float32", s.dtypes[0] == "float32", s.dtypes[0])
        want("crs_EPSG32611", str(s.crs) == ref["crs"], f"{s.crs} != {ref['crs']}")
        want("shape_match", (s.height, s.width) == ref["shape"],
             f"{(s.height, s.width)} != {ref['shape']}")
        got = tuple(round(v, 6) for v in s.transform)[:6]
        want("transform_match", got == ref["transform"], f"{got} != {ref['transform']}")
        a = s.read(1)

    finite = np.isfinite(a)
    want("finite_inside_footprint", bool(finite[tmpl_foot].all()),
         f"{int((~finite[tmpl_foot]).sum())} NaN inside footprint")
    outside_vals = a[~tmpl_foot]
    nz_out = int(((outside_vals != 0.0) & np.isfinite(outside_vals)).sum())
    want("outside_footprint_is_zero_or_nan", nz_out == 0,
         f"{nz_out} non-zero finite pixels outside the footprint")
    v = a[finite]
    want("values_in_0_1", bool(v.size and v.min() >= 0.0 and v.max() <= 1.0),
         f"min={float(v.min()) if v.size else float('nan')} "
         f"max={float(v.max()) if v.size else float('nan')}")
    want("not_all_zero", bool(v.size and (v > 0).any()), "raster is entirely zero")
    checks["emitted_pixels"] = int((v > 0).sum())
    checks["finite_pixels"] = int(finite.sum())
    checks["footprint_pixels"] = int(tmpl_foot.sum())
    return FormatCheck(ok=not failures, checks=checks, failures=failures,
                       sha256=sha256_file(path), bytes=int(path.stat().st_size))


# -----------------------------------------------------------------------------------------
# registry duplicate check
# -----------------------------------------------------------------------------------------


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation on the shared finite support."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return float("nan")
    from scipy.stats import rankdata
    ra = rankdata(a[m])
    rb = rankdata(b[m])
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def dot_overlap(a: np.ndarray, b: np.ndarray, footprint: np.ndarray, tol_px: float = 3.0,
                thresh: float = 0.0) -> dict:
    """Fraction of A's emitted dots lying within ``tol_px`` of one of B's emitted dots."""
    from scipy.ndimage import binary_dilation, distance_transform_edt
    pa = (np.nan_to_num(a, nan=0.0) > thresh) & footprint
    pb = (np.nan_to_num(b, nan=0.0) > thresh) & footprint
    na, nb = int(pa.sum()), int(pb.sum())
    if na == 0 or nb == 0:
        return dict(n_a=na, n_b=nb, within=0.0, jaccard=0.0)
    d = distance_transform_edt(~pb)
    within = int((d[pa] <= tol_px).sum())
    inter = int((pa & pb).sum())
    union = int((pa | pb).sum())
    return dict(n_a=na, n_b=nb, within=within / na, jaccard=inter / union if union else 0.0)
