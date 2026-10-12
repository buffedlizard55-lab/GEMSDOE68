"""Study-area grid, data loading and GeoTIFF export for GEMSDOE68.

Pinned constants (verified against the restored sample_submission.tif /
existing_faults.tif, see registry/restore_receipt.json):

* CRS EPSG:32611 (WGS 84 / UTM 11N), shape 3730 x 3292, 100 m cells
* transform (100, 0, 243350, 0, -100, 4508550)
* footprint = finite cells of sample_submission.tif = 5,167,373 cells
* catalogue = existing_faults.tif == 1 inside footprint = 60,988 cells
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

EPSG = 32611
HEIGHT, WIDTH = 3730, 3292
SHAPE = (HEIGHT, WIDTH)
TRANSFORM = Affine(100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
PIXEL_M = 100.0

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / ".cache" / "gems_data"

BAND_NAMES = [
    "mag_anom", "rtp", "tmi_hg", "geod_2ndinv", "iso_grav_anom_slope", "tc",
    "geod_shearrate", "geod_dilaterate", "tmi_vg", "deq_n100a15", "iso_grav_anom_vg",
    "det_elev", "iso_grav_anom", "tmi", "depth_to_base_surf", "ieq_n100a15",
    "cond_surf", "iso_grav_anom_hg", "det_elev_slope",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(src, what: str) -> None:
    if str(src.crs) != f"EPSG:{EPSG}":
        raise ValueError(f"{what}: CRS {src.crs} != EPSG:{EPSG}")
    if (src.height, src.width) != SHAPE:
        raise ValueError(f"{what}: shape {(src.height, src.width)} != {SHAPE}")
    if tuple(src.transform)[:6] != tuple(TRANSFORM)[:6]:
        raise ValueError(f"{what}: transform {tuple(src.transform)[:6]} != {tuple(TRANSFORM)[:6]}")


@dataclass(frozen=True)
class Grid:
    footprint: np.ndarray   # bool
    catalogue: np.ndarray   # bool, mapped faults inside footprint

    @property
    def shape(self):
        return SHAPE


def load_grid(data_dir: Path | None = None) -> Grid:
    d = Path(data_dir) if data_dir else DATA
    with rasterio.open(d / "existing_faults.tif") as src:
        _check(src, "existing_faults.tif")
        cat = src.read(1)
    with rasterio.open(d / "sample_submission.tif") as src:
        _check(src, "sample_submission.tif")
        sub = src.read(1)
    footprint = np.isfinite(sub)
    # labels.tif / existing_faults.tif: -1 outside footprint, 0 background, 1 mapped fault
    if not np.array_equal(cat >= 0, footprint):
        raise ValueError("existing_faults.tif validity does not match sample_submission footprint")
    return Grid(footprint=footprint, catalogue=(cat == 1) & footprint)


def read_band(index: int, data_dir: Path | None = None) -> np.ndarray:
    """Read one of the 19 feature bands (1-based) as float32 with sentinels -> NaN."""
    d = Path(data_dir) if data_dir else DATA
    with rasterio.open(d / "training_features.tif") as src:
        _check(src, "training_features.tif")
        a = src.read(index).astype(np.float32)
        nd = src.nodata
    if nd is not None:
        a[a <= -3.0e38] = np.nan
    a[~np.isfinite(a)] = np.nan
    return a


def write_geotiff(path: Path, arr: np.ndarray, *, nodata: float | None = None) -> dict:
    """Write a float32 single-band GeoTIFF on the pinned grid; re-read and report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype != np.float32 or arr.shape != SHAPE:
        raise ValueError("array must be float32 on the pinned grid")
    finite = np.isfinite(arr)
    if nodata is None and not finite.all():
        raise ValueError("all-finite export requested but array has NaN/inf")
    if (arr[finite] < 0).any() or (arr[finite] > 1).any():
        raise ValueError("values must be in [0,1]; writer does not clip")
    profile = dict(driver="GTiff", height=HEIGHT, width=WIDTH, count=1, dtype="float32",
                   crs=f"EPSG:{EPSG}", transform=TRANSFORM, tiled=True, blockxsize=256,
                   blockysize=256, compress="deflate", predictor=2)
    if nodata is not None:
        profile["nodata"] = float(nodata)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, "predicted_new_fault_probability")
    with rasterio.open(path) as src:
        back = src.read(1)
        info = dict(path=str(path), bytes=path.stat().st_size, sha256=sha256_file(path),
                    crs=str(src.crs), shape=[src.height, src.width], dtype=src.dtypes[0],
                    transform=[float(v) for v in tuple(src.transform)[:6]], nodata=src.nodata)
    if not np.array_equal(np.isfinite(back), finite) or not np.array_equal(back[finite], arr[finite]):
        raise IOError("re-read differs from written array")
    return info
