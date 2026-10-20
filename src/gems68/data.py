"""Grid, footprint and the cached feature stack.

Everything is anchored to the organizer grid read off the mirrored competition rasters:

* 3730 x 3292 pixels, EPSG:32611 (UTM zone 11N), 100 m pixels
* transform (100, 0, 243350, 0, -100, 4508550)
* ``labels.tif`` / ``existing_faults.tif`` are byte-identical (sha256 7ba308cc...);
  values {-1, 0, 1} with nodata -1.  -1 marks pixels outside the footprint.
* ``sample_submission.tif`` is float32 with nodata=NaN, finite exactly on the footprint.

The footprint derived from ``labels != -1`` is *identical* to the footprint derived from
``isfinite(sample_submission)``: 5,167,373 pixels (checked in tests/test_grid.py).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
CACHE_DATA = REPO / ".cache" / "gems_data"
BRIDGE = DATA / "bridge" if (DATA / "bridge").exists() else CACHE_DATA
OFFICIAL = DATA / "official" if (DATA / "official").exists() else CACHE_DATA
WORK = REPO / ".cache" / "work"

FEATURES_TIF = (BRIDGE / "gems-geodawn-numerical-features.tif"
                if (BRIDGE / "gems-geodawn-numerical-features.tif").exists()
                else CACHE_DATA / "training_features.tif")
LABELS_TIF = OFFICIAL / "labels.tif"
EXISTING_TIF = OFFICIAL / "existing_faults.tif"
SAMPLE_TIF = OFFICIAL / "sample_submission.tif"

FEATURES_SHA256 = "4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5"
LABELS_SHA256 = "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093"
SAMPLE_SHA256 = "2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc"

STACK_NPY = WORK / "feature_stack_f32.npy"
STACK_META = WORK / "feature_stack_meta.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Grid:
    height: int
    width: int
    crs: str
    transform: tuple
    footprint: np.ndarray  # bool, footprint pixels
    nodata_sub: float


def load_grid(sample_path: Path = SAMPLE_TIF) -> Grid:
    with rasterio.open(sample_path) as src:
        band = src.read(1)
        crs = str(src.crs)
        tr = tuple(float(v) for v in src.transform)[:6]
        nod = float(src.nodatavals[0])
    return Grid(band.shape[0], band.shape[1], crs, tr, np.isfinite(band), nod)


def load_labels(path: Path = LABELS_TIF) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.int8)


def band_names(path: Path = FEATURES_TIF) -> list[str]:
    with rasterio.open(path) as src:
        return [src.tags(i + 1).get("band_name", f"band{i+1:02d}") for i in range(src.count)]


def read_band(path: Path, index: int) -> np.ndarray:
    """Read one 1-based band, replacing the float32 sentinel nodata with NaN."""
    with rasterio.open(path) as src:
        a = src.read(index).astype(np.float32)
        nod = src.nodatavals[index - 1]
    if nod is not None:
        a[~np.isfinite(a) | (a <= nod * 0.5)] = np.nan
    return a


# -----------------------------------------------------------------------------------------
# Feature stack
# -----------------------------------------------------------------------------------------

CHANNELS: list[str] = []


def _grad(a: np.ndarray, fill: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Central differences, NaN-safe (NaN -> fill after differencing)."""
    b = np.where(np.isfinite(a), a, fill)
    gy, gx = np.gradient(b.astype(np.float64))
    return gx.astype(np.float32), gy.astype(np.float32)


def _z(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = a[mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros_like(a, dtype=np.float32)
    lo, hi = np.nanpercentile(v, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    out = (a - lo) / (hi - lo)
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, -4.0, 5.0).astype(np.float32)


DERIV_BANDS = ["det_elev", "rtp", "iso_grav_anom", "cond_surf", "depth_to_base_surf"]
SMOOTH_BANDS = ["tmi_hg", "iso_grav_anom_hg", "det_elev_slope", "tc", "rtp"]


def channel_plan(raw_names: list[str]) -> list[str]:
    """Deterministic channel order.  Pure function of the source band names."""
    out = list(raw_names)
    for nm in DERIV_BANDS:
        out += [f"d{nm}_x", f"d{nm}_y", f"gmag_{nm}"]
    out += ["st_cos2", "st_sin2", "st_coherence"]
    for nm in SMOOTH_BANDS:
        out += [f"s3_{nm}", f"s9_{nm}"]
    return out


def build_feature_stack(path: Path = FEATURES_TIF, force: bool = False) -> dict:
    """Build (or load) the cached float32 feature stack as an on-disk .npy memmap.

    Channels: the 19 raw GeoDAWN bands (percentile-clipped), first derivatives of the five
    structurally informative bands, a structure-tensor orientation triplet from the
    detrended elevation and the reduced-to-pole magnetics, and 3 px / 9 px Gaussian smooths
    of the five edge / tilt channels.  Every value is finite; the footprint mask lives
    beside the stack, not inside it.

    Memory-safe: the destination is an ``open_memmap`` file and each channel is written then
    released, so peak RSS is a handful of full-grid float32 arrays (~50 MB each) instead of
    the whole 2.3 GB cube.  This box has 3 GB of RAM and the naive build was OOM-killed here
    before this rewrite.
    """
    if STACK_META.exists() and STACK_NPY.exists() and not force:
        meta = json.loads(STACK_META.read_text())
        if meta.get("source_sha256") == sha256_file(path):
            return meta
    WORK.mkdir(parents=True, exist_ok=True)
    names = band_names(path)
    with rasterio.open(path) as src:
        H, W = src.height, src.width
        nod = src.nodatavals[0]

    # The footprint is the organizer's own: isfinite(sample_submission).  Verified in
    # tests/test_grid.py to equal labels != -1 exactly (5,167,373 pixels).
    footprint = load_grid().footprint
    np.save(WORK / "footprint.npy", footprint)

    plan = channel_plan(names)
    slot = {nm: i for i, nm in enumerate(plan)}
    if STACK_NPY.exists():
        STACK_NPY.unlink()
    stack = np.lib.format.open_memmap(STACK_NPY, mode="w+", dtype=np.float32,
                                      shape=(len(plan), H, W))

    def put(name: str, arr: np.ndarray) -> None:
        a = np.asarray(arr, dtype=np.float32)
        a[~np.isfinite(a)] = 0.0
        stack[slot[name]] = a
        del a

    for i, nm in enumerate(names, start=1):
        band = read_band(path, i)
        put(nm, _z(band, footprint))
        if nm in DERIV_BANDS:
            gx, gy = _grad(band)
            put(f"d{nm}_x", gx)
            put(f"d{nm}_y", gy)
            put(f"gmag_{nm}", np.hypot(gx, gy))
            del gx, gy
        if nm in SMOOTH_BANDS:
            z = _z(band, footprint)
            put(f"s3_{nm}", ndimage.gaussian_filter(z.astype(np.float64), 3.0))
            put(f"s9_{nm}", ndimage.gaussian_filter(z.astype(np.float64), 9.0))
            del z
        del band

    # structure tensor from detrended elevation + reduced-to-pole magnetics (fault fabric)
    gx1, gy1 = _grad(read_band(path, names.index("det_elev") + 1))
    gx2, gy2 = _grad(read_band(path, names.index("rtp") + 1))
    gx = (0.5 * (gx1 + gx2)).astype(np.float32)
    gy = (0.5 * (gy1 + gy2)).astype(np.float32)
    del gx1, gy1, gx2, gy2
    jxx = ndimage.gaussian_filter((gx * gx).astype(np.float64), 3.0)
    jyy = ndimage.gaussian_filter((gy * gy).astype(np.float64), 3.0)
    jxy = ndimage.gaussian_filter((gx * gy).astype(np.float64), 3.0)
    del gx, gy
    tr = jxx + jyy
    disc = np.sqrt(np.maximum(tr * tr - 4 * (jxx * jyy - jxy * jxy), 0.0))
    l1 = 0.5 * (tr + disc)
    l2 = np.maximum(0.5 * (tr - disc), 0.0)
    put("st_coherence", np.clip(np.where(l1 > 1e-12, (l1 - l2) / (l1 + 1e-12), 0.0), 0, 1))
    two_theta = np.arctan2(2 * jxy, jxx - jyy)
    put("st_cos2", np.cos(two_theta))
    put("st_sin2", np.sin(two_theta))
    del jxx, jyy, jxy, tr, disc, l1, l2, two_theta
    stack.flush()
    del stack

    meta = dict(channels=plan, n_channels=len(plan), height=H, width=W,
                footprint_px=int(footprint.sum()), source=str(path),
                source_sha256=sha256_file(path), source_nodata=float(nod),
                raw_bands=names, path=str(STACK_NPY),
                footprint_path=str(WORK / "footprint.npy"),
                grid=dict(crs="EPSG:32611", pixel_m=100.0,
                          transform=(100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)))
    STACK_META.write_text(json.dumps(meta, indent=1))
    return meta



def load_feature_stack() -> tuple[np.ndarray, dict]:
    meta = json.loads(STACK_META.read_text())
    arr = np.load(STACK_NPY, mmap_mode="r")
    return arr, meta


def load_footprint() -> np.ndarray:
    if (WORK / "footprint.npy").exists():
        return np.load(WORK / "footprint.npy")
    return load_grid().footprint


def rows_of(stack: np.ndarray, ys: np.ndarray, xs: np.ndarray, block: int = 200_000) -> np.ndarray:
    """Gather pixel rows from a (C, H, W) memmap without materialising the whole cube."""
    n = ys.size
    out = np.empty((n, stack.shape[0]), dtype=np.float32)
    for s in range(0, n, block):
        e = min(n, s + block)
        out[s:e] = stack[:, ys[s:e], xs[s:e]].T
    return out
