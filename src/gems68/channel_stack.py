"""Cached fold-invariant channel stack for the GEMSDOE68 auto-context lane.

This module is the lane-68 channel builder.  ``gems68.features`` is a different
module that arrived from the sibling lane and is left untouched.

Everything in this module is a *fold-invariant* channel: it depends only on the
raster data, never on which catalogue segments are withheld.  Catalogue-context
features (which must be derived from the visible faults only) live in
:mod:`gems68.catalogue_context` and are recomputed for every holdout fold.

Sources (all already bridged into ``data/`` by the shared template):

* ``data/bridge/training_features.tif`` -- the 19 DrivenData feature bands
  (sha256 ``4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5``,
  see ``registry/data_manifest.json``).
* ``data/external/lidar_scarp_features_u8.tif`` -- 2 m lidar-derived scarp
  channels aggregated to the 100 m grid (band semantics in
  ``data/external/lidar_scarp_features.json``).
* ``data/external/geodawn_rad_u8.tif`` / ``geodawn_extensions_u8.tif`` --
  GeoDAWN airborne radiometric and magnetic grids, USGS data release
  doi:10.5066/P93LGLVQ, resampled to the 100 m competition grid.

Quantisation is inverted exactly as documented in the sidecar JSON:
``q = 1 + round(254 * t(clip(x / xmax, 0, 1)))`` with ``t`` in {sqrt, linear};
``q == 0`` means "no sample" and is mapped to NaN.

The stack is built strip-by-strip into a memory-mapped ``.npy`` so the build
never needs the full 44-band stack resident (the sandbox has 3 GB of RAM).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
CACHE_DATA = REPO_ROOT / ".cache" / "gems_data"
BRIDGE = DATA / "bridge" if (DATA / "bridge" / "training_features.tif").exists() else CACHE_DATA
EXTERNAL = DATA / "external" if (DATA / "external" / "lidar_scarp_features_u8.tif").exists() else (CACHE_DATA / "external")
CACHE = REPO_ROOT / ".cache" / "channels"

HEIGHT, WIDTH = 3730, 3292
STRIP = 256

CORE_BANDS = (
    "mag_anom", "rtp", "tmi_hg", "geod_2ndinv", "iso_grav_anom_slope", "tc",
    "geod_shearrate", "geod_dilaterate", "tmi_vg", "deq_n100a15",
    "iso_grav_anom_vg", "det_elev", "iso_grav_anom", "tmi",
    "depth_to_base_surf", "ieq_n100a15", "cond_surf", "iso_grav_anom_hg",
    "det_elev_slope",
)

# channels handed to stage 2 verbatim ("the raw gradient/tilt/scarp channels")
GRADIENT_CHANNELS = ("det_elev_gx", "det_elev_gy", "det_elev_gmag",
                     "det_elev_slope", "mag_gx", "mag_gy", "tmi_hg",
                     "iso_grav_anom_hg")
TILT_CHANNELS = ("tc", "tilt_grad_mag")
SCARP_CHANNELS = ("ex_max", "ex_mean", "step_max", "lapneg_max", "lappos_max",
                  "downface_max", "upface_max", "cross_max", "relief", "coh100",
                  "strike")
RAD_CHANNELS = ("K", "Th", "U", "TC")
EXT_CHANNELS = ("ThK", "UK", "UTh", "TMI_up150")

CHANNEL_NAMES = (list(CORE_BANDS)
                 + ["det_elev_gx", "det_elev_gy", "det_elev_gmag",
                    "mag_gx", "mag_gy", "tilt_grad_mag"]
                 + list(SCARP_CHANNELS)
                 + ["lidar_valid"]
                 + ["rad_" + c for c in RAD_CHANNELS]
                 + ["ext_" + c for c in EXT_CHANNELS])

CACHE_FILE = CACHE / "channels_f32.npy"
CACHE_NAMES = CACHE / "channels_names.json"
SENTINEL = -1e38


def _central_grad(a: np.ndarray, axis: int) -> np.ndarray:
    """Central difference with edge replication; NaN propagates."""
    if axis == 1:
        g = np.empty_like(a)
        g[:, 1:-1] = (a[:, 2:] - a[:, :-2]) * 0.5
        g[:, 0] = a[:, 1] - a[:, 0]
        g[:, -1] = a[:, -1] - a[:, -2]
    else:
        g = np.empty_like(a)
        g[1:-1, :] = (a[2:, :] - a[:-2, :]) * 0.5
        g[0, :] = a[1, :] - a[0, :]
        g[-1, :] = a[-1, :] - a[-2, :]
    return g


def _sidecar_map(path: Path) -> dict[str, object]:
    """Decode a sidecar's quantisation description.

    Returns either ``[xmax, transform]`` (lidar sidecar: physical units) or the
    string ``"rank"`` (GeoDAWN sidecars: uint8 1..255 spans the in-footprint
    1st..99th percentile of each mosaicked channel, so the byte is a *rank*, not
    a physical unit).  A rank is a monotone transform of the physical value,
    which is all a tree model needs; the physical value is not recoverable from
    the byte without the percentile table, so it is never claimed.
    """
    spec = json.loads(Path(str(path).replace("_u8.tif", ".json")).read_text())
    qmap = spec["quantisation"]
    if isinstance(qmap, dict):
        return qmap
    order = spec.get("bands") or spec["channels"]
    return {name: "rank" for name in order}


def build_channels(verbose: bool = True) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Build the stack strip-by-strip into a memory-mapped float32 array."""
    import rasterio

    CACHE.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(CACHE_FILE, mode="w+", dtype=np.float32,
                                   shape=(len(CHANNEL_NAMES), HEIGHT, WIDTH))
    idx = {n: i for i, n in enumerate(CHANNEL_NAMES)}

    scarp_map = _sidecar_map(EXTERNAL / "lidar_scarp_features_u8.tif")
    rad_map = _sidecar_map(EXTERNAL / "geodawn_rad_u8.tif")
    ext_map = _sidecar_map(EXTERNAL / "geodawn_extensions_u8.tif")

    feats = rasterio.open(BRIDGE / "training_features.tif")
    scarp = rasterio.open(EXTERNAL / "lidar_scarp_features_u8.tif")
    rad = rasterio.open(EXTERNAL / "geodawn_rad_u8.tif")
    ext = rasterio.open(EXTERNAL / "geodawn_extensions_u8.tif")
    sub = rasterio.open(BRIDGE / "sample_submission.tif")

    def dequant(q: np.ndarray, name: str, qmap) -> np.ndarray:
        q = q.astype(np.float32)
        spec = qmap[name]
        if spec == "rank":
            val = np.clip((q - 1.0) / 254.0, 0.0, 1.0)
        else:
            xmax, transform = spec
            xmax = float(xmax)
            t = np.clip((q - 1.0) / 254.0, 0.0, 1.0)
            val = xmax * t * t if transform == "sqrt" else xmax * t
        val[q <= 0] = np.nan
        return val

    try:
        for r0 in range(0, HEIGHT, STRIP):
            r1 = min(HEIGHT, r0 + STRIP)
            h = r1 - r0
            win = rasterio.windows.Window(0, r0, WIDTH, h)
            fp = np.isfinite(sub.read(1, window=win))
            block: dict[str, np.ndarray] = {}
            for i in range(1, 20):
                name = feats.tags(i).get("band_name") or f"band{i:02d}"
                a = feats.read(i, window=win).astype(np.float32)
                a[a < SENTINEL] = np.nan
                block[name] = a
            gx = _central_grad(block["det_elev"], 1)
            gy = _central_grad(block["det_elev"], 0)
            block["det_elev_gx"], block["det_elev_gy"] = gx, gy
            block["det_elev_gmag"] = np.hypot(gx, gy)
            block["mag_gx"] = _central_grad(block["mag_anom"], 1)
            block["mag_gy"] = _central_grad(block["mag_anom"], 0)
            block["tilt_grad_mag"] = np.hypot(_central_grad(block["tmi_hg"], 1),
                                              _central_grad(block["tmi_hg"], 0))
            for i in range(1, scarp.count + 1):
                name = scarp.descriptions[i - 1]
                if name == "valid":
                    # lidar coverage flag: 1 where a 2 m sample exists in the cell
                    block["lidar_valid"] = (scarp.read(i, window=win) > 0).astype(np.float32)
                elif name in scarp_map:
                    block[name] = dequant(scarp.read(i, window=win), name, scarp_map)
            for i in range(1, rad.count + 1):
                name = rad.descriptions[i - 1]
                if name in rad_map:
                    block["rad_" + name] = dequant(rad.read(i, window=win), name, rad_map)
            for i in range(1, ext.count + 1):
                name = ext.descriptions[i - 1]
                if name in ext_map:
                    block["ext_" + name] = dequant(ext.read(i, window=win), name, ext_map)
            for name, arr in block.items():
                arr[~fp] = np.nan
                mm[idx[name], r0:r1, :] = arr
            del block, gx, gy
            mm.flush()
            if verbose:
                print(f"  strip {r0:5d}-{r1:5d} done", flush=True)
    finally:
        for ds in (feats, scarp, rad, ext, sub):
            ds.close()
    missing = [n for n in CHANNEL_NAMES if not np.isfinite(mm[idx[n]]).any()]
    if missing:
        raise RuntimeError(f"channels built empty: {missing}")
    CACHE_NAMES.write_text(json.dumps(CHANNEL_NAMES, indent=1))
    return list(CHANNEL_NAMES), mm, None


def load_channels(use_cache: bool = True):
    """Return (names, read-only memmap stack, footprint bool array)."""
    import rasterio

    CACHE.mkdir(parents=True, exist_ok=True)
    if use_cache and CACHE_FILE.exists() and CACHE_NAMES.exists():
        names = json.loads(CACHE_NAMES.read_text())
        if names != list(CHANNEL_NAMES):
            raise RuntimeError("cached channel names differ from the pinned list; rebuild")
        mm = np.load(CACHE_FILE, mmap_mode="r")
    else:
        names, mm, _ = build_channels()
    with rasterio.open(BRIDGE / "sample_submission.tif") as src:
        footprint = np.isfinite(src.read(1))
    return names, mm, footprint


def channel_index() -> dict[str, int]:
    return {n: i for i, n in enumerate(CHANNEL_NAMES)}
