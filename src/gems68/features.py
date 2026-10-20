"""Feature engineering for the auto-context lane (Experiment 3: de-blobbed + 2m LiDAR scarp + GeoDAWN extensions).

Four families of Stage-2/Stage-3 inputs (verbatim from the lane assignment):

1. Stage-(k-1) probabilities sampled at +-1..15 px offsets along and across the
   local orientation (built in ``autocontext.py``).
2. Multi-scale anisotropic smooths AND 1D curvilinear ridge-vs-blob contrast
   of the Stage-(k-1) probability surface (dropping the sigma=10 px isotropic
   smooth per IR-68-BLOB-01).
3. Raw gradient / tilt / scarp channels handed directly to Stage 2 & Stage 3,
   including 2 m USGS 3DEP LiDAR fault-scarp channels and GeoDAWN radiometric /
   magnetic extension channels.
4. Catalogue-context features computed with the target block's labels masked:
   distance, orientation, alignment, along-strike tip-continuation, and spacing
   (density at 1 km and 3 km) to the nearest visible mapped fault.

Per IR-68-BLOB-01 (next-session directive from Experiment 2), Stage 1 drops the
five smooth regional seismicity/strain-rate bands (``ieq_n100a15``,
``deq_n100a15``, ``geod_2ndinv``, ``geod_dilaterate``, ``geod_shearrate``) and
replaces raw regional potential levels with 1.5 km (sigma=15 px) high-pass
residuals so trees cannot partition on 30-100 km basin/range coordinates.
"""
from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

from .grid import BAND_NAMES, DATA, Grid, read_band

REGIONAL_DROP_BANDS = frozenset({
    "ieq_n100a15",
    "deq_n100a15",
    "geod_2ndinv",
    "geod_dilaterate",
    "geod_shearrate",
})

POTENTIAL_HP_BANDS = frozenset({
    "det_elev",
    "iso_grav_anom",
    "tmi",
    "rtp",
    "mag_anom",
    "depth_to_base_surf",
    "cond_surf",
})

GRAD_BAND_NAMES = [
    "elev_grad", "elev_lap", "elev_ridge_s1", "elev_ridge_s2", "elev_tilt",
    "det_elev_slope", "tmi_hg", "tmi_vg", "rtp_ridge_s2",
    "iso_grav_anom_hg", "iso_grav_anom_slope", "grav_ridge_s2", "tc",
    "scarp_ex_max", "scarp_step_max", "scarp_upface_max", "scarp_coh100",
    "scarp_antislope_excess", "scarp_crest_toe_dipole",
    "ext_UK_hp15", "tmi_shallow_dipole",
]

CAT_CONTEXT_NAMES = [
    "cat_dist",
    "cat_strike_cos2",
    "cat_strike_sin2",
    "cat_dens_1km",
    "cat_dens_3km",
    "cat_spacing_ratio",
    "cat_align",
    "cat_along_tip_drop",
]

CACHE_DIR = DATA.parent / "exp3_phys_cache_v1"


def _fill_nan(a: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Replace NaN inside footprint by footprint median; zero outside."""
    out = np.zeros_like(a, dtype=np.float32)
    vals = a[footprint]
    med = float(np.nanmedian(vals)) if np.isfinite(vals).any() else 0.0
    out[footprint] = np.where(np.isfinite(vals), vals, med)
    return out


def _highpass(a: np.ndarray, footprint: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Local high-pass residual x - G_sigma(x) normalized by mask weight inside footprint."""
    w = footprint.astype(np.float32)
    num = ndi.gaussian_filter(np.where(footprint, a, 0.0).astype(np.float32), sigma, mode="nearest")
    den = np.maximum(ndi.gaussian_filter(w, sigma, mode="nearest"), 1e-5)
    res = np.zeros_like(a, dtype=np.float32)
    res[footprint] = a[footprint] - (num[footprint] / den[footprint])
    return res


def _grad_mag(a: np.ndarray, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gy = ndi.gaussian_filter(a, sigma, order=(1, 0), mode="nearest")
    gx = ndi.gaussian_filter(a, sigma, order=(0, 1), mode="nearest")
    return gy, gx, np.hypot(gy, gx).astype(np.float32)


def _hessian_ridge(a: np.ndarray, sigma: float) -> np.ndarray:
    """Largest-magnitude principal curvature (scarp/ridge detector at scale sigma)."""
    hyy = ndi.gaussian_filter(a, sigma, order=(2, 0), mode="nearest") * (sigma ** 2)
    hxx = ndi.gaussian_filter(a, sigma, order=(0, 2), mode="nearest") * (sigma ** 2)
    hxy = ndi.gaussian_filter(a, sigma, order=(1, 1), mode="nearest") * (sigma ** 2)
    tr = hyy + hxx
    det_term = np.hypot(hyy - hxx, 2.0 * hxy)
    l1 = 0.5 * (tr + det_term)
    l2 = 0.5 * (tr - det_term)
    return np.where(np.abs(l1) >= np.abs(l2), -l1, -l2).astype(np.float32)


def _tilt_angle(a: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Miller-Singh tilt angle atan(dz / |grad_h|), bounded in [-pi/2, pi/2]."""
    _, _, gh = _grad_mag(a, sigma)
    dz = -ndi.gaussian_filter(a, sigma, order=(2, 0), mode="nearest") - ndi.gaussian_filter(
        a, sigma, order=(0, 2), mode="nearest"
    )
    return np.arctan2(dz, gh + 1e-6).astype(np.float32)


def local_orientation(
    elev: np.ndarray,
    grav: np.ndarray,
    rtp: np.ndarray,
    scarp_coh: np.ndarray | None = None,
    scarp_strike_rad: np.ndarray | None = None,
    smooth_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Dominant strike unit vector (ay, ax) at every pixel."""
    jyy = np.zeros_like(elev, dtype=np.float32)
    jxx = np.zeros_like(elev, dtype=np.float32)
    jxy = np.zeros_like(elev, dtype=np.float32)
    for ch, w in ((elev, 1.0), (grav, 0.7), (rtp, 0.7)):
        gy, gx, gm = _grad_mag(ch, sigma=1.5)
        scale = float(np.percentile(gm, 95)) + 1e-6
        gy = gy / scale
        gx = gx / scale
        jyy += w * ndi.gaussian_filter(gy * gy, smooth_sigma, mode="nearest")
        jxx += w * ndi.gaussian_filter(gx * gx, smooth_sigma, mode="nearest")
        jxy += w * ndi.gaussian_filter(gy * gx, smooth_sigma, mode="nearest")
    grad_theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    strike = grad_theta + (np.pi * 0.5)
    if scarp_coh is not None and scarp_strike_rad is not None:
        c2 = np.cos(2.0 * strike) + 0.8 * scarp_coh * np.cos(2.0 * scarp_strike_rad)
        s2 = np.sin(2.0 * strike) + 0.8 * scarp_coh * np.sin(2.0 * scarp_strike_rad)
        strike = 0.5 * np.arctan2(s2, c2)
    ay = np.sin(strike).astype(np.float32)
    ax = np.cos(strike).astype(np.float32)
    return ay, ax


@dataclass
class PhysicalFeatures:
    X1: np.ndarray               # (N_footprint, F1) float32
    names1: list[str]
    ay: np.ndarray               # (H, W) float32 unit strike vector y
    ax: np.ndarray               # (H, W) float32 unit strike vector x
    grad_idx: np.ndarray         # (len(GRAD_BAND_NAMES),) int64 indices into X1
    grad_names: list[str]

    @property
    def grad_cols(self) -> np.ndarray:
        return self.X1[:, self.grad_idx]


def build_physical_features(
    grid: Grid,
    bands: dict[str, np.ndarray] | None = None,
    *,
    use_cache: bool = True,
    data_dir: Path | None = None,
) -> PhysicalFeatures:
    """Build fold-invariant Stage-1 physical feature matrix + local orientation field with low peak RAM."""
    meta_file = CACHE_DIR / "meta.json"
    if use_cache and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        X1 = np.load(CACHE_DIR / "X1.npy")
        ay = np.load(CACHE_DIR / "ay.npy")
        ax = np.load(CACHE_DIR / "ax.npy")
        return PhysicalFeatures(
            X1=X1,
            names1=meta["names1"],
            ay=ay,
            ax=ax,
            grad_idx=np.asarray(meta["grad_idx"], dtype=np.int64),
            grad_names=meta["grad_names"],
        )

    fp = grid.footprint
    n_fp = grid.n_footprint
    ddir = Path(data_dir) if data_dir else DATA
    ext_dir = ddir / "external"

    # Build ordered list of feature names first so we can pre-allocate X1 once
    comp_names = []
    for bname in BAND_NAMES:
        if bname in REGIONAL_DROP_BANDS:
            continue
        comp_names.append(f"{bname}_hp15" if bname in POTENTIAL_HP_BANDS else bname)
    derived_names = [
        "elev_grad", "elev_lap", "elev_ridge_s1", "elev_ridge_s2", "elev_tilt",
        "rtp_ridge_s2", "grav_ridge_s2",
    ]
    ext_names = [
        "scarp_ex_max", "scarp_ex_mean", "scarp_step_max",
        "scarp_lapneg_max", "scarp_lappos_max",
        "scarp_downface_max", "scarp_upface_max", "scarp_cross_max",
        "scarp_relief", "scarp_coh100", "scarp_valid",
        "scarp_antislope_excess", "scarp_crest_toe_dipole",
        "rad_K_hp15", "rad_Th_hp15", "rad_U_hp15", "rad_TC_hp15", "rad_K_grad",
        "ext_ThK_hp15", "ext_UK_hp15", "ext_UTh_hp15", "ext_TMI_up150_hp15",
        "tmi_shallow_dipole",
    ]
    names1 = comp_names + derived_names + ext_names
    name_to_col = {nm: i for i, nm in enumerate(names1)}
    X1 = np.empty((n_fp, len(names1)), dtype=np.float32)

    elev = grav = rtp = tmi_hp_fp = None
    for idx, bname in enumerate(BAND_NAMES, start=1):
        if bname in REGIONAL_DROP_BANDS:
            continue
        raw = bands[bname] if (bands is not None and bname in bands) else read_band(idx, ddir)
        filled = _fill_nan(raw, fp)
        del raw
        if bname == "det_elev":
            elev = filled
        elif bname == "iso_grav_anom":
            grav = filled
        elif bname == "rtp":
            rtp = filled
        if bname in POTENTIAL_HP_BANDS:
            hp = _highpass(filled, fp, sigma=15.0)
            fname = f"{bname}_hp15"
            X1[:, name_to_col[fname]] = hp[fp]
            if bname == "tmi":
                tmi_hp_fp = hp[fp].copy()
            del hp
        else:
            X1[:, name_to_col[bname]] = filled[fp]
        if bname not in ("det_elev", "iso_grav_anom", "rtp"):
            del filled

    # Derived structural filters
    _, _, eg = _grad_mag(elev, sigma=1.0)
    X1[:, name_to_col["elev_grad"]] = eg[fp]
    del eg
    elap = (
        ndi.gaussian_filter(elev, 1.5, order=(2, 0), mode="nearest")
        + ndi.gaussian_filter(elev, 1.5, order=(0, 2), mode="nearest")
    ).astype(np.float32)
    X1[:, name_to_col["elev_lap"]] = elap[fp]
    del elap
    er1 = _hessian_ridge(elev, sigma=1.2)
    X1[:, name_to_col["elev_ridge_s1"]] = er1[fp]
    del er1
    er2 = _hessian_ridge(elev, sigma=2.5)
    X1[:, name_to_col["elev_ridge_s2"]] = er2[fp]
    del er2
    etilt = _tilt_angle(elev, sigma=2.0)
    X1[:, name_to_col["elev_tilt"]] = etilt[fp]
    del etilt
    rr2 = _hessian_ridge(rtp, sigma=2.0)
    X1[:, name_to_col["rtp_ridge_s2"]] = rr2[fp]
    del rr2
    gr2 = _hessian_ridge(grav, sigma=2.5)
    X1[:, name_to_col["grav_ridge_s2"]] = gr2[fp]
    del gr2
    gc.collect()

    # 2 m USGS 3DEP LiDAR scarp features
    lidar_tif = ext_dir / "lidar_scarp_features_u8.tif"
    lidar_json = ext_dir / "lidar_scarp_features.json"
    scarp_coh_full = np.zeros(fp.shape, dtype=np.float32)
    scarp_valid_full = np.zeros(fp.shape, dtype=np.float32)
    scarp_strike_rad = np.zeros(fp.shape, dtype=np.float32)
    if lidar_tif.exists() and lidar_json.exists():
        meta = json.loads(lidar_json.read_text())
        q_rules = meta["quantisation"]
        with rasterio.open(lidar_tif) as ds:
            for idx, bname in enumerate(meta["bands"], start=1):
                raw_u8 = ds.read(idx)
                xmax, mode = q_rules[bname]
                valid_px = (raw_u8 > 0) & fp
                t = np.zeros(fp.shape, dtype=np.float32)
                t[valid_px] = (raw_u8[valid_px].astype(np.float32) - 1.0) / 254.0
                val = ((t * t) * float(xmax) if mode == "sqrt" else t * float(xmax)).astype(np.float32)
                if bname == "strike":
                    scarp_strike_rad = np.deg2rad(val).astype(np.float32)
                else:
                    X1[:, name_to_col[f"scarp_{bname}"]] = val[fp]
                    if bname == "coh100":
                        scarp_coh_full = val
                    elif bname == "valid":
                        scarp_valid_full = val
                del raw_u8, t
    else:
        for bname in ("ex_max", "ex_mean", "step_max", "lapneg_max", "lappos_max",
                      "downface_max", "upface_max", "cross_max", "relief", "coh100", "valid"):
            X1[:, name_to_col[f"scarp_{bname}"]] = 0.0

    X1[:, name_to_col["scarp_antislope_excess"]] = (
        X1[:, name_to_col["scarp_upface_max"]] - 0.5 * X1[:, name_to_col["scarp_downface_max"]]
    )
    X1[:, name_to_col["scarp_crest_toe_dipole"]] = (
        np.sqrt(
            np.maximum(
                X1[:, name_to_col["scarp_lapneg_max"]] * X1[:, name_to_col["scarp_lappos_max"]],
                0.0,
            )
        )
        * X1[:, name_to_col["scarp_coh100"]]
    )

    scarp_coh_w = (scarp_coh_full * scarp_valid_full).astype(np.float32)
    ay, ax = local_orientation(elev, grav, rtp, scarp_coh_w, scarp_strike_rad)
    del elev, grav, rtp, scarp_coh_full, scarp_valid_full, scarp_strike_rad, scarp_coh_w
    gc.collect()

    # GeoDAWN radiometrics
    rad_tif = ext_dir / "geodawn_rad_u8.tif"
    if rad_tif.exists():
        with rasterio.open(rad_tif) as ds:
            for idx, ch in enumerate(("K", "Th", "U", "TC"), start=1):
                u8 = ds.read(idx)
                valid_px = (u8 > 0) & fp
                arr = np.zeros(fp.shape, dtype=np.float32)
                arr[valid_px] = (u8[valid_px].astype(np.float32) - 128.0) / 127.0
                hp = _highpass(arr, valid_px, sigma=15.0)
                X1[:, name_to_col[f"rad_{ch}_hp15"]] = hp[fp]
                del hp
                if ch == "K":
                    _, _, kg = _grad_mag(arr, sigma=1.5)
                    X1[:, name_to_col["rad_K_grad"]] = kg[fp]
                    del kg
                del u8, arr
    else:
        for ch in ("K", "Th", "U", "TC"):
            X1[:, name_to_col[f"rad_{ch}_hp15"]] = 0.0
        X1[:, name_to_col["rad_K_grad"]] = 0.0

    # GeoDAWN extensions
    ext_tif = ext_dir / "geodawn_extensions_u8.tif"
    if ext_tif.exists():
        with rasterio.open(ext_tif) as ds:
            for idx, ch in enumerate(("ThK", "UK", "UTh", "TMI_up150"), start=1):
                u8 = ds.read(idx)
                valid_px = (u8 > 0) & fp
                arr = np.zeros(fp.shape, dtype=np.float32)
                arr[valid_px] = (u8[valid_px].astype(np.float32) - 128.0) / 127.0
                hp = _highpass(arr, valid_px, sigma=15.0)
                X1[:, name_to_col[f"ext_{ch}_hp15"]] = hp[fp]
                del u8, arr, hp
    else:
        for ch in ("ThK", "UK", "UTh", "TMI_up150"):
            X1[:, name_to_col[f"ext_{ch}_hp15"]] = 0.0

    up_fp = X1[:, name_to_col["ext_TMI_up150_hp15"]]
    tmi_std = float(np.std(tmi_hp_fp)) + 1e-6
    up_std = float(np.std(up_fp)) + 1e-6
    X1[:, name_to_col["tmi_shallow_dipole"]] = ((tmi_hp_fp / tmi_std) - (up_fp / up_std)).astype(np.float32)
    del tmi_hp_fp
    gc.collect()

    grad_idx = np.asarray([name_to_col[nm] for nm in GRAD_BAND_NAMES], dtype=np.int64)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(CACHE_DIR / "X1.npy", X1)
        np.save(CACHE_DIR / "ay.npy", ay)
        np.save(CACHE_DIR / "ax.npy", ax)
        meta_file.write_text(
            json.dumps(
                dict(
                    names1=names1,
                    grad_idx=grad_idx.tolist(),
                    grad_names=list(GRAD_BAND_NAMES),
                ),
                indent=2,
            )
        )
    return PhysicalFeatures(
        X1=X1,
        names1=names1,
        ay=ay,
        ax=ax,
        grad_idx=grad_idx,
        grad_names=list(GRAD_BAND_NAMES),
    )


def catalogue_context(
    visible_catalogue: np.ndarray,
    ay: np.ndarray | None = None,
    ax: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Compute leak-free catalogue-context maps from ONLY the visible fault mask."""
    vis = np.asarray(visible_catalogue, dtype=bool)
    if not vis.any():
        raise ValueError("visible_catalogue is empty")
    vf = vis.astype(np.float32)
    gy = ndi.sobel(vf, axis=0, mode="nearest")
    gx = ndi.sobel(vf, axis=1, mode="nearest")
    theta = np.arctan2(gy, gx) + (np.pi * 0.5)
    cos2_on = np.cos(2.0 * theta).astype(np.float32)
    sin2_on = np.sin(2.0 * theta).astype(np.float32)

    dist, (ny, nx) = ndi.distance_transform_edt(~vis, return_indices=True)
    dist = dist.astype(np.float32)
    near_cos2 = cos2_on[ny, nx]
    near_sin2 = sin2_on[ny, nx]
    dens_1km = ndi.gaussian_filter(vf, sigma=10.0, mode="constant", cval=0.0).astype(np.float32)
    dens_3km = ndi.gaussian_filter(vf, sigma=30.0, mode="constant", cval=0.0).astype(np.float32)
    spacing_ratio = (dens_1km / (dens_3km + 1e-4)).astype(np.float32)

    if ay is not None and ax is not None:
        loc_cos2 = (ax * ax - ay * ay).astype(np.float32)
        loc_sin2 = (2.0 * ay * ax).astype(np.float32)
        align = (loc_cos2 * near_cos2 + loc_sin2 * near_sin2).astype(np.float32)

        h, w = dist.shape
        yy, xx = np.indices((h, w), dtype=np.int32)
        dy = np.rint(12.0 * ay).astype(np.int32)
        dx = np.rint(12.0 * ax).astype(np.int32)
        y_f = np.clip(yy + dy, 0, h - 1)
        x_f = np.clip(xx + dx, 0, w - 1)
        y_b = np.clip(yy - dy, 0, h - 1)
        x_b = np.clip(xx - dx, 0, w - 1)
        raw_drop = dist - np.minimum(dist[y_f, x_f], dist[y_b, x_b])
        tip_drop = (raw_drop * np.exp(-dist / 15.0)).astype(np.float32)
    else:
        align = np.zeros_like(dist, dtype=np.float32)
        tip_drop = np.zeros_like(dist, dtype=np.float32)

    # Clamp Euclidean catalogue distance to [3.5, 12.0] px:
    # - below 3.5 px: prevents trees from learning the 3 px holdout-collar boundary as a
    #   negative predictor (which would erase Stage-2 probability on visible fault traces);
    # - above 12.0 px (1.2 km): prevents trees from creating a 1-2 km distance halo that
    #   starves >1 km basin-interior fault strands.
    dist_clamped = np.clip(dist, 3.5, 12.0).astype(np.float32)
    return [dist_clamped, near_cos2, near_sin2, dens_1km, dens_3km, spacing_ratio, align, tip_drop]


# ---------------------------------------------------------------------------
# Compatibility helpers restored at the PR #11 merge (2026-10-20).
# PR #9 rewrote this module for the scarp-ridge experiment and dropped the
# exp2-era helpers below; the Exp-3-ridgecrest / Exp-4 emission code
# (gems68.emission.crest_mask) and scripts/build_features.py still use them.
# They are restored verbatim from 06fc4f3 (only G.* references adapted to the
# names this module now imports).  PR #9's code above is untouched; its
# catalogue_context(visible, ay, ax) differs from the exp2-era one-arg version
# and is intentionally NOT shadowed here.
# ---------------------------------------------------------------------------

PHYSICS_NAMES = [
    "elev_grad", "elev_lap", "elev_ridge_s1", "elev_ridge_s2", "elev_tilt",
    "rtp_ridge_s2", "grav_ridge_s2",
]
CONTEXT_NAMES = ["cat_dist", "cat_strike_cos2", "cat_strike_sin2", "cat_dens_1km", "cat_dens_3km"]

def _fill(a: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=True)
    bad = ~np.isfinite(a)
    if bad.any():
        med = float(np.nanmedian(a[footprint]))
        a[bad] = med
    return a


def hessian_ridge(a: np.ndarray, sigma: float) -> np.ndarray:
    """Signed principal curvature with the largest magnitude (Hessian eigenvalue)."""
    hxx = ndi.gaussian_filter(a, sigma, order=(0, 2))
    hyy = ndi.gaussian_filter(a, sigma, order=(2, 0))
    hxy = ndi.gaussian_filter(a, sigma, order=(1, 1))
    tr = 0.5 * (hxx + hyy)
    det = np.sqrt(np.maximum(0.25 * (hxx - hyy) ** 2 + hxy ** 2, 0))
    l1, l2 = tr + det, tr - det
    out = np.where(np.abs(l1) > np.abs(l2), l1, l2)
    return out.astype(np.float32)


def raw_stack(footprint: np.ndarray) -> list[np.ndarray]:
    return [_fill(read_band(i + 1), footprint) for i in range(19)]


def physics_stack(bands: list[np.ndarray]) -> list[np.ndarray]:
    elev = bands[BAND_NAMES.index("det_elev")]
    rtp = bands[BAND_NAMES.index("rtp")]
    grav = bands[BAND_NAMES.index("iso_grav_anom")]
    gx = ndi.sobel(elev, axis=1) / 8.0
    gy = ndi.sobel(elev, axis=0) / 8.0
    grad = np.hypot(gx, gy).astype(np.float32)
    lap = ndi.laplace(ndi.gaussian_filter(elev, 1.0)).astype(np.float32)
    tilt = np.arctan(grad / 100.0).astype(np.float32)  # 100.0 = PIXEL_M
    return [grad, lap, hessian_ridge(elev, 1.0), hessian_ridge(elev, 2.0), tilt,
            hessian_ridge(rtp, 2.0), hessian_ridge(grav, 2.0)]


def orientation_field(a: np.ndarray, sigma_grad: float = 1.0, sigma_win: float = 3.0):
    """Structure-tensor orientation (theta of the dominant *line* direction).

    Returns (cos2t, sin2t, coherence, theta) where theta is the direction along
    the linear feature (perpendicular to the dominant gradient)."""
    gx = ndi.gaussian_filter(a, sigma_grad, order=(0, 1))
    gy = ndi.gaussian_filter(a, sigma_grad, order=(1, 0))
    jxx = ndi.gaussian_filter(gx * gx, sigma_win)
    jyy = ndi.gaussian_filter(gy * gy, sigma_win)
    jxy = ndi.gaussian_filter(gx * gy, sigma_win)
    # gradient direction = 0.5*atan2(2jxy, jxx-jyy); line direction is +90 deg
    two_t = np.arctan2(2 * jxy, jxx - jyy)
    theta = 0.5 * two_t + np.pi / 2
    lam = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    coh = lam / (jxx + jyy + 1e-12)
    return (np.cos(2 * theta).astype(np.float32), np.sin(2 * theta).astype(np.float32),
            coh.astype(np.float32), theta.astype(np.float32))


def stage1_matrix(bands, phys, footprint) -> np.ndarray:
    cols = bands + phys
    X = np.empty((int(footprint.sum()), len(cols)), np.float32)
    for j, c in enumerate(cols):
        X[:, j] = c[footprint]
    return X


STAGE1_NAMES = BAND_NAMES + PHYSICS_NAMES
