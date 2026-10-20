"""Leak-free multi-physics, LiDAR scarp, GeoDAWN, GDR-1391 geothermal vent,
Andersonian kinematic & confound channel stack for GEMSDOE68.

STRICT ZERO-LEAKAGE INVARIANT
-----------------------------
Not a single function in this module reads ``labels.tif``, ``existing_faults.tif``,
``sample_submission.tif`` values, or any fault-distance column (e.g.
``dist_known_fault_px`` in ``gdr_wellspring_in_footprint.csv`` and
``gdr_volcanic_vents_in_footprint.csv`` is explicitly dropped on load).
Every channel is strictly label-blind and fold-invariant, and is audited
individually against withheld holdout fault segments via ``leakage_canary``
(rule: max(AUC, 1 - AUC) > 0.90 is flagged as leakage until disproven).

Official public data sources integrated (all verified & SHA-256 pinned):
  1. Competition 19-band geophysical/topographic stack (DrivenData #306).
  2. USGS 3DEP 1m/2m LiDAR scarp morphology stack (11 channels + valid mask).
  3. USGS GeoDAWN airborne radiometric & magnetic extensions (DOI: 10.5066/P93LGLVQ;
     K, Th, U, TC, Th/K, U/K, U/Th, 150 m upward-continued TMI).
  4. DOE INGENIOUS Great Basin Regional Dataset Compilation (GDR #1391,
     https://gdr.openei.org/submissions/1391, DOI: 10.15121/1881483, CC BY 4.0):
       - ``gdr_wellspring_in_footprint.csv``: 27,092 springs/wells thermal +
         quartz/chalcedony/Na-K-Ca geothermometry enthalpy surfaces;
       - ``2m_temperature_probe_INGENIOUS_regional_data.zip``: 2,782 in-footprint
         shallow 2-meter temperature probe ``F2mDAB`` anomaly stations;
       - ``paleo_geothermal_regional.zip``: siliceous sinter, travertine & tufa
         paleo-geothermal hot-spring vent mounds;
       - ``gdr_volcanic_vents_in_footprint.csv``: Quaternary volcanic vents.
  5. USGS State Geologic Map Compilation (SGMC) pre-Quaternary bedrock fault traces
     (``derived_sgmc_faults_100m_u8.tif``).
  6. US Census TIGER road distance (``tiger_road_distance_m.tif``) & GeoDAWN survey
     acquisition seam distance (``acquisition_block_id_100m.tif``) to suppress
     cultural road/railway scarp mimics and airborne survey block boundary seams.
"""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as rio_transform
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from . import channel_stack as CS
from . import features as F68
from . import grid as G68

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = REPO_ROOT / "data" / "external"
STACK_CACHE = REPO_ROOT / ".cache" / "s68_stack"

# Core local sharp + LiDAR + GeoDAWN channels (Experiment 1: H68-AC-O2)
EXP1_LOCAL_NAMES = [
    # High-pass local topography & Hessian ridges (no raw basin floor level!)
    "det_elev_hp", "det_elev_slope", "det_elev_gmag", "elev_lap",
    "elev_ridge_s1", "elev_ridge_s2", "elev_tilt",
    # High-pass local potential-field & subsurface derivatives
    "mag_hp", "rtp_hp", "tmi_hg", "tmi_vg", "tc", "tilt_grad_mag", "rtp_ridge_s2",
    "iso_grav_hp", "iso_grav_anom_hg", "iso_grav_anom_vg", "iso_grav_anom_slope", "grav_ridge_s2",
    "cond_hp", "basement_grad",
    # 11 USGS 3DEP 1m/2m LiDAR scarp channels + coverage indicator
    "ex_max", "ex_mean", "step_max", "lapneg_max", "lappos_max",
    "downface_max", "upface_max", "cross_max", "relief", "coh100", "strike", "lidar_valid",
    # 8 USGS GeoDAWN radiometric & magnetic extension channels + K alteration residual
    "rad_K", "rad_Th", "rad_U", "rad_TC", "rad_K_hp",
    "ext_ThK", "ext_UK", "ext_UTh", "ext_TMI_up150",
]

# Additional Geothermal Vent + Anderson Kinematic + Confound channels (Experiment 2 & 3)
EXP2_VENT_PHYSICS_NAMES = [
    # DOE INGENIOUS GDR-1391 Geothermal Vent & Upflow Conduit channels (label-free!)
    "gdr_spring_enthalpy_k15",
    "gdr_hot_spring_logd",
    "gdr_probe2m_dab_k8",
    "gdr_probe2m_anom_logd",
    "gdr_paleovent_logd",
    "gdr_volcvent_logd",
    "vent_alteration_coupling",
    # Andersonian stress/strain kinematics + subsurface persistence + bedrock grain
    "ext_TMI_up150_hg",
    "potential_concordance",
    "anderson_nne_align",
    "walker_nw_align",
    "geod_strain_index",
    "geod_dilat_norm",
    "crustal_seis_rate",
    "sgmc_bedrock_logd",
    # Cultural & survey-seam false-positive suppressors
    "tiger_road_logd",
    "geodawn_seam_logd",
]

ALL_STACK_NAMES = EXP1_LOCAL_NAMES + EXP2_VENT_PHYSICS_NAMES

# Compact physical channel subset passed verbatim alongside Auto-Context features in Stage >= 2
STAGE2_PHYS_EXP1 = [
    "det_elev_gmag", "elev_lap", "elev_ridge_s1", "elev_ridge_s2", "elev_tilt",
    "tmi_hg", "tc", "tilt_grad_mag", "iso_grav_anom_hg", "grav_ridge_s2", "cond_hp",
    "ex_max", "step_max", "lapneg_max", "lappos_max", "downface_max", "relief", "coh100",
    "lidar_valid", "rad_K_hp", "ext_ThK", "ext_TMI_up150",
]

STAGE2_PHYS_EXP2 = STAGE2_PHYS_EXP1 + [
    "gdr_spring_enthalpy_k15", "gdr_hot_spring_logd", "gdr_probe2m_dab_k8",
    "gdr_paleovent_logd", "vent_alteration_coupling",
    "ext_TMI_up150_hg", "potential_concordance", "anderson_nne_align",
    "geod_strain_index", "sgmc_bedrock_logd",
    "tiger_road_logd", "geodawn_seam_logd",
]


def _parse_shp_points(shp_bytes: bytes) -> list[tuple[float, float]]:
    pos = 100
    pts: list[tuple[float, float]] = []
    n = len(shp_bytes)
    while pos + 8 <= n:
        _, content_len = struct.unpack(">ii", shp_bytes[pos:pos + 8])
        pos += 8
        rec = shp_bytes[pos:pos + content_len * 2]
        pos += content_len * 2
        if len(rec) >= 20 and struct.unpack("<i", rec[:4])[0] in (1, 11, 21):
            pts.append(struct.unpack("<dd", rec[4:20]))
        else:
            pts.append((float("nan"), float("nan")))
    return pts


def _idw_grid(points_rc: np.ndarray, values: np.ndarray, shape: tuple[int, int],
              scale_px: float = 15.0, k: int = 8, sigma: float = 2.0) -> np.ndarray:
    if points_rc.shape[0] == 0:
        return np.zeros(shape, dtype=np.float32)
    tree = cKDTree(points_rc.astype(np.float64))
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    pts = np.column_stack([yy.ravel().astype(np.float64), xx.ravel().astype(np.float64)])
    kk = min(k, points_rc.shape[0])
    out = np.empty(pts.shape[0], dtype=np.float32)
    for s in range(0, pts.shape[0], 600_000):
        e = min(pts.shape[0], s + 600_000)
        d, idx = tree.query(pts[s:e], k=kk)
        if kk == 1:
            d = d[:, None]
            idx = idx[:, None]
        w = 1.0 / (1.0 + (d / scale_px) ** 2)
        out[s:e] = ((w * values[idx]).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-12)).astype(np.float32)
    arr = out.reshape(shape)
    if sigma > 0:
        arr = ndi.gaussian_filter(arr, sigma=sigma).astype(np.float32)
    return arr


def _log_dist_to_points(rows_idx: np.ndarray, cols_idx: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    ok = (rows_idx >= 0) & (rows_idx < shape[0]) & (cols_idx >= 0) & (cols_idx < shape[1])
    mask[rows_idx[ok], cols_idx[ok]] = True
    if not mask.any():
        return np.full(shape, 8.0, dtype=np.float32)
    return np.log1p(ndi.distance_transform_edt(~mask)).astype(np.float32)


def _rank_norm(arr: np.ndarray, fp: np.ndarray) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.float32)
    vals = arr[fp]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty_like(vals, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, vals.size, dtype=np.float32)
    out[fp] = ranks
    return out


def build_or_load_stack(force: bool = False) -> tuple[list[str], np.memmap, np.ndarray, dict]:
    """Build or load the 59-channel footprint matrix ``X`` of shape ``(N_fp, 59)``."""
    STACK_CACHE.mkdir(parents=True, exist_ok=True)
    x_path = STACK_CACHE / "X_fp59.npy"
    meta_path = STACK_CACHE / "meta.json"

    grid = G68.load_grid()
    fp = grid.footprint
    h, w = fp.shape
    n_fp = int(fp.sum())

    if not force and x_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("names") == ALL_STACK_NAMES and meta.get("n_fp") == n_fp:
            mm = np.load(x_path, mmap_mode="r")
            return list(ALL_STACK_NAMES), mm, fp, meta

    # Ensure base 45-channel memmap exists first
    cs_names, cs_mm, _ = CS.load_channels(use_cache=True)
    cs_idx = {n: i for i, n in enumerate(cs_names)}

    mm = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32,
                                   shape=(n_fp, len(ALL_STACK_NAMES)))
    col_idx = {n: j for j, n in enumerate(ALL_STACK_NAMES)}

    def get_cs(name: str, fill_zero: bool = False) -> np.ndarray:
        a = np.array(cs_mm[cs_idx[name]], dtype=np.float32, copy=True)
        bad = ~np.isfinite(a)
        if bad.any():
            fill = 0.0 if fill_zero else float(np.nanmedian(a[fp & ~bad]))
            a[bad] = fill
        return a

    def put(name: str, arr2d: np.ndarray) -> None:
        v = np.asarray(arr2d, dtype=np.float32)[fp]
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        mm[:, col_idx[name]] = v

    # 1. Local sharp topography + Hessian ridges (no basin floor altitude memorization!)
    det_elev = get_cs("det_elev")
    det_elev_hp = (det_elev - ndi.gaussian_filter(det_elev, sigma=12.0)).astype(np.float32)
    put("det_elev_hp", det_elev_hp)
    put("det_elev_slope", get_cs("det_elev_slope"))
    det_gmag = get_cs("det_elev_gmag")
    put("det_elev_gmag", det_gmag)
    put("elev_lap", ndi.laplace(ndi.gaussian_filter(det_elev, 1.0)).astype(np.float32))
    put("elev_ridge_s1", F68.hessian_ridge(det_elev, 1.0))
    put("elev_ridge_s2", F68.hessian_ridge(det_elev, 2.0))
    put("elev_tilt", np.arctan(det_gmag / 100.0).astype(np.float32))

    # Local orientation field of sharp topography + potential gradients
    c2_topo, s2_topo, _, _ = F68.orientation_field(det_elev, sigma_grad=1.0, sigma_win=3.0)

    # 2. Local potential-field & subsurface basement step derivatives
    mag = get_cs("mag_anom")
    rtp = get_cs("rtp")
    grav = get_cs("iso_grav_anom")
    cond = get_cs("cond_surf")
    base = get_cs("depth_to_base_surf")

    put("mag_hp", mag - ndi.gaussian_filter(mag, sigma=12.0))
    put("rtp_hp", rtp - ndi.gaussian_filter(rtp, sigma=12.0))
    tmi_hg = get_cs("tmi_hg")
    put("tmi_hg", tmi_hg)
    put("tmi_vg", get_cs("tmi_vg"))
    put("tc", get_cs("tc"))
    put("tilt_grad_mag", get_cs("tilt_grad_mag"))
    put("rtp_ridge_s2", F68.hessian_ridge(rtp, 2.0))

    put("iso_grav_hp", grav - ndi.gaussian_filter(grav, sigma=15.0))
    grav_hg = get_cs("iso_grav_anom_hg")
    put("iso_grav_anom_hg", grav_hg)
    put("iso_grav_anom_vg", get_cs("iso_grav_anom_vg"))
    put("iso_grav_anom_slope", get_cs("iso_grav_anom_slope"))
    put("grav_ridge_s2", F68.hessian_ridge(grav, 2.0))

    cond_hp = (cond - ndi.gaussian_filter(cond, sigma=12.0)).astype(np.float32)
    put("cond_hp", cond_hp)
    bx = ndi.sobel(base, axis=1) / 8.0
    by = ndi.sobel(base, axis=0) / 8.0
    put("basement_grad", np.hypot(bx, by).astype(np.float32))

    # 3. 11 USGS 3DEP 1m/2m LiDAR scarp channels + valid flag
    lidar_valid = get_cs("lidar_valid", fill_zero=True)
    put("lidar_valid", lidar_valid)
    for sname in CS.SCARP_CHANNELS:
        put(sname, get_cs(sname, fill_zero=True))

    # 4. 8 USGS GeoDAWN radiometric & magnetic extension channels + K alteration high-pass
    rad_K = get_cs("rad_K")
    put("rad_K", rad_K)
    put("rad_Th", get_cs("rad_Th"))
    put("rad_U", get_cs("rad_U"))
    put("rad_TC", get_cs("rad_TC"))
    rad_K_hp = (rad_K - ndi.gaussian_filter(rad_K, sigma=10.0)).astype(np.float32)
    put("rad_K_hp", rad_K_hp)
    put("ext_ThK", get_cs("ext_ThK"))
    put("ext_UK", get_cs("ext_UK"))
    put("ext_UTh", get_cs("ext_UTh"))
    tmi_up150 = get_cs("ext_TMI_up150")
    put("ext_TMI_up150", tmi_up150)

    # 5. Official DOE INGENIOUS GDR-1391 Geothermal Vent & Hydrothermal Conduit channels
    #    STRICTLY DROP dist_known_fault_px on load so zero fault-label information enters!
    ws_df = pd.read_csv(EXTERNAL / "gdr_wellspring_in_footprint.csv").drop(
        columns=["dist_known_fault_px"], errors="ignore"
    )
    for col in ("temp_c", "geothermquartz_c", "geothermchalc_c", "geothermcat_c"):
        ws_df[col] = pd.to_numeric(ws_df[col], errors="coerce")
    ws_df = ws_df.dropna(subset=["row", "col"])
    ws_site = ws_df.groupby(["row", "col"], as_index=False).agg(
        temp_c=("temp_c", "max"),
        tq=("geothermquartz_c", "max"),
        tc=("geothermchalc_c", "max"),
        tk=("geothermcat_c", "max"),
    )
    t_max = ws_site[["temp_c", "tq", "tc", "tk"]].max(axis=1)
    valid_t = t_max.notna()
    enth = np.clip(t_max[valid_t].to_numpy(dtype=np.float64) - 15.0, 0.0, 220.0)
    pts_ws = ws_site.loc[valid_t, ["row", "col"]].to_numpy(dtype=np.float64)
    spring_enth = _idw_grid(pts_ws, enth, (h, w), scale_px=15.0, k=8, sigma=2.0)
    put("gdr_spring_enthalpy_k15", spring_enth)

    hot_mask = (
        (ws_site["temp_c"].fillna(-999) >= 30.0)
        | (ws_site["tq"].fillna(-999) >= 80.0)
        | (ws_site["tc"].fillna(-999) >= 70.0)
        | (ws_site["tk"].fillna(-999) >= 90.0)
    )
    hot_r = np.rint(ws_site.loc[hot_mask, "row"].to_numpy()).astype(int)
    hot_c = np.rint(ws_site.loc[hot_mask, "col"].to_numpy()).astype(int)
    hot_logd = _log_dist_to_points(hot_r, hot_c, (h, w))
    put("gdr_hot_spring_logd", hot_logd)

    # 2m shallow temperature probe survey (Coolbaugh / Sladek / GBCGE INGENIOUS)
    with rasterio.open(REPO_ROOT / "data" / "bridge" / "sample_submission.tif") as sref:
        tr = sref.transform
    with zipfile.ZipFile(EXTERNAL / "2m_temperature_probe_INGENIOUS_regional_data.zip") as zf:
        shp_b = zf.read("2m_temperature_probe_INGENIOUS_regional_data/2m_temperature_probe_n83geo/2m_temperature_probe_n83geo.shp")
        dbf_b = zf.read("2m_temperature_probe_INGENIOUS_regional_data/2m_temperature_probe_n83geo/2m_temperature_probe_n83geo.dbf")
    pts_2m = _parse_shp_points(shp_b)
    num_rec, header_len, rec_len = struct.unpack("<IHH", dbf_b[4:12])
    fields = []
    fpos = 32
    while dbf_b[fpos] != 0x0D:
        fields.append((dbf_b[fpos:fpos + 11].split(b"\x00", 1)[0].decode("latin1"),
                       dbf_b[fpos + 16]))
        fpos += 32
    dab_vals, lon_2m, lat_2m = [], [], []
    rpos = header_len
    for idx_rec in range(num_rec):
        rec = dbf_b[rpos + 1:rpos + rec_len]
        rpos += rec_len
        off = 0
        row_dict = {}
        for fname, flen in fields:
            row_dict[fname] = rec[off:off + flen].decode("latin1").strip()
            off += flen
        try:
            dab = float(row_dict.get("F2mDAB", "nan"))
        except ValueError:
            dab = float("nan")
        dab_vals.append(dab)
        lon_2m.append(pts_2m[idx_rec][0])
        lat_2m.append(pts_2m[idx_rec][1])
    ux2, uy2 = rio_transform("EPSG:4269", "EPSG:32611", lon_2m, lat_2m)
    c2m = np.floor((np.asarray(ux2) - tr.c) / tr.a).astype(int)
    r2m = np.floor((np.asarray(uy2) - tr.f) / tr.e).astype(int)
    dab_arr = np.asarray(dab_vals, dtype=np.float64)
    ok2 = np.isfinite(dab_arr) & (r2m >= 0) & (r2m < h) & (c2m >= 0) & (c2m < w)
    probe_grid = _idw_grid(np.column_stack([r2m[ok2], c2m[ok2]]),
                           np.clip(dab_arr[ok2], -3.0, 15.0), (h, w), scale_px=10.0, k=6, sigma=2.0)
    put("gdr_probe2m_dab_k8", probe_grid)
    anom2 = ok2 & (dab_arr >= 1.5)
    probe_logd = _log_dist_to_points(r2m[anom2], c2m[anom2], (h, w))
    put("gdr_probe2m_anom_logd", probe_logd)

    # Paleo-geothermal sinter / travertine / tufa hot-spring vent mounds
    with zipfile.ZipFile(EXTERNAL / "paleo_geothermal_regional.zip") as zf:
        with zf.open("Paleo_geothermal_final.csv") as fh:
            paleo = pd.read_csv(fh, encoding="cp1252")
    uxp, uyp = rio_transform("EPSG:4269", "EPSG:32611",
                             paleo["Long_deg_NAD83"].tolist(), paleo["Lat_deg_NAD83"].tolist())
    cp = np.floor((np.asarray(uxp) - tr.c) / tr.a).astype(int)
    rp = np.floor((np.asarray(uyp) - tr.f) / tr.e).astype(int)
    paleo_logd = _log_dist_to_points(rp, cp, (h, w))
    put("gdr_paleovent_logd", paleo_logd)

    # Quaternary volcanic vents (drop dist_known_fault_px!)
    vv = pd.read_csv(EXTERNAL / "gdr_volcanic_vents_in_footprint.csv").drop(
        columns=["dist_known_fault_px"], errors="ignore"
    )
    put("gdr_volcvent_logd", _log_dist_to_points(
        np.rint(vv["row"].to_numpy()).astype(int),
        np.rint(vv["col"].to_numpy()).astype(int),
        (h, w),
    ))

    # Coupled hydrothermal conduit index: thermal/vent proximity x K-alteration + clay cap conductivity
    vent_prox = np.maximum(np.exp(-np.exp(hot_logd) / 20.0), np.exp(-np.exp(paleo_logd) / 20.0))
    alter_rank = 0.5 * _rank_norm(rad_K_hp, fp) + 0.5 * _rank_norm(cond_hp, fp)
    put("vent_alteration_coupling", (vent_prox * alter_rank).astype(np.float32))

    # 6. Andersonian stress/strain kinematics + subsurface persistence + bedrock grain
    up_gx = ndi.sobel(tmi_up150, axis=1) / 8.0
    up_gy = ndi.sobel(tmi_up150, axis=0) / 8.0
    up_hg = np.hypot(up_gx, up_gy).astype(np.float32)
    put("ext_TMI_up150_hg", up_hg)

    pot_conc = (
        _rank_norm(det_gmag, fp)
        * (0.5 * _rank_norm(grav_hg, fp) + 0.5 * _rank_norm(tmi_hg, fp))
    ).astype(np.float32)
    put("potential_concordance", pot_conc)

    # Canonical Northern Basin-and-Range strike (022 deg E of N => row-col azimuth)
    # In grid coords (row=South, col=East): azimuth alpha CW from North has unit line vector
    # (dy=-cos(alpha), dx=sin(alpha)), angle theta = atan2(dy, dx).
    for nm, az_deg in (("anderson_nne_align", 22.0), ("walker_nw_align", 328.0)):
        rad_az = np.deg2rad(az_deg)
        th_ref = float(np.arctan2(-np.cos(rad_az), np.sin(rad_az)))
        align = (c2_topo * np.cos(2.0 * th_ref) + s2_topo * np.sin(2.0 * th_ref)).astype(np.float32)
        put(nm, align)

    s2inv = get_cs("geod_2ndinv")
    dilat = get_cs("geod_dilaterate")
    shear = get_cs("geod_shearrate")
    put("geod_strain_index", np.log1p(np.maximum(s2inv, 0.0) + np.abs(shear)).astype(np.float32))
    put("geod_dilat_norm", (dilat / (np.abs(s2inv) + 1e-3)).astype(np.float32))

    ieq = get_cs("ieq_n100a15")
    deq = get_cs("deq_n100a15")
    put("crustal_seis_rate", (0.5 * _rank_norm(ieq, fp) + 0.5 * _rank_norm(deq, fp)).astype(np.float32))

    with rasterio.open(EXTERNAL / "derived_sgmc_faults_100m_u8.tif") as ds:
        sgmc = ds.read(1) > 0
    put("sgmc_bedrock_logd", np.log1p(ndi.distance_transform_edt(~sgmc)).astype(np.float32))

    # 7. Cultural road embankment & GeoDAWN block seam confound suppressors
    with rasterio.open(EXTERNAL / "audit_sources" / "tiger_road_distance_m.tif") as ds:
        road_m = np.nan_to_num(ds.read(1).astype(np.float32), nan=5000.0)
    put("tiger_road_logd", np.log1p(np.maximum(road_m, 0.0) / 100.0).astype(np.float32))

    with rasterio.open(EXTERNAL / "audit_sources" / "acquisition_block_id_100m.tif") as ds:
        blk_id = ds.read(1)
    seam = (ndi.maximum_filter(blk_id, size=3) != ndi.minimum_filter(blk_id, size=3)) & (blk_id > 0)
    put("geodawn_seam_logd", np.log1p(ndi.distance_transform_edt(~seam)).astype(np.float32))

    mm.flush()

    stats = {
        "n_wellspring_sites_with_temp_or_geotherm": int(valid_t.sum()),
        "n_hot_spring_or_geotherm_anomalies": int(hot_mask.sum()),
        "n_2m_probe_stations_in_grid": int(ok2.sum()),
        "n_2m_probe_anomalous_ge_1_5c": int(anom2.sum()),
        "n_paleo_geothermal_vents_in_footprint": int(((rp >= 0) & (rp < h) & (cp >= 0) & (cp < w) & fp[np.clip(rp, 0, h - 1), np.clip(cp, 0, w - 1)]).sum()),
        "n_quaternary_volcanic_vents": int(len(vv)),
        "label_columns_dropped": ["dist_known_fault_px"],
    }
    meta = {
        "names": ALL_STACK_NAMES,
        "exp1_names": EXP1_LOCAL_NAMES,
        "exp2_added_names": EXP2_VENT_PHYSICS_NAMES,
        "n_fp": n_fp,
        "n_channels": len(ALL_STACK_NAMES),
        "gdr_stats": stats,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return list(ALL_STACK_NAMES), np.load(x_path, mmap_mode="r"), fp, meta
