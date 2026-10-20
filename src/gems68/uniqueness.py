"""Lane-drift / duplicate checks against registry rasters (protocol rule 1).

* Spearman rank correlation of the candidate SURFACE against each registry
  raster on a fixed random sample of footprint pixels (threshold 0.90).
* Dot overlap: fraction of the candidate's dots lying within 3 px of any
  positive pixel of a registry raster (threshold 70 %), reported together with
  that raster's own 3 px footprint coverage and the lift = overlap / coverage.
  A registry raster whose 3 px halo covers more than ``COVERAGE_MAX`` of the
  footprint (e.g. a regular 5 px lattice, 99.9 %) overlaps *every* candidate by
  construction and cannot define a lane, so it is excluded from the duplicate
  verdict (IR-68-UNIQ-01).  Both checks run on the surface before placement and
  on the final dots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.stats import spearmanr

RHO_MAX = 0.90
OVERLAP_MAX = 0.70
COVERAGE_MAX = 0.70


def load_registry(paths: list[Path]) -> dict[str, np.ndarray]:
    out = {}
    for p in paths:
        with rasterio.open(p) as src:
            a = src.read(1).astype(np.float32)
        a[~np.isfinite(a)] = 0.0
        out[p.name] = a
    return out


def surface_rank_corr(surface: np.ndarray, registry: dict, footprint: np.ndarray,
                      n_sample: int = 400_000, seed: int = 68) -> dict:
    yy, xx = np.nonzero(footprint)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(yy), size=min(n_sample, len(yy)), replace=False)
    s = surface[yy[pick], xx[pick]]
    res = {}
    for name, a in registry.items():
        r = a[yy[pick], xx[pick]]
        rho = spearmanr(s, r).correlation if np.std(r) > 0 else 0.0
        res[name] = float(0.0 if not np.isfinite(rho) else rho)
    return res


def dot_overlap(dots: np.ndarray, registry: dict, footprint: np.ndarray,
                radius_px: float = 3.0) -> dict:
    """{name: {overlap, coverage, lift, n_registry_dots}} per registry raster."""
    res = {}
    n = int(dots.sum())
    n_fp = int(footprint.sum())
    for name, a in registry.items():
        pos = a > 0
        if not pos.any():
            res[name] = dict(overlap=0.0, coverage=0.0, lift=0.0, n_registry_dots=0)
            continue
        near = ndi.distance_transform_edt(~pos) <= radius_px
        ov = float((dots & near).sum() / max(n, 1))
        cov = float((near & footprint).sum() / max(n_fp, 1))
        res[name] = dict(overlap=ov, coverage=cov, lift=float(ov / max(cov, 1e-9)),
                         n_registry_dots=int(pos.sum()))
    return res


def check_against_paths(surface: np.ndarray, dots: np.ndarray, paths: list[Path],
                        footprint: np.ndarray, n_sample: int = 400_000, seed: int = 68) -> list[dict]:
    """Streaming variant of surface_rank_corr + dot_overlap: one registry raster
    in memory at a time (the in-memory ``load_registry`` needs ~50 MB per raster,
    which does not scale to the full 60+ raster registry in a 4 GB sandbox).
    Returns one row per raster with |Spearman| on the fixed footprint sample,
    dot overlap within 3 px, that raster's 3 px footprint coverage, and lift."""
    import hashlib

    yy, xx = np.nonzero(footprint)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(yy), size=min(n_sample, len(yy)), replace=False)
    ys, xs = yy[pick], xx[pick]
    s = surface[ys, xs]
    n_dots = int(dots.sum())
    rows = []
    for p in paths:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        with rasterio.open(p) as src:
            a = src.read(1).astype(np.float32)
            same_grid = (src.height, src.width) == surface.shape
        if not same_grid:
            rows.append(dict(name=p.name, sha256=h.hexdigest(), same_grid=False))
            continue
        a[~np.isfinite(a)] = 0.0
        r = a[ys, xs]
        rho = float(spearmanr(s, r).correlation) if np.std(r) > 0 else 0.0
        if not np.isfinite(rho):
            rho = 0.0
        pos = a > 0
        if pos.any():
            near = ndi.distance_transform_edt(~pos) <= 3.0
            ov = float((dots & near).sum() / max(n_dots, 1))
            cov = float((near & footprint).sum() / float(footprint.sum()))
        else:
            ov, cov = 0.0, 0.0
        rows.append(dict(name=p.name, sha256=h.hexdigest(), same_grid=True,
                         spearman=rho, abs_spearman=abs(rho), dot_overlap_3px=ov,
                         registry_coverage_3px=cov, lift=float(ov / max(cov, 1e-9)),
                         n_registry_dots=int(pos.sum())))
        del a
    return rows


def verdict(rhos: dict, overlaps: dict) -> dict:
    worst_rho = max(rhos.items(), key=lambda kv: kv[1]) if rhos else ("", 0.0)
    informative = {k: v for k, v in overlaps.items() if v["coverage"] <= COVERAGE_MAX}
    excluded = sorted(k for k in overlaps if k not in informative)
    worst_ov = (max(informative.items(), key=lambda kv: kv[1]["overlap"]) if informative else ("", dict(overlap=0.0)))
    worst_ov = (worst_ov[0], worst_ov[1]["overlap"])
    dup = worst_rho[1] > RHO_MAX or worst_ov[1] > OVERLAP_MAX
    return dict(max_rank_corr=worst_rho, max_dot_overlap=worst_ov,
                thresholds=dict(rank_corr=RHO_MAX, dot_overlap=OVERLAP_MAX, registry_coverage_max=COVERAGE_MAX),
                excluded_uninformative_rasters=excluded, duplicate=bool(dup))


def check_uniqueness(
    candidate_surface: np.ndarray,
    candidate_dots: np.ndarray,
    footprint: np.ndarray,
    registry_dirs: list[Path],
    *,
    max_rank_corr: float = RHO_MAX,
    max_dot_overlap: float = OVERLAP_MAX,
    n_sample: int = 300_000,
    seed: int = 68,
) -> dict:
    """Stream-check surface Spearman rank correlation and 3 px dot overlap against all prior rasters."""
    yy, xx = np.nonzero(footprint)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(yy), size=min(n_sample, len(yy)), replace=False)
    s_sub = candidate_surface[yy[pick], xx[pick]].astype(np.float64)
    n_dots = int(candidate_dots.sum())
    n_fp = int(footprint.sum())

    seen_names: set[str] = set()
    per_raster = {}
    for d in registry_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.tif")):
            if p.name in seen_names:
                continue
            seen_names.add(p.name)
            with rasterio.open(p) as src:
                if (src.height, src.width) != footprint.shape:
                    continue
                a = src.read(1).astype(np.float32)
            a[~np.isfinite(a)] = 0.0
            r_sub = a[yy[pick], xx[pick]].astype(np.float64)
            rho = float(spearmanr(s_sub, r_sub).correlation) if np.std(r_sub) > 0 else 0.0
            if not np.isfinite(rho):
                rho = 0.0
            pos = a > 0
            if pos.any():
                near = ndi.distance_transform_edt(~pos) <= 3.0
                ov = float((candidate_dots & near).sum() / max(n_dots, 1))
                cov = float((near & footprint).sum() / max(n_fp, 1))
                lift = float(ov / max(cov, 1e-9))
                n_reg = int(pos.sum())
            else:
                ov = cov = lift = 0.0
                n_reg = 0
            per_raster[p.name] = dict(
                surface_spearman=round(rho, 4),
                dot_overlap_3px=round(ov, 4),
                registry_3px_footprint_coverage=round(cov, 4),
                lift_over_coverage=round(lift, 3),
                n_registry_positive_px=n_reg,
                informative_for_lane_gate=bool(cov <= COVERAGE_MAX),
            )
            del a

    worst_rho_name, worst_rho_val = max(
        ((k, v["surface_spearman"]) for k, v in per_raster.items()),
        key=lambda kv: kv[1],
        default=("", 0.0),
    )
    inf_items = [(k, v["dot_overlap_3px"]) for k, v in per_raster.items() if v["informative_for_lane_gate"]]
    worst_ov_name, worst_ov_val = max(inf_items, key=lambda kv: kv[1], default=("", 0.0))
    passes = bool(worst_rho_val < max_rank_corr and worst_ov_val < max_dot_overlap)
    return dict(
        n_registry_rasters_checked=len(per_raster),
        max_surface_spearman=worst_rho_val,
        max_surface_spearman_vs=worst_rho_name,
        surface_spearman_threshold=max_rank_corr,
        max_dot_overlap_3px=worst_ov_val,
        max_dot_overlap_vs=worst_ov_name,
        dot_overlap_threshold=max_dot_overlap,
        informative_coverage_ceiling=COVERAGE_MAX,
        passes_uniqueness=passes,
        per_raster=per_raster,
    )

