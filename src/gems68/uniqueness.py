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


def verdict(rhos: dict, overlaps: dict) -> dict:
    # Protocol rule 1 is a magnitude rule: a strongly *anti*-correlated surface
    # is the same lane with the sign flipped.  Rank by |rho| (fixed 2026-10-20;
    # the previous signed max could not flag rho <= -0.90).
    worst_rho = max(rhos.items(), key=lambda kv: abs(kv[1])) if rhos else ("", 0.0)
    informative = {k: v for k, v in overlaps.items() if v["coverage"] <= COVERAGE_MAX}
    excluded = sorted(k for k in overlaps if k not in informative)
    worst_ov = (max(informative.items(), key=lambda kv: kv[1]["overlap"]) if informative else ("", dict(overlap=0.0)))
    worst_ov = (worst_ov[0], worst_ov[1]["overlap"])
    dup = abs(worst_rho[1]) > RHO_MAX or worst_ov[1] > OVERLAP_MAX
    return dict(max_rank_corr=worst_rho, max_dot_overlap=worst_ov,
                thresholds=dict(rank_corr=RHO_MAX, dot_overlap=OVERLAP_MAX, registry_coverage_max=COVERAGE_MAX),
                excluded_uninformative_rasters=excluded, duplicate=bool(dup))
