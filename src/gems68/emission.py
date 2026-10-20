"""Binary dot emission from a probability surface.

Why dots and why binary (derivation in knowledge/01_why_02778.md):
DTI = T / (0.2*(T + S - M) + 0.8*|G|).  Scaling every emitted value by
lambda < 1 scales T and (S - M) together and lowers DTI, so for a fixed set of
emitted pixels the value 1.0 is optimal.  A dot pays iff its realised kernel
credit exceeds 0.2*DTI, so mass must be spent only on the best-ranked pixels and
never inside the catalogue halo (the 0.2708 -> 0.2778 ladder removed d = 1 px and
d = 2 px dots and gained both times).

Rule (identical for every candidate so comparisons are at matched mass):
  1. zero the surface within ``exclude_px`` of the (visible) catalogue;
  2. keep 5x5 local maxima of the surface (dots >= ~3 px apart, one kernel
     radius, so each dot covers distinct ground);
  3. take the top ``n_dots`` by surface value -> value 1.0.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

EXCLUDE_PX = 2.0


def exclusion_mask(catalogue: np.ndarray, exclude_px: float = EXCLUDE_PX) -> np.ndarray:
    return ndi.distance_transform_edt(~np.asarray(catalogue, bool)) <= exclude_px


def emit_dots(
    surface: np.ndarray,
    allowed: np.ndarray,
    n_dots_or_catalogue: int | np.ndarray | None = None,
    nms: int = 5,
    *,
    budget: int | None = None,
    nms_size: int | None = None,
    exclusion_px: float = EXCLUDE_PX,
) -> np.ndarray:
    """Emit binary dots from ``surface`` using 5x5 NMS and optional catalogue collar exclusion.

    Supports both:
      - ``emit_dots(surface, allowed, n_dots, nms=5)``
      - ``emit_dots(surface, allowed, catalogue, budget=..., nms_size=5, exclusion_px=2.0)``
    """
    eff_allowed = np.asarray(allowed, dtype=bool)
    if isinstance(n_dots_or_catalogue, np.ndarray):
        cat = np.asarray(n_dots_or_catalogue, dtype=bool)
        if exclusion_px > 0 and cat.any():
            eff_allowed = eff_allowed & ~exclusion_mask(cat, exclusion_px)
        n_dots = int(budget) if budget is not None else 0
    else:
        n_dots = int(budget) if budget is not None else int(n_dots_or_catalogue or 0)
    eff_nms = int(nms_size) if nms_size is not None else int(nms)

    s = np.where(eff_allowed, np.nan_to_num(surface.astype(np.float32), nan=0.0), 0.0)
    mx = ndi.maximum_filter(s, size=eff_nms, mode="constant", cval=0.0)
    cand = (s >= mx) & (s > 0) & eff_allowed
    yy, xx = np.nonzero(cand)
    if len(yy) == 0 or n_dots <= 0:
        return np.zeros(surface.shape, bool)
    vals = s[yy, xx]
    if n_dots < len(vals):
        keep = np.argpartition(-vals, n_dots - 1)[:n_dots]
        yy, xx = yy[keep], xx[keep]
    out = np.zeros(surface.shape, bool)
    out[yy, xx] = True
    return out


def spatial_block_terms(
    emitted: np.ndarray,
    truth: np.ndarray,
    active: np.ndarray,
    block_side: int = 200,
) -> np.ndarray:
    """Compute (tp, fp, fn, n_truth) per 200x200 px (20 km) spatial cluster."""
    from .metric import kernel

    p = np.asarray(emitted, dtype=bool) & np.asarray(active, dtype=bool)
    g = np.asarray(truth, dtype=bool) & np.asarray(active, dtype=bool)
    h, w = g.shape
    ncols = (w + block_side - 1) // block_side
    nrows = (h + block_side - 1) // block_side
    terms = np.zeros((ncols * nrows, 4), dtype=np.float64)

    if g.any():
        dp = ndi.distance_transform_edt(~p) if p.any() else np.full(g.shape, np.inf)
        cov = kernel(dp[g])
        ty, tx = np.nonzero(g)
        tids = (ty // block_side) * ncols + (tx // block_side)
        terms[:, 0] = np.bincount(tids, weights=cov, minlength=len(terms))
        terms[:, 2] = np.bincount(tids, weights=1.0 - cov, minlength=len(terms))
        terms[:, 3] = np.bincount(tids, weights=np.ones(len(ty)), minlength=len(terms))
        dg = ndi.distance_transform_edt(~g)
    else:
        dg = np.full(g.shape, np.inf)

    if p.any():
        py, px = np.nonzero(p)
        pids = (py // block_side) * ncols + (px // block_side)
        fp_w = 1.0 - kernel(dg[p])
        terms[:, 1] = np.bincount(pids, weights=fp_w, minlength=len(terms))
    return terms


def quota(n_total: int, footprint: np.ndarray, region: np.ndarray) -> int:
    """Share of a global dot budget proportional to a region's footprint area."""
    return int(round(n_total * float((footprint & region).sum()) / float(footprint.sum())))
