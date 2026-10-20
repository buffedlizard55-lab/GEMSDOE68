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


def emit_dots(surface: np.ndarray, allowed: np.ndarray, n_dots: int, nms: int = 5) -> np.ndarray:
    s = np.where(allowed, np.nan_to_num(surface.astype(np.float32), nan=0.0), 0.0)
    mx = ndi.maximum_filter(s, size=nms, mode="constant", cval=0.0)
    cand = (s >= mx) & (s > 0) & allowed
    yy, xx = np.nonzero(cand)
    if len(yy) == 0:
        return np.zeros(surface.shape, bool)
    vals = s[yy, xx]
    if n_dots < len(vals):
        keep = np.argpartition(-vals, n_dots - 1)[:n_dots]
        yy, xx = yy[keep], xx[keep]
    out = np.zeros(surface.shape, bool)
    out[yy, xx] = True
    return out


def quota(n_total: int, footprint: np.ndarray, region: np.ndarray) -> int:
    """Share of a global dot budget proportional to a region's footprint area."""
    return int(round(n_total * float((footprint & region).sum()) / float(footprint.sum())))


# ---------------------------------------------------------------------------
# Exp-3 emission variants (preregistered in knowledge/02_exp3_preregistration_and_hypotheses.md).
# V0 above is unchanged.  V1/V2/V3 share one greedy min-spacing selector; they
# differ only in the candidate set / ranking score handed to it.
# ---------------------------------------------------------------------------

SPACING_PX = 2.8  # champion ladder's winning thinning scale (d2.8 -> 0.2600 rung)


def greedy_spacing(score: np.ndarray, allowed: np.ndarray, candidates: np.ndarray,
                   n_dots: int, min_dist_px: float = SPACING_PX) -> np.ndarray:
    """Top-``n_dots`` candidate pixels by ``score`` under an exact pairwise
    minimum-spacing constraint (greedy, descending score; grid-hash accelerated).

    Ties are broken by raster order so the result is deterministic.  Every pair
    of emitted dots is >= ``min_dist_px`` apart (verified in tests)."""
    cand = np.asarray(candidates, bool) & np.asarray(allowed, bool)
    yy, xx = np.nonzero(cand)
    out = np.zeros(score.shape, bool)
    if len(yy) == 0 or n_dots <= 0:
        return out
    vals = np.nan_to_num(score[yy, xx].astype(np.float64), nan=-np.inf)
    order = np.lexsort((xx, yy, -vals))  # descending value, then raster order
    yy, xx = yy[order], xx[order]
    cell = float(min_dist_px)
    r2 = cell * cell
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    kept_y: list[int] = []
    kept_x: list[int] = []
    for y, x in zip(yy.tolist(), xx.tolist()):
        by, bx = int(y // cell), int(x // cell)
        ok = True
        for gy in (by - 1, by, by + 1):
            for gx in (bx - 1, bx, bx + 1):
                for (ky, kx) in buckets.get((gy, gx), ()):  # exact check vs neighbours
                    if (y - ky) ** 2 + (x - kx) ** 2 < r2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            buckets.setdefault((by, bx), []).append((y, x))
            kept_y.append(y)
            kept_x.append(x)
            if len(kept_y) >= n_dots:
                break
    out[np.asarray(kept_y), np.asarray(kept_x)] = True
    return out


def crest_mask(surface: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Across-strike ridge crests of ``surface`` (H68-A).

    A pixel is a crest pixel iff its surface value is >= the bilinearly sampled
    surface at +/-1 px and +/-2 px ACROSS the local line orientation (structure
    tensor of the surface itself, same parameters as the stage-k context:
    sigma_grad 1.5, sigma_win 4).  Along-strike neighbours never suppress, so a
    continuous fault-parallel crest survives intact."""
    from . import features as F

    s = np.where(allowed, np.nan_to_num(surface.astype(np.float32), nan=0.0), 0.0)
    _, _, _, theta = F.orientation_field(s, 1.5, 4.0)
    yy, xx = np.nonzero(allowed & (s > 0))
    th = theta[yy, xx].astype(np.float64)
    cy, cx = -np.cos(th), np.sin(th)  # across-line unit vector (row, col); along = (sin, cos)
    centre = s[yy, xx].astype(np.float64)
    keep = np.ones(len(yy), bool)
    for d in (1.0, 2.0):
        for sgn in (1.0, -1.0):
            sampled = ndi.map_coordinates(
                s, [yy + sgn * d * cy, xx + sgn * d * cx], order=1, mode="nearest")
            keep &= centre >= sampled - 1e-12
    out = np.zeros(surface.shape, bool)
    out[yy[keep], xx[keep]] = True
    return out


def greedy_spacing_stratified(score: np.ndarray, allowed: np.ndarray, candidates: np.ndarray,
                              zone_id: np.ndarray, zone_quotas: dict[int, int],
                              min_dist_px: float = SPACING_PX) -> np.ndarray:
    """Greedy >=``min_dist_px`` spacing (global, exact) with per-zone dot quotas.

    Single pass in descending score order (raster-order tie-break): a candidate
    is accepted iff it is >= min_dist from every accepted dot AND its zone's
    quota is unexhausted.  Used by Exp-4 (H68-G) with near/far zones split at
    10 px from the visible catalogue and quotas proportional to zone area."""
    cand = np.asarray(candidates, bool) & np.asarray(allowed, bool)
    yy, xx = np.nonzero(cand)
    out = np.zeros(score.shape, bool)
    if len(yy) == 0:
        return out
    vals = np.nan_to_num(score[yy, xx].astype(np.float64), nan=-np.inf)
    order = np.lexsort((xx, yy, -vals))
    yy, xx = yy[order], xx[order]
    zones = zone_id[yy, xx]
    remaining = dict(zone_quotas)
    cell = float(min_dist_px)
    r2 = cell * cell
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    kept_y: list[int] = []
    kept_x: list[int] = []
    total = sum(zone_quotas.values())
    for y, x, z in zip(yy.tolist(), xx.tolist(), zones.tolist()):
        if remaining.get(int(z), 0) <= 0:
            continue
        by, bx = int(y // cell), int(x // cell)
        ok = True
        for gy in (by - 1, by, by + 1):
            for gx in (bx - 1, bx, bx + 1):
                for (ky, kx) in buckets.get((gy, gx), ()):
                    if (y - ky) ** 2 + (x - kx) ** 2 < r2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            buckets.setdefault((by, bx), []).append((y, x))
            kept_y.append(y)
            kept_x.append(x)
            remaining[int(z)] -= 1
            if len(kept_y) >= total:
                break
    out[np.asarray(kept_y), np.asarray(kept_x)] = True
    return out


def rank01(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Percentile rank in [0,1] of ``values`` over ``mask`` (0 outside)."""
    out = np.zeros(values.shape, np.float32)
    idx = np.flatnonzero(np.asarray(mask, bool).ravel())
    if idx.size == 0:
        return out
    v = np.nan_to_num(values.ravel()[idx].astype(np.float64), nan=-np.inf)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(idx.size, np.float64)
    ranks[order] = np.arange(idx.size)
    out.ravel()[idx] = (ranks / max(idx.size - 1, 1)).astype(np.float32)
    return out
