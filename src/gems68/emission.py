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
