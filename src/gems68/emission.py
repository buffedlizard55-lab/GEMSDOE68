"""Binary dot emission from a probability surface.

Why dots and why binary (see knowledge/01_why_02778_and_can_we_beat_it.md):
DTI = T / (0.2*(T + S - M) + 0.8*|G|). For a fixed set of pixels,
unit intensity increases truth coverage but also false-positive mass. Thinning
can improve the objective when deleted mass was low-credit; it can also lose
true recall. The family scores motivating 38,000 are owner-reported, not
organizer receipts for this method.

Fixed emission rule (identical across holdout stages):
  1. exclude the visible catalogue and 2 px around it;
  2. identify 5x5 local maxima and GREEDILY resolve equal-valued plateaus;
  3. rank by descending value, break ties by flat index, enforce at least
     3 px Chebyshev separation and keep up to the fixed quota;
  4. emit each selected pixel at 1.0.

Old code retained entire equal-valued LightGBM plateaus and could place
adjacent pixels despite a 5x5 NMS setting. This is a shared-tool correction,
not a new post-hoc budget or a private lane-specific emitter.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

EXCLUDE_PX = 2.0
EMITTER_VERSION = "gems68-plateau-safe-nms-v2"


def exclusion_mask(catalogue: np.ndarray, exclude_px: float = EXCLUDE_PX) -> np.ndarray:
    return ndi.distance_transform_edt(~np.asarray(catalogue, bool)) <= exclude_px


def emit_dots(surface: np.ndarray, allowed: np.ndarray, n_dots: int, nms: int = 5) -> np.ndarray:
    """Take rank-ordered local peaks, enforcing NMS even on tied score plateaus.

    ``maximum_filter`` identifies *all* plateau pixels as maxima. Selecting a
    top-k subset without further suppression silently violates the advertised
    5×5 NMS spacing for piecewise-constant tree probabilities. Coordinates are
    used only for reproducible tie-breaking, never for fitting or validation.
    """
    if nms < 1 or nms % 2 != 1 or n_dots < 0:
        raise ValueError("nms must be a positive odd window and n_dots >= 0")
    scores = np.asarray(surface, dtype=np.float32)
    allowed = np.asarray(allowed, dtype=bool)
    if scores.shape != allowed.shape or scores.ndim != 2:
        raise ValueError("surface/allowed must share a 2D grid")
    if not np.isfinite(scores[allowed]).all() or (scores[allowed] < 0).any():
        raise ValueError("allowed scores must be non-negative and finite")
    out = np.zeros(scores.shape, bool)
    if n_dots == 0:
        return out
    s = np.where(allowed, scores, 0.0)
    maxima = ndi.maximum_filter(s, size=nms, mode="constant", cval=0.0)
    flat = np.flatnonzero((s >= maxima) & (s > 0) & allowed)
    if flat.size == 0:
        return out
    vals = s.ravel()[flat]
    # Stable, transparent tie break: highest value, then smallest flat index.
    order = np.lexsort((flat, -vals))
    suppressed = np.zeros(scores.shape, bool)
    radius = nms // 2
    h, w = scores.shape
    count = 0
    for yx in flat[order]:
        y, x = divmod(int(yx), w)
        if suppressed[y, x]:
            continue
        out[y, x] = True
        suppressed[max(0, y - radius):min(h, y + radius + 1),
                   max(0, x - radius):min(w, x + radius + 1)] = True
        count += 1
        if count == n_dots:
            break
    return out


def quota(n_total: int, footprint: np.ndarray, region: np.ndarray) -> int:
    """Share of a global dot budget proportional to a region's footprint area."""
    return int(round(n_total * float((footprint & region).sum()) / float(footprint.sum())))
