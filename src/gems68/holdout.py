"""Hide-and-recover, spatially blocked holdout (protocol rule 2).

PROVENANCE: ``quadrant_ids``, ``_pick_segments`` and ``buffered_component_draw``
are reused from the shared template
https://github.com/buffedlizard55-lab/GEMSDOE65/blob/main/lane65-sarprop/src/gems65_sarprop/template/holdout.py
(split_version 'buffered-whole-components-loqo-v2'), trimmed to what this lane
calls.  Semantics are unchanged:

* footprint split into four quadrants (NW, NE, SW, SE) at the median footprint
  row/col; each quadrant region is eroded by ``boundary_px`` so withheld truth
  never touches the fold boundary;
* whole 8-connected catalogue components are withheld (20 % of each
  quadrant's catalogue pixels), only components wholly inside the eroded region;
* ``visible`` = catalogue minus a ``buffer_px`` collar around every withheld
  component -> every catalogue-derived feature must be computed from ``visible``;
* ``masked_known`` = catalogue minus hidden = pixels masked out of scoring
  pixel-exactly (the organiser masks mapped faults; dots there earn nothing).

Scoring: pooled DTI, alpha 0.2, beta 0.8, 300 m triangular kernel, via
``evaluate_holdout.evaluate``.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

FOLD_NAMES = ("NW", "NE", "SW", "SE")
STRUCT3 = np.ones((3, 3), bool)
DOMAIN_ERODE = 12


def quadrant_ids(footprint: np.ndarray) -> np.ndarray:
    fp = np.asarray(footprint, bool)
    yy, xx = np.nonzero(fp)
    ym, xm = int(np.median(yy)), int(np.median(xx))
    gy, gx = np.ogrid[: fp.shape[0], : fp.shape[1]]
    q = np.full(fp.shape, -1, np.int8)
    q[(gy < ym) & (gx < xm) & fp] = 0
    q[(gy < ym) & (gx >= xm) & fp] = 1
    q[(gy >= ym) & (gx < xm) & fp] = 2
    q[(gy >= ym) & (gx >= xm) & fp] = 3
    return q


def _pick_segments(rng, sizes: np.ndarray, ids: np.ndarray, target_px: float) -> np.ndarray:
    if ids.size == 0:
        return ids
    perm = rng.permutation(ids)
    cum = np.cumsum(sizes[perm])
    k = int(np.searchsorted(cum, target_px)) + 1
    return perm[: min(k, perm.size)]


def buffered_component_draw(grid, seed=20, hide_frac=0.20, buffer_px=3, boundary_px=DOMAIN_ERODE):
    """Label-blind quadrant draw withholding whole catalogue components with a feature buffer."""
    if not 0 < hide_frac < 1 or buffer_px < 0 or boundary_px < buffer_px:
        raise ValueError("invalid holdout fraction/buffer")
    cat = np.asarray(grid.catalogue, bool)
    quad = quadrant_ids(grid.footprint)
    comp, count = ndi.label(cat, STRUCT3)
    sizes = np.bincount(comp.ravel(), minlength=count + 1)
    hidden = np.zeros(cat.shape, bool)
    regions, receipts = [], []
    for q, name in enumerate(FOLD_NAMES):
        region = ndi.binary_erosion(quad == q, iterations=boundary_px) if boundary_px else quad == q
        region &= grid.footprint
        candidate = np.unique(comp[cat & region])
        clipped = np.unique(comp[cat & ~region])
        eligible = np.setdiff1d(candidate[candidate > 0], clipped, assume_unique=True)
        rng = np.random.default_rng(seed * 10_000 + q)
        chosen = _pick_segments(rng, sizes, eligible, hide_frac * float((cat & region).sum()))
        withheld = np.isin(comp, chosen) & cat
        if (withheld & ~region).any():
            raise AssertionError("a withheld component was clipped")
        hidden |= withheld
        regions.append(region)
        receipts.append(dict(fold=name, seed=seed, held_components=int(len(chosen)),
                             withheld_positive_pixels=int(withheld.sum()),
                             eligible_components=int(len(eligible)),
                             boundary_buffer_px=boundary_px, feature_buffer_px=buffer_px))
    if not hidden.any():
        raise ValueError("no eligible withheld components")
    collar = ndi.distance_transform_edt(~hidden) <= buffer_px
    visible = cat & ~collar
    masked_known = cat & ~hidden
    minimum = float(ndi.distance_transform_edt(~visible)[hidden].min())
    if minimum <= buffer_px:
        raise AssertionError("context buffer was not honored")
    folds = [dict(fold=i, name=FOLD_NAMES[i], region=regions[i], truth=hidden & regions[i],
                  visible=visible, masked_known=masked_known, hidden_all=hidden,
                  receipt={**receipts[i], "nearest_visible_to_truth_px": minimum,
                           "split_version": "buffered-whole-components-loqo-v2"})
             for i in range(4)]
    return folds, quad
