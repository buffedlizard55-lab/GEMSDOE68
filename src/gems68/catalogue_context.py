"""Catalogue-context features, always derived from the *visible* faults only.

Protocol rule 2 of the parallel-run brief: never derive a catalogue-based
feature from a withheld fault.  Every function here takes the visible catalogue
mask for the fold (or inner block) being predicted, so the target block's labels
are masked out of the geometry that the model is allowed to see.

Features (see :data:`gems68.blocks.CATALOGUE_CONTEXT`):

``d_cat`` / ``log_d_cat``   distance to the nearest visible fault, in 100 m px
``n_cat_3/10/30``           visible-fault pixel count within 3/10/30 px
                            (300 m / 1 km / 3 km) -- a spacing proxy
``fault_cos2/sin2/coh``     structure-tensor orientation of the visible
                            catalogue: the local mapped strike and its coherence
``align``                   cos(2*(theta_local - theta_fault)): does the local
                            geophysical lineation run parallel to the mapped
                            strike?
``d_cat_fwd/bwd``           ``d_cat`` sampled 30 px along the local orientation,
                            forward and backward -- relay / stepover geometry
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .blocks import CATALOGUE_CONTEXT, structure_tensor_orientation

FORWARD_PX = 30


def catalogue_context(visible: np.ndarray, footprint: np.ndarray,
                      local_cos2: np.ndarray, local_sin2: np.ndarray) -> dict[str, np.ndarray]:
    vis = np.asarray(visible, bool) & np.asarray(footprint, bool)
    out: dict[str, np.ndarray] = {}

    d = ndi.distance_transform_edt(~vis).astype(np.float32)
    out["d_cat"] = d
    out["log_d_cat"] = np.log1p(d).astype(np.float32)

    v = vis.astype(np.float32)
    for r in (3, 10, 30):
        out[f"n_cat_{r}"] = ndi.uniform_filter(v, size=2 * r + 1, mode="constant").astype(np.float32)

    fcos2, fsin2, fcoh = structure_tensor_orientation(vis.astype(np.float32), sigma=3.0)
    out["fault_cos2"], out["fault_sin2"], out["fault_coh"] = fcos2, fsin2, fcoh

    # cos(2*(theta_local - theta_fault)) = cos2l*cos2f + sin2l*sin2f
    out["align"] = (local_cos2 * fcos2 + local_sin2 * fsin2).astype(np.float32)

    gx = np.sqrt(np.clip(0.5 * (1.0 + local_cos2), 0.0, 1.0))
    gy = np.sqrt(np.clip(0.5 * (1.0 - local_cos2), 0.0, 1.0)) * np.where(local_sin2 >= 0, 1.0, -1.0)
    h, w = d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    for key, sgn in (("d_cat_fwd", 1.0), ("d_cat_bwd", -1.0)):
        dy = np.rint(gy * sgn * FORWARD_PX).astype(np.int64)
        dx = np.rint(gx * sgn * FORWARD_PX).astype(np.int64)
        idx = (np.clip(yy + dy, 0, h - 1) * w + np.clip(xx + dx, 0, w - 1)).ravel()
        out[key] = d.ravel()[idx].reshape(h, w).astype(np.float32)

    missing = [k for k in CATALOGUE_CONTEXT if k not in out]
    if missing:
        raise RuntimeError(f"catalogue context missing {missing}")
    return out
