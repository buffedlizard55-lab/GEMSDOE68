"""Block-wise feature engineering for the auto-context lane.

Two feature families are produced here:

* **stage 1** -- the fold-invariant channel stack (45 channels: the 19 official
  DrivenData bands, 6 derived gradient channels, 11 lidar scarp channels, 8
  GeoDAWN radiometric/magnetic channels and the lidar coverage flag).
* **stage 2/3** -- the auto-context inputs of Tu (CVPR 2008,
  doi:10.1109/CVPR.2008.4587436): the previous stage's probability map sampled
  at +/-1..15 px along and across the *local* orientation, multi-scale
  anisotropic smooths of that map, the raw gradient/tilt/scarp channels, and
  catalogue-context features computed with the target block's labels masked.

Everything is evaluated in row strips with a halo, so the full 45-band stack is
never resident in RAM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

PAD_STAGE2 = 18          # halo needed for the +/-15 px offset samples
OFFSETS = (1, 2, 3, 5, 8, 12, 15)
ALONG_WIDTHS = (4, 8)         # oriented box half-widths along local orientation
ACROSS_WIDTHS = (2, 4)         # oriented box half-widths across local orientation
ISO_SIGMAS = (2.0, 8.0)


# --------------------------------------------------------------------------- io
def read_rows(stack, index, rows: slice, pad: int, channels,
              cols: tuple[int, int] | None = None) -> np.ndarray:
    """Read ``channels`` for global rows ``rows`` plus ``pad`` halo rows.

    Returns an array of shape ``(len(channels), nrows + 2*pad, ncols)``; rows
    outside the raster are replicated at the border so gradients stay defined.
    ``cols`` optionally restricts the columns (used to score only a bounding
    box instead of the whole raster).
    """
    h, w = stack.shape[1], stack.shape[2]
    r0, r1 = rows.start, rows.stop
    a, b = max(0, r0 - pad), min(h, r1 + pad)
    c0, c1 = (0, w) if cols is None else cols
    out = np.empty((len(channels), b - a, c1 - c0), dtype=np.float32)
    for j, name in enumerate(channels):
        out[j] = np.asarray(stack[index[name], a:b, c0:c1])
    # rows that fall outside the raster are replicated at the border; the block
    # must always be exactly (r1 - r0) + 2*pad rows tall.
    top = max(0, a - (r0 - pad))
    bot = max(0, (r1 + pad) - b)
    if top:
        out = np.concatenate([np.repeat(out[:, :1, :], top, axis=1), out], axis=1)
    if bot:
        out = np.concatenate([out, np.repeat(out[:, -1:, :], bot, axis=1)], axis=1)
    return out


# ------------------------------------------------------------------- orientation
def structure_tensor_orientation(field: np.ndarray, sigma: float = 3.0):
    """Local orientation of a 2-D field from its structure tensor.

    Returns ``(cos2theta, sin2theta, coherence)`` where theta is the dominant
    gradient direction.  NaN cells are filled with 0 before filtering so the
    filters stay finite, and the coherence is set to 0 there.
    """
    f = np.nan_to_num(np.asarray(field, dtype=np.float64), nan=0.0)
    gx = ndi.sobel(f, axis=1, mode="nearest")
    gy = ndi.sobel(f, axis=0, mode="nearest")
    jxx = ndi.gaussian_filter(gx * gx, sigma, mode="nearest")
    jyy = ndi.gaussian_filter(gy * gy, sigma, mode="nearest")
    jxy = ndi.gaussian_filter(gx * gy, sigma, mode="nearest")
    tmp = jxx - jyy
    lam1 = 0.5 * (jxx + jyy) + np.sqrt(0.25 * tmp * tmp + jxy * jxy)
    lam2 = 0.5 * (jxx + jyy) - np.sqrt(0.25 * tmp * tmp + jxy * jxy)
    coherence = np.clip((lam1 - lam2) / (lam1 + lam2 + 1e-12), 0.0, 1.0)
    # dominant gradient direction
    ang = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    return (np.cos(2.0 * ang).astype(np.float32),
            np.sin(2.0 * ang).astype(np.float32),
            coherence.astype(np.float32))


class _StripIndex:
    """Precomputed flat-index base for one row strip, reused by every gather."""

    __slots__ = ("flat", "h", "w", "base")

    def __init__(self, field: np.ndarray, rows: slice, pad: int,
                 cols: tuple[int, int] | None = None):
        r0, r1 = rows.start, rows.stop
        a, b = max(0, r0 - pad), min(field.shape[0], r1 + pad)
        c0, c1 = (0, field.shape[1]) if cols is None else cols
        sub = np.ascontiguousarray(field[a:b, c0:c1])
        top, bot = max(0, a - (r0 - pad)), max(0, (r1 + pad) - b)
        if top:
            sub = np.concatenate([np.repeat(sub[:1], top, axis=0), sub], axis=0)
        if bot:
            sub = np.concatenate([sub, np.repeat(sub[-1:], bot, axis=0)], axis=0)
        self.flat = sub.ravel()
        self.h, self.w = sub.shape
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        self.base = (yy * self.w + xx).astype(np.int32)

    def gather(self, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
        idx = self.base + (dy.astype(np.int32) * self.w + dx.astype(np.int32))
        np.clip(idx, 0, self.flat.size - 1, out=idx)
        return self.flat[idx]


def _halo(field: np.ndarray, rows: slice, pad: int,
          cols: tuple[int, int] | None = None) -> np.ndarray:
    r0, r1 = rows.start, rows.stop
    a, b = max(0, r0 - pad), min(field.shape[0], r1 + pad)
    c0, c1 = (0, field.shape[1]) if cols is None else cols
    sub = np.asarray(field[a:b, c0:c1])
    top, bot = max(0, a - (r0 - pad)), max(0, (r1 + pad) - b)
    if top:
        sub = np.concatenate([np.repeat(sub[:1], top, axis=0), sub], axis=0)
    if bot:
        sub = np.concatenate([sub, np.repeat(sub[-1:], bot, axis=0)], axis=0)
    return sub


def _orient_vectors(cos2: np.ndarray, sin2: np.ndarray, rows: slice, pad: int,
                    cols: tuple[int, int] | None = None):
    c2 = _halo(cos2, rows, pad, cols).astype(np.float64)
    s2 = _halo(sin2, rows, pad, cols).astype(np.float64)
    gx = np.sqrt(np.clip(0.5 * (1.0 + c2), 0.0, 1.0))
    gy = np.sqrt(np.clip(0.5 * (1.0 - c2), 0.0, 1.0)) * np.where(s2 >= 0, 1.0, -1.0)
    return gx, gy


def core_rows(block: np.ndarray, pad: int, nrows: int) -> np.ndarray:
    """Crop the halo off a padded strip: ``block`` is (C, nrows + 2*pad, W)."""
    if block.ndim == 2:
        return block[pad:pad + nrows]
    return block[:, pad:pad + nrows, :]


def build_context_block(prev: np.ndarray, rows: slice, cos2: np.ndarray,
                        sin2: np.ndarray,
                        cols: tuple[int, int] | None = None) -> np.ndarray:
    """Auto-context channels derived from the previous stage's probability map.

    Channel order is :func:`context_feature_names`: the map itself; samples at
    +/-1,2,3,5,8,12,15 px along and across the local orientation; oriented box
    means along/across at several half-widths (the multi-scale anisotropic
    smooths); and two isotropic Gaussian smooths.  The returned array is already
    cropped to the requested rows.
    """
    pad = PAD_STAGE2
    nrows = rows.stop - rows.start
    si = _StripIndex(prev, rows, pad, cols)
    gx, gy = _orient_vectors(cos2, sin2, rows, pad, cols)
    channels = [si.flat.reshape(si.h, si.w)]
    for d in OFFSETS:                                   # along orientation
        dy = np.rint(gy * d)
        dx = np.rint(gx * d)
        channels.append(si.gather(dy, dx))
    for d in OFFSETS:                                   # across orientation
        dy = np.rint(gx * d)
        dx = -np.rint(gy * d)
        channels.append(si.gather(dy, dx))
    for wdt in ALONG_WIDTHS + ACROSS_WIDTHS:             # anisotropic box means
        acc = np.zeros((si.h, si.w), np.float64)
        for d in range(-wdt, wdt + 1):
            acc += si.gather(np.rint(gy * d), np.rint(gx * d))
        channels.append((acc / (2 * wdt + 1)).astype(np.float32))
    sub = _halo(prev, rows, pad, cols).astype(np.float64)
    for sg in ISO_SIGMAS:                               # isotropic smooths
        sm = ndi.gaussian_filter(np.nan_to_num(sub, nan=0.0), sg, mode="nearest")
        channels.append(sm.astype(np.float32))
    out = np.stack([c.astype(np.float32) for c in channels], axis=0)
    return out[:, pad:pad + nrows, :]


# ------------------------------------------------------------------ feature sets
STAGE1_CHANNELS = (
    "mag_anom", "rtp", "tmi_hg", "geod_2ndinv", "iso_grav_anom_slope", "tc",
    "geod_shearrate", "geod_dilaterate", "tmi_vg", "deq_n100a15",
    "iso_grav_anom_vg", "det_elev", "iso_grav_anom", "tmi",
    "depth_to_base_surf", "ieq_n100a15", "cond_surf", "iso_grav_anom_hg",
    "det_elev_slope", "det_elev_gx", "det_elev_gy", "det_elev_gmag",
    "mag_gx", "mag_gy", "tilt_grad_mag", "ex_max", "ex_mean", "step_max",
    "lapneg_max", "lappos_max", "downface_max", "upface_max", "cross_max",
    "relief", "coh100", "strike", "rad_K", "rad_Th", "rad_U", "rad_TC",
    "ext_ThK", "ext_UK", "ext_UTh", "ext_TMI_up150",
)

# "the raw gradient/tilt/scarp channels" handed to stage 2
STAGE2_RAW = (
    "det_elev_gx", "det_elev_gy", "det_elev_gmag", "det_elev_slope",
    "mag_gx", "mag_gy", "tmi_hg", "iso_grav_anom_hg",
    "tc", "tilt_grad_mag",
    "ex_max", "ex_mean", "step_max", "lapneg_max", "lappos_max",
    "downface_max", "upface_max", "cross_max", "relief", "coh100", "strike",
)

CATALOGUE_CONTEXT = (
    "d_cat", "log_d_cat", "n_cat_3", "n_cat_10", "n_cat_30",
    "fault_cos2", "fault_sin2", "fault_coh", "align",
    "d_cat_fwd", "d_cat_bwd",
)


def context_feature_names(prev: str) -> list[str]:
    names = [prev]
    for d in OFFSETS:
        names.append(f"{prev}_along{d}")
    for d in OFFSETS:
        names.append(f"{prev}_across{d}")
    for wdt in ALONG_WIDTHS:
        names.append(f"{prev}_boxalong{wdt}")
    for wdt in ACROSS_WIDTHS:
        names.append(f"{prev}_boxacross{wdt}")
    for s in ISO_SIGMAS:
        names.append(f"{prev}_iso{int(s)}")
    return names


def stage2_feature_names(prev: str) -> list[str]:
    return context_feature_names(prev) + list(STAGE2_RAW) + list(CATALOGUE_CONTEXT)
