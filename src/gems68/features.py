"""Feature channels for the auto-context lane.

Three families, all on the 100 m grid, all float32, NaN -> band median inside
the footprint (sentinel -3.4e38 in the organiser stack is converted to NaN by
``grid.read_band``):

1. ``raw_stack``        19 organiser bands (GeoDAWN magnetics/radiometrics-derived
                        products, gravity, geodesy, seismicity, DEM, MT) as shipped.
2. ``physics_stack``    gradient / tilt / scarp (ridge) channels derived from the
                        detrended elevation, magnetics and gravity bands:
                          * |grad det_elev| (Sobel), Laplacian(det_elev)
                          * Hessian ridge strength of det_elev at sigma 1, 2 px
                            (largest-magnitude eigenvalue, sign kept: scarp crest
                            vs scarp base)
                          * tilt angle of det_elev = atan(|grad|/pixel) (Miller &
                            Singh 1994 formulation applied to topography)
                          * Hessian ridge strength of rtp and iso_grav_anom (sigma 2)
3. ``catalogue_context`` computed ONLY from the visible catalogue mask passed in
                        (hide-and-recover rule): distance to nearest visible
                        fault, local strike of the nearest fault (structure tensor
                        orientation of the smoothed visible mask), and fault
                        density at 1 km / 3 km (a spacing proxy).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import grid as G

PHYSICS_NAMES = [
    "elev_grad", "elev_lap", "elev_ridge_s1", "elev_ridge_s2", "elev_tilt",
    "rtp_ridge_s2", "grav_ridge_s2",
]
CONTEXT_NAMES = ["cat_dist", "cat_strike_cos2", "cat_strike_sin2", "cat_dens_1km", "cat_dens_3km"]


def _fill(a: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=True)
    bad = ~np.isfinite(a)
    if bad.any():
        med = float(np.nanmedian(a[footprint]))
        a[bad] = med
    return a


def hessian_ridge(a: np.ndarray, sigma: float) -> np.ndarray:
    """Signed principal curvature with the largest magnitude (Hessian eigenvalue)."""
    hxx = ndi.gaussian_filter(a, sigma, order=(0, 2))
    hyy = ndi.gaussian_filter(a, sigma, order=(2, 0))
    hxy = ndi.gaussian_filter(a, sigma, order=(1, 1))
    tr = 0.5 * (hxx + hyy)
    det = np.sqrt(np.maximum(0.25 * (hxx - hyy) ** 2 + hxy ** 2, 0))
    l1, l2 = tr + det, tr - det
    out = np.where(np.abs(l1) > np.abs(l2), l1, l2)
    return out.astype(np.float32)


def raw_stack(footprint: np.ndarray) -> list[np.ndarray]:
    return [_fill(G.read_band(i + 1), footprint) for i in range(19)]


def physics_stack(bands: list[np.ndarray]) -> list[np.ndarray]:
    elev = bands[G.BAND_NAMES.index("det_elev")]
    rtp = bands[G.BAND_NAMES.index("rtp")]
    grav = bands[G.BAND_NAMES.index("iso_grav_anom")]
    gx = ndi.sobel(elev, axis=1) / 8.0
    gy = ndi.sobel(elev, axis=0) / 8.0
    grad = np.hypot(gx, gy).astype(np.float32)
    lap = ndi.laplace(ndi.gaussian_filter(elev, 1.0)).astype(np.float32)
    tilt = np.arctan(grad / G.PIXEL_M).astype(np.float32)
    return [grad, lap, hessian_ridge(elev, 1.0), hessian_ridge(elev, 2.0), tilt,
            hessian_ridge(rtp, 2.0), hessian_ridge(grav, 2.0)]


def orientation_field(a: np.ndarray, sigma_grad: float = 1.0, sigma_win: float = 3.0):
    """Structure-tensor orientation (theta of the dominant *line* direction).

    Returns (cos2t, sin2t, coherence, theta) where theta is the direction along
    the linear feature (perpendicular to the dominant gradient)."""
    gx = ndi.gaussian_filter(a, sigma_grad, order=(0, 1))
    gy = ndi.gaussian_filter(a, sigma_grad, order=(1, 0))
    jxx = ndi.gaussian_filter(gx * gx, sigma_win)
    jyy = ndi.gaussian_filter(gy * gy, sigma_win)
    jxy = ndi.gaussian_filter(gx * gy, sigma_win)
    # gradient direction = 0.5*atan2(2jxy, jxx-jyy); line direction is +90 deg
    two_t = np.arctan2(2 * jxy, jxx - jyy)
    theta = 0.5 * two_t + np.pi / 2
    lam = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    coh = lam / (jxx + jyy + 1e-12)
    return (np.cos(2 * theta).astype(np.float32), np.sin(2 * theta).astype(np.float32),
            coh.astype(np.float32), theta.astype(np.float32))


def catalogue_context(visible: np.ndarray) -> list[np.ndarray]:
    """Context features from the VISIBLE catalogue only."""
    vis = np.asarray(visible, bool)
    dist = ndi.distance_transform_edt(~vis).astype(np.float32)
    sm = ndi.gaussian_filter(vis.astype(np.float32), 2.0)
    c2, s2, coh, _ = orientation_field(sm, 1.0, 4.0)
    # propagate the strike of the nearest fault outward (nearest-pixel indices)
    idx = ndi.distance_transform_edt(~vis, return_distances=False, return_indices=True)
    c2n = c2[idx[0], idx[1]]
    s2n = s2[idx[0], idx[1]]
    d1 = ndi.gaussian_filter(vis.astype(np.float32), 10.0).astype(np.float32)
    d3 = ndi.gaussian_filter(vis.astype(np.float32), 30.0).astype(np.float32)
    return [dist, c2n, s2n, d1, d3]


def stage1_matrix(bands, phys, footprint) -> np.ndarray:
    cols = bands + phys
    X = np.empty((int(footprint.sum()), len(cols)), np.float32)
    for j, c in enumerate(cols):
        X[:, j] = c[footprint]
    return X


STAGE1_NAMES = G.BAND_NAMES + PHYSICS_NAMES
