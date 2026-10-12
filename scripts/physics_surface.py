#!/usr/bin/env python3
"""
Physics-based Fault Detection Surface — NO catalogue leakage
=============================================================
Build geological fault indicator surfaces from the 19 GeoDAWN bands only.
No distance-to-fault, no catalogue-derived features.
These surfaces detect structural signatures that could indicate unmapped faults.
"""
import numpy as np
import rasterio
import time
import gc
from scipy.ndimage import gaussian_filter, distance_transform_edt

t0 = time.time()
H, W = 3730, 3292

def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

log("=== PHYSICS-BASED FAULT SURFACES ===")

# Load features
features_mmap = np.load('data/features_mmap.npy', mmap_mode='r')
footprint = np.load('data/footprint.npy')

with rasterio.open('data/existing_faults.tif') as src:
    labels_raw = src.read(1)
labels_bool = labels_raw >= 1

# Helper: robust normalization to [0,1] within footprint
def robust_norm(arr, foot, pct=99.5):
    out = np.zeros_like(arr, dtype=np.float32)
    v = foot & np.isfinite(arr)
    if not v.any():
        return out
    vals = arr[v]
    hi = float(np.percentile(vals, pct))
    if hi <= 0:
        hi = float(np.max(np.abs(vals))) + 1e-9
    out[v] = np.clip(arr[v] / hi, 0.0, 1.0)
    return out

# ============================================================================
# Surface 1: Multi-scale ridge detection on magnetic/gravity/topography
# ============================================================================
log("Surface 1: Multi-physics ridge detection...")

def line_response(a, sigma, foot):
    """Frangi-style curvilinear response using Hessian eigenvalues."""
    smoothed = gaussian_filter(a, sigma, mode='nearest')
    axx = gaussian_filter(smoothed, sigma, order=(0, 2), mode='nearest')
    ayy = gaussian_filter(smoothed, sigma, order=(2, 0), mode='nearest')
    axy = gaussian_filter(smoothed, sigma, order=(1, 1), mode='nearest')
    tmp = np.sqrt(np.maximum(((axx - ayy) * 0.5) ** 2 + axy ** 2, 0.0))
    l1 = (axx + ayy) * 0.5 + tmp
    l2 = (axx + ayy) * 0.5 - tmp
    small = np.where(np.abs(l1) >= np.abs(l2), l2, l1)
    resp = -small  # positive on lines
    resp[~foot] = 0
    return resp.astype(np.float32)

# Key bands for ridge detection
# Band indices (0-based in features_mmap): 
# 0:mag_anom, 1:rtp, 5:tc(tilt), 11:det_elev, 12:iso_grav, 13:tmi, 16:cond_surf, 18:det_elev_slope

ridge_surfaces = []
key_bands_ridge = [0, 1, 5, 11, 12, 13, 16]

for bi in key_bands_ridge:
    band = features_mmap[bi].copy()
    for sigma in [1.0, 2.0, 3.0]:
        r = line_response(band, sigma, footprint)
        r = robust_norm(r, footprint)
        ridge_surfaces.append(r)
    del band
    gc.collect()

# OR-fusion of all ridge surfaces
ridge_fusion = np.zeros((H, W), dtype=np.float32)
for r in ridge_surfaces:
    ridge_fusion = np.maximum(ridge_fusion, r)
del ridge_surfaces
gc.collect()
log(f"  Ridge fusion: {ridge_fusion[footprint].mean():.4f} mean")

# ============================================================================
# Surface 2: Gradient coherence (structural boundaries)
# ============================================================================
log("Surface 2: Gradient coherence...")

grad_coh = np.zeros((H, W), dtype=np.float32)
for bi in [0, 1, 11, 12, 16]:  # mag, rtp, det_elev, grav, cond
    band = features_mmap[bi].copy()
    smoothed = gaussian_filter(band, 2.0, mode='nearest')
    gy = np.gradient(smoothed, axis=0)
    gx = np.gradient(smoothed, axis=1)
    mag = np.hypot(gy, gx)
    mag[~footprint] = 0
    
    # Orientation coherence via structure tensor
    jxx = gaussian_filter(gx * gx, 3.0, mode='nearest')
    jyy = gaussian_filter(gy * gy, 3.0, mode='nearest')
    jxy = gaussian_filter(gx * gy, 3.0, mode='nearest')
    tr = jxx + jyy + 1e-12
    coh = np.sqrt(np.maximum(((jxx - jyy) * 0.5) ** 2 + jxy ** 2, 0.0)) / tr
    coh[~footprint] = 0
    
    combined = (robust_norm(mag, footprint) * robust_norm(coh, footprint))
    grad_coh += combined
    del band, smoothed, gy, gx, mag, jxx, jyy, jxy, tr, coh, combined
    gc.collect()

grad_coh /= len([0, 1, 11, 12, 16])
grad_coh = robust_norm(grad_coh, footprint)
log(f"  Gradient coherence: {grad_coh[footprint].mean():.4f} mean")

# ============================================================================
# Surface 3: Conductivity-basement structural boundary
# ============================================================================
log("Surface 3: Conductivity-basement boundary...")

cond = features_mmap[16].copy()  # cond_surf
base_depth = features_mmap[14].copy()  # depth_to_base_surf

# Gradient of conductivity
cond_smooth = gaussian_filter(cond, 1.5, mode='nearest')
cond_gy = np.gradient(cond_smooth, axis=0)
cond_gx = np.gradient(cond_smooth, axis=1)
cond_grad = np.hypot(cond_gy, cond_gx)
cond_grad[~footprint] = 0

# Gradient of basement depth
base_smooth = gaussian_filter(base_depth, 1.5, mode='nearest')
base_gy = np.gradient(base_smooth, axis=0)
base_gx = np.gradient(base_smooth, axis=1)
base_grad = np.hypot(base_gy, base_gx)
base_grad[~footprint] = 0

# Combine: strong gradient in both = structural boundary
cond_boundary = robust_norm(cond_grad, footprint) * robust_norm(base_grad, footprint)
cond_boundary[~footprint] = 0
cond_boundary = robust_norm(cond_boundary, footprint)
log(f"  Cond-boundary: {cond_boundary[footprint].mean():.4f} mean")

del cond, base_depth, cond_smooth, cond_gy, cond_gx, cond_grad
del base_smooth, base_gy, base_gx, base_grad
gc.collect()

# ============================================================================
# Surface 4: Tilt/curvature edge detection (tc band is literally tilt angle)
# ============================================================================
log("Surface 4: Tilt curvature edges...")

tc = features_mmap[5].copy()  # tc = tilt angle/total curvature
# High tilt angle = strong edge = potential fault
tc_norm = robust_norm(np.abs(tc), footprint)

# Multi-scale: detect edges at different scales
tc_edges = np.zeros((H, W), dtype=np.float32)
for sigma in [0.5, 1.0, 2.0]:
    tc_smooth = gaussian_filter(np.abs(tc), sigma, mode='nearest')
    tc_edges += robust_norm(tc_smooth, footprint)
tc_edges /= 3.0
tc_edges[~footprint] = 0
log(f"  Tilt edges: {tc_edges[footprint].mean():.4f} mean")

del tc, tc_norm, tc_smooth, tc_edges
gc.collect()

# ============================================================================
# Surface 5: Topographic lineaments (detrended elevation ridges/valleys)
# ============================================================================
log("Surface 5: Topographic lineaments...")

det_elev = features_mmap[11].copy()  # detrended elevation

# Ridge response at multiple scales
topo_ridges = np.zeros((H, W), dtype=np.float32)
for sigma in [1.5, 3.0, 6.0]:
    r = line_response(det_elev, sigma, footprint)
    topo_ridges += robust_norm(r, footprint)
topo_ridges /= 3.0

# Valley response (negate elevation first)
topo_valleys = np.zeros((H, W), dtype=np.float32)
for sigma in [1.5, 3.0, 6.0]:
    r = line_response(-det_elev, sigma, footprint)
    topo_valleys += robust_norm(r, footprint)
topo_valleys /= 3.0

# Combine ridges and valleys (faults can be either)
topo_lineaments = np.maximum(topo_ridges, topo_valleys)
topo_lineaments[~footprint] = 0
log(f"  Topo lineaments: {topo_lineaments[footprint].mean():.4f} mean")

del det_elev, topo_ridges, topo_valleys
gc.collect()

# ============================================================================
# Surface 6: Gravity-magnetic correlation (both respond to density/magnetic boundaries)
# ============================================================================
log("Surface 6: Gravity-magnetic correlation...")

grav = features_mmap[12].copy()  # iso_grav_anom
mag = features_mmap[1].copy()   # rtp

grav_smooth = gaussian_filter(grav, 2.0, mode='nearest')
mag_smooth = gaussian_filter(mag, 2.0, mode='nearest')

# Gradient alignment: where both gradients point the same way = structural boundary
grav_gy = np.gradient(grav_smooth, axis=0)
grav_gx = np.gradient(grav_smooth, axis=1)
mag_gy = np.gradient(mag_smooth, axis=0)
mag_gx = np.gradient(mag_smooth, axis=1)

# Cosine similarity of gradient directions
grav_mag = np.hypot(grav_gy, grav_gx) + 1e-9
mag_mag = np.hypot(mag_gy, mag_gx) + 1e-9
cos_sim = (grav_gy * mag_gy + grav_gx * mag_gx) / (grav_mag * mag_mag)
cos_sim = np.clip(cos_sim, 0, 1)  # only positive alignment matters

# Weight by gradient magnitudes
gm_corr = cos_sim * robust_norm(grav_mag, footprint) * robust_norm(mag_mag, footprint)
gm_corr[~footprint] = 0
gm_corr = robust_norm(gm_corr, footprint)
log(f"  Grav-mag correlation: {gm_corr[footprint].mean():.4f} mean")

del grav, mag, grav_smooth, mag_smooth, grav_gy, grav_gx, mag_gy, mag_gx
del grav_mag, mag_mag, cos_sim
gc.collect()

# ============================================================================
# Surface 7: Slope anomaly (det_elev_slope high where fault scarp preserved)
# ============================================================================
log("Surface 7: Slope anomalies...")

slope = features_mmap[18].copy()  # det_elev_slope
slope_norm = robust_norm(np.abs(slope), footprint)

# Look for linear slope anomalies
for sigma in [1.0, 2.0]:
    slope_r = line_response(np.abs(slope), sigma, footprint)
    slope_norm += 0.5 * robust_norm(slope_r, footprint)
    del slope_r
    
slope_norm[~footprint] = 0
slope_norm = robust_norm(slope_norm, footprint)
log(f"  Slope anomalies: {slope_norm[footprint].mean():.4f} mean")

del slope
gc.collect()

# ============================================================================
# ENSEMBLE: Weighted combination of all surfaces
# ============================================================================
log("=== ENSEMBLE COMBINATION ===")

# The surfaces we have (recompute names for clarity):
# 1. ridge_fusion (multi-physics ridges)
# 2. grad_coh (gradient coherence)
# 3. cond_boundary (conductivity-basement boundary)
# 4. tc_edges (tilt curvature - recomputed below)
# 5. topo_lineaments (topographic lineaments)
# 6. gm_corr (gravity-magnetic correlation)
# 7. slope_norm (slope anomalies)

# Recompute tc_edges and slope_norm since we deleted them
tc = features_mmap[5].copy()
tc_edges = np.zeros((H, W), dtype=np.float32)
for sigma in [0.5, 1.0, 2.0]:
    tc_smooth = gaussian_filter(np.abs(tc), sigma, mode='nearest')
    tc_edges += robust_norm(tc_smooth, footprint)
tc_edges /= 3.0
tc_edges[~footprint] = 0
del tc, tc_smooth

slope = features_mmap[18].copy()
slope_norm = robust_norm(np.abs(slope), footprint)
for sigma in [1.0, 2.0]:
    slope_r = line_response(np.abs(slope), sigma, footprint)
    slope_norm += 0.5 * robust_norm(slope_r, footprint)
    del slope_r
slope_norm[~footprint] = 0
slope_norm = robust_norm(slope_norm, footprint)
del slope
gc.collect()

# Equal-weight ensemble
ensemble = (
    0.20 * ridge_fusion +
    0.15 * grad_coh +
    0.15 * cond_boundary +
    0.15 * tc_edges +
    0.15 * topo_lineaments +
    0.10 * gm_corr +
    0.10 * slope_norm
)
ensemble[~footprint] = 0
ensemble = robust_norm(ensemble, footprint)

log(f"Ensemble: mean={ensemble[footprint].mean():.4f}, max={ensemble[footprint].max():.4f}")

# ============================================================================
# AUTO-CONTEXT REFINEMENT (Stage 2)
# ============================================================================
log("=== AUTO-CONTEXT REFINEMENT ===")

# Stage 2: refine the ensemble using spatial context
# - Smooths of the ensemble at multiple scales
# - Offset samples
# - Orientation coherence of the ensemble

# Multi-scale smooths
ctx = [ensemble.copy()]
for sigma in [1.0, 2.0, 4.0, 8.0]:
    sm = gaussian_filter(ensemble, sigma, mode='nearest')
    sm[~footprint] = 0
    ctx.append(sm)

# Gradient of ensemble
gy_e = np.gradient(ensemble, axis=0)
gx_e = np.gradient(ensemble, axis=1)
grad_e = np.hypot(gy_e, gx_e).astype(np.float32)
grad_e[~footprint] = 0
ctx.append(grad_e)
del gy_e, gx_e

# Laplacian (ridge detector on the ensemble itself)
lap_e = np.gradient(np.gradient(ensemble, axis=0), axis=0) + np.gradient(np.gradient(ensemble, axis=1), axis=1)
lap_e = np.abs(lap_e).astype(np.float32)
lap_e[~footprint] = 0
ctx.append(lap_e)
del lap_e

# Offset samples at multiple distances
for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
    for off in [1, 3, 5, 10]:
        shifted = np.roll(np.roll(ensemble, -dy * off, axis=0), -dx * off, axis=1)
        shifted[~footprint] = 0
        ctx.append(shifted)

# Max of offsets (measures local ridge strength)
max_offset = np.zeros((H, W), dtype=np.float32)
for c in ctx[5:]:  # skip the first 5 (smooths)
    max_offset = np.maximum(max_offset, c)
ctx.append(max_offset)

# Mean of offsets (measures continuity)
mean_offset = np.zeros((H, W), dtype=np.float32)
n_offsets = 0
for c in ctx[5:-1]:
    mean_offset += c
    n_offsets += 1
mean_offset /= max(n_offsets, 1)
ctx.append(mean_offset)

# Orientation coherence of the ensemble
gx2 = gaussian_filter(ensemble, 2.0, order=(0, 1), mode='nearest')
gy2 = gaussian_filter(ensemble, 2.0, order=(1, 0), mode='nearest')
jxx2 = gaussian_filter(gx2 * gx2, 3.0, mode='nearest')
jyy2 = gaussian_filter(gy2 * gy2, 3.0, mode='nearest')
jxy2 = gaussian_filter(gx2 * gy2, 3.0, mode='nearest')
tr2 = jxx2 + jyy2 + 1e-12
coh2 = np.sqrt(np.maximum(((jxx2 - jyy2) * 0.5) ** 2 + jxy2 ** 2, 0.0)) / tr2
coh2[~footprint] = 0
ctx.append(coh2.astype(np.float32))
del gx2, gy2, jxx2, jyy2, jxy2, tr2

# Refined surface: weighted combination of context features
# Weight the original ensemble highest, context features lower
refined = (
    0.40 * ctx[0] +      # original ensemble
    0.10 * ctx[1] +      # smooth sigma=1
    0.10 * ctx[2] +      # smooth sigma=2
    0.05 * ctx[5] +      # gradient
    0.05 * ctx[6] +      # laplacian
    0.10 * max_offset +  # max offset (ridge strength)
    0.10 * mean_offset + # mean offset (continuity)
    0.10 * coh2          # orientation coherence
)
refined[~footprint] = 0
refined = robust_norm(refined, footprint)

log(f"Refined surface: mean={refined[footprint].mean():.4f}, max={refined[footprint].max():.4f}")

# ============================================================================
# SAVE for emission
# ============================================================================
np.save('data/physics_surface.npy', ensemble)
np.save('data/refined_surface.npy', refined)
log("=== SURFACES SAVED ===")
