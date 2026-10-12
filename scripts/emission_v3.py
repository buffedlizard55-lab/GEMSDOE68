#!/usr/bin/env python3
"""
Emission v3: Efficient dot placement with proper spacing.
Uses a spatial hash grid for O(1) neighbor checking.
"""
import numpy as np
import rasterio
import time
import gc
import json
import hashlib
import os
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

t0 = time.time()

def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

log("=== EMISSION V3 ===")

# Load
refined = np.load('data/refined_surface.npy')
footprint = np.load('data/footprint.npy')
H, W = 3730, 3292

with rasterio.open('data/existing_faults.tif') as src:
    labels_raw = src.read(1)
labels_bool = labels_raw >= 1

# ============================================================================
# DOT PLACEMENT using spatial hash grid
# ============================================================================
log("=== DOT PLACEMENT (spatial hash) ===")

# Get candidates sorted by probability
candidates_mask = footprint & ~labels_bool & (refined > 0)
cand_ys, cand_xs = np.where(candidates_mask)
cand_probs = refined[cand_ys, cand_xs]

order = np.argsort(-cand_probs)
cand_ys = cand_ys[order]
cand_xs = cand_xs[order]

log(f"  Candidates: {len(cand_ys):,}")

# Spatial hash grid approach
# Cell size = min_dist, so we only need to check 3x3 neighborhood
min_dist = 2.83
cell_size = min_dist
grid_h = int(np.ceil(H / cell_size)) + 2
grid_w = int(np.ceil(W / cell_size)) + 2

# For each cell, store the coordinates of kept points
# Use a dict: (cy, cx) -> list of (y, x)
grid = {}
target_n = 38000

kept_y = []
kept_x = []
search_r = 1  # search radius in cells (we need to check 3x3)

def cell_of(y, x):
    return int(y / cell_size), int(x / cell_size)

def is_too_close(y, x, cy, cx):
    """Check if (y,x) is within min_dist of any kept point in nearby cells."""
    for dy in range(-search_r - 1, search_r + 2):
        for dx in range(-search_r - 1, search_r + 2):
            key = (cy + dy, cx + dx)
            if key in grid:
                for py, px in grid[key]:
                    if (y - py)**2 + (x - px)**2 < min_dist**2:
                        return True
    return False

n = 0
for i in range(len(cand_ys)):
    y, x = int(cand_ys[i]), int(cand_xs[i])
    cy, cx = cell_of(y, x)
    
    if not is_too_close(y, x, cy, cx):
        kept_y.append(y)
        kept_x.append(x)
        key = (cy, cx)
        if key not in grid:
            grid[key] = []
        grid[key].append((y, x))
        n += 1
        
        if n >= target_n:
            break
    
    if (i + 1) % 100000 == 0:
        log(f"    Processed {i+1:,}, kept {n:,}")

kept_arr = np.array(list(zip(kept_y, kept_x)))
n_dots = len(kept_arr)
log(f"  Kept: {n_dots} dots")

# Spacing stats
if n_dots > 1:
    tree = cKDTree(kept_arr)
    nn_dists, _ = tree.query(kept_arr, k=2)
    nn = nn_dists[:, 1]
    log(f"  Spacing: min={nn.min():.2f}, median={np.median(nn):.2f}, 5th={np.percentile(nn, 5):.2f}, max={nn.max():.2f}")

# ============================================================================
# FLANK PRUNE
# ============================================================================
log("=== FLANK PRUNE ===")

dist_to_cat = distance_transform_edt(~labels_bool)
flank_buffer = 2
dists_at_dots = dist_to_cat[kept_arr[:, 0], kept_arr[:, 1]]
near_cat = dists_at_dots <= flank_buffer
n_pruned = near_cat.sum()
kept_arr = kept_arr[~near_cat]
log(f"  Pruned {n_pruned} dots within {flank_buffer} px of catalogue")
log(f"  Final: {len(kept_arr):,} dots")

# ============================================================================
# BUILD & WRITE
# ============================================================================
log("=== BUILD & WRITE ===")

submission = np.zeros((H, W), dtype=np.float32)
submission[kept_arr[:, 0], kept_arr[:, 1]] = 1.0
submission[~footprint] = 0.0

assert np.isfinite(submission).all()
assert (submission >= 0).all() and (submission <= 1).all()

n_pos = int((submission > 0).sum())

# Uniqueness
log("  Uniqueness check...")
registry_paths = [
    'gemsdoe32_ref/docs/downloads/gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif',
    'gemsdoe32_ref/docs/downloads/gems32-probe-S1-ANCHOR-identical-to-live-02600.tif',
]
our_dots = set(zip(kept_arr[:, 0].tolist(), kept_arr[:, 1].tolist()))
max_overlap = 0.0
for rpath in registry_paths:
    if os.path.exists(rpath):
        with rasterio.open(rpath) as src:
            reg = src.read(1)
        reg_dots = set(zip(*np.where(reg > 0)))
        overlap = len(our_dots & reg_dots)
        frac = overlap / max(len(our_dots), 1)
        log(f"    vs {os.path.basename(rpath)}: {overlap} ({frac:.1%})")
        max_overlap = max(max_overlap, frac)

uniqueness = "PASS" if max_overlap <= 0.70 else ("CAUTION" if max_overlap <= 0.90 else "FAIL")

# Write
import time as time_mod
timestamp = time_mod.strftime("%Y%m%dT%H%M%SZ", time_mod.gmtime())
os.makedirs('docs/downloads', exist_ok=True)

submission_name = f"gemsdoe68-autocontext-s2-{timestamp}"
out_filename = f"{submission_name}-zeros.tif"
out_path = os.path.join('docs/downloads', out_filename)

with rasterio.open(
    out_path, 'w',
    driver='GTiff', dtype='float32', count=1,
    crs='EPSG:32611',
    transform=rasterio.transform.from_origin(243350.0, 4508550.0, 100, 100),
    width=3292, height=3730,
    nodata=None,
    compress='deflate', tiled=True, blockxsize=256, blockysize=256,
) as dst:
    dst.write(submission, 1)

sha = hashlib.sha256(open(out_path, 'rb').read()).hexdigest()
file_size = os.path.getsize(out_path)

# NaN variant
nan_filename = f"{submission_name}-nan.tif"
nan_path = os.path.join('docs/downloads', nan_filename)
submission_nan = submission.copy()
submission_nan[~footprint] = np.nan
with rasterio.open(
    nan_path, 'w',
    driver='GTiff', dtype='float32', count=1,
    crs='EPSG:32611',
    transform=rasterio.transform.from_origin(243350.0, 4508550.0, 100, 100),
    width=3292, height=3730,
    nodata=np.nan,
    compress='deflate', tiled=True, blockxsize=256, blockysize=256,
) as dst:
    dst.write(submission_nan, 1)
sha_nan = hashlib.sha256(open(nan_path, 'rb').read()).hexdigest()

# Format audit
with rasterio.open(out_path) as src:
    check = src.read(1)
    m_crs = str(src.crs)
    m_h, m_w = src.height, src.width
    m_dtype = src.dtypes[0]

checks = {
    'all_finite': bool(np.isfinite(check).all()),
    'all_in_01': bool((check >= 0).all() and (check <= 1).all()),
    'no_nan_in_footprint': bool(np.isfinite(check[footprint]).all()),
    'zero_outside_footprint': bool((check[~footprint] == 0).all()),
    'crs_correct': m_crs == 'EPSG:32611',
    'shape_correct': m_h == 3730 and m_w == 3292,
    'dtype_float32': m_dtype == 'float32',
}
all_pass = all(checks.values())

# Run card
run_card = {
    "hypothesis": "Auto-context stacked refinement: 7 physics surfaces + spatial auto-context",
    "mechanism": "Multi-physics ridge detection + gradient coherence + conductivity boundaries + tilt curvature + topo lineaments + grav-mag correlation + slope anomalies → ensemble → auto-context spatial refinement",
    "non_fault_mimic": "Stratigraphic contacts in tilted sedimentary sequences",
    "n_dots": n_pos,
    "n_pruned": int(n_pruned),
    "uniqueness": uniqueness,
    "max_overlap": round(max_overlap, 4),
    "sha256": sha,
    "sha256_nan": sha_nan,
    "bytes": file_size,
    "format_pass": all_pass,
    "verdict": "promote" if (all_pass and uniqueness == "PASS") else "negative",
    "submission_name": submission_name,
    "portal_name": "GEMSDOE68-autocontext-s2",
    "portal_note": f"Auto-context physics {n_pos}px sha256:{sha[:16]} UNSCORED",
}

with open('data/run_card.json', 'w') as f:
    json.dump(run_card, f, indent=2)

# Save final filenames for the site
with open('data/submission_info.json', 'w') as f:
    json.dump({
        'out_filename': out_filename,
        'nan_filename': nan_filename,
        'out_path': out_path,
        'nan_path': nan_path,
        'sha256': sha,
        'sha256_nan': sha_nan,
        'n_dots': n_pos,
        'submission_name': submission_name,
    }, f, indent=2)

log(f"\n{'='*60}")
log(f"Written: {out_path}")
log(f"SHA-256: {sha}")
log(f"Size: {file_size:,} bytes")
log(f"Dots: {n_pos}")
log(f"Format: {'PASS' if all_pass else 'FAIL'}")
for k, v in checks.items():
    log(f"  {k}: {'✓' if v else '✗'}")
log(f"Uniqueness: {uniqueness} ({max_overlap:.1%})")
log(f"VERDICT: {run_card['verdict']}")
log(f"{'='*60}")
