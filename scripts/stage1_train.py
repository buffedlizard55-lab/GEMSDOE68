#!/usr/bin/env python3
"""
Auto-context Stacked Refinement — Stage 1
==========================================
Memory-efficient: loads bands one at a time, builds features incrementally,
trains LightGBM with spatial blocking.
"""
import numpy as np
import rasterio
import time
import gc
from scipy.ndimage import gaussian_filter, distance_transform_edt
import lightgbm as lgb
import json
import os

t0 = time.time()
H, W = 3730, 3292
N_PIXELS = H * W

def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

log("=== STAGE 1: Memory-efficient feature loading ===")

# Step 1: Load features band-by-band, compute gradient/ridge features on-the-fly
# Keep only: original 19 bands (float16 to save memory) + key engineered features
features_path = 'data/gems-geodawn-numerical-features.tif'

# First pass: get footprint and band medians
with rasterio.open(features_path) as src:
    b1 = src.read(1)
    nodata_val = src.nodata
    
footprint = np.isfinite(b1) & (b1 > nodata_val * 0.5)
del b1
gc.collect()
log(f"Footprint: {footprint.sum():,} valid pixels")

# Second pass: read all 19 bands, standardize, keep as float32 but in a memory-mapped file
# Use a memory-mapped array to avoid holding everything in RAM
mmap_path = 'data/features_mmap.npy'
n_original_bands = 19

# Pre-compute how many extra features we'll add
# Gradient of 8 key bands x 2 scales = 16
# Smooth of 5 key bands x 3 scales = 15  
# Distance to fault = 1
# Tilt/slope derivatives = 3 (already in original)
# Total extra: ~35
n_extra = 35
n_total_bands = n_original_bands + n_extra

# Create memory-mapped array
features_mmap = np.lib.format.open_memmap(
    mmap_path, mode='w+', dtype=np.float32,
    shape=(n_total_bands, H, W)
)

# Load original bands
log("Loading original bands...")
with rasterio.open(features_path) as src:
    for b in range(1, src.count + 1):
        band = src.read(b).astype(np.float32)
        # Fill nodata with 0
        invalid = ~footprint | ~np.isfinite(band) | (band < nodata_val * 0.5)
        band[invalid] = 0.0
        features_mmap[b-1] = band
        del band

gc.collect()
log(f"Original bands loaded: {n_original_bands}")

# Standardize each original band in-place
log("Standardizing original bands...")
for b in range(n_original_bands):
    vals = features_mmap[b][footprint]
    mu = vals.mean()
    sd = vals.std() + 1e-9
    features_mmap[b] = np.where(footprint, (features_mmap[b] - mu) / sd, 0.0).astype(np.float32)

gc.collect()

# Step 3: Compute engineered features
log("Computing engineered features...")
bi = n_original_bands

# Band indices for key features
# 0:mag_anom, 1:rtp, 5:tc, 11:det_elev, 12:iso_grav, 13:tmi, 16:cond_surf, 18:det_elev_slope
key_bands = [0, 1, 5, 11, 12, 13, 16, 18]

# Gradient features: key_bands x 2 scales
for band_idx in key_bands:
    for sigma in [1.0, 2.0]:
        if bi >= n_total_bands:
            break
        smoothed = gaussian_filter(features_mmap[band_idx], sigma, mode='nearest')
        gy = np.gradient(smoothed, axis=0)
        gx = np.gradient(smoothed, axis=1)
        grad = np.hypot(gy, gx).astype(np.float32)
        grad[~footprint] = 0
        # Standardize
        vals = grad[footprint]
        mu, sd = vals.mean(), vals.std() + 1e-9
        features_mmap[bi] = np.where(footprint, (grad - mu) / sd, 0.0).astype(np.float32)
        bi += 1
        del smoothed, gy, gx, grad
        gc.collect()

log(f"  Gradient features: {bi - n_original_bands}")

# Smooth features: 5 key bands x 3 scales
for band_idx in [0, 1, 5, 12, 16]:
    for sigma in [2.0, 4.0, 8.0]:
        if bi >= n_total_bands:
            break
        s = gaussian_filter(features_mmap[band_idx], sigma, mode='nearest')
        s[~footprint] = 0
        vals = s[footprint]
        mu, sd = vals.mean(), vals.std() + 1e-9
        features_mmap[bi] = np.where(footprint, (s - mu) / sd, 0.0).astype(np.float32)
        bi += 1
        del s
        gc.collect()

log(f"  Smooth features: {bi - n_original_bands - 16}")

# Distance to fault
log("Computing distance to fault...")
with rasterio.open('data/existing_faults.tif') as src:
    labels_raw = src.read(1)
labels = labels_raw >= 1
fault_dists = distance_transform_edt(~labels).astype(np.float32)
fault_dists[~footprint] = 0
if bi < n_total_bands:
    vals = fault_dists[footprint]
    mu, sd = vals.mean(), vals.std() + 1e-9
    features_mmap[bi] = np.where(footprint, (fault_dists - mu) / sd, 0.0).astype(np.float32)
    bi += 1
del fault_dists
gc.collect()

# Tilt of magnetic (already computed as band 5:tc, but let's add the tilt angle of det_elev)
if bi < n_total_bands:
    elev_smooth = gaussian_filter(features_mmap[11], 1.0, mode='nearest')
    gy = np.gradient(elev_smooth, axis=0)
    gx = np.gradient(elev_smooth, axis=1)
    tilt = np.arctan2(np.hypot(gy, gx), 1.0).astype(np.float32)
    tilt[~footprint] = 0
    vals = tilt[footprint]
    mu, sd = vals.mean(), vals.std() + 1e-9
    features_mmap[bi] = np.where(footprint, (tilt - mu) / sd, 0.0).astype(np.float32)
    bi += 1
    del elev_smooth, gy, gx, tilt
    gc.collect()

# Fill remaining slots with zeros
for j in range(bi, n_total_bands):
    features_mmap[j] = 0

n_feat = bi  # actual number of features used
log(f"Total features computed: {n_feat}")

# Flush mmap
features_mmap.flush()

# Step 4: Load labels
log("Loading labels...")
y = labels.astype(np.float32)  # target
log(f"Positive pixels (catalogue faults): {labels.sum():,}")

# Step 5: Spatial blocking (4 quadrants)
block_ids = np.zeros((H, W), dtype=np.int32)
block_ids[:H//2, :W//2] = 0
block_ids[:H//2, W//2:] = 1
block_ids[H//2:, :W//2] = 2
block_ids[H//2:, W//2:] = 3

# Step 6: Train Stage 1 with spatial blocked CV
log("=== Training Stage 1 (4-fold spatial CV) ===")

oof_probs = np.zeros((H, W), dtype=np.float32)
models_s1 = []

rng = np.random.RandomState(42)

for fold in range(4):
    log(f"  Fold {fold}...")
    val_mask = (block_ids == fold) & footprint
    train_mask = (block_ids != fold) & footprint
    
    # Get positive/negative indices for training
    tr_pos = np.argwhere(train_mask & labels)
    tr_neg = np.argwhere(train_mask & ~labels)
    
    # Subsample negatives (10:1 ratio)
    n_neg_sample = min(len(tr_pos) * 10, len(tr_neg))
    neg_idx = rng.choice(len(tr_neg), size=n_neg_sample, replace=False)
    
    # Build training arrays
    pos_coords = tr_pos
    neg_coords = tr_neg[neg_idx]
    train_coords = np.vstack([pos_coords, neg_coords])
    train_labels = np.concatenate([np.ones(len(pos_coords)), np.zeros(len(neg_coords))]).astype(np.float32)
    
    # Extract features for training pixels
    X_train = np.zeros((len(train_coords), n_feat), dtype=np.float32)
    for b in range(n_feat):
        X_train[:, b] = features_mmap[b][train_coords[:, 0], train_coords[:, 1]]
    
    # Validation pixels (all footprint in this block)
    val_coords = np.argwhere(val_mask)
    X_val = np.zeros((len(val_coords), n_feat), dtype=np.float32)
    for b in range(n_feat):
        X_val[:, b] = features_mmap[b][val_coords[:, 0], val_coords[:, 1]]
    
    val_labels = labels[val_coords[:, 0], val_coords[:, 1]].astype(np.float32)
    
    # Train LightGBM
    dtrain = lgb.Dataset(X_train, label=train_labels)
    dval = lgb.Dataset(X_val, label=val_labels, reference=dtrain)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 100,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42 + fold,
    }
    
    model = lgb.train(
        params, dtrain,
        num_boost_round=300,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
    )
    
    models_s1.append(model)
    
    # OOF predictions
    oof_pred = model.predict(X_val).astype(np.float32)
    oof_probs[val_coords[:, 0], val_coords[:, 1]] = oof_pred
    
    log(f"    {model.num_trees()} trees, train={len(train_coords):,}, val={len(val_coords):,}")
    
    del X_train, X_val, train_coords, val_coords
    gc.collect()

log(f"Stage 1 OOF: min={oof_probs[footprint].min():.4f}, max={oof_probs[footprint].max():.4f}")

# Save stage 1 results
np.save('data/oof_probs.npy', oof_probs)
np.save('data/footprint.npy', footprint)
np.save('data/labels.npy', labels)
np.save('data/block_ids.npy', block_ids)
log("Stage 1 results saved")

# Also save feature info for stage 2
with open('data/feature_info.json', 'w') as f:
    json.dump({
        'n_feat': int(n_feat),
        'n_original': int(n_original_bands),
        'H': H, 'W': W,
        'footprint_px': int(footprint.sum()),
        'label_px': int(labels.sum()),
    }, f)

log("=== Stage 1 complete ===")
