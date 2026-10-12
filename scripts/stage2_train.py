#!/usr/bin/env python3
"""
Auto-context Stacked Refinement — Stage 2
==========================================
Stage 2 classifier on Stage 1 probabilities + spatial context features:
  - Stage-1 probs at ±1-15 px offsets along/across local orientation
  - Multi-scale anisotropic smooths
  - Raw gradient/tilt/scarp channels
  - Catalogue-context features (distance/orientation to nearest fault)
"""
import numpy as np
import rasterio
import time
import gc
import json
from scipy.ndimage import gaussian_filter, distance_transform_edt
import lightgbm as lgb

t0 = time.time()

def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

log("=== STAGE 2: Auto-context refinement ===")

# Load Stage 1 outputs
oof_probs = np.load('data/oof_probs.npy')
footprint = np.load('data/footprint.npy')
labels = np.load('data/labels.npy')
block_ids = np.load('data/block_ids.npy')

with open('data/feature_info.json') as f:
    info = json.load(f)

H, W = info['H'], info['W']
n_feat_s1 = info['n_feat']
log(f"Stage 1 OOF loaded, footprint: {footprint.sum():,}")

# Load feature mmap for raw channels
features_mmap = np.load('data/features_mmap.npy', mmap_mode='r')
log(f"Feature mmap loaded: {features_mmap.shape}")

# ============================================================================
# Stage 2 Feature Engineering
# ============================================================================
log("Building Stage 2 features...")

# Feature 0: Stage 1 OOF probability
s2_feats = [oof_probs.copy()]
s2_names = ['s1_oof_prob']

# Feature 1: Stage 1 OOF prob, smoothed at multiple scales
for sigma in [1.0, 2.0, 4.0, 8.0, 16.0]:
    sm = gaussian_filter(oof_probs, sigma, mode='nearest')
    sm[~footprint] = 0
    s2_feats.append(sm.astype(np.float32))
    s2_names.append(f's1_smooth_s{sigma}')

log(f"  Smooths: {len(s2_feats)}")

# Feature 6: Gradient of Stage 1 probability map
grad_y = np.gradient(oof_probs, axis=0)
grad_x = np.gradient(oof_probs, axis=1)
grad_mag = np.hypot(grad_y, grad_x).astype(np.float32)
grad_mag[~footprint] = 0
s2_feats.append(grad_mag)
s2_names.append('s1_grad_mag')
del grad_y, grad_x

# Feature 7: Laplacian of Stage 1 probability (detects ridges)
lap = np.gradient(np.gradient(oof_probs, axis=0), axis=0) + np.gradient(np.gradient(oof_probs, axis=1), axis=1)
lap = np.abs(lap).astype(np.float32)
lap[~footprint] = 0
s2_feats.append(lap)
s2_names.append('s1_laplacian')
del lap

# Feature 8: Local orientation coherence of Stage 1 probs
# Structure tensor of s1 probability
gx = gaussian_filter(oof_probs, 2.0, order=(0, 1), mode='nearest')
gy = gaussian_filter(oof_probs, 2.0, order=(1, 0), mode='nearest')
jxx = gaussian_filter(gx * gx, 3.0, mode='nearest')
jyy = gaussian_filter(gy * gy, 3.0, mode='nearest')
jxy = gaussian_filter(gx * gy, 3.0, mode='nearest')
tr = jxx + jyy + 1e-12
coherence = np.sqrt(np.maximum(((jxx - jyy) * 0.5) ** 2 + jxy ** 2, 0.0)) / tr
coherence[~footprint] = 0
s2_feats.append(coherence.astype(np.float32))
s2_names.append('s1_orientation_coherence')
del gx, gy, jxx, jyy, jxy, tr

# Features 9-13: Offset samples along 4 directions (horizontal, vertical, 2 diagonals)
# At ±1, ±3, ±7, ±15 pixels
directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
offsets = [1, 3, 7, 15]
for dy, dx in directions:
    for off in offsets:
        shifted = np.roll(np.roll(oof_probs, -dy * off, axis=0), -dx * off, axis=1)
        shifted[~footprint] = 0
        s2_feats.append(shifted.astype(np.float32))
        s2_names.append(f's1_offset_d{dy}_{dx}_o{off}')

log(f"  Offset samples: {len(s2_feats)}")

# Distance to nearest catalogue fault
with rasterio.open('data/existing_faults.tif') as src:
    labels_raw = src.read(1)
labels_bool = labels_raw >= 1

# Distance to fault (already in features_mmap band 50, but let's also add proximity bands)
dist_fault = distance_transform_edt(~labels_bool).astype(np.float32)
dist_fault[~footprint] = 0

# Proximity features: distance in bins
for threshold in [1, 2, 3, 5, 10, 20, 30]:
    prox = np.clip(1.0 - dist_fault / threshold, 0, 1).astype(np.float32)
    prox[~footprint] = 0
    s2_feats.append(prox)
    s2_names.append(f'fault_prox_t{threshold}')

del dist_fault
gc.collect()

# Anisotropic smooths along detected orientation
# Use the structure tensor orientation to smooth along-strike vs cross-strike
theta = 0.5 * np.arctan2(2 * jxy, (jxx - jyy) + 1e-12) if 'jxy' in dir() else np.zeros_like(oof_probs)
# Recompute since we deleted above
gx2 = gaussian_filter(oof_probs, 2.0, order=(0, 1), mode='nearest')
gy2 = gaussian_filter(oof_probs, 2.0, order=(1, 0), mode='nearest')
jxx2 = gaussian_filter(gx2 * gx2, 3.0, mode='nearest')
jyy2 = gaussian_filter(gy2 * gy2, 3.0, mode='nearest')
jxy2 = gaussian_filter(gx2 * gy2, 3.0, mode='nearest')
theta = 0.5 * np.arctan2(2 * jxy2, (jxx2 - jyy2) + 1e-12)
del gx2, gy2, jxx2, jyy2, jxy2

# Along-strike smooth (anisotropic: elongated along local orientation)
for sigma_along in [3.0, 6.0, 12.0]:
    # Smooth along the mean orientation (approximate with fixed angles)
    for angle_deg in [0, 45, 90, 135]:
        angle = np.radians(angle_deg)
        # Create anisotropic kernel: elongated along angle
        # Approximate by smoothing more in one direction
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        # Rotate, anisotropic smooth, rotate back
        # For efficiency: just smooth differently in y vs x
        if abs(cos_a) > abs(sin_a):
            s_along = gaussian_filter(oof_probs, sigma=[sigma_along * abs(sin_a) + 0.5, sigma_along * abs(cos_a) + 0.5], mode='nearest')
        else:
            s_along = gaussian_filter(oof_probs, sigma=[sigma_along * abs(cos_a) + 0.5, sigma_along * abs(sin_a) + 0.5], mode='nearest')
        s_along[~footprint] = 0
        s2_feats.append(s_along.astype(np.float32))
        s2_names.append(f'aniso_s{sigma_along}_a{angle_deg}')
        del s_along

gc.collect()
log(f"  Anisotropic smooths: {len(s2_feats)}")

# Raw feature channels that are most relevant (gradient/tilt/scarp proxies)
# Band 5: tc (tilt angle), Band 18: det_elev_slope, Band 0: mag_anom gradient
for bi in [0, 5, 11, 16, 18]:  # mag_anom, tc, det_elev, cond_surf, det_elev_slope
    feat = features_mmap[bi].copy()
    feat[~footprint] = 0
    s2_feats.append(feat.astype(np.float32))
    s2_names.append(f'raw_band_{bi}')

# Gradient of key raw bands
for bi in [0, 5, 18]:  # mag_anom, tc, det_elev_slope gradients
    feat = features_mmap[bi + n_feat_s1 - 35 + 16].copy() if (bi + n_feat_s1 - 35 + 16) < features_mmap.shape[0] else np.zeros((H, W), dtype=np.float32)
    # Use pre-computed gradient features from mmap
    # gradient features start at band 19 in mmap
    grad_idx = 19 + [0, 1, 5, 11, 12, 13, 16, 18].index(bi) * 2  # first scale
    if grad_idx < features_mmap.shape[0]:
        feat = features_mmap[grad_idx].copy()
    feat[~footprint] = 0
    s2_feats.append(feat.astype(np.float32))
    s2_names.append(f'raw_grad_band_{bi}')

gc.collect()
log(f"  Total S2 features: {len(s2_feats)}")

# Stack all S2 features
n_s2 = len(s2_feats)
# Save as memory-mapped to avoid memory spike
s2_mmap = np.lib.format.open_memmap(
    'data/s2_features.npy', mode='w+', dtype=np.float32,
    shape=(n_s2, H, W)
)
for i, feat in enumerate(s2_feats):
    s2_mmap[i] = feat
    if i % 10 == 0:
        gc.collect()
s2_mmap.flush()
del s2_feats
gc.collect()

log(f"Stage 2 feature stack: {n_s2} channels saved to mmap")

# ============================================================================
# Train Stage 2 with spatial blocked CV
# ============================================================================
log("=== Training Stage 2 (4-fold spatial CV on S2 features) ===")

oof_s2 = np.zeros((H, W), dtype=np.float32)
rng = np.random.RandomState(123)

for fold in range(4):
    log(f"  Fold {fold}...")
    val_mask = (block_ids == fold) & footprint
    train_mask = (block_ids != fold) & footprint
    
    # Get positive/negative indices for training
    tr_pos = np.argwhere(train_mask & labels_bool)
    tr_neg = np.argwhere(train_mask & ~labels_bool)
    
    # Subsample negatives (20:1 ratio for Stage 2 - more negative examples)
    n_neg_sample = min(len(tr_pos) * 20, len(tr_neg))
    neg_idx = rng.choice(len(tr_neg), size=n_neg_sample, replace=False)
    
    # Build training arrays
    pos_coords = tr_pos
    neg_coords = tr_neg[neg_idx]
    train_coords = np.vstack([pos_coords, neg_coords])
    train_labels = np.concatenate([np.ones(len(pos_coords)), np.zeros(len(neg_coords))]).astype(np.float32)
    
    # Extract S2 features
    X_train = np.zeros((len(train_coords), n_s2), dtype=np.float32)
    for b in range(n_s2):
        X_train[:, b] = s2_mmap[b][train_coords[:, 0], train_coords[:, 1]]
    
    # Validation
    val_coords = np.argwhere(val_mask)
    X_val = np.zeros((len(val_coords), n_s2), dtype=np.float32)
    for b in range(n_s2):
        X_val[:, b] = s2_mmap[b][val_coords[:, 0], val_coords[:, 1]]
    
    val_labels = labels_bool[val_coords[:, 0], val_coords[:, 1]].astype(np.float32)
    
    # Train LightGBM Stage 2
    dtrain = lgb.Dataset(X_train, label=train_labels)
    dval = lgb.Dataset(X_val, label=val_labels, reference=dtrain)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 7,
        'min_child_samples': 50,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 100 + fold,
    }
    
    model = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )
    
    # OOF predictions
    oof_pred = model.predict(X_val).astype(np.float32)
    oof_s2[val_coords[:, 0], val_coords[:, 1]] = oof_pred
    
    log(f"    {model.num_trees()} trees, val_binary_logloss: min(oof_s2[footprint])={oof_s2[footprint].min():.4f}, max={oof_s2[footprint].max():.4f}")
    
    del X_train, X_val, train_coords, val_coords
    gc.collect()

log(f"Stage 2 OOF: min={oof_s2[footprint].min():.4f}, max={oof_s2[footprint].max():.4f}")

# Save Stage 2 results
np.save('data/oof_s2.npy', oof_s2)
log("=== Stage 2 complete ===")
