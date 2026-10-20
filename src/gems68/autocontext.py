"""Auto-context stacked refinement (Tu, CVPR 2008, doi:10.1109/CVPR.2008.4587436;
TPAMI 2010, doi:10.1109/TPAMI.2009.186).

Stage-2 and Stage-3 feature construction on the footprint:

* ``p_self`` -- Stage-(k-1) out-of-fold probability at x
* ``p_along+d``, ``p_along-d`` for d in (1, 2, 3, 5, 8, 12, 15) px along local strike
* ``p_across+d``, ``p_across-d`` for d in (1, 2, 3, 5, 8, 12, 15) px across local strike
* multi-scale smooths of ``p_{k-1}``: isotropic sigma in (1.5, 3, 5) px (dropping
  the blob-inducing sigma=10 px smooth per IR-68-BLOB-01) and anisotropic
  (L_along, W_across) in ((3, 1), (6, 1.5), (12, 2), (15, 3)) px
* 1D curvilinear ridge-vs-blob contrast features (IR-68-BLOB-01 fix):
  ``p_aniso_contrast_6``, ``p_aniso_contrast_12``, ``p_ridge_convexity_3``,
  ``p_along_bridge``, ``p_echelon_side``
* raw gradient / tilt / 2 m LiDAR scarp / GeoDAWN extension channels from ``features.py``
* catalogue-context features computed with the target block's labels masked

Every probability fed into Stage 2 / Stage 3 is strictly out-of-fold.
"""
from __future__ import annotations

from dataclasses import dataclass as _dataclass

import numpy as np
from scipy import ndimage as ndi

from .features import CAT_CONTEXT_NAMES, GRAD_BAND_NAMES, PhysicalFeatures

OFFSETS_PX = (1, 2, 3, 5, 8, 12, 15)
ISO_SIGMAS = (1.5, 3.0, 5.0)
ANISO_SCALES = ((3, 1.0), (6, 1.5), (12, 2.0), (15, 3.0))

RIDGE_CONTRAST_NAMES = [
    "p_aniso_contrast_6",
    "p_aniso_contrast_12",
    "p_ridge_convexity_3",
    "p_along_bridge",
    "p_echelon_side",
]

LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    boosting_type="gbdt",
    num_leaves=63,
    max_depth=7,
    learning_rate=0.07,
    min_data_in_leaf=200,
    feature_fraction=0.75,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    verbose=-1,
    n_jobs=2,
    seed=2026068,
)
N_ROUNDS = 180
NEG_PER_POS = 8


def stage2_feature_names() -> list[str]:
    names = ["p_self"]
    for d in OFFSETS_PX:
        names.extend([f"p_along+{d}", f"p_along-{d}", f"p_across+{d}", f"p_across-{d}"])
    for s in ISO_SIGMAS:
        names.append(f"p_sm{str(s).replace('.0', '').replace('.', '_')}")
    for la, wc in ANISO_SCALES:
        names.append(f"p_aniso_L{la}_W{str(wc).replace('.0', '').replace('.', '_')}")
    names.extend(RIDGE_CONTRAST_NAMES)
    names.extend(GRAD_BAND_NAMES)
    names.extend(CAT_CONTEXT_NAMES)
    return names


def precompute_iso_smooths(pmap: np.ndarray) -> dict[float, np.ndarray]:
    """Precompute isotropic Gaussian smooths of ``pmap`` once for all sigmas needed."""
    sigmas = sorted({*ISO_SIGMAS, *(wc for _, wc in ANISO_SCALES)})
    return {
        float(s): ndi.gaussian_filter(pmap, sigma=float(s), mode="constant", cval=0.0).astype(np.float32)
        for s in sigmas
    }


def _sample_offsets(
    pmap: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    ay: np.ndarray,
    ax: np.ndarray,
) -> np.ndarray:
    """Sample pmap at +-d along (ay, ax) and across (-ax, ay) for d in OFFSETS_PX."""
    h, w = pmap.shape
    ay_f = ay[rows, cols]
    ax_f = ax[rows, cols]
    cy_f = -ax_f
    cx_f = ay_f
    out = np.empty((rows.size, 4 * len(OFFSETS_PX)), dtype=np.float32)
    col = 0
    for d in OFFSETS_PX:
        for vy, vx in ((ay_f, ax_f), (-ay_f, -ax_f), (cy_f, cx_f), (-cy_f, -cx_f)):
            ry = np.clip(np.rint(rows + d * vy).astype(np.int32), 0, h - 1)
            rx = np.clip(np.rint(cols + d * vx).astype(np.int32), 0, w - 1)
            out[:, col] = pmap[ry, rx]
            col += 1
    return out


def _anisotropic_smooths_from_cache(
    iso_cache: dict[float, np.ndarray],
    rows: np.ndarray,
    cols: np.ndarray,
    ay: np.ndarray,
    ax: np.ndarray,
) -> np.ndarray:
    """Multi-scale isotropic + locally-oriented anisotropic smooths at ``(rows, cols)``."""
    first_map = next(iter(iso_cache.values()))
    h, w = first_map.shape
    ay_f = ay[rows, cols]
    ax_f = ax[rows, cols]
    out = np.empty((rows.size, len(ISO_SIGMAS) + len(ANISO_SCALES)), dtype=np.float32)
    c = 0
    for s in ISO_SIGMAS:
        out[:, c] = iso_cache[float(s)][rows, cols]
        c += 1
    for la, wc in ANISO_SCALES:
        base = iso_cache[float(wc)]
        acc = base[rows, cols].copy()
        steps = (-la, -la // 2, la // 2, la)
        for t in steps:
            ry = np.clip(np.rint(rows + t * ay_f).astype(np.int32), 0, h - 1)
            rx = np.clip(np.rint(cols + t * ax_f).astype(np.int32), 0, w - 1)
            acc += base[ry, rx]
        out[:, c] = acc * (1.0 / (1 + len(steps)))
        c += 1
    return out


def context_matrix_rows(
    pmap: np.ndarray,
    iso_cache: dict[float, np.ndarray],
    fp_idx: np.ndarray,
    fp_rows: np.ndarray,
    fp_cols: np.ndarray,
    phys: PhysicalFeatures,
    cat_cols: list[np.ndarray],
) -> np.ndarray:
    """Assemble Stage-2 / Stage-3 features ONLY for the requested footprint indices ``fp_idx``."""
    rows = fp_rows[fp_idx]
    cols = fp_cols[fp_idx]
    n_off = 4 * len(OFFSETS_PX)
    n_sm = len(ISO_SIGMAS) + len(ANISO_SCALES)
    n_rc = len(RIDGE_CONTRAST_NAMES)
    n_gr = len(phys.grad_idx)
    n_cat = len(cat_cols)
    total = 1 + n_off + n_sm + n_rc + n_gr + n_cat
    out = np.empty((fp_idx.size, total), dtype=np.float32)

    p_self = pmap[rows, cols]
    out[:, 0] = p_self
    offs = _sample_offsets(pmap, rows, cols, phys.ay, phys.ax)
    out[:, 1 : 1 + n_off] = offs
    sms = _anisotropic_smooths_from_cache(iso_cache, rows, cols, phys.ay, phys.ax)
    c = 1 + n_off
    out[:, c : c + n_sm] = sms
    c += n_sm

    p_sm5 = sms[:, 2]
    p_aniso_6 = sms[:, 4]
    p_aniso_12 = sms[:, 5]
    out[:, c + 0] = p_aniso_6 - p_sm5
    out[:, c + 1] = p_aniso_12 - p_sm5
    out[:, c + 2] = p_self - 0.5 * (offs[:, 10] + offs[:, 11])
    fwd_bridge = np.maximum(np.maximum(offs[:, 8], offs[:, 12]), offs[:, 16])
    bwd_bridge = np.maximum(np.maximum(offs[:, 9], offs[:, 13]), offs[:, 17])
    out[:, c + 3] = np.minimum(fwd_bridge, bwd_bridge)
    out[:, c + 4] = np.maximum(
        np.maximum(offs[:, 18], offs[:, 19]),
        np.maximum(offs[:, 22], offs[:, 23]),
    )
    c += n_rc

    for j, gcol in enumerate(phys.grad_idx):
        out[:, c + j] = phys.X1[fp_idx, int(gcol)]
    c += n_gr
    for k, col_arr in enumerate(cat_cols):
        out[:, c + k] = col_arr[fp_idx]
    return out


def context_matrix(
    pmap: np.ndarray,
    footprint: np.ndarray,
    phys: PhysicalFeatures,
    cat_cols: list[np.ndarray],
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Full-footprint wrapper around :func:`context_matrix_rows`."""
    fp_rows, fp_cols = np.nonzero(footprint)
    fp_idx = np.arange(fp_rows.size, dtype=np.int64)
    iso_cache = precompute_iso_smooths(pmap)
    res = context_matrix_rows(pmap, iso_cache, fp_idx, fp_rows, fp_cols, phys, cat_cols)
    if out is not None:
        out[:] = res
        return out
    return res


def to_map(vec: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    m = np.zeros(footprint.shape, np.float32)
    m[footprint] = np.nan_to_num(vec, nan=0.0)
    return m


@_dataclass(frozen=True)
class FoldSpec:
    fold_id: int
    hidden: np.ndarray
    visible: np.ndarray
    known: np.ndarray
    train_mask: np.ndarray
    n_hidden: int


def quadrant_blocks(footprint: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(footprint)
    out = np.full(footprint.shape, -1, np.int8)
    if ys.size == 0:
        return out
    ym, xm = int(np.median(ys)), int(np.median(xs))
    out[ys, xs] = ((ys >= ym).astype(np.int8) * 2 + (xs >= xm).astype(np.int8))
    return out


def make_folds(labels: np.ndarray, footprint: np.ndarray, buffer_px: int = 3) -> list[FoldSpec]:
    """Partition >=3 px connected catalogue components into 4 spatial quadrant folds."""
    pos = (labels == 1) & footprint
    quads = quadrant_blocks(footprint)
    lab, n = ndi.label(pos, structure=np.ones((3, 3), bool))
    counts = np.bincount(lab.ravel())
    comp_quad = {}
    for cid in range(1, n + 1):
        if counts[cid] < 3:
            continue
        ys, xs = np.nonzero(lab == cid)
        cy, cx = int(np.round(ys.mean())), int(np.round(xs.mean()))
        q = int(quads[cy, cx])
        if q < 0:
            q = int(quads[ys[0], xs[0]])
        comp_quad[cid] = max(q, 0)
    folds = []
    for q in range(4):
        cids = [cid for cid, cq in comp_quad.items() if cq == q]
        hidden = np.isin(lab, cids) if cids else np.zeros_like(footprint, dtype=bool)
        visible = pos & ~hidden
        buf = (
            ndi.binary_dilation(hidden, iterations=buffer_px)
            if hidden.any()
            else np.zeros_like(footprint, dtype=bool)
        )
        train_mask = footprint & ~buf
        folds.append(
            FoldSpec(
                fold_id=q,
                hidden=hidden,
                visible=visible,
                known=visible.copy(),
                train_mask=train_mask,
                n_hidden=int(hidden.sum()),
            )
        )
    return folds
