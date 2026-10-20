#!/usr/bin/env python3
"""Density-null control for the 3-px dot-overlap duplicate rule.

The rule is "more than 70 % of this raster's dots lie within 3 px of one registry
raster's dots".  A registry raster that itself contains hundreds of thousands of
dots spread over the whole footprint makes that fraction large for *any*
candidate, including a completely unrelated one, because the statistic measures
the registry raster's density rather than any similarity.

This script measures that density effect directly: it draws random dot sets of
the same size as the candidate from the same eligible region and reports the
overlap the rule would return for them.  If the observed overlap is at the
random-null level, the rule firing is an artefact of registry density, not
evidence of duplication.

Usage::

    python scripts/lane_density_null.py candidate.tif registry.tif [--draws 200]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems.validate_submission import footprint_from_features  # noqa: E402

DOT_RADIUS_PX = 3.0
OVERLAP_LIMIT = 0.70


def _read(path):
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
    return arr


def _dots(arr):
    return np.argwhere(np.isfinite(arr) & (arr > 0))


def _overlap(candidate_dots, registry_dots, radius=DOT_RADIUS_PX):
    from scipy.spatial import cKDTree

    if len(candidate_dots) == 0 or len(registry_dots) == 0:
        return float("nan")
    d, _ = cKDTree(registry_dots).query(candidate_dots, k=1)
    return float(np.mean(d <= radius))


def main(argv=None) -> int:
    from scipy.ndimage import distance_transform_edt
    import rasterio

    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", nargs="?", default=None)
    ap.add_argument("registry", nargs="?", default=None)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--min-cat-dist-px", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    with rasterio.open(ROOT / "data" / "bridge" / "existing_faults.tif") as s:
        cat = np.nan_to_num(s.read(1), nan=0.0) > 0.5
    d_cat = distance_transform_edt(~cat)

    if args.candidate is None:
        # Batch mode across the two space-filling registry rasters -> evidence/ac68_density_null.json
        rc = json.loads((ROOT / "evidence" / "run_card.json").read_text())
        cand_path = ROOT / "docs" / "downloads" / Path(rc["submission"]["primary_zeros_tif"]).name
        cand = _read(cand_path)
        official = footprint_from_features(ROOT / "data" / "bridge" / "training_features.tif", cand.shape)
        pa = (np.nan_to_num(cand, nan=0.0) > 0.0) & official
        k_cand = int(pa.sum())
        elig = official & (d_cat > args.min_cat_dist_px)
        idx_el = np.flatnonzero(elig)
        rng = np.random.default_rng(68)
        out = {"candidate": cand_path.name, "k_candidate": k_cand, "n_footprint": int(official.sum()),
               "n_eligible_gt_2px": int(elig.sum()), "n_trials": args.draws, "results": {}}
        for name in ("gems61-lane-buried-step-gravinversion-hgm-r3-k400k-b2zeros.tif",
                     "13gems_20261001_r13-lattice-s5_v2_nan-outside.tif"):
            p_reg = ROOT / "registry" / "rasters" / "rasters" / name
            if not p_reg.is_file():
                p_reg = ROOT / ".cache" / "gems_data" / "scored" / name
            reg = _read(p_reg)
            pb = (np.nan_to_num(reg, nan=0.0) > 0.0) & official
            d = distance_transform_edt(~pb)
            halo = (d <= DOT_RADIUS_PX) & official
            halo_fp = float(halo.sum()) / int(official.sum())
            halo_el = float((halo & elig).sum()) / int(elig.sum())
            flat = halo.ravel()
            hits = [float(flat[rng.choice(idx_el, size=k_cand, replace=False)].mean()) for _ in range(args.draws)]
            obs = float((d[pa] <= DOT_RADIUS_PX).mean())
            out["results"][name] = dict(
                n_b=int(pb.sum()),
                halo_3px_fraction_of_footprint=round(halo_fp, 4),
                halo_3px_fraction_of_eligible_gt2px=round(halo_el, 4),
                random_expected_within_3px=round(float(np.mean(hits)), 4),
                random_sd=round(float(np.std(hits, ddof=1)), 5),
                candidate_observed_within_3px=round(obs, 4),
                excess_over_random=round(obs - float(np.mean(hits)), 4),
            )
        (ROOT / "evidence" / "ac68_density_null.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return 0

    cand = _read(args.candidate)
    reg = _read(args.registry)
    cand_dots = _dots(cand)
    reg_dots = _dots(reg)
    observed = _overlap(cand_dots, reg_dots)

    official = footprint_from_features(ROOT / "data" / "bridge" / "training_features.tif",
                                       cand.shape)
    eligible = np.isfinite(cand) & (cand >= 0) & official & (d_cat > args.min_cat_dist_px)
    reg_mask = (np.nan_to_num(reg, nan=0.0) > 0.0) & official
    halo = (distance_transform_edt(~reg_mask) <= DOT_RADIUS_PX) & official
    idx_pool = np.flatnonzero(eligible)
    flat = halo.ravel()
    rng = np.random.default_rng(args.seed)
    n = len(cand_dots)
    nulls = [float(flat[rng.choice(idx_pool, size=min(n, len(idx_pool)), replace=False)].mean())
             for _ in range(args.draws)]
    nulls = np.asarray(nulls, float)
    null_mean = float(np.nanmean(nulls))
    null_sd = float(np.nanstd(nulls))
    z = (observed - null_mean) / null_sd if null_sd > 0 else float("inf")

    rep = {
        "candidate": Path(args.candidate).name,
        "registry": Path(args.registry).name,
        "candidate_dots": int(n),
        "registry_dots": int(len(reg_dots)),
        "registry_dot_density_per_1000_cells": round(
            1000.0 * len(reg_dots) / max(1, int(np.isfinite(reg).sum())), 4),
        "observed_overlap_within_3px": observed,
        "overlap_limit": OVERLAP_LIMIT,
        "literal_rule_fires": bool(observed > OVERLAP_LIMIT),
        "null_draws": int(args.draws),
        "null_mean_overlap": null_mean,
        "null_sd_overlap": null_sd,
        "null_p95_overlap": float(np.nanpercentile(nulls, 95)),
        "observed_minus_null_sd": z,
        "interpretation": (
            "the observed overlap is at or below the eligible-domain random-null level, so the rule firing is "
            "an artefact of the registry raster's own dot density rather than evidence "
            "of duplication" if z < 2.0 else
            "the observed overlap exceeds the random-null level; treat as genuine overlap"),
    }
    print(json.dumps(rep, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
