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
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("registry")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    cand = _read(args.candidate)
    reg = _read(args.registry)
    cand_dots = _dots(cand)
    reg_dots = _dots(reg)
    observed = _overlap(cand_dots, reg_dots)

    official = footprint_from_features(ROOT / "data" / "bridge" / "training_features.tif",
                                       cand.shape)
    eligible = np.isfinite(cand) & (cand >= 0)
    pool = np.argwhere(eligible & official)
    rng = np.random.default_rng(args.seed)
    n = len(cand_dots)
    nulls = []
    for _ in range(args.draws):
        pick = pool[rng.choice(pool.shape[0], size=min(n, pool.shape[0]), replace=False)]
        nulls.append(_overlap(pick, reg_dots))
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
            "the observed overlap is at the random-null level, so the rule firing is "
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
