#!/usr/bin/env python3
"""Compare candidate surfaces or final emissions with supplied registry rasters.

Stages:
  surface  — pre-placement: continuous-surface Spearman rank correlation and exact pixel/hash agreement;
  emitted  — post-placement: the same checks plus final-dot overlap within 3 px.

Absolute rank correlation >0.90, >70% of candidate dots within 3 px of one registry raster, or
an exact pixel duplicate flags lane drift. Empty/constant comparisons are INCONCLUSIVE, not a pass.
All inputs must already share one aligned grid; the tool never reprojects rasters.

Examples:
  python scripts/lane_check.py --stage surface surface.tif registry1.tif registry2.tif
  python scripts/lane_check.py --stage emitted candidate.tif registry1.tif registry2.tif
Exit code: 0 = no drift vs supplied registry; 2 = duplicate/drift; 3 = inconclusive; 1 = error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gems.submission_writer import sha256_file  # noqa: E402

CORR_LIMIT = 0.90
DOT_OVERLAP_LIMIT = 0.70
DOT_RADIUS_PX = 3.0


def _read(path):
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
                arr[np.isnan(arr)] = np.nan
            else:
                arr[arr == nodata] = np.nan
        meta = {
            "shape": (src.height, src.width),
            "transform": tuple(src.transform)[:6],
            "crs": src.crs,
        }
    return arr, meta


def _assert_aligned(candidate_meta: dict, registry_meta: dict, path: str | Path) -> None:
    if candidate_meta["shape"] != registry_meta["shape"]:
        raise ValueError(f"grid shape mismatch for {path}: {registry_meta['shape']} != {candidate_meta['shape']}")
    if candidate_meta["crs"] is None or registry_meta["crs"] is None:
        raise ValueError(f"missing CRS on candidate or registry raster {path}")
    if candidate_meta["crs"] != registry_meta["crs"]:
        raise ValueError(f"CRS mismatch for {path}: {registry_meta['crs']} != {candidate_meta['crs']}")
    if not np.allclose(candidate_meta["transform"], registry_meta["transform"], rtol=0.0, atol=1e-9):
        raise ValueError(f"transform mismatch for {path}; align to the official grid before comparison")


def spearman_on_common(a: np.ndarray, b: np.ndarray) -> float:
    """Exact Spearman rank correlation over common finite cells."""
    from scipy.stats import spearmanr

    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return float("nan")
    x, y = a[m], b[m]
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def dot_fraction_near(dots_a: np.ndarray, dots_b: np.ndarray, radius: float = DOT_RADIUS_PX) -> float:
    """Fraction of candidate dots with a registry dot within ``radius`` pixels."""
    from scipy.spatial import cKDTree

    if len(dots_a) == 0:
        return float("nan")
    if len(dots_b) == 0:
        return 0.0
    d, _ = cKDTree(dots_b).query(dots_a, k=1)
    return float(np.mean(d <= radius))


def exact_pixel_agreement(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Exact-equality fraction over the union of valid cells; shared NaN exterior is excluded."""
    comparable = np.isfinite(a) | np.isfinite(b)
    n = int(comparable.sum())
    if n == 0:
        return float("nan"), 0
    same = np.isfinite(a) & np.isfinite(b) & (a == b)
    return float(same[comparable].mean()), n


def _json_number(value):
    return float(value) if np.isfinite(value) else None


def lane_check(candidate: str | Path, registry: list[str | Path], stage: str = "emitted") -> dict:
    if stage not in {"surface", "emitted"}:
        raise ValueError("stage must be 'surface' or 'emitted'")
    a, candidate_meta = _read(candidate)
    dots_a = np.argwhere(np.isfinite(a) & (a > 0)) if stage == "emitted" else None
    rows = []
    for r in registry:
        b, registry_meta = _read(r)
        _assert_aligned(candidate_meta, registry_meta, r)
        corr = spearman_on_common(a, b)
        agreement, compared_cells = exact_pixel_agreement(a, b)
        if stage == "emitted":
            dots_b = np.argwhere(np.isfinite(b) & (b > 0))
            fraction = dot_fraction_near(dots_a, dots_b)
            dot_overlap_drift = np.isfinite(fraction) and fraction > DOT_OVERLAP_LIMIT
            registry_dot_count = int(len(dots_b))
        else:
            fraction = None
            dot_overlap_drift = False
            registry_dot_count = None
        corr_drift = np.isfinite(corr) and abs(corr) > CORR_LIMIT
        exact_duplicate = np.isfinite(agreement) and agreement == 1.0
        drift = bool(corr_drift or dot_overlap_drift or exact_duplicate)
        rows.append({
            "registry": Path(r).name,
            "sha256": sha256_file(r),
            "spearman_rho": _json_number(corr),
            "absolute_spearman_rho": _json_number(abs(corr)) if np.isfinite(corr) else None,
            "exact_pixel_agreement": _json_number(agreement),
            "compared_cells": compared_cells,
            "registry_dots": registry_dot_count,
            "candidate_dots_within_3px": _json_number(fraction) if fraction is not None else None,
            "exact_duplicate": bool(exact_duplicate),
            "drift": drift,
        })
    return {
        "candidate": Path(candidate).name,
        "candidate_sha256": sha256_file(candidate),
        "candidate_dots": int(len(dots_a)) if dots_a is not None else None,
        "stage": stage,
        "limits": {"absolute_spearman_rho": CORR_LIMIT, "dots_within_3px": DOT_OVERLAP_LIMIT,
                   "dot_radius_px": DOT_RADIUS_PX},
        "comparisons": rows,
        "verdict": _verdict(rows, stage),
    }


def _verdict(rows, stage: str) -> str:
    if not rows:
        return "INCONCLUSIVE - no registry rasters supplied"
    if any(r["drift"] for r in rows):
        return "LANE DRIFT - log duplicate and stop"
    if any(r["spearman_rho"] is None or r["exact_pixel_agreement"] is None for r in rows):
        return "INCONCLUSIVE - empty or constant comparison; not a pass"
    if stage == "emitted" and any(r["candidate_dots_within_3px"] is None for r in rows):
        return "INCONCLUSIVE - no candidate dots to compare; not a pass"
    return "no drift vs supplied registry rasters; completeness not established by this tool"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("candidate")
    ap.add_argument("registry", nargs="+")
    ap.add_argument("--stage", choices=("surface", "emitted"), default="emitted",
                    help="surface = continuous pre-placement; emitted = final raster plus dot overlap")
    ap.add_argument("--json", help="write the report to this path")
    args = ap.parse_args(argv)
    try:
        rep = lane_check(args.candidate, args.registry, stage=args.stage)
    except Exception as exc:  # noqa: BLE001 - report raster/grid errors as exit 1
        print(f"error: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(rep, indent=2, allow_nan=False)
    print(text)
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    if rep["verdict"].startswith("LANE DRIFT"):
        return 2
    if rep["verdict"].startswith("INCONCLUSIVE"):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
