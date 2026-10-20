#!/usr/bin/env python3
"""GEMSDOE68 Pass-1 audit (2026-10-20 session, branch arena/738dbc78).

Read-only, line-by-line verification of every raster already shipped in
docs/downloads/ plus the registry champion and the dense raster that trips
the literal 3-px lane rule.  No shared tool is forked: geometry constants are
read from gems68.grid, overlaps reuse gems68.uniqueness semantics in addition
to the literal rule.

Outputs evidence/audit68_pass1.json.  Every number is measured in this
checkout from bytes on disk; nothing here is an organiser score.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems68 import grid as G  # noqa: E402

OUT = ROOT / "evidence" / "audit68_pass1.json"
SHAPE = (G.HEIGHT, G.WIDTH)
TRANSFORM = tuple(G.TRANSFORM)[:6]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(path: Path) -> dict:
    with rasterio.open(path) as src:
        a = src.read(1)
        meta = dict(
            count=src.count, dtype=str(src.dtypes[0]), crs=str(src.crs),
            shape=[src.height, src.width],
            transform=[float(v) for v in tuple(src.transform)[:6]],
            nodata=(None if src.nodata is None else
                    ("NaN" if isinstance(src.nodata, float) and np.isnan(src.nodata) else float(src.nodata))),
            band_description=list(src.descriptions),
        )
    finite = np.isfinite(a)
    vals = a[finite]
    # Footprint proxy available without competition data: the champion/zeros
    # family writes 0 outside; nan family writes NaN outside.  The official
    # footprint (5,167,373 finite sample cells per gems68.grid) cannot be
    # re-derived here without sample_submission.tif, so NaN-inside cannot be
    # certified in this pass for -nan files; that is recorded, not assumed.
    return dict(
        file=path.name, rel=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
        sha256=sha256_file(path), **meta,
        geometry_ok=bool(meta["shape"] == list(SHAPE) and meta["transform"] == list(TRANSFORM)
                         and meta["crs"] == "EPSG:32611" and meta["count"] == 1
                         and meta["dtype"] == "float32"),
        n_cells=int(a.size), n_finite=int(finite.sum()), n_nan=int(np.isnan(a).sum()),
        n_inf=int(np.isinf(a).sum()),
        min_finite=(float(vals.min()) if vals.size else None),
        max_finite=(float(vals.max()) if vals.size else None),
        n_below_0=int((vals < 0).sum()), n_above_1=int((vals > 1).sum()),
        sentinel_le_neg3e38=int((vals <= -3.0e38).sum()),
        n_gt0=int((vals > 0).sum()), n_eq1=int((vals == 1).sum()),
        unique_finite_sample=[float(v) for v in np.unique(vals)[:8]],
        is_binary=bool(vals.size and np.all((vals == 0) | (vals == 1))),
        portal_range_risk=bool(
            (vals < 0).any() or (vals > 1).any()
            or (isinstance(meta["nodata"], float) and not (0.0 <= meta["nodata"] <= 1.0))
            or meta["nodata"] == "NaN"),
    )


def dots_of(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read(1)
    return np.argwhere(np.isfinite(a) & (a > 0))


def frac_within(dots_a: np.ndarray, dots_b: np.ndarray, radius: float = 3.0) -> float:
    if len(dots_a) == 0:
        return float("nan")
    if len(dots_b) == 0:
        return 0.0
    d, _ = cKDTree(dots_b.astype(float)).query(dots_a.astype(float), k=1)
    return float(np.mean(d <= radius))


def halo_coverage(dots_b: np.ndarray, footprint_cells: int, shape=SHAPE, radius: float = 3.0) -> float:
    m = np.zeros(shape, bool)
    if len(dots_b):
        m[dots_b[:, 0], dots_b[:, 1]] = True
    near = ndi.distance_transform_edt(~m) <= radius
    return float(near.sum() / footprint_cells), int(near.sum())


def main() -> int:
    downloads = sorted((ROOT / "docs" / "downloads").glob("*.tif"))
    files = [audit(p) for p in downloads]

    champ = ROOT / "registry/rasters/rasters/gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif"
    dense = ROOT / "registry/rasters/rasters/gems61-lane-buried-step-gravinversion-hgm-r3-k400k-b2zeros.tif"
    extra = {}
    for label, p in (("champion_registry_zeros", champ), ("dense_registry_311k", dense)):
        if p.exists():
            extra[label] = audit(p)

    # Cross-check run-card sha claims against bytes on disk.
    claim_checks = []
    for card_rel in ("evidence/run-card-ac68.json", "docs/lane68/data/run-card.json", "evidence/run_card.json"):
        card_path = ROOT / card_rel
        if not card_path.exists():
            continue
        card = json.loads(card_path.read_text())
        entries = []
        if "raster" in card:
            r = card["raster"]
            entries.append((r.get("primary_path"), r.get("primary_sha256")))
            entries.append((r.get("fallback_path"), r.get("fallback_sha256")))
        if "files" in card and isinstance(card["files"], dict):
            for v in card["files"].values():
                if isinstance(v, dict) and v.get("name"):
                    entries.append((f"docs/downloads/{v['name']}", v.get("sha256")))
        for rel, claimed in entries:
            if not rel:
                continue
            p = ROOT / rel
            got = sha256_file(p) if p.exists() else None
            claim_checks.append(dict(card=card_rel, file=rel, claimed_sha256=claimed,
                                     measured_sha256=got, match=bool(got and got == claimed),
                                     card_verdict=card.get("verdict")))

    # Literal 3-px rule degeneracy: does the champion itself trip the rule
    # against the dense raster?  And random-null coverage.
    degeneracy = {}
    if champ.exists() and dense.exists():
        cd, dd = dots_of(champ), dots_of(dense)
        cov_ratio, cov_cells = halo_coverage(dd, 5_167_373)
        # restricted candidates we shipped
        cand_overlap = {}
        for name in ("GEMSDOE68-AC-stack2-20261012T042050Z-nan.tif",
                     "GEMSDOE68-AC-stack3-r3-N32000-20261020T182552Z-nan.tif",
                     "gems68-autoctx-s3-hidedraw-38000dots-20261012T022132Z-4c18def6-zeros.tif"):
            p = ROOT / "docs" / "downloads" / name
            if p.exists():
                cdots = dots_of(p)
                cand_overlap[name] = dict(
                    vs_dense_frac_within_3px=frac_within(cdots, dd),
                    vs_champion_frac_within_3px=frac_within(cdots, cd),
                    n_dots=int(len(cdots)))
        degeneracy = dict(
            champion_vs_dense_frac_within_3px=frac_within(cd, dd),
            champion_n_dots=int(len(cd)), dense_n_dots=int(len(dd)),
            dense_3px_halo_cells_fullgrid=cov_cells,
            dense_3px_halo_frac_of_official_footprint=cov_ratio,
            note="footprint denominator 5,167,373 is the pinned constant in src/gems68/grid.py "
                 "(sample_submission finite cells); halo cells counted on the full 3730x3292 grid, "
                 "so the fraction is an upper bound on in-footprint coverage",
            candidates=cand_overlap,
        )

    # Spearman of champion vs dense (sampled, fixed seed) for context.
    if champ.exists() and dense.exists():
        with rasterio.open(champ) as s:
            a = s.read(1).ravel()
        with rasterio.open(dense) as s:
            b = s.read(1).ravel()
        m = np.isfinite(a) & np.isfinite(b)
        idx = np.flatnonzero(m)
        rng = np.random.default_rng(68)
        pick = rng.choice(idx, size=min(400_000, idx.size), replace=False)
        degeneracy["champion_vs_dense_spearman_sample400k"] = float(
            spearmanr(a[pick], b[pick]).statistic)

    # ---- true-footprint checks (only if core data restored) ----------------
    footprint_checks = {}
    sample = ROOT / ".cache" / "gems_data" / "sample_submission.tif"
    if sample.exists():
        g = G.load_grid()
        fp, cat = g.footprint, g.catalogue
        for p in downloads:
            with rasterio.open(p) as s:
                a = s.read(1)
            footprint_checks[p.name] = dict(
                nan_inside_official_footprint=int((fp & np.isnan(a)).sum()),
                nonzero_outside_official_footprint=int((~fp & np.isfinite(a) & (a != 0)).sum()),
                positive_dots_on_catalogue=int((np.isfinite(a) & (a > 0) & cat).sum()),
            )

    report = dict(
        audit="GEMSDOE68 Pass-1 read-only audit; all values measured in this checkout",
        grid_expected=dict(shape=list(SHAPE), transform=list(TRANSFORM), crs="EPSG:32611",
                           official_footprint_cells_pinned=5_167_373),
        downloads=files, extra=extra, claim_checks=claim_checks,
        dense_rule_degeneracy=degeneracy,
        footprint_checks_vs_sample_submission=footprint_checks,
    )
    OUT.write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}")
    for f in files:
        print(f"{f['file'][:62]:64s} dots={f['n_gt0']:6d} binary={f['is_binary']!s:5s} "
              f"nan={f['n_nan']:8d} range=[{f['min_finite']},{f['max_finite']}] nodata={f['nodata']} "
              f"geom_ok={f['geometry_ok']} portal_risk={f['portal_range_risk']}")
    for k, f in extra.items():
        print(f"[{k}] dots={f['n_gt0']} binary={f['is_binary']} sha={f['sha256'][:16]} "
              f"range=[{f['min_finite']},{f['max_finite']}] nan={f['n_nan']}")
    for c in claim_checks:
        print("CLAIM", c["card"], Path(c["file"]).name[:50], "match=", c["match"], "verdict=", c["card_verdict"])
    if degeneracy:
        print("champion vs dense within3px =", degeneracy["champion_vs_dense_frac_within_3px"])
        print("dense halo frac footprint (upper bound) =", degeneracy["dense_3px_halo_frac_of_official_footprint"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
