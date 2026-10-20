#!/usr/bin/env python3
"""Final build: full-data out-of-fold auto-context maps -> dots -> validated GeoTIFF + run card.

Reads the gate decision from evidence/exp2_autocontext_hidedraw.json (stage and
budget are chosen there, on the holdout, never here).  Then:
  1. LOQO maps on the full catalogue (every pixel predicted out-of-fold);
  2. emission with the shared rule (exclude <= 2 px from the FULL catalogue,
     5x5 local maxima, top-N);
  3. lane-drift check on the surface AND on the dots vs every registry raster;
  4. zeros-mode float32 GeoTIFF (all finite, 0 outside footprint) as the primary
     file, NaN-outside variant as secondary; validate both; sha256;
  5. run card (protocol rule 5) with the promote / negative verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems68 import autocontext as AC, emission as E, grid as G, uniqueness as U, validate as V  # noqa: E402
from gems68.experiment import Lane, log  # noqa: E402
from gems68.holdout import quadrant_ids  # noqa: E402

# Archived experimental emitter: never write directly into public Pages.
OUT = ROOT / "outputs" / "legacy_candidates"
REG = ROOT / ".cache" / "registry"
CACHE = ROOT / ".cache" / "final"


def choose(gate: dict, budgets=(25_000, 38_000, 50_000)) -> tuple[int, int, dict]:
    """Pick (stage, budget) with the best passing gate; fall back to the best delta."""
    best = None
    for key, g in gate.items():
        n, st = key.split("_stage")
        n, st = int(n), int(st)
        if n not in budgets:
            continue
        score = (1 if g["passes"] else 0, g["dti_cand"])
        if best is None or score > best[0]:
            best = (score, st, n, g)
    return best[1], best[2], best[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=None)
    ap.add_argument("--budget", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    exp2 = json.loads((ROOT / "evidence" / "exp2_autocontext_hidedraw.json").read_text())
    stage, budget, gate = choose(exp2["gate"])
    if args.stage: stage = args.stage
    if args.budget: budget = args.budget
    log(f"chosen stage={stage} budget={budget} gate_passes={gate['passes']}")

    g = G.load_grid(); fp = g.footprint
    X1 = np.ascontiguousarray(np.load(ROOT / ".cache/features/X1.npy", mmap_mode="r"))
    phys = list(np.load(ROOT / ".cache/features/phys.npy"))
    maps_file = CACHE / f"final_maps_s{stage}.npz"
    if maps_file.is_file():
        z = np.load(maps_file); maps = {int(k[1:]): z[k] for k in z.files}
        log("resumed final maps")
    else:
        lane = Lane(g, X1, phys, quadrant_ids(fp))
        maps = lane.final(stage)
        np.savez(maps_file, **{f"s{k}": v for k, v in maps.items()})
    del X1
    surface = AC.to_map(maps[stage], fp)
    s1 = AC.to_map(maps[1], fp)

    # ---- emission
    allowed = ~E.exclusion_mask(g.catalogue) & fp
    dots = E.emit_dots(surface, allowed, budget)
    n_dots = int(dots.sum())
    from scipy import ndimage as ndi
    dcat = ndi.distance_transform_edt(~g.catalogue)
    dot_dist = dcat[dots]
    log(f"dots={n_dots} min_dist_to_catalogue_px={dot_dist.min():.2f} median={np.median(dot_dist):.2f}")

    # ---- lane drift / duplicates (surface before placement AND final dots)
    registry = U.load_registry(sorted(REG.glob("*.tif")))
    rho = U.surface_rank_corr(surface, registry, fp)
    rho_s1 = U.surface_rank_corr(s1, registry, fp)
    ov = U.dot_overlap(dots, registry, fp)
    verdict_u = U.verdict(rho, ov)
    log("uniqueness:", verdict_u)

    # ---- write files
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest8 = hashlib.sha256(np.packbits(dots).tobytes()).hexdigest()[:8]
    stem = f"gems68-autoctx-s{stage}-hidedraw-{n_dots}dots-{stamp}-{digest8}"
    arr0 = np.where(dots, np.float32(1.0), np.float32(0.0)).astype(np.float32)
    info0 = G.write_geotiff(OUT / f"{stem}-zeros.tif", arr0, nodata=None)
    arrn = arr0.copy(); arrn[~fp] = np.nan
    infon = G.write_geotiff(OUT / f"{stem}-nan.tif", arrn, nodata=float("nan"))
    val0 = V.validate(OUT / f"{stem}-zeros.tif", fp, g.catalogue)
    valn = V.validate(OUT / f"{stem}-nan.tif", fp, g.catalogue)
    # the NaN variant legitimately fails the "no_nan_anywhere" policy check; report in-footprint checks
    zip_path = OUT / f"{stem}-zeros.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(OUT / f"{stem}-zeros.tif", arcname=f"{stem}-zeros.tif")
    # surface (float, for audit) is kept out of docs/ to keep the site light
    np.save(CACHE / f"{stem}-surface.npy", surface.astype(np.float16))

    promote = bool(gate["passes"] and not verdict_u["duplicate"] and val0["all_checks_passed"])
    note = (f"GEMSDOE68 auto-context s{stage} hide-draw; {n_dots} dots >2px off catalogue; "
            f"HOLDOUT-DTI {gate['dti_cand']:.4f} vs s1 {gate['dti_stage1']:.4f}; UNSCORED")[:140]
    card = dict(
        run_card_version=1, lane="auto-context stacked refinement (GEMSDOE68)",
        hypothesis="H68-1: a stage-2 LightGBM on orientation-sampled stage-1 probabilities (+-1..15 px along/across strike), "
                   "multi-scale/anisotropic smooths, gradient/tilt/scarp channels and visible-catalogue context "
                   "(trained on withheld components, hide-draw) ranks unmapped fault pixels better than the per-pixel stage 1 at matched mass.",
        mechanism="continuity, parallelism and en-echelon spacing of Basin-and-Range normal-fault strands are learnable from "
                  "the spatial arrangement of first-stage evidence; unmapped faults are mostly continuations/splays of mapped ones.",
        non_fault_mimic="any linear non-tectonic feature favoured by stage 1 (roads, canals, shoreline terraces, lithologic contacts) "
                        "is reinforced by continuity; bedding-parallel ridges in tilted volcanic sections.",
        holdout=dict(evidence_class="HOLDOUT-DTI", evaluator_version=exp2["evaluator_version"],
                     withheld_positive_pixels=exp2["holdout"]["withheld_positive_pixels"],
                     stage=stage, budget=budget,
                     dti_stage1=gate["dti_stage1"], dti_candidate=gate["dti_cand"],
                     delta_vs_stage1=gate["delta_all"], ci95_delta=gate["ci_all"],
                     far_1km=dict(dti_stage1=gate["far_dti_stage1"], dti_candidate=gate["far_dti_cand"],
                                  delta=gate["delta_far"], ci95=gate["ci_far"]),
                     gate_passes=gate["passes"]),
        emission=dict(n_dots=n_dots, exclude_px=E.EXCLUDE_PX, nms=5, value=1.0,
                      min_dist_to_catalogue_px=float(dot_dist.min()), median_dist_px=float(np.median(dot_dist)),
                      p95_dist_px=float(np.percentile(dot_dist, 95))),
        uniqueness=dict(surface_rank_corr=rho, stage1_surface_rank_corr=rho_s1, dot_overlap_within_3px=ov,
                        **verdict_u, registry_rasters=len(registry)),
        files=dict(primary=dict(name=f"{stem}-zeros.tif", sha256=info0["sha256"], bytes=info0["bytes"],
                                zip=zip_path.name, zip_sha256=G.sha256_file(zip_path), validator=val0),
                   secondary_nan_outside=dict(name=f"{stem}-nan.tif", sha256=infon["sha256"], bytes=infon["bytes"],
                                              validator=valn)),
        validator_summary=dict(no_nan_inside_footprint=val0["checks"]["in_footprint_all_finite"],
                               values_in_0_1=val0["checks"]["range_0_1_guaranteed"],
                               crs_shape_transform_match=all(val0["checks"][k] for k in val0["checks"] if k.startswith(("crs", "dimensions", "transform"))),
                               zero_dots_on_catalogue=val0["checks"]["zero_dots_on_mapped_catalogue"]),
        submission_name=f"{stem}-zeros", submission_note=note, note_chars=len(note),
        verdict="promote" if promote else "negative",
        verdict_reason=("gate met, unique, validator green" if promote else
                        "; ".join(r for r, bad in (("holdout gate not met", not gate["passes"]),
                                                   ("duplicate of a registry raster", verdict_u["duplicate"]),
                                                   ("validator failed", not val0["all_checks_passed"])) if bad)),
        organizer_confirmed_score=None, built_utc=stamp, seconds=round(time.time() - t0, 1))
    (ROOT / "evidence" / "run_card.json").write_text(json.dumps(card, indent=2, default=float))
    (OUT / f"{stem}-runcard.json").write_text(json.dumps(card, indent=2, default=float))
    log("verdict:", card["verdict"], "|", card["verdict_reason"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
