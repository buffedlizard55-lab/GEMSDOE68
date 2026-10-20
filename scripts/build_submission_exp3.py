#!/usr/bin/env python3
"""Exp 3 final build: full-catalogue LOQO auto-context maps -> dots -> GeoTIFF.

Refuses to emit a promoted download unless the PRE-REGISTERED candidate
(stage 3 @ 38,000 dots, evidence/exp3_preregistration.json) passed its gate
on the fresh seed-31 holdout AND the leakage canary is clean.  Writes the
run card to evidence/run_card.json (same schema scripts/build_site.py reads)
plus evidence/run_card_exp3.json.
Uniqueness is checked streaming (one raster at a time) against every TIFF in
registry/rasters/ and every prior GEMSDOE68 download in docs/downloads/.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems68 import autocontext as AC, emission as E, grid as G, uniqueness as U, validate as V  # noqa: E402
from gems68.experiment import Lane, log  # noqa: E402
from gems68.holdout import quadrant_ids  # noqa: E402

AC.OFFSETS = (1, 2, 3, 5, 8, 12, 15)
AC.MODEL_SEEDS = (68, 71)
AC.N_ROUNDS = 180
STAGE, BUDGET = 3, 38_000  # defaults; overridden by CLI (Exp4: 25_000)
OUT = ROOT / "docs" / "downloads"
CACHE = ROOT / ".cache" / "final_exp3"


def streaming_uniqueness(surface, dots, fp, paths):
    rhos, ovs, hashes = {}, {}, {}
    for p in paths:
        reg = U.load_registry([p])
        rhos.update(U.surface_rank_corr(surface, reg, fp))
        ovs.update(U.dot_overlap(dots, reg, fp))
        hashes[p.name] = G.sha256_file(p)
        del reg
    return rhos, ovs, U.verdict(rhos, ovs), hashes


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="exp3_autocontext_ext_bagged.json")
    ap.add_argument("--stage", type=int, default=STAGE)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--eval-seed", type=int, default=31)
    ap.add_argument("--train-seeds", default="31,32,33")
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--tag", default="autoctx3")
    ap.add_argument("--run-card", default="run_card_exp3.json")
    a = ap.parse_args()
    stage, budget = a.stage, a.budget
    train_seeds = tuple(int(s) for s in a.train_seeds.split(","))
    cache = Path(a.cache)
    t0 = time.time()
    cache.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    exp3 = json.loads((ROOT / "evidence" / a.evidence).read_text())
    cand = exp3["preregistered_candidate"]
    assert (cand["stage"], cand["budget"]) == (stage, budget), (cand, stage, budget)
    assert exp3["eval_seed"] == a.eval_seed and tuple(exp3["train_seeds"]) == train_seeds
    gate = cand["gate"]
    flagged = [k for k, v in exp3["canary"].items() if v["flag"]]
    log(f"preregistered gate passes={gate['passes']} canary_flagged={flagged}")
    if not gate["passes"] or flagged:
        card = dict(run_card_version=1, lane="auto-context stacked refinement (GEMSDOE68 exp3)",
                    verdict="negative",
                    verdict_reason=("pre-registered holdout gate not met" if not gate["passes"]
                                    else f"leakage canary flagged: {flagged}"),
                    holdout=dict(evidence_class="HOLDOUT-DTI", evaluator_version=exp3["evaluator_version"],
                                 withheld_positive_pixels=exp3["holdout"]["withheld_positive_pixels"],
                                 stage=stage, budget=budget, gate=gate),
                    organizer_confirmed_score=None,
                    built_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        (ROOT / "evidence" / a.run_card).write_text(json.dumps(card, indent=2))
        log(f"NEGATIVE: no submission emitted. Run card: evidence/{a.run_card}")
        return 2

    g = G.load_grid()
    fp = g.footprint
    X1 = np.load(ROOT / ".cache/features/X1ext.npy", mmap_mode="r")
    names1 = json.loads((ROOT / ".cache/features/X1ext_names.json").read_text())["names"]
    phys = list(np.load(ROOT / ".cache/features/phys.npy"))
    maps_file = cache / f"final_maps_s{stage}.npz"
    if maps_file.is_file():
        z = np.load(maps_file)
        maps = {int(k[1:]): z[k] for k in z.files}
        log("resumed final maps")
    else:
        lane = Lane(g, np.ascontiguousarray(X1), phys, quadrant_ids(fp),
                    eval_seed=a.eval_seed, train_seeds=train_seeds, stage1_names=names1)
        maps = lane.final(stage)
        np.savez(maps_file, **{f"s{k}": v for k, v in maps.items()})
        del lane
    del X1
    surface = AC.to_map(maps[stage], fp)

    allowed = ~E.exclusion_mask(g.catalogue) & fp
    dots = E.emit_dots(surface, allowed, budget)
    n_dots = int(dots.sum())
    dcat = ndi.distance_transform_edt(~g.catalogue)
    dot_dist = dcat[dots]
    log(f"dots={n_dots} min_dist_px={dot_dist.min():.2f} median={np.median(dot_dist):.2f}")

    prior = sorted((ROOT / "registry/rasters").rglob("*.tif")) + sorted(OUT.glob("*.tif"))
    log(f"uniqueness vs {len(prior)} prior rasters (surface + dots, streaming)")
    rho, ov, verdict_u, hashes = streaming_uniqueness(surface, dots, fp, prior)
    log("uniqueness:", verdict_u)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest8 = hashlib.sha256(np.packbits(dots).tobytes()).hexdigest()[:8]
    stem = f"gems68-{a.tag}-ext-bagged-{n_dots}dots-{stamp}-{digest8}"
    arr0 = np.where(dots, np.float32(1.0), np.float32(0.0)).astype(np.float32)
    info0 = G.write_geotiff(OUT / f"{stem}-zeros.tif", arr0, nodata=None)
    arrn = arr0.copy()
    arrn[~fp] = np.nan
    infon = G.write_geotiff(OUT / f"{stem}-nan.tif", arrn, nodata=float("nan"))
    if info0["sha256"] in hashes.values() or infon["sha256"] in hashes.values():
        raise SystemExit("BYTE DUPLICATE of a prior raster -- aborting")
    val0 = V.validate(OUT / f"{stem}-zeros.tif", fp, g.catalogue)
    valn = V.validate(OUT / f"{stem}-nan.tif", fp, g.catalogue)
    zip_path = OUT / f"{stem}-zeros.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(OUT / f"{stem}-zeros.tif", arcname=f"{stem}-zeros.tif")
    np.save(cache / f"{stem}-surface.npy", surface.astype(np.float16))

    promote = bool(gate["passes"] and not verdict_u["duplicate"] and val0["all_checks_passed"])
    note = (f"GEMSDOE68 {a.tag} ext+bagged s{stage}; {n_dots} dots >2px off cat; "
            f"HOLDOUT-DTI {gate['dti_cand']:.4f} vs s1 {gate['dti_stage1']:.4f} seed{a.eval_seed}; UNSCORED")[:140]
    card = dict(
        run_card_version=1, lane="auto-context stacked refinement (GEMSDOE68)",
        hypothesis="H68-1/H68-4: stage-2/3 LightGBM on orientation-sampled stage-1 probabilities "
                   "(+-1,2,3,5,8,12,15 px along/across strike), multi-scale/anisotropic smooths, "
                   "gradient/tilt/scarp + LiDAR-scarp + radiometric channels and visible-catalogue "
                   "context (hide-draw trained, 2-seed bagged) ranks unmapped fault pixels better "
                   "than the per-pixel stage 1 at matched mass.",
        mechanism="continuity, parallelism and en-echelon spacing of Basin-and-Range normal-fault "
                  "strands are learnable from the spatial arrangement of first-stage evidence; "
                  "unmapped faults are mostly continuations/splays of mapped ones; LiDAR scarps "
                  "and radiometric alteration add surface/fluid-pathway evidence under thin cover.",
        non_fault_mimic="range-front/basin-margin erosional escarpments, shoreline terraces and "
                        "volcanic-flow K contrasts produce the same scarp/radiometric signatures "
                        "and are reinforced by the continuity prior.",
        holdout=dict(evidence_class="HOLDOUT-DTI", evaluator_version=exp3["evaluator_version"],
                     withheld_positive_pixels=exp3["holdout"]["withheld_positive_pixels"],
                     eval_seed=a.eval_seed, stage=stage, budget=budget,
                     dti_stage1=gate["dti_stage1"], dti_candidate=gate["dti_cand"],
                     delta_vs_stage1=gate["delta_all"], ci95_delta=gate["ci_all"],
                     far_1km=dict(dti_stage1=gate["far_dti_stage1"], dti_candidate=gate["far_dti_cand"],
                                  delta=gate["delta_far"], ci95=gate["ci_far"]),
                     gate_passes=gate["passes"],
                     canary_flagged=flagged),
        emission=dict(n_dots=n_dots, exclude_px=E.EXCLUDE_PX, nms=5, value=1.0,
                      min_dist_to_catalogue_px=float(dot_dist.min()),
                      median_dist_px=float(np.median(dot_dist)),
                      p95_dist_px=float(np.percentile(dot_dist, 95))),
        uniqueness=dict(surface_rank_corr=rho, dot_overlap_within_3px=ov, **verdict_u,
                        registry_rasters=len(prior),
                        registry_sha256=hashes),
        files=dict(primary=dict(name=f"{stem}-zeros.tif", sha256=info0["sha256"], bytes=info0["bytes"],
                                zip=zip_path.name, zip_sha256=G.sha256_file(zip_path), validator=val0),
                   secondary_nan_outside=dict(name=f"{stem}-nan.tif", sha256=infon["sha256"],
                                              bytes=infon["bytes"], validator=valn)),
        validator_summary=dict(no_nan_inside_footprint=val0["checks"]["in_footprint_all_finite"],
                               values_in_0_1=val0["checks"]["range_0_1_guaranteed"],
                               crs_shape_transform_match=all(
                                   val0["checks"][k] for k in val0["checks"]
                                   if k.startswith(("crs", "dimensions", "transform"))),
                               zero_dots_on_catalogue=val0["checks"]["zero_dots_on_mapped_catalogue"]),
        submission_name=f"{stem}-zeros", submission_note=note, note_chars=len(note),
        verdict="promote" if promote else "negative",
        verdict_reason=("pre-registered gate met on fresh seed, canary clean, unique, validator green"
                        if promote else
                        "; ".join(r for r, bad in (("holdout gate not met", not gate["passes"]),
                                                   ("duplicate of a registry raster", verdict_u["duplicate"]),
                                                   ("validator failed", not val0["all_checks_passed"])) if bad)),
        organizer_confirmed_score=None, built_utc=stamp, seconds=round(time.time() - t0, 1))
    for dest in (ROOT / "evidence/run_card.json", ROOT / "evidence" / a.run_card,
                 OUT / f"{stem}-runcard.json"):
        dest.write_text(json.dumps(card, indent=2, default=float))
    log("verdict:", card["verdict"], "|", card["verdict_reason"])
    log("primary:", card["files"]["primary"]["name"], card["files"]["primary"]["sha256"])
    return 0 if promote else 2


if __name__ == "__main__":
    sys.exit(main())
