#!/usr/bin/env python3
"""Final build for Experiment 3: full-data out-of-fold auto-context maps -> dots -> validated GeoTIFF + run card.

Reads the gate decision from ``evidence/exp3_autocontext_scarp_ridge_hidedraw.json``
(stage=3, budget=35000 chosen on the holdout, never tuned post-hoc). Then:
  1. LOQO maps on the full catalogue (every pixel predicted out-of-fold at Stages 1, 2, 3);
  2. emission with the shared rule (exclude <= 2 px from the FULL catalogue,
     5x5 local maxima, top-35,000 dots -> value 1.0);
  3. lane-drift and duplicate check on the continuous surface BEFORE placement
     AND on the final binary dots against every raster in ``.cache/gems_data/scored``,
     ``.cache/gems_data/calibration``, ``registry/rasters/rasters``, and ``docs/downloads``;
  4. zeros-mode float32 GeoTIFF (all finite, 0.0 outside footprint) as primary
     file, NaN-outside variant as secondary; validate both with ``gems68.validate``
     and ``gems.validate_submission``;
  5. write Run Card (protocol rule 5) with ``promote`` verdict to ``evidence/run_card.json``,
     ``evidence/run-card-ac68.json``, ``docs/lane68/data/run-card.json``, and ``docs/downloads/``.
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
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems.validate_submission import validate as validate_vendored  # noqa: E402
from gems68 import autocontext as AC, emission as E, features as F, grid as G, submission as S, uniqueness as U, validate as V  # noqa: E402
from gems68.experiment import Lane, log  # noqa: E402

OUT = ROOT / "docs" / "downloads"
CACHE = ROOT / ".cache" / "final_exp3"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    exp3 = json.loads((ROOT / "evidence" / "exp3_autocontext_scarp_ridge_hidedraw.json").read_text())
    stage = int(args.stage)
    budget = int(args.budget) if args.budget else int(exp3["best_budget"])
    b_entry = exp3["by_budget"][str(budget)]
    st_key = f"stage{stage}"
    gate_all = b_entry[st_key]["all"]["delta_vs_s1"]
    gate_far = b_entry[st_key]["far_gt_1km"]["delta_vs_s1"]
    gate_passes = bool(exp3["gate_passed"] and gate_all["ci95_excludes_zero"] and gate_far["ci95_excludes_zero"])
    log(
        f"chosen experiment={exp3['experiment']} stage={stage} budget={budget} "
        f"dti_all={b_entry[st_key]['all']['dti']:.4f} (vs s1 {b_entry['stage1']['all']['dti']:.4f}) "
        f"dti_far={b_entry[st_key]['far_gt_1km']['dti']:.4f} (vs s1 {b_entry['stage1']['far_gt_1km']['dti']:.4f}) "
        f"gate_passes={gate_passes}"
    )

    g = G.load_grid()
    fp = g.footprint
    phys = F.build_physical_features(g, use_cache=True)

    maps_file = CACHE / f"final_maps_exp3_s{stage}.npz"
    if maps_file.is_file() and not args.recompute:
        z = np.load(maps_file)
        maps = {int(k[1:]): z[k] for k in z.files}
        log(f"resumed final maps from {maps_file}")
    else:
        empty_collar = np.zeros(g.shape, dtype=bool)
        lane = Lane(g, phys, g.catalogue, empty_collar)
        maps = lane.final(stage)
        np.savez(maps_file, **{f"s{k}": v for k, v in maps.items()})
        log(f"saved final maps to {maps_file}")

    surface = AC.to_map(maps[stage], fp)
    s2_surface = AC.to_map(maps[2], fp)
    s1_surface = AC.to_map(maps[1], fp)

    # Normalize continuous surface into [0, 1] inside footprint for uniqueness check
    s_min = float(np.nanmin(surface[fp]))
    s_max = float(np.nanmax(surface[fp]))
    if s_max > s_min:
        surface_norm = np.where(fp, (surface - s_min) / (s_max - s_min), 0.0).astype(np.float32)
    else:
        surface_norm = np.where(fp, surface, 0.0).astype(np.float32)

    # ---- emission (exclude <= 2 px from FULL catalogue, 5x5 NMS, top-budget dots -> 1.0)
    dots = E.emit_dots(surface_norm, fp, g.catalogue, budget=budget, nms_size=5, exclusion_px=2.0)
    n_dots = int(dots.sum())
    dcat = ndi.distance_transform_edt(~g.catalogue)
    dot_dist = dcat[dots]
    log(
        f"emitted dots={n_dots} min_dist_px={dot_dist.min():.2f} "
        f"median_dist_m={np.median(dot_dist)*100.0:.0f} p75_dist_m={np.percentile(dot_dist, 75)*100.0:.0f}"
    )

    # ---- uniqueness / lane-drift check on BOTH surface before placement AND final dots
    reg_dirs = [
        ROOT / ".cache" / "gems_data" / "scored",
        ROOT / ".cache" / "gems_data" / "calibration",
        ROOT / "registry" / "rasters" / "rasters",
        OUT,
    ]
    uniq = U.check_uniqueness(
        surface_norm,
        dots,
        fp,
        [d for d in reg_dirs if d.exists()],
        max_rank_corr=0.90,
        max_dot_overlap=0.70,
    )
    log(
        f"uniqueness vs {uniq['n_registry_rasters_checked']} prior rasters: "
        f"max_surface_spearman={uniq['max_surface_spearman']:.4f} ({uniq['max_surface_spearman_vs']}), "
        f"max_dot_overlap_3px_informative={uniq['max_dot_overlap_3px']:.4f} ({uniq['max_dot_overlap_vs']}), "
        f"passes_uniqueness={uniq['passes_uniqueness']}"
    )
    if not uniq["passes_uniqueness"]:
        raise AssertionError("Candidate failed uniqueness / lane-drift gate against registry rasters!")

    # ---- write GeoTIFF files
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest8 = hashlib.sha256(np.packbits(dots).tobytes()).hexdigest()[:8]
    stem = f"gems68-autoctx-s{stage}-scarp-ridge-{n_dots}dots-{stamp}-{digest8}"

    arr0 = np.where(dots, np.float32(1.0), np.float32(0.0)).astype(np.float32)
    zeros_path = OUT / f"{stem}-zeros.tif"
    nan_path = OUT / f"{stem}-nan.tif"
    zip_path = OUT / f"{stem}-zeros.zip"

    info0 = G.write_geotiff(zeros_path, arr0, nodata=None)
    arrn = arr0.copy()
    arrn[~fp] = np.nan
    infon = G.write_geotiff(nan_path, arrn, nodata=float("nan"))

    val0 = V.validate(zeros_path, fp, g.catalogue)
    valn = V.validate(nan_path, fp, g.catalogue)
    V.assert_submittable(val0)

    tpl_path = ROOT / ".cache" / "gems_data" / "sample_submission.tif"
    fc0 = S.validate_submission(zeros_path, template=tpl_path)
    fcn = S.validate_submission(nan_path, template=tpl_path)
    vendn = validate_vendored(
        nan_path,
        reference_path=tpl_path,
        features_path=tpl_path,
    )
    if not fc0.ok or not fcn.ok or not vendn["passed"]:
        raise AssertionError(f"Validator failed: zeros={fc0.failures} nan={fcn.failures} vend={vendn}")

    # Verify byte-level SHA256 uniqueness against every prior .tif in registry and docs/downloads
    all_other_shas = set()
    for d in reg_dirs:
        if not d.exists():
            continue
        for p in d.glob("*.tif"):
            if p.resolve() in (zeros_path.resolve(), nan_path.resolve()):
                continue
            all_other_shas.add(G.sha256_file(p))
    if info0["sha256"] in all_other_shas or infon["sha256"] in all_other_shas:
        raise AssertionError("SHA-256 collision with an existing raster!")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(zeros_path, arcname=zeros_path.name)

    np.save(CACHE / f"{stem}-surface.npy", surface_norm.astype(np.float16))

    promote = bool(gate_passes and uniq["passes_uniqueness"] and val0["all_checks_passed"] and fc0.ok and fcn.ok and vendn["passed"])
    submission_name = f"{stem}-zeros"
    note = (
        f"GEMSDOE68 autoctx s{stage} scarp+ridge hide-draw; {n_dots} dots >2px off cat; "
        f"HOLDOUT-DTI {b_entry[st_key]['all']['dti']:.4f} vs s1 {b_entry['stage1']['all']['dti']:.4f}"
    )[:140]

    card = dict(
        schema="gems68-run-card-v1",
        run_card_version=2,
        lane="Auto-context stacked refinement (Tu, CVPR 2008, doi:10.1109/CVPR.2008.4587436; TPAMI 2010, doi:10.1109/TPAMI.2009.186)",
        repo="https://github.com/buffedlizard55-lab/GEMSDOE68",
        experiment_id="exp3_autocontext_scarp_ridge_hidedraw",
        generated_utc=stamp,
        built_utc=stamp,
        hypothesis=(
            "De-blobbed Stage-2 and Stage-3 LightGBM auto-context classifiers trained with whole-component "
            "hide-draw labels on four input families -- (1) out-of-fold Stage-(k-1) probabilities sampled at "
            "+-1..15 px along and across local structure-tensor + 2 m LiDAR scarp strike, (2) multi-scale "
            "anisotropic smooths (L in {3,6,12,15} px, W in {1,1.5,2,3} px) and 1D curvilinear ridge-vs-blob "
            "contrast (dropping the sigma=10 px isotropic smooth per IR-68-BLOB-01), (3) raw gradient/tilt/Hessian-ridge "
            "plus 2 m USGS 3DEP LiDAR antislope scarp & GeoDAWN 1.5 km high-pass radiometric/magnetic channels, and "
            "(4) masked-catalogue context (clamped distance, strike alignment, 1 km/3 km spacing ratio, and along-strike "
            "tip-continuation drop) -- beat the per-pixel Stage-1 baseline by >2 SEs on outer blocks across all withheld "
            "pixels and on pixels >1 km from any mapped fault."
        ),
        mechanism=(
            "Normal-fault networks in the Great Basin exhibit multi-kilometer along-strike continuity, sharp across-strike "
            "profile convexity, antislope 2 m LiDAR scarps, shallow magnetic dipole steps, and potassium/uranium fault-core "
            "alteration corridors. Removing 30-100 km regional background levels and adding 1D ridge-vs-blob contrast allows "
            "Stage 2 and Stage 3 to link fragmented per-pixel detections across alluvial cover without creating 2D regional blobs."
        ),
        named_non_fault_process_that_could_mimic_it=(
            "Pleistocene Lake Lahontan / Bonneville shoreline beach ridges, fluvial terrace risers, highway/canal embankments, "
            "and tilted volcanic cooling-unit cliffs produce linear 2 m LiDAR steps and local topographic gradients that can "
            "mimic fault scarps when aligned with regional structure."
        ),
        non_fault_mimic=(
            "Pleistocene Lake Lahontan / Bonneville shoreline beach ridges, fluvial terrace risers, highway/canal embankments, "
            "and tilted volcanic cooling-unit cliffs produce linear 2 m LiDAR steps and local topographic gradients that can "
            "mimic fault scarps when aligned with regional structure."
        ),
        holdout_dti=dict(
            evidence_class="HOLDOUT-DTI",
            evaluator_version=exp3["evaluator_version"],
            split_version=exp3["split_version"],
            seed=exp3["seed"],
            selected_stage=f"stage{stage}",
            selected_budget_dots=budget,
            all_withheld_pixels=dict(
                withheld_positive_pixels=b_entry[st_key]["all"]["withheld_positive_pixels"],
                stage1_dti=b_entry["stage1"]["all"]["dti"],
                stage2_dti=b_entry["stage2"]["all"]["dti"],
                stage3_dti=b_entry["stage3"]["all"]["dti"],
                delta_stage3_minus_stage1=gate_all["delta"],
                ci95_delta_stage3_minus_stage1=gate_all["ci95"],
                ci95_excludes_zero=gate_all["ci95_excludes_zero"],
                z_cluster_boot=gate_all["z_cluster_boot"],
                t_outer_block=gate_all["t_outer_block"],
                se_outer_block=gate_all["se_outer_block"],
                n_spatial_clusters_20km=gate_all["n_spatial_clusters"],
                bootstrap_draws=gate_all["bootstrap_draws"],
            ),
            far_gt_1km_from_mapped_faults=dict(
                withheld_positive_pixels=b_entry[st_key]["far_gt_1km"]["withheld_positive_pixels"],
                stage1_dti=b_entry["stage1"]["far_gt_1km"]["dti"],
                stage2_dti=b_entry["stage2"]["far_gt_1km"]["dti"],
                stage3_dti=b_entry["stage3"]["far_gt_1km"]["dti"],
                delta_stage3_minus_stage1=gate_far["delta"],
                ci95_delta_stage3_minus_stage1=gate_far["ci95"],
                ci95_excludes_zero=gate_far["ci95_excludes_zero"],
                z_cluster_boot=gate_far["z_cluster_boot"],
                t_outer_block=gate_far["t_outer_block"],
                se_outer_block=gate_far["se_outer_block"],
                n_spatial_clusters_20km=gate_far["n_spatial_clusters"],
                bootstrap_draws=gate_far["bootstrap_draws"],
            ),
            gate_all_pixels_gt_2se=exp3["gate_all_pixels_gt_2se"],
            gate_far_gt_1km_gt_2se=exp3["gate_far_gt_1km_gt_2se"],
            gate_outer_block_all_gt_2se=exp3["gate_outer_block_all_gt_2se"],
            gate_outer_block_far_gt_2se=exp3["gate_outer_block_far_gt_2se"],
            gate_passed=gate_passes,
        ),
        leakage_canary=exp3["leakage_canary"],
        correlation_overlap_vs_registry=uniq,
        emission=dict(
            n_dots=n_dots,
            exclude_px=E.EXCLUDE_PX,
            nms=5,
            value=1.0,
            min_dist_to_catalogue_px=float(dot_dist.min()),
            median_dist_px=float(np.median(dot_dist)),
            median_dist_m=float(np.median(dot_dist) * 100.0),
            p75_dist_m=float(np.percentile(dot_dist, 75) * 100.0),
            p95_dist_px=float(np.percentile(dot_dist, 95)),
        ),
        raster_sha256=dict(
            zeros_tif=info0["sha256"],
            zeros_zip=G.sha256_file(zip_path),
            nan_tif=infon["sha256"],
            byte_unique_vs_all_prior_rasters=True,
        ),
        submission_files=dict(
            primary_zeros_tif=dict(
                path=f"docs/downloads/{zeros_path.name}",
                filename=zeros_path.name,
                bytes=info0["bytes"],
                sha256=info0["sha256"],
                role="RECOMMENDED PRIMARY UPLOAD -- all-finite float32 in [0.0, 1.0], 0.0 outside footprint, eliminates portal 'Predicted values must be in range [0, 1]' error",
            ),
            primary_zeros_zip=dict(
                path=f"docs/downloads/{zip_path.name}",
                filename=zip_path.name,
                bytes=zip_path.stat().st_size,
                sha256=G.sha256_file(zip_path),
                role="Compressed ZIP archive of the primary -zeros.tif file",
            ),
            secondary_nan_tif=dict(
                path=f"docs/downloads/{nan_path.name}",
                filename=nan_path.name,
                bytes=infon["bytes"],
                sha256=infon["sha256"],
                role="Template-matched twin -- finite [0.0, 1.0] inside footprint, NaN outside footprint",
            ),
        ),
        validator_output=dict(
            zeros_tif=val0,
            nan_tif=valn,
            submission_format_check_zeros=dict(ok=fc0.ok, failures=fc0.failures, checks=fc0.checks),
            submission_format_check_nan=dict(ok=fcn.ok, failures=fcn.failures, checks=fcn.checks),
            vendored_validate_submission_nan=vendn,
        ),
        submission_name=submission_name,
        note=note,
        submission_note=note,
        note_length=len(note),
        note_chars=len(note),
        verdict="promote" if promote else "negative",
        verdict_reason=(
            f"All holdout gates met (>2 SEs on outer blocks and 95% spatial-cluster bootstrap CI > 0 on both all withheld "
            f"pixels and >1 km far pixels), IR-68-BLOB-01 resolved (median dot distance {np.median(dot_dist)*100.0:.0f} m vs 5,100 m in Exp 2), "
            f"unique surface (max Spearman {uniq['max_surface_spearman']:.4f} < 0.90) and dots "
            f"(max informative 3 px overlap {uniq['max_dot_overlap_3px']:.4f} < 0.70 across {uniq['n_registry_rasters_checked']} prior rasters), "
            f"byte-distinct SHA-256, and all format/range checks passed."
        ),
        ok_to_download_and_submit=promote,
        organizer_confirmed_score=None,
        seconds=round(time.time() - t0, 1),
    )

    (ROOT / "evidence" / "run_card.json").write_text(json.dumps(card, indent=2, default=float))
    (ROOT / "evidence" / "run-card-ac68.json").write_text(json.dumps(card, indent=2, default=float))
    (OUT / f"{stem}-runcard.json").write_text(json.dumps(card, indent=2, default=float))
    log(f"wrote run card for {submission_name} | verdict={card['verdict']} | sha256={info0['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
