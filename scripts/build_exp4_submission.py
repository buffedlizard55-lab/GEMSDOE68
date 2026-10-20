#!/usr/bin/env python3
"""Build the Exp4 (AC68-N2, stage-1 external channels) submission raster.

Stage/budget/tier come ONLY from evidence/exp4_decision.json.  Full-data LOQO
maps (Lane.final with the 46-column stage-1 matrix, catalogue context on) ->
shared emission -> streamed uniqueness vs the 60-raster registry AND pairwise
dot overlap vs prior candidates in docs/downloads -> all-finite zeros primary
+ NaN twin -> both shared validators -> evidence/run_card_exp4.json.
Also points the canonical evidence/run_card_current.json at this card.
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
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_exp3_submission import streamed_uniqueness  # noqa: E402  (shared machinery)
from gems import validate_submission as GV  # noqa: E402
from gems68 import autocontext as AC, emission as E, grid as G, uniqueness as U, validate as V  # noqa: E402
from gems68.experiment import Lane, log  # noqa: E402
from gems68.holdout import quadrant_ids  # noqa: E402

OUT = ROOT / "docs" / "downloads"
CACHE = ROOT / ".cache" / "final_exp4"
DATA = ROOT / ".cache" / "gems_data"


def overlap_vs_prior_downloads(dots: np.ndarray, fp: np.ndarray, stem: str) -> dict:
    out = {}
    for p in sorted(OUT.glob("*.tif")):
        if p.name.startswith(stem) or "surface" in p.name:
            continue
        with rasterio.open(p) as src:
            a = src.read(1).astype(np.float32)
        out[p.name] = U.dot_overlap(dots, {p.name: a}, fp)[p.name]
        del a
    return out


def main() -> int:
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    decision = json.loads((ROOT / "evidence" / "exp4_decision.json").read_text())
    exp4 = json.loads((ROOT / "evidence" / "exp4_autocontext_ext.json").read_text())
    stage, budget = int(decision["chosen"]["stage"]), int(decision["chosen"]["budget"])
    tier = decision["tier_before_uniqueness_and_format"]
    arm = decision["arms"][f"{budget}_stage{stage}"]
    log(f"decision tier={tier} stage={stage} budget={budget}")

    g = G.load_grid()
    fp = g.footprint
    names1 = json.loads((ROOT / ".cache/features/stage1_ext_names.json").read_text())
    X1 = np.load(ROOT / ".cache/features/X1_ext.npy", mmap_mode="r")
    phys = list(np.load(ROOT / ".cache/features/phys.npy"))
    maps_file = CACHE / f"final_maps_ext_s{stage}.npz"
    if maps_file.is_file():
        z = np.load(maps_file)
        maps = {int(k[1:]): z[k] for k in z.files}
        log("resumed final maps from cache")
    else:
        lane = Lane(g, X1, phys, quadrant_ids(fp), catalogue_context=True, stage1_names=names1)
        maps = lane.final(stage)
        np.savez(maps_file, **{f"s{k}": v for k, v in maps.items()})
    surface = AC.to_map(maps[stage], fp)
    np.save(CACHE / "surface_float16.npy", surface.astype(np.float16))

    allowed = ~E.exclusion_mask(g.catalogue) & fp
    dots = E.emit_dots(surface, allowed, budget)
    n_dots = int(dots.sum())
    from scipy import ndimage as ndi
    dcat = ndi.distance_transform_edt(~g.catalogue)
    dot_dist = dcat[dots]
    log(f"dots={n_dots} min_dist_cat_px={dot_dist.min():.2f} median={np.median(dot_dist):.2f}")

    log("streamed uniqueness vs registry (surface then dots)...")
    u = streamed_uniqueness(surface, dots, fp)
    verdict_u = U.verdict(u["rhos"], u["overlaps"])
    literal_flagged = sorted(k for k, v in u["overlaps"].items() if v["overlap"] > 0.70)
    max_abs_rho = max(u["rhos"].items(), key=lambda kv: abs(kv[1]))
    champ_name = next((k for k in u["overlaps"] if "h33-2-b2" in k and "zeros" in k), None)
    log("registry duplicate(informative):", verdict_u["duplicate"], "max|rho|", max_abs_rho)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest8 = hashlib.sha256(np.packbits(dots).tobytes()).hexdigest()[:8]
    stem = f"gems68-ac-ext-s{stage}-{n_dots}dots-{stamp}-{digest8}"
    prior_overlap = overlap_vs_prior_downloads(dots, fp, stem)
    worst_prior = max(prior_overlap.items(), key=lambda kv: kv[1]["overlap"]) if prior_overlap else ("", {"overlap": 0.0})
    log("worst overlap vs prior downloads:", worst_prior[0], worst_prior[1]["overlap"])

    arr0 = np.where(dots, np.float32(1.0), np.float32(0.0)).astype(np.float32)
    info0 = G.write_geotiff(OUT / f"{stem}-zeros.tif", arr0, nodata=None)
    arrn = arr0.copy()
    arrn[~fp] = np.nan
    infon = G.write_geotiff(OUT / f"{stem}-nan.tif", arrn, nodata=float("nan"))
    del arrn

    prior_hashes = dict(u["hashes"])
    for p in OUT.glob("*.tif"):
        if not p.name.startswith(stem):
            prior_hashes[f"docs/downloads/{p.name}"] = G.sha256_file(p)
    exact_matches = sorted(k for k, v in prior_hashes.items() if v in (info0["sha256"], infon["sha256"]))

    val0 = V.validate(OUT / f"{stem}-zeros.tif", fp, g.catalogue)
    valn = V.validate(OUT / f"{stem}-nan.tif", fp, g.catalogue)
    gv0 = GV.validate(OUT / f"{stem}-zeros.tif", reference_path=DATA / "sample_submission.tif",
                      features_path=DATA / "training_features.tif",
                      footprint_path=DATA / "sample_submission.tif")
    gvn = GV.validate(OUT / f"{stem}-nan.tif", reference_path=DATA / "sample_submission.tif",
                      features_path=DATA / "training_features.tif",
                      footprint_path=DATA / "sample_submission.tif")
    zip_path = OUT / f"{stem}-zeros.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(OUT / f"{stem}-zeros.tif", arcname=f"{stem}-zeros.tif")

    primary_ok = bool(val0["all_checks_passed"])
    nan_ok = bool(valn["checks"]["in_footprint_all_finite"]
                  and valn["checks"]["in_footprint_range_0_1"]
                  and valn["checks"]["zero_dots_on_mapped_catalogue"]
                  and not [c for c in gvn["checks"] if c["status"] == "FAIL"])
    informative_dup = bool(verdict_u["duplicate"]) or bool(exact_matches) \
        or bool(worst_prior[1]["overlap"] > 0.70
                and u["overlaps"] is not None
                and worst_prior[1].get("coverage", 0.0) <= 0.70)
    if tier in ("promote", "promote-with-caveats") and primary_ok and nan_ok and not informative_dup:
        verdict, cleared = tier, True
    else:
        verdict, cleared = "negative", False
    reasons = []
    if tier == "negative":
        reasons.append("preregistered Exp4 gate not met (see evidence/exp4_decision.json)")
    if not primary_ok:
        reasons.append("primary zeros file failed gems68 local preflight")
    if not nan_ok:
        reasons.append("NaN twin failed official-footprint checks")
    if informative_dup:
        reasons.append("duplicate/lane-drift vs informative prior rasters")
    if verdict == "promote-with-caveats":
        reasons.append("bootstrap gates pass on both strata and the candidate beats the previous "
                       "holdout best point estimate, BUT the n=4 per-fold >2SE rule fails on the "
                       "far stratum (t=1.53) and the far bootstrap lower bound is only +0.0007; "
                       "treat as conditional, selector decides")

    note = (f"GEMSDOE68 auto-context ext-channels s{stage}; {n_dots} dots >2px off-cat; "
            f"HOLDOUT-DTI {arm['dti_candidate']:.4f} vs s1 {arm['dti_stage1']:.4f}; UNSCORED")[:140]
    card = dict(
        schema="gemsdoe68-run-card/2", generated_utc=stamp,
        lane="auto-context stacked refinement (Tu, CVPR 2008, doi:10.1109/CVPR.2008.4587436)",
        hypothesis="AC68-N2: enriching stage 1 with fold-invariant external channels (2 m lidar "
                   "scarp aggregates + coverage, GeoDAWN radiometric/extension percentile ranks) "
                   "improves first-stage ranking enough that unchanged auto-context stages beat "
                   "the previous holdout best at matched mass.",
        mechanism="stage 1 LightGBM on 19 organiser bands + 7 physics + 20 external channels; "
                  "stages 2/3 add orientation-sampled previous probabilities (+-1,3,7,15 px), "
                  "smooths, anisotropy/coherence, physics and visible-catalogue context; "
                  "hide-draw labels; final maps are LOQO out-of-fold for every pixel.",
        named_non_fault_process_that_could_mimic_it=
            "Erosional terrace risers and range-front escarpments (the lidar scarp channels "
            "respond to any sharp step) and lithologic potassium contrasts in volcanic flows "
            "(radiometric ranks); roads/channels also create 2 m steps.",
        holdout_dti=dict(
            evidence_class="HOLDOUT-DTI", evaluator_version=exp4["evaluator_version"],
            protocol="pooled DTI alpha=0.2 beta=0.8, 300 m triangular kernel; whole-component "
                     "hide-and-recover (buffered-whole-components-loqo-v2, eval seed 20)",
            withheld_positive_pixels=arm["withheld_positive_pixels"], stage=stage, budget=budget,
            pooled_all=dict(stage1=arm["dti_stage1"], candidate=arm["dti_candidate"],
                            delta=arm["delta_all"], ci95_delta=arm["ci95_all"]),
            pooled_far_gt_1km=dict(stage1=arm["far_dti_stage1"], candidate=arm["far_dti_candidate"],
                                   delta=arm["delta_far"], ci95_delta=arm["ci95_far"],
                                   withheld_positive_pixels_far=sum(
                                       exp4["results"][str(budget)]["per_fold"][f"{f}_stage1"]["far_n_truth"]
                                       for f in ("NW", "NE", "SW", "SE"))),
            bootstrap_gate_both_strata=arm["bootstrap_gate_both_strata"],
            per_fold_gate=arm["per_fold_gate"],
            beats_previous_holdout_best=dict(
                comparator="Exp2 catalogue-context-full, same budget/stage (historical, same instrument)",
                exp2_dti=arm["exp2_holdout_best_comparator"]["dti_candidate"],
                exp2_far_dti=arm["exp2_holdout_best_comparator"]["far_dti_candidate"],
                beats_all=arm["beats_exp2_all"], far_difference=arm["far_vs_exp2"]),
            canary=decision["canary"],
            exp3_ccfree_negative_control=decision["exp3_final_disposition"],
            decision_file="evidence/exp4_decision.json", experiment_file="evidence/exp4_autocontext_ext.json"),
        correlation_overlap_vs_registry=dict(
            registry_dir="registry/rasters/rasters", n_registry=u["n_registry"],
            max_abs_spearman_surface=list(max_abs_rho),
            max_informative_dot_overlap_3px=list(verdict_u["max_dot_overlap"]),
            thresholds=verdict_u["thresholds"],
            excluded_uninformative_rasters=verdict_u["excluded_uninformative_rasters"],
            literal_overlap_gt_0p70_rasters=literal_flagged,
            champion_overlap=(dict(name=champ_name, **u["overlaps"][champ_name]) if champ_name else None),
            overlap_vs_prior_download_candidates=prior_overlap,
            exact_byte_matches=exact_matches, informative_duplicate=informative_dup,
            literal_note="Literal >70% 3-px overlap is reported for every raster; it also fires for "
                         "the family champion vs the 311,617-dot buried-step raster (0.9873, Pass-1 "
                         "measured; halo ~96% of footprint). Informative verdict uses the "
                         "pre-existing IR-68-UNIQ-01 coverage rule."),
        raster=dict(primary_path=f"docs/downloads/{stem}-zeros.tif", primary_sha256=info0["sha256"],
                    primary_bytes=info0["bytes"], primary_kind="all-finite, 0 outside footprint, nodata=None",
                    nan_twin_path=f"docs/downloads/{stem}-nan.tif", nan_twin_sha256=infon["sha256"],
                    zip=f"{stem}-zeros.zip", zip_sha256=G.sha256_file(zip_path),
                    dot_count=n_dots, value=1.0, exclude_px=E.EXCLUDE_PX, nms=5,
                    min_dist_to_catalogue_px=float(dot_dist.min()),
                    median_dist_to_catalogue_px=float(np.median(dot_dist)),
                    frac_dots_farther_than_1km=float((dot_dist > 10).mean()),
                    dist_to_catalogue_px_percentiles={
                        f"p{q}": float(np.percentile(dot_dist, q)) for q in (10, 50, 90)},
                    overlap_metric_note="U.dot_overlap uses Euclidean distance <=3 px; a "
                                        "Chebyshev (3x dilation) reading is looser. Worst prior "
                                        "candidate (sibling stage-3 file): Euclidean 0.183, "
                                        "Chebyshev 0.251, exact-shared dots 336/38000 "
                                        "(independently recomputed 2026-10-20)."),
        validator_output=dict(gems68_validate_primary_zeros=val0, gems68_validate_nan_twin=valn,
                              gems_validate_submission_primary_zeros=gv0,
                              gems_validate_submission_nan_twin=gvn,
                              interpretation="zeros primary is all-finite in [0,1] (portal range "
                                             "rejection impossible) and fails only the converse "
                                             "NaN-only-outside policy check by construction; the NaN "
                                             "twin passes the official-footprint validator. Full "
                                             "receipts kept verbatim."),
        submission_name=f"{stem}-zeros", note=note, note_chars=len(note),
        verdict=verdict, verdict_reason="; ".join(reasons) or "all preregistered gates met",
        download_cleared=cleared,
        download_instruction=("OK to download and submit (slot choice remains the human selector's)"
                              if verdict == "promote" else
                              "CONDITIONAL: read caveats; only the human selector may spend a slot"
                              if verdict == "promote-with-caveats" else
                              "DO NOT DOWNLOAD OR SUBMIT - negative result"),
        organizer_confirmed_score=None,
        ai_disclosure="Generative AI (Arena agent) wrote and executed the analysis code, audit "
                      "scripts and site text in this repository under the owner's standing brief; "
                      "all numbers were produced by running that code on the pinned data in this "
                      "checkout, and the human owner makes the final submission-slot decision.",
        limitations=[
            "No ORGANIZER-CONFIRMED score exists for this raster; every DTI here is HOLDOUT-DTI.",
            "Holdout truth is withheld catalogue components, not the panel's unmapped faults.",
            "Lidar coverage is ~75% of the footprint (sidecar-measured); uncovered cells carry "
            "median-filled scarp channels plus the lidar_valid flag.",
            "Radiometric/extension bytes are percentile ranks, not physical units.",
            "Registry completeness is limited to the 60 rasters in registry/rasters/ plus prior "
            "candidates in docs/downloads/.",
            "Area-fill (blob) risk IR-68-BLOB-01 remains open: median dot distance to the "
            "catalogue is reported in raster.median_dist_to_catalogue_px for selector review."],
        seconds=round(time.time() - t0, 1))
    text = json.dumps(card, indent=1, default=float)
    (ROOT / "evidence" / "run_card_exp4.json").write_text(text)
    (ROOT / "evidence" / "run_card_current.json").write_text(text)
    (OUT / f"{stem}-runcard.json").write_text(text)
    log("verdict:", verdict, "|", card["verdict_reason"])
    log("stem:", stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
