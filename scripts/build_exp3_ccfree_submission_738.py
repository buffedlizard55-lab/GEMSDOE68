#!/usr/bin/env python3
"""Build the Exp3 (AC68-N1, catalogue-context-free) submission raster.

Stage/budget/tier come ONLY from evidence/exp3_decision.json (the preregistered
rule applied to the holdout); nothing is chosen here.  Full-data LOQO maps ->
shared emission rule -> streamed uniqueness vs the 60-raster in-repo registry
(surface rank-correlation BEFORE placement and final-dot overlap AFTER, both
literal and coverage-informative readings) -> all-finite zeros primary +
official-footprint NaN twin -> both shared validators -> run card.

The primary file is all-finite with nodata=None so the portal error
"Predicted values must be in range [0, 1]" is mathematically impossible on any
reading of the bytes (12,279,160 finite cells in [0,1] after re-read).
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
from gems import validate_submission as GV  # noqa: E402
from gems68 import autocontext_738 as AC, emission as E, grid as G, uniqueness as U, validate as V  # noqa: E402
from gems68.experiment_738 import Lane, log  # noqa: E402
from gems68.holdout import quadrant_ids  # noqa: E402

OUT = ROOT / "docs" / "downloads"
CACHE = ROOT / ".cache" / "final_exp3"
REGISTRY = ROOT / "registry" / "rasters" / "rasters"
DATA = ROOT / ".cache" / "gems_data"


def streamed_uniqueness(surface: np.ndarray, dots: np.ndarray, fp: np.ndarray) -> dict:
    """Per-raster application of gems68.uniqueness (60 full rasters do not fit
    resident together in 3.8 GB, so the registry dict is a singleton per file;
    surface_rank_corr re-derives the same seeded sample on every call)."""
    paths = sorted(REGISTRY.glob("*.tif"))
    rhos: dict[str, float] = {}
    overlaps: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    for i, p in enumerate(paths):
        with rasterio.open(p) as src:
            a = src.read(1).astype(np.float32)
        one = {p.name: a}
        rhos.update(U.surface_rank_corr(surface, one, fp))
        overlaps.update(U.dot_overlap(dots, one, fp))
        hashes[p.name] = G.sha256_file(p)
        del a, one
        if (i + 1) % 15 == 0:
            log(f"uniqueness {i + 1}/{len(paths)}")
    return dict(n_registry=len(paths), rhos=rhos, overlaps=overlaps, hashes=hashes)


def main() -> int:
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    decision = json.loads((ROOT / "evidence" / "exp3_decision.json").read_text())
    exp3 = json.loads((ROOT / "evidence" / "exp3_autocontext_ccfree.json").read_text())
    stage, budget = int(decision["chosen"]["stage"]), int(decision["chosen"]["budget"])
    tier = decision["tier_before_uniqueness_and_format"]
    arm = decision["arms"][f"{budget}_stage{stage}"]
    log(f"decision tier={tier} stage={stage} budget={budget}")

    g = G.load_grid()
    fp = g.footprint
    X1 = np.ascontiguousarray(np.load(ROOT / ".cache/features/X1.npy", mmap_mode="r"))
    phys = list(np.load(ROOT / ".cache/features/phys.npy"))
    maps_file = CACHE / f"final_maps_ccfree_s{stage}.npz"
    if maps_file.is_file():
        z = np.load(maps_file)
        maps = {int(k[1:]): z[k] for k in z.files}
        log("resumed final maps from cache")
    else:
        lane = Lane(g, X1, phys, quadrant_ids(fp), catalogue_context=False)
        maps = lane.final(stage)
        np.savez(maps_file, **{f"s{k}": v for k, v in maps.items()})
    del X1
    surface = AC.to_map(maps[stage], fp)

    allowed = ~E.exclusion_mask(g.catalogue) & fp
    dots = E.emit_dots(surface, allowed, budget)
    n_dots = int(dots.sum())
    from scipy import ndimage as ndi
    dcat = ndi.distance_transform_edt(~g.catalogue)
    dot_dist = dcat[dots]
    log(f"dots={n_dots} min_dist_cat_px={dot_dist.min():.2f} median={np.median(dot_dist):.2f}")

    log("streamed uniqueness vs in-repo registry (surface, then dots)...")
    u = streamed_uniqueness(surface, dots, fp)
    verdict_u = U.verdict(u["rhos"], u["overlaps"])
    literal_flagged = sorted(k for k, v in u["overlaps"].items() if v["overlap"] > 0.70)
    max_abs_rho = max(u["rhos"].items(), key=lambda kv: abs(kv[1]))
    champ_name = next((k for k in u["overlaps"] if "h33-2-b2" in k and "zeros" in k), None)
    log("uniqueness:", verdict_u["duplicate"], "max|rho|", max_abs_rho,
        "literal>0.70:", literal_flagged)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest8 = hashlib.sha256(np.packbits(dots).tobytes()).hexdigest()[:8]
    stem = f"gems68-ac-ccfree-s{stage}-{n_dots}dots-{stamp}-{digest8}"
    arr0 = np.where(dots, np.float32(1.0), np.float32(0.0)).astype(np.float32)
    info0 = G.write_geotiff(OUT / f"{stem}-zeros.tif", arr0, nodata=None)
    arrn = arr0.copy()
    arrn[~fp] = np.nan
    infon = G.write_geotiff(OUT / f"{stem}-nan.tif", arrn, nodata=float("nan"))
    del arrn

    # exact byte-duplicate check vs every prior raster we hold
    prior_hashes = dict(u["hashes"])
    for p in OUT.glob("*.tif"):
        if not p.name.startswith(stem):
            prior_hashes[f"docs/downloads/{p.name}"] = G.sha256_file(p)
    exact_matches = sorted(k for k, v in prior_hashes.items()
                           if v in (info0["sha256"], infon["sha256"]))

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

    primary_format_ok = bool(val0["all_checks_passed"])
    nan_official_ok = bool(
        valn["checks"]["in_footprint_all_finite"] and valn["checks"]["in_footprint_range_0_1"]
        and valn["checks"]["zero_dots_on_mapped_catalogue"]
        and not [c for c in gvn["checks"] if c["status"] == "FAIL"])
    informative_dup = bool(verdict_u["duplicate"]) or bool(exact_matches)
    if tier in ("promote", "promote-with-caveats") and primary_format_ok and nan_official_ok \
            and not informative_dup:
        verdict = tier
        download_cleared = True
    else:
        verdict = "negative"
        download_cleared = False
    reasons = []
    if tier == "negative":
        reasons.append("preregistered holdout gate not met")
    if not primary_format_ok:
        reasons.append("primary zeros file failed gems68 local preflight")
    if not nan_official_ok:
        reasons.append("NaN twin failed official-footprint checks")
    if informative_dup:
        reasons.append("duplicate/lane-drift vs informative registry rasters")
    if verdict == "promote-with-caveats":
        reasons.append("bootstrap gates pass both strata but the n=4 per-fold >2SE rule fails on at "
                       "least one stratum (low power); selector must read the caveats")

    note = (f"GEMSDOE68 auto-context CC-free s{stage}; {n_dots} dots >2px off-cat; "
            f"HOLDOUT-DTI {arm['dti_candidate']:.4f} vs s1 {arm['dti_stage1']:.4f}; UNSCORED")[:140]
    card = dict(
        schema="gemsdoe68-run-card/2", generated_utc=stamp,
        lane="auto-context stacked refinement (Tu, CVPR 2008, doi:10.1109/CVPR.2008.4587436)",
        hypothesis="AC68-N1: removing catalogue-context features from stage>=2 preserves/improves "
                   "auto-context gains >1 km from mapped faults, because proximity to the "
                   "incomplete published catalogue is a mapping artefact, not a property of "
                   "unmapped faults.",
        mechanism="stage>=2 LightGBM sees only orientation-sampled previous-stage probabilities "
                  "(+-1,3,7,15 px along/across local strike), isotropic smooths sigma=2/5/10, "
                  "anisotropy/coherence and 7 physics gradient/tilt/ridge channels; training "
                  "labels are hide-draw withheld components; every final pixel is predicted "
                  "out-of-fold (LOQO quadrants, 12 px exclusion band).",
        named_non_fault_process_that_could_mimic_it=
            "Range-front / basin-margin erosional escarpments and terraced alluvial margins: same "
            "detrended-elevation gradient, Laplacian and ridge responses as fault scarps, and "
            "continuity stacking can connect them into long lineaments.",
        holdout_dti=dict(
            evidence_class="HOLDOUT-DTI", evaluator_version=exp3["evaluator_version"],
            protocol="pooled DTI alpha=0.2 beta=0.8, 300 m triangular kernel; whole-component "
                     "hide-and-recover (buffered-whole-components-loqo-v2, eval seed 20)",
            withheld_positive_pixels=arm["withheld_positive_pixels"],
            stage=stage, budget=budget,
            pooled_all=dict(stage1=arm["dti_stage1"], candidate=arm["dti_candidate"],
                            delta=arm["delta_all"], ci95_delta=arm["ci95_all"]),
            pooled_far_gt_1km=dict(stage1=arm["far_dti_stage1"], candidate=arm["far_dti_candidate"],
                                   delta=arm["delta_far"], ci95_delta=arm["ci95_far"],
                                   withheld_positive_pixels_far=sum(
                                       exp3["results"][str(budget)]["per_fold"][f"{f}_stage1"]["far_n_truth"]
                                       for f in ("NW", "NE", "SW", "SE"))),
            bootstrap_gate_both_strata=arm["bootstrap_gate_both_strata"],
            per_fold_gate=arm["per_fold_gate"],
            historical_ccfull_comparator_exp2=arm["exp2_ccfull_same_instrument"],
            canary=decision["canary"],
            decision_file="evidence/exp3_decision.json",
            experiment_file="evidence/exp3_autocontext_ccfree.json"),
        correlation_overlap_vs_registry=dict(
            registry_dir="registry/rasters/rasters", n_registry=u["n_registry"],
            max_abs_spearman_surface=list(max_abs_rho),
            max_informative_dot_overlap_3px=list(verdict_u["max_dot_overlap"]),
            thresholds=verdict_u["thresholds"],
            excluded_uninformative_rasters=verdict_u["excluded_uninformative_rasters"],
            literal_overlap_gt_0p70_rasters=literal_flagged,
            champion_overlap=(dict(name=champ_name, **u["overlaps"][champ_name]) if champ_name else None),
            exact_byte_matches=exact_matches,
            informative_duplicate=informative_dup,
            literal_note="The literal >70% rule also fires for the family champion itself vs the "
                         "311,617-dot buried-step raster (Pass-1 measured fraction 0.9873; its 3-px "
                         "halo covers ~96% of the footprint). The informative verdict applies the "
                         "pre-existing IR-68-UNIQ-01 coverage rule; the literal firing is reported, "
                         "never hidden."),
        raster=dict(
            primary_path=f"docs/downloads/{stem}-zeros.tif", primary_sha256=info0["sha256"],
            primary_bytes=info0["bytes"], primary_kind="all-finite, 0 outside footprint, nodata=None",
            nan_twin_path=f"docs/downloads/{stem}-nan.tif", nan_twin_sha256=infon["sha256"],
            zip=f"{stem}-zeros.zip", zip_sha256=G.sha256_file(zip_path),
            dot_count=n_dots, value=1.0, exclude_px=E.EXCLUDE_PX, nms=5,
            min_dist_to_catalogue_px=float(dot_dist.min()),
            median_dist_to_catalogue_px=float(np.median(dot_dist))),
        validator_output=dict(
            gems68_validate_primary_zeros=val0, gems68_validate_nan_twin=valn,
            gems_validate_submission_primary_zeros=gv0,
            gems_validate_submission_nan_twin=gvn,
            interpretation="zeros primary passes the all-finite preflight and cannot trigger a "
                           "[0,1] range rejection; it fails only the other shared validator's "
                           "NaN-only-outside policy check by construction. The NaN twin is the "
                           "converse. Both full receipts are kept verbatim."),
        submission_name=f"{stem}-zeros", note=note, note_chars=len(note),
        verdict=verdict, verdict_reason="; ".join(reasons) or "all preregistered gates met",
        download_cleared=download_cleared,
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
            "Registry completeness is limited to the 60 rasters committed in registry/rasters/.",
            "Mirror data were hash-verified against registry/data_manifest.json pins; pins prove "
            "mirror consistency, not organiser authentication."],
        seconds=round(time.time() - t0, 1))
    (ROOT / "evidence" / "run_card_exp3.json").write_text(json.dumps(card, indent=1, default=float))
    (OUT / f"{stem}-runcard.json").write_text(json.dumps(card, indent=1, default=float))
    log("verdict:", verdict, "|", card["verdict_reason"])
    log("stem:", stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
