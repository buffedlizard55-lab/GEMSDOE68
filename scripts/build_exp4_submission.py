#!/usr/bin/env python3
"""Exp-4 release build.  Reads the mechanical decision in evidence/exp4_stratified.json
(Exp-3, evidence/exp3_ridgecrest.json, was negative; its cached outer maps are reused).

If the decision is negative: write the negative run card and NO GeoTIFF.
If promote: full-catalogue LOQO maps (Lane.final), emit the chosen arm at 38,000 dots
with >2 px catalogue exclusion, then:
  * PRIMARY  -zeros.tif : all-finite {0,1}, 0 outside the sample footprint, no NaN
    anywhere (cannot trip any "[0,1]" range implementation; the 0.2778 champion is
    itself a -zeros file).  Validated with gems68.validate (sample footprint).
  * SECONDARY -nan.tif  : same dots, NaN outside the sample footprint only.
  * BOTH also run through the shared gems.validate_submission; every check result is
    recorded verbatim in the run card (its footprint is training_features band 1,
    3,061 cells smaller than the sample footprint -- see IR log).
  * Lane check (streaming) vs registry/rasters/rasters/*.tif AND docs/downloads/*.tif.
Writes evidence/run_card_exp4.json (+ copy next to the downloads).
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
from gems.validate_submission import validate as shared_validate  # noqa: E402
from gems68 import autocontext as AC, emission as E, grid as G, uniqueness as U, validate as V  # noqa: E402
from gems68 import features as F  # noqa: E402
from gems68.experiment import Lane, log  # noqa: E402
from gems68.holdout import quadrant_ids  # noqa: E402

OUT = ROOT / "docs" / "downloads"
CACHE = ROOT / ".cache" / "exp4final"
BUDGET = 38_000


def emit_chosen(arm: str, maps: dict, lane_phys, fp, allowed, catalogue):
    """Final-map emission.  V6/V7 = Exp-4 stratified spacing with zones split at
    10 px from the FULL published catalogue and quotas proportional to zone
    allowed-area over the whole footprint (the full-map analogue of the holdout
    per-region rule in scripts/run_exp4.py)."""
    stage = int(arm[1])
    surface = AC.to_map(maps[stage], fp)
    variant = arm[3:]
    if variant == "V0":
        return E.emit_dots(surface, allowed, BUDGET), surface
    if variant not in ("V6", "V7"):
        raise ValueError(arm)
    zone_far = ndi.distance_transform_edt(~np.asarray(catalogue, bool)) > 10
    area_far = int((allowed & zone_far).sum())
    area_near = int((allowed & ~zone_far).sum())
    n_far = int(round(BUDGET * area_far / max(area_far + area_near, 1)))
    quotas = {0: BUDGET - n_far, 1: n_far}
    cand = allowed if variant == "V6" else E.crest_mask(surface, allowed)
    dots = E.greedy_spacing_stratified(surface, allowed, cand, zone_far.astype(np.int8), quotas)
    return dots, surface


def density_null(dots: np.ndarray, registry_row: dict, allowed: np.ndarray, fp, n_draws=50, seed=7):
    """Random same-size dot sets from the allowed region scored against the flagged
    registry raster's 3 px halo (same rule as scripts/lane68_density_null.py)."""
    reg_path = next((ROOT / "registry/rasters/rasters").glob(registry_row["name"]))
    import rasterio
    with rasterio.open(reg_path) as s:
        a = s.read(1)
    near = ndi.distance_transform_edt(~(np.isfinite(a) & (a > 0))) <= 3.0
    pool = np.flatnonzero((allowed & fp).ravel())
    rng = np.random.default_rng(seed)
    n = int(dots.sum())
    vals = []
    near_flat = near.ravel()
    for _ in range(n_draws):
        pick = rng.choice(pool, size=n, replace=False)
        vals.append(float(near_flat[pick].mean()))
    vals = np.asarray(vals)
    obs = registry_row["dot_overlap_3px"]
    return dict(registry=registry_row["name"], observed=obs, null_draws=n_draws,
                null_mean=float(vals.mean()), null_sd=float(vals.std()),
                null_p95=float(np.percentile(vals, 95)),
                observed_minus_null_sd=float((obs - vals.mean()) / max(vals.std(), 1e-12)))


def main() -> int:
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    exp3 = json.loads((ROOT / "evidence/exp3_ridgecrest.json").read_text())
    exp4 = json.loads((ROOT / "evidence/exp4_stratified.json").read_text())
    dec = exp4["decision"]
    arm = dec["chosen_arm"]
    base_card = dict(
        schema="gemsdoe68-run-card/exp4", generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        lane="auto-context stacked refinement (Tu, CVPR 2008, doi:10.1109/CVPR.2008.4587436)",
        hypothesis="H68-G: auto-context dots emitted with exact >=2.8 px spacing AND near/far zone quotas proportional to allowed area (zone split 10 px from the visible catalogue) beat both the per-pixel stage 1 and the previous best (stage-3 5x5-NMS) at matched mass in every stratum, because spacing removes redundant kernel coverage while stratification restores the spatial diversification that pure value-greedy packing destroys (Exp-3 measured far allocation collapsing 52%->22%; the champion allocates ~70% beyond 1 km).",
        mechanism="Stage 1 LightGBM (19 GeoDAWN bands + 7 gradient/tilt/scarp physics channels) out-of-fold by spatial quadrant; stages 2-3 LightGBM on previous-stage probabilities sampled +-1..15 px along/across local orientation, multi-scale smooths, physics channels and visible-catalogue context, trained hide-draw (positives = whole components withheld from the context catalogue). Exp-3/Exp-4 change only emission (see knowledge/02, knowledge/03).",
        named_non_fault_process_that_could_mimic_it="Erosional escarpment and Lake Lahontan shoreline-terrace crests, plus bedding-parallel ridges in tilted volcanics: all produce across-strike probability crests and scarp-like gradient/tilt responses without being active faults.",
        evidence_class="HOLDOUT-DTI", organizer_confirmed_score=None,
        decision=dec, replication=exp3["replication"])
    if dec["verdict"] != "promote" or arm is None:
        card = {**base_card, "verdict": "negative", "download_cleared": False,
                "download_instruction": "DO NOT DOWNLOAD OR SUBMIT — Exp-4 gate not met; no new raster was built.",
                "raster": None}
        (ROOT / "evidence/run_card_exp4.json").write_text(json.dumps(card, indent=1, default=float))
        log("negative: no raster built")
        return 0

    stage = int(arm[1])
    g = G.load_grid()
    fp = g.footprint
    X1 = np.ascontiguousarray(np.load(ROOT / ".cache/features/X1.npy", mmap_mode="r"))
    phys = list(np.load(ROOT / ".cache/features/phys.npy"))
    maps_file = CACHE / f"final_maps_s{stage}.npz"
    if maps_file.is_file():
        z = np.load(maps_file)
        maps = {int(k[1:]): z[k] for k in z.files}
        log("resumed final maps")
    else:
        lane = Lane(g, X1, phys, quadrant_ids(fp))
        maps = lane.final(stage)
        np.savez(maps_file, **{f"s{k}": v for k, v in maps.items()})
        del lane
    del X1

    allowed = ~E.exclusion_mask(g.catalogue) & fp
    dots, surface = emit_chosen(arm, maps, phys, fp, allowed, g.catalogue)
    n_dots = int(dots.sum())
    dcat = ndi.distance_transform_edt(~g.catalogue)[dots]
    log(f"arm={arm} dots={n_dots} min_dcat={dcat.min():.3f} median_dcat={np.median(dcat):.1f}")

    # lane check: committed registry + every prior candidate in docs/downloads
    reg_paths = sorted((ROOT / "registry/rasters/rasters").glob("*.tif"))
    prior_paths = sorted(OUT.glob("*.tif"))
    rows_reg = U.check_against_paths(surface, dots, reg_paths, fp)
    rows_prior = U.check_against_paths(surface, dots, prior_paths, fp)
    log(f"lane check done: {len(rows_reg)} registry + {len(rows_prior)} prior candidates")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest8 = hashlib.sha256(np.packbits(dots).tobytes()).hexdigest()[:8]
    tag = {"V0": "v0nms5", "V6": "v6strat28", "V7": "v7stratcrest28"}[arm[3:]]
    stem = f"gems68-ac{stage}-{tag}-{n_dots}dots-{stamp}-{digest8}"
    arr0 = np.where(dots, np.float32(1.0), np.float32(0.0)).astype(np.float32)
    info0 = G.write_geotiff(OUT / f"{stem}-zeros.tif", arr0, nodata=None)
    arrn = arr0.copy()
    arrn[~fp] = np.nan
    infon = G.write_geotiff(OUT / f"{stem}-nan.tif", arrn, nodata=float("nan"))
    zip_path = OUT / f"{stem}-zeros.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(OUT / f"{stem}-zeros.tif", arcname=f"{stem}-zeros.tif")

    data = ROOT / ".cache/gems_data"
    val0_local = V.validate(OUT / f"{stem}-zeros.tif", fp, g.catalogue)
    valn_local = V.validate(OUT / f"{stem}-nan.tif", fp, g.catalogue)
    val0_shared = shared_validate(OUT / f"{stem}-zeros.tif", data / "sample_submission.tif", data / "training_features.tif")
    valn_shared = shared_validate(OUT / f"{stem}-nan.tif", data / "sample_submission.tif", data / "training_features.tif")

    def lane_summary(rows):
        same = [r for r in rows if r.get("same_grid")]
        worst_rho = max(same, key=lambda r: r["abs_spearman"]) if same else None
        worst_ov = max(same, key=lambda r: r["dot_overlap_3px"]) if same else None
        return dict(n=len(rows), n_same_grid=len(same), worst_abs_spearman=worst_rho, worst_dot_overlap=worst_ov,
                    literal_breaches=[r["name"] for r in same if r["abs_spearman"] > 0.90 or r["dot_overlap_3px"] > 0.70],
                    rows=rows)

    ls_reg, ls_prior = lane_summary(rows_reg), lane_summary(rows_prior)
    nulls = []
    for r in rows_reg:
        if r.get("same_grid") and r["dot_overlap_3px"] > 0.70:
            nulls.append(density_null(dots, r, allowed, fp))
    hash_clash = [r["name"] for r in rows_reg + rows_prior if r["sha256"] in (info0["sha256"], infon["sha256"])]
    rho_breach = [r["name"] for r in rows_reg + rows_prior if r.get("same_grid") and r["abs_spearman"] > 0.90]
    nondeg_ov_breach = [r["name"] for r in rows_reg
                        if r.get("same_grid") and r["dot_overlap_3px"] > 0.70 and r["registry_coverage_3px"] <= 0.70]
    mech_ok = bool(val0_local["all_checks_passed"] and not hash_clash and not rho_breach and not nondeg_ov_breach)
    cleared = bool(mech_ok)  # holdout gate already decided in exp3 decision

    hold = exp4["results"]["38000"]["scopes"]
    note = (f"GEMSDOE68 autoctx s{stage} {arm[3:]} stratified 2.8px spacing; {n_dots} dots >2px off cat; "
            f"HOLDOUT-DTI {hold['all']['scores'][arm]['dti']:.4f} vs s1 {hold['all']['scores']['s1_V0']['dti']:.4f}; UNSCORED")[:140]
    card = {**base_card,
            "verdict": "promote" if cleared else "negative",
            "download_cleared": cleared,
            "download_instruction": ("OK TO DOWNLOAD the -zeros.tif primary and submit (slot choice remains the selector's step)."
                                     if cleared else "DO NOT DOWNLOAD OR SUBMIT — a mechanical/uniqueness check failed; see run card."),
            "chosen_arm": arm, "budget_requested": BUDGET,
            "holdout_dti": dict(
                evidence_class="HOLDOUT-DTI", evaluator_version=exp4["evaluator_version"],
                withheld_positive_pixels=exp4["withheld_positive_pixels"],
                budget=38000,
                all={a: hold["all"]["scores"][a] for a in ("s1_V0", "s3_V0", arm)},
                far={a: hold["far"]["scores"][a] for a in ("s1_V0", "s3_V0", arm)},
                farfull={a: hold["farfull"]["scores"][a] for a in ("s1_V0", "s3_V0", arm)},
                paired_vs_s1_all=hold["all"]["paired_vs"][arm]["s1_V0"],
                paired_vs_s1_far=hold["far"]["paired_vs"][arm]["s1_V0"],
                paired_vs_current_best_all=hold["all"]["paired_vs"][arm].get("s3_V0")),
            "leakage_canary": dict(max_abs_auc_deviation=max(max(v["auc"], 1 - v["auc"]) for v in exp3["canary"].values()),
                                   flagged=[k for k, v in exp3["canary"].items() if v["flag"]]),
            "emission": dict(rule=arm, n_dots=n_dots, exclude_px=2.0, min_dist_to_catalogue_px=float(dcat.min()),
                             median_dist_to_catalogue_px=float(np.median(dcat)),
                             p95_dist_to_catalogue_px=float(np.percentile(dcat, 95))),
            "correlation_overlap_vs_registry": dict(registry=ls_reg, prior_downloads=ls_prior,
                                                    density_nulls=nulls, exact_hash_clashes=hash_clash,
                                                    rho_breaches=rho_breach, nondegenerate_overlap_breaches=nondeg_ov_breach),
            "raster": dict(primary=dict(name=f"{stem}-zeros.tif", **{k: info0[k] for k in ("sha256", "bytes")},
                                        zip=zip_path.name, zip_sha256=G.sha256_file(zip_path)),
                           secondary_nan=dict(name=f"{stem}-nan.tif", **{k: infon[k] for k in ("sha256", "bytes")})),
            "validator_output": dict(primary_local_sample_footprint=val0_local, secondary_local_sample_footprint=valn_local,
                                     primary_shared_features_footprint=val0_shared, secondary_shared_features_footprint=valn_shared),
            "submission_name": f"{stem}-zeros", "note": note, "note_chars": len(note),
            "seconds": round(time.time() - t0, 1)}
    (ROOT / "evidence/run_card_exp4.json").write_text(json.dumps(card, indent=1, default=float))
    (OUT / f"{stem}-runcard.json").write_text(json.dumps(card, indent=1, default=float))
    log(f"verdict={card['verdict']} cleared={cleared} primary={stem}-zeros.tif sha={info0['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
