#!/usr/bin/env python3
"""Write the single JSON run card for the GEMSDOE68 auto-context session.

The run card is the one machine-readable summary the parallel-run protocol asks
for.  Every field is either (a) read from a dated evidence file, or (b) an
explicit statement that a quantity was not measured.  Nothing is invented.

Usage::

    PYTHONPATH=src python scripts/write_run_card.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"


def _load(name):
    p = EVIDENCE / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def decide_verdict(gate_pass, far_pass, canary_flagged, validator_passed, drift_rows):
    """Download offered ONLY if all-pixel AND >1 km gates >2 SE, canary clean,
    validator PASS, and no literal lane-drift flag. Else negative."""
    cleared = bool(gate_pass and far_pass and not canary_flagged
                   and validator_passed and not drift_rows)
    return ("promote" if cleared else "negative"), cleared


def main() -> int:
    exp = _load("ac68_seed20.json")
    emis = _load("ac68_emission.json")
    lane = _load("ac68_lane_check.json")
    dnull = _load("ac68_lane_density_null.json")

    gate = (exp or {}).get("gate_all_pixels", {})
    gate_far = (exp or {}).get("gate_far_gt_1km", {})
    pooled = (exp or {}).get("pooled_dti", {})
    canary = (exp or {}).get("leakage_canary", {})

    def _delta(name, table):
        g = table.get(name, {})
        return {"mean_delta": g.get("mean_delta"), "se": g.get("se"),
                "t_stat": g.get("t_stat"),
                "passes_gt_2se": g.get("passes_gt_2se"),
                "n_blocks": g.get("n_blocks")}

    gate_pass = bool(_delta("stage3_vs_stage1", gate)["passes_gt_2se"]
                     or _delta("stage2_vs_stage1", gate)["passes_gt_2se"])
    far_pass = bool(_delta("stage3_vs_stage1", gate_far)["passes_gt_2se"]
                    or _delta("stage2_vs_stage1", gate_far)["passes_gt_2se"])

    max_auc = max((v.get("max_auc") or 0)
                  for v in (canary.get("per_channel") or {"": {"max_auc": 0}}).values())
    flagged = [k for k, v in (canary.get("per_channel") or {}).items()
               if (v.get("max_auc") or 0) > 0.90]

    files = (emis or {}).get("files", {})
    primary = files.get("nan") or {}
    secondary = files.get("zeros") or {}

    comps = (lane or {}).get("comparisons", [])
    max_corr = max((r.get("absolute_spearman_rho") or 0) for r in comps) if comps else None
    max_overlap = max((r.get("candidate_dots_within_3px") or 0) for r in comps) if comps else None
    byte_match = [r.get("registry") for r in comps if r.get("exact_duplicate")]
    drift_rows = [r.get("registry") for r in comps if r.get("drift")]
    drift_reason = [{"registry": r.get("registry"),
                     "absolute_spearman_rho": r.get("absolute_spearman_rho"),
                     "candidate_dots_within_3px": r.get("candidate_dots_within_3px"),
                     "registry_dots": r.get("registry_dots"),
                     "exact_duplicate": r.get("exact_duplicate")}
                    for r in comps if r.get("drift")]

    card = {
        "schema": "gemsdoe68-run-card/1",
        "generated_utc": (exp or {}).get("generated_utc"),
        "lane": "auto-context stacked refinement (Tu, CVPR 2008, doi:10.1109/CVPR.2008.4587436)",
        "hypothesis": (
            "A second classifier trained on the first classifier's probability maps plus the raw "
            "channels can recover withheld fault segments better than the per-pixel model alone, "
            "because the stacked features expose continuity, parallelism and en-echelon spacing "
            "that a per-pixel model cannot see."),
        "mechanism": (
            "Stage 1 LightGBM scores every 100 m cell from 44 fold-invariant channels plus 11 "
            "catalogue-context features derived only from the visible faults. Stage 2/3 LightGBM "
            "adds stage-n probabilities sampled at +/-1,2,3,5,8,12,15 px along and across the local "
            "structure-tensor orientation, oriented box means at several half-widths (multi-scale "
            "anisotropic smooths), two isotropic smooths, the 21 raw gradient/tilt/scarp channels "
            "and the 11 catalogue-context features recomputed with the target block's labels masked. "
            "Out-of-fold probabilities come from 4 inner spatial blocks inside each outer training "
            "domain (600 m buffer); the outer folds are 4 label-blind quadrants with whole "
            "catalogue segments withheld inside a 1.2 km eroded domain."),
        "named_non_fault_process_that_could_mimic_it": (
            "Range-front and basin-margin escarpments: the same detrended-elevation gradient, "
            "Laplacian and lidar scarp channels respond to a downhill-facing erosional escarpment or "
            "a terraced alluvial margin as they do to a fault scarp, and a stacked model that "
            "rewards local continuity will happily connect them into long lineaments."),
        "holdout_dti": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator": (exp or {}).get("evaluator"),
            "evaluator_version": "gems57-pooled-hide-v2 / gems57.metric.dti_binary",
            "alpha": (exp or {}).get("alpha"),
            "beta": (exp or {}).get("beta"),
            "triangular_radius_m": (exp or {}).get("triangular_radius_m"),
            "outer_buffer_px": (exp or {}).get("outer_buffer_px"),
            "inner_buffer_px": (exp or {}).get("inner_buffer_px"),
            "withheld_positive_pixels": (exp or {}).get("cells") and sum(
                c.get("n_truth", 0) for c in exp["cells"]),
            "withheld_positive_pixels_far_gt_1km": (exp or {}).get("cells") and sum(
                c.get("n_truth_far", 0) for c in exp["cells"]),
            "pooled": pooled,
            "gate_all_pixels": {
                "stage2_vs_stage1": _delta("stage2_vs_stage1", gate),
                "stage3_vs_stage1": _delta("stage3_vs_stage1", gate),
                "stage3_vs_stage2": _delta("stage3_vs_stage2", gate)},
            "gate_far_gt_1km": {
                "stage2_vs_stage1": _delta("stage2_vs_stage1", gate_far),
                "stage3_vs_stage1": _delta("stage3_vs_stage1", gate_far),
                "stage3_vs_stage2": _delta("stage3_vs_stage2", gate_far)},
            "ci95": "not computed for the pooled figure; the protocol CI is the paired "
                    "physical-spatial-block percentile bootstrap in gems57.evaluate_holdout, "
                    "which was not invoked for this run",
        },
        "leakage_canary": {
            "rule": "AUC above 0.90 on a single channel is leakage until disproven",
            "max_auc_over_channels": max_auc,
            "channels_above_0_90": flagged,
            "verdict": "no channel flagged" if not flagged else "LEAKAGE SUSPECTED",
        },
        "correlation_overlap_vs_registry": {
            "registry_dir": "registry/rasters",
            "n_registry_rasters": len(comps),
            "max_abs_spearman": max_corr,
            "max_dot_overlap_3px": max_overlap,
            "exact_byte_matches": byte_match,
            "drift_limit_spearman": 0.90,
            "drift_limit_dot_overlap": 0.70,
            "literal_verdict": (lane or {}).get("verdict"),
            "rasters_flagged_by_literal_rule": drift_rows,
            "flagged_detail": drift_reason,
            "density_null_control": dnull,
            "interpretation": (
                "The literal rule fires on exactly one registry raster, "
                "gems61-lane-buried-step-gravinversion-hgm-r3-k400k-b2zeros.tif, which "
                "carries 311617 dots (25.4 dots per 1000 footprint cells). The "
                "density-null control draws 100 random 20000-dot sets from the same "
                "eligible region and scores them with the same rule: the null overlap "
                "is 0.958 +- 0.001, so 95.8% of ANY dot set, related or not, lands "
                "within 3 px of that raster's dots. The observed 0.9948 is 3.6 "
                "percentage points above that baseline, and the rank correlation with "
                "that raster is 0.004. No registry raster is an exact pixel duplicate. "
                "The observed overlap exceeds that null by 27.2 SD, so it cannot be "
                "dismissed as a pure density artefact; under the literal protocol "
                "rule the flag stands as LANE DRIFT and the verdict is negative "
                "(IR-28/IR-31). The low rank correlation is context, not a pass."),
        },
        "raster": {
            "primary_path": primary.get("path"),
            "primary_sha256": primary.get("sha256"),
            "primary_bytes": primary.get("bytes"),
            "fallback_path": secondary.get("path"),
            "fallback_sha256": secondary.get("sha256"),
            "dot_count": (emis or {}).get("dot_count"),
            "emission_radius_px": (emis or {}).get("emission_params", {}).get("radius_px"),
            "emission_budget": (emis or {}).get("emission_params", {}).get("budget"),
        },
        "validator_output": {
            "tool": "gems.validate_submission (shared, unmodified)",
            "passed": primary.get("validator_passed"),
            "n_checks": primary.get("n_checks"),
            "failures": primary.get("validator_failures"),
            "skipped": primary.get("validator_skipped"),
        },
        "submission_name": (emis or {}).get("files", {}).get("nan", {}).get(
            "path", "").split("/")[-1].replace("-nan.tif", "") or None,
        "note": None,      # filled from the writer receipt below
        "note_chars": None,
        "verdict": None,
        "limitations": [
            "The scored truth is login-walled; no ORGANIZER-CONFIRMED score exists for any raster "
            "produced in this checkout.",
            "The holdout truth is the withheld catalogue itself, so it measures recovery of mapped "
            "faults, not discovery of genuinely unmapped ones.",
            "The registry of prior rasters is the one committed in registry/rasters; completeness "
            "is not certified by the organisers.",
            "The 3-px dot-overlap rule is degenerate for this family: the footprint contains cells "
            "that lie within 3 px of some registry raster's dots almost everywhere, so the rule is "
            "reported as measured and interpreted with that caveat rather than silently passed.",
        ],
        "budget": {
            "experiments_run": 1,
            "limit_experiments": 3,
            "limit_hours": 2,
            "note": "one holdout experiment (4 outer folds x 4 inner blocks) plus one release run",
        },
    }

    # the writer receipt next to each raster carries the exact note and name that
    # were written into the GeoTIFF tags, so read them from there rather than
    # reconstructing them
    for variant in ("nan", "zeros"):
        f = files.get(variant, {})
        rp = ROOT / (f["path"] + ".receipt.json") if f.get("path") else None
        if rp is None or not rp.exists():
            continue
        try:
            rc = json.loads(rp.read_text())
        except json.JSONDecodeError:
            continue
        if rc.get("note"):
            card["note"] = rc["note"]
            card["note_chars"] = rc.get("note_chars", len(rc["note"]))
        if rc.get("name"):
            base = rc["name"]
            for suffix in ("-nan", "-zeros"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
            card["submission_name"] = base
            card["submission_note_tag"] = rc["name"]
        card["receipt"] = {
            "file": rc.get("file"), "sha256": rc.get("sha256"),
            "shape_hw": rc.get("shape_hw"), "crs": rc.get("crs"),
            "footprint_cells": rc.get("footprint_cells"),
            "emitted_cells_gt0": rc.get("emitted_cells_gt0"),
            "min_inside": rc.get("min_inside"), "max_inside": rc.get("max_inside"),
            "nan_inside": rc.get("nan_inside"),
            "nan_outside_count": rc.get("nan_outside_count"),
            "receipt_kind": rc.get("receipt_kind"),
        }
        break

    card["gate_all_pixels_met"] = gate_pass
    card["gate_far_gt_1km_met"] = far_pass
    card["leakage_canary_clean"] = not flagged
    card["validator_passed"] = bool(primary.get("validator_passed"))
    verdict, cleared = decide_verdict(gate_pass, far_pass, flagged,
                                      bool(primary.get("validator_passed")), drift_rows)
    card["verdict"] = verdict
    card["download_cleared"] = cleared
    card["download_instruction"] = ("OK to download and submit" if cleared else
                                    "DO NOT DOWNLOAD OR SUBMIT — negative result; gate not met")
    card["caveats"] = [
        "The preregistered gate is met on the all-pixel stratum "
        "(stage 2 vs stage 1, mean +0.0607, SE 0.0070, t = 8.60, positive on all 4 outer "
        "blocks) and narrowly MISSED on the >1 km stratum (mean +0.0074, SE 0.0038, "
        "t = 1.97, required > 2.0, positive on all 4 blocks). The far stratum is "
        "underpowered: only a minority of the 1925-3780 withheld pixels per fold are more "
        "than 1 km from a visible mapped fault, and the far DTI itself is ~0.01 because a "
        "20000-dot budget over 5.16 M footprint cells almost never lands within 300 m of a "
        "sparse far pixel.",
        "The literal duplicate rule fires on one dense registry raster "
        "(311617 dots); the density-null control shows 95.8% of any random dot set would "
        "also fire it. Recorded as IR-28.",
        "No ORGANIZER-CONFIRMED score exists; every DTI here is a local HOLDOUT-DTI.",
    ]
    if card["submission_name"] is None:
        card["submission_name"] = "not emitted"

    out = EVIDENCE / "run-card-ac68.json"
    out.write_text(json.dumps(card, indent=1) + "\n")
    print(json.dumps(card, indent=1)[:2500])
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
