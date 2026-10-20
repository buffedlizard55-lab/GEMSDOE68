#!/usr/bin/env python3
"""Build ONE canonical, negative-by-default research run card from raw evidence.

Never self-promote: the documented selector and independent provenance/review
are separate. Historical cards may not supply missing H68-M1 evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence/ac68_m1_holdout.json"
SURFACE = ROOT / "evidence/ac68_m1_preplacement.json"
EMITTED = ROOT / "evidence/ac68_m1_candidate.json"
OUT = ROOT / "evidence/run-card-2026-10-20.json"


def digest(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def stats(group: dict) -> dict:
    return {key: {"value": item["dti"], "ci95": item["ci95"],
                  "withheld_positive_pixels": item["withheld_positive_pixels"]}
            for key, item in group["scores"].items()}


def gate(pair: dict, bootstrap_ci95: list[float]) -> dict:
    return {"delta_mean": pair["mean_delta"], "se": pair["se"],
            "bootstrap_paired_delta_ci95": bootstrap_ci95,
            "t_stat": pair["t_stat"], "n_outer": pair["n_blocks"],
            "per_outer_delta": pair["per_block"], "pass": pair["passes_gt_2se"],
            "rule": "positive mean > 2 × paired outer-block SE"}


def uniq_state(report: dict | None, label: str) -> dict:
    if report is None:
        return {"status": "NOT RUN — no new " + label + " candidate", "worst_raster": None,
                "comparison_count": 0, "complete_accessible_scan": False}
    rows = report.get("comparisons") or []
    flagged = [r for r in rows if r.get("drift")]
    worst = flagged[0] if flagged else max(rows, key=lambda r: abs(r.get("spearman_rho") or 0), default=None)
    return {"status": "FAIL — literal duplicate" if flagged else (
                    "ACCESSIBLE-SUBSET PASS; organizer completeness unknown"
                    if report.get("verdict", "").startswith("no drift") and report.get("complete_accessible_scan")
                    else "INCONCLUSIVE — no cleared full scan"),
            "verdict": report.get("verdict"),
            "worst_raster": (worst or {}).get("registry"),
            "worst_raster_sha256": (worst or {}).get("sha256"),
            "worst_abs_spearman": max((r["absolute_spearman_rho"] for r in rows
                                       if r["absolute_spearman_rho"] is not None), default=None),
            "worst_dot_overlap": max((r["candidate_dots_within_3px"] for r in rows
                                      if r["candidate_dots_within_3px"] is not None), default=None),
            "comparison_count": len(rows),
            "registry_rasters_supplied": report.get("rasters_supplied"),
            "complete_accessible_scan": report.get("complete_accessible_scan") is True,
            "raster_sha256": report.get("candidate_sha256")}


def build(source: dict, surface: dict | None, emitted: dict | None) -> dict:
    s = source["summary"]
    stage = stats(s["pooled_all"])
    stage_far = stats(s["pooled_far"])
    g_all = gate(s["paired_outer_all"]["stage2_vs_stage1"],
                 s["pooled_all"]["paired_differences"]["stage1"]["ci95"])
    g_far = gate(s["paired_outer_far"]["stage2_vs_stage1"],
                 s["pooled_far"]["paired_differences"]["stage1"]["ci95"])
    leak = s["canary"]
    high_auc = [r for r in leak["not_cleared"] if r["auc"] is not None]
    unavailable = [r for r in leak["not_cleared"] if r["auc"] is None]
    leak_status = "FAIL — AUC > 0.90" if high_auc else (
        "INCOMPLETE — non-finite channels" if unavailable else "PASS")
    passed = (g_all["pass"] is True and g_far["pass"] is True and
              leak_status == "PASS")
    rows = source.get("canary_results")
    full_canary = isinstance(rows, list) and len(rows) == leak["features_tested"]
    if full_canary and (len([r for r in rows if r.get("auc") is None]) != len(unavailable) or
                        len([r for r in rows if r.get("auc") is not None and r.get("flagged")]) != len(high_auc)):
        raise ValueError("per-feature canary list disagrees with aggregated leakage summary")
    validator = (emitted or {}).get("validator") or {}
    raster_path = ROOT / "evidence/candidates" / (emitted or {}).get("candidate", "NOT_PRESENT")
    real_raster = (raster_path.is_file() and emitted and emitted.get("candidate_sha256") == digest(raster_path))
    restrictions = []
    if not g_all["pass"]:
        restrictions.append("All-withheld paired stage-2 minus stage-1 gain did not exceed 2 outer-block SE.")
    if not g_far["pass"]:
        restrictions.append(">1 km paired stage-2 minus stage-1 gain did not exceed 2 outer-block SE; no density/power exception.")
    if high_auc:
        restrictions.append(f"{len(high_auc)} feature/fold canaries exceeded the absolute-direction AUC .90 leakage stop.")
    if unavailable:
        restrictions.append(f"{len(unavailable)} feature/fold canaries contained non-finite values, so an AUC and complete individual-feature safety check are unavailable; this is NOT evidence of AUC > .90.")
    if not full_canary:
        restrictions.append("The complete per-feature AUC list is missing from the experiment artifact; aggregate canary counts alone cannot clear a release.")
    for label, report in (("pre-placement surface", surface), ("final-dot raster", emitted)):
        if report is None:
            restrictions.append(f"{label} uniqueness was NOT RUN because the scientific gate stopped candidate generation.")
        elif not report.get("verdict", "").startswith("no drift") or not report.get("complete_accessible_scan"):
            restrictions.append(f"{label} literal originality gate negative or incomplete; see its comparison report.")
    if not validator.get("passed"):
        restrictions.append("No strict official-grid/footprint format validator pass on a new raster.")
    if emitted and not real_raster:
        restrictions.append("Candidate file SHA does not match the final-scan artifact (or file absent).")
    restrictions.extend((
        "Owner-mirror SHA pins do not authenticate original DrivenData files, native 1 m DEM, or a cultural mask.",
        "Accessible prior rasters are not certified as the complete organizer-wide submission registry.",
        "No independent scientific-lead review or weekly-slot selector approval; no organizer receipt or private score.",
    ))
    card = {
        "schema_version": "gems68-strict-run-card-v1", "experiment_id": "H68-M1",
        "created_utc": source["generated_utc"],
        "hypothesis": source["hypothesis"],
        "mechanism": "Magnetic-susceptibility discontinuity aligned auto-context may bridge buried faults missed by mapped traces.",
        "named_non_fault_mimics": ["magnetic lithologic contacts", "east-west airborne-survey striping"],
        "evidence_class": "HOLDOUT-DTI (not organizer-confirmed)",
        "holdout_source": str(SOURCE.relative_to(ROOT)), "holdout_source_sha256": digest(SOURCE),
        "evaluator_version": source["evaluator_version"], "evaluator_hashes": source["evaluator_hashes"],
        "emitter_version": source.get("emitter_version"),
        "emitter_sha256": source.get("emitter_sha256"),
        "metric": source["metric"], "split": source["split"],
        "outer_blocks": len(source["folds"]), "inner_blocks_per_outer": 4,
        "withheld_positive_pixels": source["n_withheld"],
        "withheld_positive_pixels_far_gt_1km": source["n_withheld_far"],
        "stage_dti": stage, "stage_dti_far": stage_far,
        "paired_gate": {"all": g_all, "far_gt_1km": g_far, "scientific_gate_pass": passed},
        "leakage_canary": {"status": leak_status,
                           "feature_fold_checks_attempted": leak["features_tested"],
                           "max_auc_measurable": leak["max_directional_auc"],
                           "count_over_0_90": len(high_auc),
                           "count_non_finite": len(unavailable),
                           "flagged_or_undefined": leak["not_cleared"],
                           "full_feature_auc_list_in_report": full_canary},
        "per_outer_fold": [{"fold": f["fold"], "n_truth": f["n_truth"],
                            "n_truth_far": f["n_truth_far"], "quota": f["quota"],
                            "dti": {k: v["dti"] for k, v in f["dti"].items()},
                            "dti_far": {k: v["dti"] for k, v in f["dti_far"].items()}}
                           for f in source["folds"]],
        "uniqueness": {"surface": uniq_state(surface, "surface"),
                       "final_dots": uniq_state(emitted, "dot")},
        "format": {"status": ("PASS STRICT LOCAL (portal not tested)" if validator.get("passed")
                              else ("FAILED STRICT LOCAL; no cleared candidate" if validator
                                    else "NOT RUN — stopped before candidate")),
                   "strict_validator_passed": validator.get("passed") is True,
                   "outside_fill": "nan" if real_raster else None,
                   "validator_report": validator if validator else None},
        "raster": {"path": str(raster_path.relative_to(ROOT)) if real_raster else None,
                   "sha256": emitted["candidate_sha256"] if real_raster else None,
                   "evidence_only": True},
        "name": None, "note": None,
        "input_provenance_verified": False, "organizer_confirmed": None,
        "selector_status": "not requested", "download_allowed": False,
        "verdict": "negative", "restrictions": restrictions,
        "budget": {"preregistered_hypotheses_tested": 1,
                   "first_execution_stopped_before_scoring": True,
                   "provisional_pre_plateau_safe_nms_archived": True,
                   "fit_attempts": 3, "scored_attempts": 2,
                   "submission_slots_used": 0,
                   "measured_fit_seconds": source["elapsed_seconds"],
                   "max_experiments": 3, "max_elapsed_seconds": 7200},
        "review_passes": {"iterative_passes_planned": 3,
                          "independent_scientific_leads": 0},
        "caveat": source["caveat"],
    }
    return card


def main() -> int:
    if not SOURCE.is_file():
        raise SystemExit("strict experiment has not produced a result; do not create a synthetic score")
    source = json.loads(SOURCE.read_text())
    surface = json.loads(SURFACE.read_text()) if SURFACE.is_file() else None
    emitted = json.loads(EMITTED.read_text()) if EMITTED.is_file() else None
    card = build(source, surface, emitted)
    OUT.write_text(json.dumps(card, indent=2, allow_nan=False) + "\n")
    print("H68-M1:", card["evidence_class"], "stage1", card["stage_dti"]["stage1"],
          "stage2", card["stage_dti"]["stage2"], "gate", card["paired_gate"],
          "verdict", card["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
