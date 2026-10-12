"""Fail-closed research-to-download gate. This is NOT the competition's upload validator.

A local format preflight or a subset registry scan can never be called release clearance.
This gate combines *recorded* scientific decisions with on-disk format and uniqueness
receipts. Human review must authenticate the inputs, registry certificate and reviewers;
boolean fields alone do not prove any scientific claim. See README.md release protocol.
"""
from __future__ import annotations

from pathlib import Path
import math

from .submission_writer import sha256_file


def _positive_gain(record: object) -> bool:
    """Require a paired, positive, uncertainty-bounded gain (not a projection)."""
    if not isinstance(record, dict):
        return False
    ci = record.get("ci95")
    try:
        value = float(record["value"])
        lo, hi = map(float, ci)
        return all(math.isfinite(x) for x in (value, lo, hi)) and 0 < lo <= value <= hi
    except (TypeError, ValueError, KeyError):
        return False


def assess_release(candidate: str | Path, card: dict, validator: dict,
                   surface: dict, emitted: dict, surface_sha256: str | None = None) -> dict:
    """Check documentary and mechanical prerequisites; never infer missing evidence as pass.

    The candidate is checked by file SHA-256; the continuous surface requires a separate
    pinned SHA-256 and a completed *pre-placement* audit. The returned verdict must be
    reviewed by a scientific lead before any file is copied into a public downloads path.
    """
    path = Path(candidate)
    reasons: list[str] = []
    raster = card.get("raster") or {}
    submission = card.get("submission") or {}
    holdout = card.get("holdout_dti") or {}
    canary = card.get("leakage_canary") or {}
    reviews = card.get("independent_reviews") or {}
    provenance = card.get("input_provenance") or {}
    budget = card.get("budget") or {}

    if card.get("promote") is not True or card.get("promotion_decision") != "promote":
        reasons.append("selector has not promoted the candidate")
    expected = raster.get("sha256")
    digest = sha256_file(path) if path.is_file() else None
    if not expected or digest != expected or raster.get("file") != path.name or submission.get("file") != path.name:
        reasons.append("candidate file/name/SHA-256 does not match the canonical run card")
    required = {"single band", "dtype float32", "CRS EPSG:32611", "shape matches",
                "resolution 100 m", "transform matches reference", "CRS matches reference",
                "values within [0, 1]", "no NaN inside footprint",
                "NaN-only outside footprint", "features grid shape matches",
                "features transform matches reference grid", "features CRS matches reference grid"}
    passed_checks = {c.get("check") for c in validator.get("checks", []) if c.get("status") == "PASS"}
    if not validator.get("passed") or validator.get("sha256") != digest or \
            validator.get("skipped_checks_mean_unverified") or \
            any(c.get("status") not in ("PASS", "INFO") for c in validator.get("checks", [])) or \
            not required.issubset(passed_checks):
        reasons.append("official reference/features format and footprint validation is incomplete or failed")
    if validator.get("file") is None or Path(validator["file"]).resolve() != path.resolve():
        reasons.append("validator report does not identify the candidate file")

    for label, report, phase in (("pre-placement surface", surface, "surface"),
                                 ("final dots", emitted, "emitted")):
        rho = report.get("worst_abs_spearman_full_footprint")
        overlap = report.get("worst_dot_overlap")
        expected_n = report.get("registry_rasters_expected")
        if report.get("unique") is not True or report.get("organizer_registry_complete") is not True \
                or report.get("complete_accessible_scan") is not True \
                or report.get("stop_required") is not False or report.get("phase") != phase \
                or not isinstance(expected_n, int) or expected_n <= 0 \
                or report.get("registry_rasters_checked") != expected_n \
                or report.get("missing_or_invalid") != 0 or report.get("source_errors") \
                or report.get("duplicate_count") != 0 \
                or report.get("byte_unique_among_checked") is not True \
                or report.get("pixel_unique_among_checked") is not True \
                or not isinstance(rho, (int, float)) or not math.isfinite(rho) or rho > .90 \
                or (phase == "emitted" and (not isinstance(overlap, (int, float))
                    or not math.isfinite(overlap) or overlap > .70)):
            reasons.append(f"{label} has no complete-registry uniqueness pass")
    if not surface_sha256 or surface_sha256 != raster.get("surface_sha256") or \
            surface.get("candidate_file_sha256") != surface_sha256:
        reasons.append("pre-placement surface audit is not tied to a pinned surface file")
    if emitted.get("candidate_file_sha256") != digest or digest is None:
        reasons.append("final-dot audit is not tied to the submitted candidate bytes")

    try:
        value = float(holdout["value"])
        lo, hi = map(float, holdout["ci95"])
        n = int(holdout["withheld_positive_count"])
        science_ok = (holdout.get("label") == "HOLDOUT-DTI" and
                      bool(holdout.get("evaluator_version")) and n > 0 and
                      all(math.isfinite(x) for x in (value, lo, hi)) and
                      0 <= lo <= value <= hi <= 1)
    except (KeyError, TypeError, ValueError):
        science_ok = False
    if not science_ok or not _positive_gain(holdout.get("gain_vs_frozen_best")) or \
            not _positive_gain(holdout.get("off_range_front_gain")):
        reasons.append("qualified HOLDOUT-DTI, paired gain over frozen best and off-front gain with 95% CIs missing")

    features = canary.get("feature_results")
    if not isinstance(features, list) or not features or any(
        not isinstance(f, dict) or not f.get("name") or not isinstance(f.get("auc"), (int, float))
        or not math.isfinite(f["auc"]) or f["auc"] < 0 or f["auc"] > 0.90 for f in features
    ) or canary.get("status") != "PASS":
        reasons.append("feature-alone leakage canary incomplete or AUC > 0.90")
    reviewer_ids = reviews.get("reviewer_ids")
    if not isinstance(reviewer_ids, list) or len(reviewer_ids) < 2 or \
            not all(isinstance(i, str) and i.strip() for i in reviewer_ids) or \
            len(set(str(i) for i in reviewer_ids)) < 2 or \
            not isinstance(reviews.get("completed"), int) or reviews["completed"] < 2 or \
            reviews.get("status") != "PASS":
        reasons.append("two independent scientific-lead reviews not documented")
    if not all(provenance.get(k) is True for k in
               ("official_grid_verified", "native_1m_dem_verified", "cultural_mask_verified")):
        reasons.append("official grid/native DEM/cultural mask provenance not verified")
    if not isinstance(budget.get("new_experiments_used"), int) or \
            not 0 <= budget["new_experiments_used"] <= 3 or \
            budget.get("submission_slots_used") not in (0, None):
        reasons.append("research budget or unused submission slot not established")
    name, note = submission.get("name"), submission.get("note")
    if not isinstance(name, str) or not name.strip() or not isinstance(note, str) or len(note) > 140:
        reasons.append("unique submission name or <=140-character note absent")
    return {"cleared_for_review": not reasons, "download_allowed": False,
            "candidate_sha256": digest, "blockers": reasons,
            "meaning": "Passing mechanical checks only permits an independent release review; it never automatically publishes or submits."}
