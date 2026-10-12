#!/usr/bin/env python3
"""Emit the single JSON run card required by the parallel-run protocol (item 5)."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
ROOT = Path(__file__).resolve().parents[1]
def load(n):
    p = ROOT / "evidence" / n
    return json.loads(p.read_text()) if p.exists() else None
hold, build = load("holdout.json"), load("submission_build.json")
gate = (hold or {}).get("gate", {})
fc = (build or {}).get("outputs", {}).get("zeros", {}).get("format_check", {})
reg = (build or {}).get("registry_verdict", {})
if build is None:
    verdict = "not_run"
elif not gate:
    verdict = "negative"          # gate never evaluated -> cannot promote
else:
    verdict = "promote" if (gate.get("passes") and fc.get("ok") and reg.get("lane_ok")) else "negative"
card = dict(
    schema=1, generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    repo="buffedlizard55-lab/GEMSDOE68", lane="auto-context stacked refinement",
    hypothesis=("A second classifier trained on the first classifier's out-of-fold probability "
                "map plus the original channels learns fault-scale spatial regularities "
                "(continuity along strike, parallelism, en echelon spacing) that a per-pixel "
                "model cannot represent, and places its top-ranked mass closer to unmapped "
                "faults per unit of emitted mass."),
    mechanism=("Stage 1: per-pixel LightGBM on 47 GeoDAWN-derived channels, spatially blocked "
               "buffered K-fold, out-of-fold only. Stage 2: LightGBM on (a) stage-1 probability "
               "sampled at +-1,2,4,7,11,15 px along and across the structure-tensor strike, "
               "(b) oriented means at 6 and 18 px along and across, (c) the raw gradient/tilt/"
               "scarp channels, (d) catalogue context (log distance, 3 px and 10 px kernel "
               "weights, along/across alignment and 5/20 px density of the nearest mapped "
               "fault). Emission: local-maximum dots, 2.4 px minimum separation, 40,000 budget, "
               "hard exclusion within 200 m of the mapped catalogue."),
    mimicking_non_fault_process=("Linear magnetic and gravity lineaments produced by dyke "
                                 "swarms, basalt flow fronts, lithologic contacts and alluvial-fan "
                                 "edges, plus flight-line artefacts in the airborne GeoDAWN "
                                 "grids, all produce straight, parallel, en echelon anomalies "
                                 "with no fault. The structure tensor cannot distinguish a fault "
                                 "fabric from a dyke fabric; only the catalogue-context term and "
                                 "the topographic scarp/tilt channels separate them."),
    holdout_dti=dict(
        evaluator=(hold or {}).get("evaluator", "not_run"),
        withheld_positives=(hold or {}).get("withheld_positives_total"),
        withheld_segments=(hold or {}).get("withheld_segments_total"),
        n_folds=(hold or {}).get("n_folds"),
        stage1=(hold or {}).get("stage1_pooled", {}).get("dti"),
        stage2=(hold or {}).get("stage2_pooled", {}).get("dti"),
        stage2_far1km=(hold or {}).get("stage2_pooled_far1km", {}).get("dti"),
        delta=gate.get("delta"), se=gate.get("se"), z=gate.get("z"),
        ci95=None if gate.get("se") is None else [round(gate["delta"] - 1.96 * gate["se"], 6),
                                                  round(gate["delta"] + 1.96 * gate["se"], 6)],
        label="HOLDOUT-DTI on a catalogue proxy (labels.tif is byte-identical to "
              "existing_faults.tif, IR-68-02). NOT organizer-confirmed.",
        leakage_canary_max_auc=((hold or {}).get("canary_top20") or [{}])[0].get("auc"),
        leakage_canary_flagged=len((hold or {}).get("canary_flags") or [])),
    registry=dict(worst_spearman=reg.get("worst_spearman"), worst_overlap3px=reg.get("worst_overlap"),
                  n_rasters_compared=len((build or {}).get("registry") or []),
                  lane_ok=reg.get("lane_ok"),
                  note="api.github.com rate-limited the shared token, so no registry rasters "
                       "could be fetched this session (IR-68-06); the lane check is INCOMPLETE."),
    raster_sha256=(build or {}).get("outputs", {}).get("zeros", {}).get("sha256"),
    validator=fc,
    submission_name=(build or {}).get("base_name"),
    submission_note=(build or {}).get("note"),
    verdict=verdict,
    verdict_reason=("no holdout gate evaluation completed this session, so the promotion "
                    "condition (stage 2 beats stage 1 by > 2 SE on outer blocks, including "
                    "> 1 km from any mapped fault) is NOT met; and the lane duplicate check "
                    "could not be run. The raster is published for audit as a negative result.")
        if verdict != "promote" else "gate met, validator passed, lane unique",
)
(ROOT / "evidence" / "run_card.json").write_text(json.dumps(card, indent=1))
print(json.dumps(dict(verdict=verdict, sha256=card["raster_sha256"]), indent=1))
