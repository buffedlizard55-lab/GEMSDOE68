#!/usr/bin/env python3
"""Private H68-M1 candidate path — only if *both* preregistered holdout gates pass.

Continuous surface is screened before placement; emitted candidate screened
separately. Stop on the first literal duplicate. This never copies a TIFF into
GitHub Pages or uses a DrivenData slot. No parameter optimization is allowed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gems import submission_writer as SW, validate_submission as V  # noqa: E402
from gems57 import grid as G57                                     # noqa: E402
from gems68 import channel_stack as F, emission as E              # noqa: E402
from lane68_strict_experiment import BUDGET, EXCLUDE_PX, orientation  # noqa: E402
from lane68_emit_submission import build_final_stack              # noqa: E402
from lane68_check import lane_check                               # noqa: E402


REPORT = ROOT / "evidence/ac68_m1_holdout.json"
SURFACE_REPORT = ROOT / "evidence/ac68_m1_preplacement.json"
FINAL_REPORT = ROOT / "evidence/ac68_m1_candidate.json"
NAME = "GEMSDOE68-H68M1-magtangent-20261020-stage2-38k"
NOTE = "Magnetic-edge tangent auto-context; 38k dots; buffered OOF stage2; no organizer score"


def prior_rasters() -> list[Path]:
    pinned = json.loads((ROOT / "registry/data_manifest.json").read_text())
    scored = [ROOT / ".cache/gems_data" / e["dest"] for e in pinned["files"]
              if e["group"] == "scored" and e["dest"].endswith(".tif")]
    for path in scored:
        if not path.is_file():
            raise ValueError(f"missing prior raster from manifest: {path}; run restore_data.py --group all")
    paths = (list((ROOT / "registry/rasters").rglob("*.tif")) +
             list((ROOT / "evidence/archives").rglob("*.tif")) + scored)
    if len(paths) < 60:
        raise ValueError("no credible prior-raster corpus; cannot pass an originality gate")
    return sorted({p.resolve() for p in paths})


def main() -> int:
    if not REPORT.is_file():
        raise SystemExit("strict preregistered holdout absent: no candidate generated")
    science = json.loads(REPORT.read_text())
    gate = science["summary"]["gate"]
    if not gate.get("meets_protocol"):
        print("STOP: preregistered stage-2 gate or individual canary failed; NO CANDIDATE GENERATED.")
        return 2
    if len(NOTE) > 140:
        raise AssertionError("note too long")
    registry = prior_rasters()
    grid = G57.load_grid()
    names, stack, fp = F.load_channels()
    c, s, _ = orientation(stack, F.channel_index(), fp)
    print(f"Candidate computation: {len(names)} pinned channels, {len(registry)} prior rasters", flush=True)
    fields = build_final_stack(grid, stack, F.channel_index(), fp, c, s,
                               neg_ratio=3, max_stage=2)
    p2 = np.asarray(fields["p2"], dtype=np.float32)
    if (not np.isfinite(p2[grid.footprint]).all() or
            np.nanmin(p2[grid.footprint]) < 0 or np.nanmax(p2[grid.footprint]) > 1):
        raise ValueError("full-field probability outside [0,1] or non-finite inside official footprint")
    surface_path = ROOT / "outputs/candidates/H68-M1-preplacement-continuous.tif"
    surface_path.parent.mkdir(parents=True, exist_ok=True)
    SW.write_submission(surface_path, p2, grid.footprint,
                        ROOT / "data/bridge/sample_submission.tif", overwrite=True,
                        name=NAME + "-surface", note=NOTE)
    print("Checking continuous surface before dot placement (no model tuning):", flush=True)
    surface = lane_check(surface_path, registry, stage="surface", stop_on_first_drift=True)
    SURFACE_REPORT.write_text(json.dumps(surface, indent=2, allow_nan=False) + "\n")
    print("surface", surface["verdict"], "checked", surface["rasters_checked"],
          "/", surface["rasters_supplied"], flush=True)
    if not surface["verdict"].startswith("no drift") or not surface["complete_accessible_scan"]:
        print("STOP: continuous surface is duplicate/inconclusive; no dots placed.", flush=True)
        return 3

    eligible = grid.footprint & (ndi.distance_transform_edt(~grid.catalogue) > EXCLUDE_PX)
    dots = E.emit_dots(p2, eligible, BUDGET, nms=5)
    if (dots > 0).sum() != BUDGET:
        raise ValueError("unexpected emitted budget; do not post-hoc adjust placement")
    out = ROOT / "evidence/candidates" / f"{NAME}-UNRELEASED-nan.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    receipt = SW.write_submission(out, dots.astype(np.float32), grid.footprint,
                                  ROOT / "data/bridge/sample_submission.tif",
                                  overwrite=True, name=NAME, note=NOTE,
                                  outside_fill="nan")
    strict = V.validate(out, ROOT / "data/bridge/sample_submission.tif",
                        ROOT / "data/bridge/training_features.tif")
    # A continuous prior is rank/hash comparable, but 5 million positive
    # surface cells must not be misrepresented as emitted fault dots.
    surfaces = [p for p in registry if "surface" in p.name.lower()]
    final = lane_check(out, registry, stage="emitted", skip_dot_overlap_for=surfaces,
                       stop_on_first_drift=True)
    final["validator"] = strict
    final["writer_receipt"] = receipt
    final["surface_candidate_sha256"] = surface["candidate_sha256"]
    final["surfaces_without_dot_overlap"] = [str(p) for p in surfaces]
    final["not_organizer_authenticated"] = True
    final["not_a_public_download"] = True
    FINAL_REPORT.write_text(json.dumps(final, indent=2, allow_nan=False) + "\n")
    print("final", final["verdict"], "validator", strict["passed"],
          "checked", final["rasters_checked"], "/", final["rasters_supplied"], flush=True)
    if not strict["passed"] or not final["verdict"].startswith("no drift") or \
            not final["complete_accessible_scan"]:
        print("STOP: final candidate is NOT CLEARED and must not be downloaded or submitted.", flush=True)
        return 4
    print("Mechanical checks passed in an accessible subset, but organizer completeness, "
          "independent review and selector approval remain outstanding; NOT CLEARED.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
