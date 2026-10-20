"""A failed preregistered gate cannot start a full-data fit or emit a TIFF."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import lane68_m1_candidate as candidate  # noqa: E402


def test_negative_gate_stops_before_loading_training_data(tmp_path, monkeypatch):
    holdout = tmp_path / "failed-holdout.json"
    holdout.write_text(json.dumps({"summary": {"gate": {"meets_protocol": False}}}))
    monkeypatch.setattr(candidate, "REPORT", holdout)
    monkeypatch.setattr(candidate, "SURFACE_REPORT", tmp_path / "surface.json")
    monkeypatch.setattr(candidate, "FINAL_REPORT", tmp_path / "dots.json")
    monkeypatch.setattr(candidate, "prior_rasters", lambda: (_ for _ in ()).throw(
        AssertionError("no registry scan or model fit after negative holdout")))
    assert candidate.main() == 2
    assert not list(tmp_path.glob("*.tif"))
    assert not (tmp_path / "surface.json").exists()
    assert not (tmp_path / "dots.json").exists()
