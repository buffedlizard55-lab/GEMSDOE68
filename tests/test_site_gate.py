"""A negative research result cannot be published via any current or legacy URL."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_site_verified import build, site_release_allowed  # noqa: E402


def test_negative_card_hides_every_download_link(tmp_path):
    card = {"verdict": "negative", "experiment_id": "H68-M1",
            "paired_gate": {"all": {"delta_mean": .01, "se": .004, "t_stat": 2.5, "pass": True},
                            "far_gt_1km": {"delta_mean": -.01, "se": .003,
                                           "t_stat": -3.3, "pass": False}},
            "restrictions": ["far gate did not pass"],
            "raster": {"path": "docs/downloads/stale-DO-NOT-SUBMIT.tif"}}
    (tmp_path / "docs" / "downloads").mkdir(parents=True)
    (tmp_path / "docs" / "downloads" / "stale-DO-NOT-SUBMIT.tif").write_bytes(b"old")
    pages = build(card, {"observed_utc_date": "2026-10-20", "sources": []},
                  {"observed_utc_date": "2026-10-20",
                   "url": "https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/",
                   "top_public": {"participant": "observed", "displayed_score": .4413}},
                  [], root=tmp_path)
    assert not site_release_allowed(card, tmp_path)
    assert "DO NOT SUBMIT" in pages["index.html"]
    for name, page in pages.items():
        assert 'href="downloads/' not in page, name
        assert '.tif"' not in page, name
        assert "Verdict: promote-with-caveats" not in page, name
    assert "HISTORICAL PAGE WITHDRAWN" in pages["lane68/submission-guide.html"]


def test_manual_flag_is_not_enough_to_override_shared_gate(tmp_path):
    path = tmp_path / "docs" / "downloads" / "candidate.tif"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a valid raster")
    card = {"verdict": "promote", "selector_status": "approved", "download_allowed": True,
            "name": "new", "note": "short", "input_provenance_verified": True,
            "paired_gate": {"all": {"pass": True}, "far_gt_1km": {"pass": True}},
            "leakage_canary": {"status": "PASS", "max_auc_measurable": .5,
                               "count_non_finite": 0, "count_over_0_90": 0,
                               "full_feature_auc_list_in_report": True},
            "uniqueness": {"surface": {"status": "PASS"}, "final_dots": {"status": "PASS"}},
            "format": {"strict_validator_passed": True, "outside_fill": "nan"},
            "raster": {"path": "docs/downloads/candidate.tif",
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
    assert not site_release_allowed(card, tmp_path)  # requires genuine release evidence
