"""Release-page fail-closed checks; intentionally use only the standard library."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lane68_build_site", ROOT / "scripts" / "lane68_build_site.py"
)
site = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(site)


def _card():
    return json.loads((ROOT / "docs/lane68/data/run-card.json").read_text())


def test_negative_run_card_has_no_download_action():
    card = _card()
    assert card["verdict"] == "negative"
    assert card["download_cleared"] is False
    assert "DO NOT DOWNLOAD OR SUBMIT" in site._release_status_block(card)
    assert not re.search(r'href=["\'][^"\']+\.tif(?:f)?(?:\.zip)?["\']',
                         site._release_status_block(card), re.I)


def test_run_card_verdict_requires_linkage_surface_check_and_ci():
    spec = importlib.util.spec_from_file_location(
        "lane68_write_run_card", ROOT / "scripts" / "lane68_write_run_card.py"
    )
    writer = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(writer)
    assert writer.decide_verdict(True, True, False, True, []) == ("negative", False)
    assert writer.decide_verdict(True, True, False, True, [], True, True, True) == ("promote", True)


def test_only_explicit_promote_and_true_clearance_can_render_download():
    card = _card()
    card["verdict"] = "negative"
    card["download_cleared"] = True
    assert "No download is offered" in site._release_status_block(card)
    card["verdict"] = "promote"
    card["download_cleared"] = True
    rendered = site._release_status_block(card)
    assert 'href="../downloads/' in rendered
    assert "download>" in rendered


def test_latest_emission_provenance_irregularities_are_explicit():
    card = _card()
    assert card["holdout_emission_linkage"]["holdout_run_sha256_for_emitted_surface"] is None
    assert card["raster"]["official_minus_emission_footprint_cells"] == 1540
    assert "intentional domain difference" in " ".join(card["limitations"])
    assert card["holdout_dti"]["ci95"].startswith("not computed")
    assert card["correlation_overlap_vs_registry"]["literal_verdict"].startswith("LANE DRIFT")


def test_candidate_bytes_are_not_an_exact_registry_raster_copy():
    card = _card()
    candidate = ROOT / card["raster"]["primary_path"]
    actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert actual == card["raster"]["primary_sha256"]
    registry = ROOT / "registry/rasters"
    registry_hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in registry.rglob("*.tif")
    }
    assert actual not in registry_hashes


def test_user_facing_pages_do_not_link_to_unapproved_tiffs():
    pages = [
        ROOT / "docs/index.html",
        ROOT / "docs/executive-summary.html",
        ROOT / "docs/how-to-submit.html",
        ROOT / "docs/lane68/index.html",
        ROOT / "docs/lane68/executive-summary.html",
        ROOT / "docs/lane68/submission-guide.html",
    ]
    for page in pages:
        text = page.read_text()
        assert "DO NOT DOWNLOAD" in text or "No file is offered" in text, page
        assert not re.search(r'href=["\'][^"\']+\.tif(?:f)?(?:\.zip)?["\']', text, re.I), page


def test_generated_feed_separates_candidate_path_from_public_download():
    feed = json.loads((ROOT / "docs/lane68/data/feed.json").read_text())
    assert feed["download_cleared"] is False
    assert feed["download"] is None
    assert feed["candidate_artifact"].endswith(".tif")
