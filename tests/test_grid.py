"""Grid, footprint and mirror-integrity tests."""
from __future__ import annotations

import numpy as np
import pytest

from gems68.data import (LABELS_SHA256, SAMPLE_SHA256, FEATURES_SHA256, FEATURES_TIF, LABELS_TIF,
                         SAMPLE_TIF, EXISTING_TIF, band_names, channel_plan, load_grid,
                         load_labels, sha256_file)

pytestmark = pytest.mark.skipif(not SAMPLE_TIF.exists(), reason="mirrors not restored")


def test_mirrors_match_the_published_pins():
    assert sha256_file(LABELS_TIF) == LABELS_SHA256
    assert sha256_file(SAMPLE_TIF) == SAMPLE_SHA256
    if FEATURES_TIF.exists():
        assert sha256_file(FEATURES_TIF) == FEATURES_SHA256


def test_labels_and_existing_faults_are_byte_identical():
    """IR-68-02: the mirrored training target IS the mapped-fault catalogue."""
    if not EXISTING_TIF.exists():
        pytest.skip("existing_faults.tif not restored")
    assert sha256_file(LABELS_TIF) == sha256_file(EXISTING_TIF)


def test_grid_matches_the_submission_format():
    g = load_grid()
    assert (g.height, g.width) == (3730, 3292)
    assert g.crs == "EPSG:32611"
    assert g.transform == (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
    assert int(g.footprint.sum()) == 5167373


def test_footprint_from_labels_equals_footprint_from_sample():
    lab = load_labels()
    g = load_grid()
    assert set(np.unique(lab).tolist()) == {-1, 0, 1}
    assert (lab >= 0).sum() == g.footprint.sum()
    assert bool(((lab >= 0) == g.footprint).all())
    assert int((lab == 1).sum()) == 60988


def test_channel_plan_covers_every_source_band():
    names = band_names()
    plan = channel_plan(names)
    assert len(plan) == 47
    assert len(set(plan)) == len(plan)
    for n in names:
        assert n in plan
