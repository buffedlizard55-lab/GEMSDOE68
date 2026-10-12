import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems68 import grid as G  # noqa: E402
from gems68.holdout import buffered_component_draw  # noqa: E402

pytestmark = pytest.mark.skipif(not (G.DATA / "existing_faults.tif").is_file(), reason="data not restored")


def test_holdout_invariants():
    g = G.load_grid()
    folds, quad = buffered_component_draw(g, seed=20)
    hidden = folds[0]["hidden_all"]; vis = folds[0]["visible"]; known = folds[0]["masked_known"]
    assert not (hidden & vis).any()                  # withheld never visible
    assert not (hidden & known).any()                # withheld never masked as known
    assert (vis | hidden) .sum() <= g.catalogue.sum()
    assert folds[0]["receipt"]["nearest_visible_to_truth_px"] > 3
    for f in folds:
        assert f["truth"].any() and not (f["truth"] & ~f["region"]).any()
