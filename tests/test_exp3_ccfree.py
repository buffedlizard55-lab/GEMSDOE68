"""AC68-N1 ablation switch: catalogue-context columns must be *absent*,
not zeroed, and the default path must be byte-for-byte the old behaviour."""
import numpy as np

from gems68 import autocontext as AC
from gems68 import features as F


def test_context_names_default_includes_catalogue():
    names = AC.context_names()
    assert names[-len(F.CONTEXT_NAMES):] == F.CONTEXT_NAMES
    assert len(names) == 4 * len(AC.OFFSETS) + 8 + len(F.PHYSICS_NAMES) + len(F.CONTEXT_NAMES)


def test_context_names_ccfree_drops_exactly_the_catalogue_columns():
    full = AC.context_names(include_catalogue=True)
    free = AC.context_names(include_catalogue=False)
    assert free == full[: len(full) - len(F.CONTEXT_NAMES)]
    assert not any(n in free for n in F.CONTEXT_NAMES)


def test_lane_stage1_names_width_guard():
    # AC68-N2: an enriched stage-1 matrix must be labelled with matching names
    import pytest

    from gems68.experiment import Lane

    class _Grid:
        footprint = np.ones((8, 8), bool)
        catalogue = np.zeros((8, 8), bool)

    X_bad = np.zeros((64, 30), np.float32)
    with pytest.raises(ValueError, match="stage-1 feature names"):
        Lane.__init__(object.__new__(Lane), _Grid(), X_bad, [], np.zeros((8, 8), np.int8))


def test_context_matrix_ccfree_width_and_no_catalogue_leak():
    rng = np.random.default_rng(0)
    fp = np.ones((40, 40), bool)
    pmap = rng.random((40, 40), dtype=np.float32)
    phys = [rng.random((40, 40), dtype=np.float32) for _ in F.PHYSICS_NAMES]
    X_free = AC.context_matrix(pmap, fp, phys, [])
    assert X_free.shape == (1600, len(AC.context_names(include_catalogue=False)))
    # the same matrix must be a strict column-prefix of the full one
    ctx = [rng.random((40, 40), dtype=np.float32) for _ in F.CONTEXT_NAMES]
    X_full = AC.context_matrix(pmap, fp, phys, ctx)
    assert X_full.shape == (1600, len(AC.context_names(include_catalogue=True)))
    np.testing.assert_allclose(X_full[:, : X_free.shape[1]], X_free, rtol=0, atol=0)
