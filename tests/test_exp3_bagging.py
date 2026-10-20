"""Exp 3 machinery: seed-bagging helpers and extended-offset context names."""
import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")
from gems68 import autocontext_bagged as AC  # noqa: E402


def test_context_names_follow_offsets(monkeypatch):
    monkeypatch.setattr(AC, "OFFSETS", (1, 2, 3, 5, 8, 12, 15))
    names = AC.context_names()
    assert names[:4] == ["p_along+1", "p_along-1", "p_across+1", "p_across-1"]
    assert "p_along+15" in names and "p_across-12" in names
    assert len(names) == 7 * 4 + 8 + 7 + 5  # offsets + summaries + physics + catalogue ctx


def test_bagged_fit_predict_averages_and_shapes(monkeypatch):
    rng = np.random.default_rng(0)
    monkeypatch.setattr(AC, "MODEL_SEEDS", (68, 71))
    monkeypatch.setattr(AC, "N_ROUNDS", 5)
    X = rng.normal(size=(2000, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int8)
    idx = np.arange(2000)
    p, boosters = AC.fit_predict(X, y, idx, idx[:100], rng, ["a", "b", "c", "d"])
    assert len(boosters) == 2
    assert p.shape == (100,) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    manual = np.mean([b.predict(X[:100]) for b in boosters], axis=0)
    np.testing.assert_allclose(p, manual, rtol=1e-6)
    gains = AC.mean_gain(boosters)
    assert set(gains) == {"a", "b", "c", "d"}


def test_default_single_seed(monkeypatch):
    # module default must stay (68,) so Exp 1/2 remain reproducible
    assert AC.MODEL_SEEDS == (68,)
