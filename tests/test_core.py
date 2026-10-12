import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems68 import metric, emission as E, grid as G  # noqa: E402


def test_metric_matches_official_worked_example():
    # DrivenData page 967: TP_w 3.00, FP_w 1.89, FN_w 2.00 -> 0.60 (exact 0.6027)
    assert metric.dti_from_components(3.0, 1.89, 2.0) == pytest.approx(0.6027, abs=5e-4)


def test_metric_exact_equals_bruteforce():
    rng = np.random.default_rng(7)
    p = (rng.random((30, 30)) < 0.06) * rng.random((30, 30))
    g = rng.random((30, 30)) < 0.04
    a, b = metric.dti_exact(p, g), metric.dti_bruteforce(p, g)
    assert a["dti"] == pytest.approx(b["dti"], abs=1e-9)
    assert a["tp"] == pytest.approx(b["tp"], abs=1e-9) and a["fp"] == pytest.approx(b["fp"], abs=1e-9)


def test_binary_value_one_is_optimal_for_fixed_dots():
    rng = np.random.default_rng(3)
    g = rng.random((40, 40)) < 0.03
    dots = rng.random((40, 40)) < 0.03
    full = metric.dti_exact(dots.astype(float), g)["dti"]
    half = metric.dti_exact(dots * 0.5, g)["dti"]
    assert full > half


def test_emit_dots_respects_budget_exclusion_and_spacing():
    rng = np.random.default_rng(1)
    s = rng.random((60, 60)).astype(np.float32)
    cat = np.zeros((60, 60), bool); cat[30, :] = True
    allowed = ~E.exclusion_mask(cat)
    d = E.emit_dots(s, allowed, 25)
    assert d.sum() == 25
    assert not (d & ~allowed).any()
    # 5x5 NMS: no two dots at Chebyshev distance <= 2 unless equal values (continuous -> none)
    yy, xx = np.nonzero(d)
    for i in range(len(yy)):
        for j in range(i + 1, len(yy)):
            assert max(abs(yy[i] - yy[j]), abs(xx[i] - xx[j])) > 2


def test_write_geotiff_rejects_out_of_range(tmp_path):
    arr = np.zeros(G.SHAPE, np.float32); arr[0, 0] = 1.5
    with pytest.raises(ValueError):
        G.write_geotiff(tmp_path / "x.tif", arr)


def test_write_geotiff_rejects_nan_in_zeros_mode(tmp_path):
    arr = np.zeros(G.SHAPE, np.float32); arr[0, 0] = np.nan
    with pytest.raises(ValueError):
        G.write_geotiff(tmp_path / "x.tif", arr, nodata=None)
