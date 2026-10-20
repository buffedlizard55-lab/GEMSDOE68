"""The matched-budget emitter must really enforce NMS on tree-score plateaus."""
from __future__ import annotations

import numpy as np
import pytest

from gems68.emission import EMITTER_VERSION, emit_dots


def test_nms_plateau_is_separated_and_deterministic():
    scores = np.ones((40, 40), np.float32) * .5  # every pixel is a tied local maximum
    scores[18, 19] = 1.0
    allowed = np.ones_like(scores, bool)
    allowed[18, 19] = True
    dots = emit_dots(scores, allowed, 38, nms=5)
    assert EMITTER_VERSION == "gems68-plateau-safe-nms-v2"
    assert dots.sum() == 38
    assert dots[18, 19]  # highest score, regardless of tie order
    assert np.array_equal(dots, emit_dots(scores, allowed, 38, nms=5))
    xy = np.argwhere(dots)
    for i, (y, x) in enumerate(xy):
        assert all(max(abs(y - yy), abs(x - xx)) >= 3 for yy, xx in xy[i + 1:])


def test_nms_refuses_missing_legal_peaks_rather_than_faking_budget():
    scores = np.ones((3, 3), np.float32)
    dots = emit_dots(scores, np.ones_like(scores, bool), 2, nms=5)
    assert dots.sum() == 1  # matched-budget experiment must stop on shortfall


def test_nms_rejects_non_finite_allowed_scores():
    a = np.ones((5, 5), np.float32)
    a[2, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        emit_dots(a, np.ones_like(a, bool), 1)
    assert not emit_dots(a, np.isfinite(a) & (a > 0), 0).any()
    with pytest.raises(ValueError, match="odd"):
        emit_dots(np.ones((5, 5), np.float32), np.ones((5, 5), bool), 1, nms=4)
