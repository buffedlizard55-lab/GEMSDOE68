"""Single-channel canary semantics, including absent (constant) LiDAR channels."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lane68_strict_experiment import feature_canary_auc  # noqa: E402


def test_constant_channel_has_auc_half_not_a_leakage_alarm():
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert feature_canary_auc(labels, np.zeros(6))[0] == 0.5
    assert feature_canary_auc(labels, np.zeros(6))[1] is False
    assert feature_canary_auc(labels, np.array([0, 0, 0, 1, 1, 1]))[1] is True
    assert feature_canary_auc(labels, np.array([1, 1, 1, 0, 0, 0]))[1] is True
    assert feature_canary_auc(labels, np.array([0, 0, 0, 1, 1, np.nan])) == (None, True)
