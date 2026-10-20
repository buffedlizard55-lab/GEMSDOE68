"""No private duplicate-gate exception for dense prior *surfaces*."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lane68_check import lane_check  # noqa: E402


def _raster(path, data):
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0],
                       width=data.shape[1], count=1, dtype="float32",
                       crs="EPSG:32611", transform=from_origin(250000, 4500000, 100, 100)) as out:
        out.write(data.astype(np.float32), 1)


def test_surface_is_rank_and_hash_compared_without_false_dot_overlap(tmp_path):
    candidate = np.zeros((24, 24), np.float32)
    candidate[3, 4] = candidate[12, 20] = candidate[20, 2] = 1
    surface = (np.arange(24 * 24).reshape(24, 24) + 1).astype(np.float32) / 600.0
    me, old_surface, exact = (tmp_path / n for n in
                               ("me.tif", "older-surface.tif", "same.tif"))
    _raster(me, candidate)
    _raster(old_surface, surface)
    _raster(exact, candidate)
    check = lane_check(me, [old_surface], stage="emitted", skip_dot_overlap_for=[old_surface])
    assert check["complete_accessible_scan"]
    assert check["comparisons"][0]["dot_overlap_applicable"] is False
    assert check["comparisons"][0]["candidate_dots_within_3px"] is None
    assert check["comparisons"][0]["sha256"]  # rank + byte identity still checked
    assert check["verdict"].startswith("no drift")
    bad = lane_check(me, [old_surface, exact, old_surface], stage="emitted",
                     skip_dot_overlap_for=[old_surface], stop_on_first_drift=True)
    assert bad["rasters_checked"] == 2
    assert bad["stopped_after_duplicate"]
    assert bad["verdict"].startswith("LANE DRIFT")
    assert bad["comparisons"][-1]["exact_duplicate"]
