"""Writer + validator + duplicate-check tests, run on a synthetic 60x80 grid."""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from gems68.submission import (dot_overlap, spearman, validate_submission, write_submission,
                               zip_submission)
from gems68 import metric


@pytest.fixture()
def template(tmp_path):
    p = tmp_path / "sample_submission.tif"
    H, W = 60, 80
    fp = np.zeros((H, W), bool)
    fp[5:55, 5:75] = True
    a = np.where(fp, 0.0, np.nan).astype(np.float32)
    with rasterio.open(p, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
                       crs="EPSG:32611", transform=from_origin(243350, 4508550, 100, 100),
                       nodata=float("nan"), compress="lzw") as dst:
        dst.write(a, 1)
    return p, fp


def test_zeros_twin_passes_every_check(template, tmp_path):
    tpl, fp = template
    vals = np.zeros(fp.shape, np.float32)
    ys, xs = np.nonzero(fp)
    vals[ys[:100], xs[:100]] = 1.0
    out = write_submission(vals, fp, tmp_path / "s-zeros.tif", outside="zeros", template=tpl)
    fc = validate_submission(out["path"], template=tpl)
    assert fc.ok, fc.failures
    assert fc.checks["emitted_pixels"] == 100


def test_nan_twin_is_masked_like_the_template(template, tmp_path):
    tpl, fp = template
    vals = np.where(fp, 0.3, 0.0).astype(np.float32)
    out = write_submission(vals, fp, tmp_path / "s-nan.tif", outside="nan", template=tpl)
    a = rasterio.open(out["path"]).read(1)
    assert bool(np.isnan(a[~fp]).all())
    assert bool(np.isfinite(a[fp]).all())
    assert validate_submission(out["path"], template=tpl).ok


def test_out_of_range_values_are_caught(template, tmp_path):
    tpl, fp = template
    vals = np.where(fp, 1.7, 0.0).astype(np.float32)  # deliberately wrong
    out = write_submission(vals, fp, tmp_path / "bad.tif", outside="zeros", template=tpl)
    a = rasterio.open(out["path"]).read(1)
    assert a.max() <= 1.0  # the writer clips, which is the point
    assert validate_submission(out["path"], template=tpl).ok


def test_nan_inside_footprint_is_caught(template, tmp_path):
    tpl, fp = template
    vals = np.where(fp, 0.5, 0.0).astype(np.float32)
    ys, xs = np.nonzero(fp)
    vals[ys[0], xs[0]] = np.nan
    with rasterio.open(tpl) as s:
        prof = s.profile
    p = tmp_path / "naninside.tif"
    with rasterio.open(p, "w", **prof) as dst:
        dst.write(vals, 1)
    fc = validate_submission(p, template=tpl)
    assert not fc.ok
    assert any("finite_inside_footprint" in f for f in fc.failures)


def test_zip_contains_one_geotiff(template, tmp_path):
    tpl, fp = template
    out = write_submission(np.where(fp, 0.1, 0.0).astype(np.float32), fp,
                           tmp_path / "s.tif", outside="zeros", template=tpl)
    z = zip_submission(out["path"])
    import zipfile
    with zipfile.ZipFile(z["path"]) as zf:
        assert len(zf.namelist()) == 1
        assert zf.namelist()[0].endswith(".tif")


def test_spearman_is_for_surfaces_and_overlap_for_dots():
    """Rank correlation is meaningless on a sparse binary field (ties dominate and two
    similar dot sets correlate *negatively*), so the lane check uses Spearman on the
    continuous surface and 3 px overlap on the dots."""
    rng = np.random.default_rng(0)
    fp = np.ones((60, 60), bool)
    surf = rng.random((60, 60)).astype(np.float32)
    assert spearman(surf, surf + 0.01 * rng.random((60, 60)).astype(np.float32)) > 0.9
    dots_a = np.zeros((40, 40), np.float32); dots_b = np.zeros((40, 40), np.float32)
    fp2 = np.ones((40, 40), bool)
    dots_a[10:30, 10] = 1.0
    dots_b[10:30, 11] = 1.0
    assert dot_overlap(dots_a, dots_b, fp2)["within"] > 0.9
    c = np.zeros((40, 40), np.float32); c[5, 35] = 1.0
    assert dot_overlap(dots_a, c, fp2)["within"] < 0.1
