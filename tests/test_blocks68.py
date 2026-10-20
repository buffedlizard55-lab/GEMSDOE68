"""Tests for the GEMSDOE68 block-wise feature engineering.

These are pure-shape / pure-maths tests: they need no competition data and no
model, so they run anywhere and catch the class of bug that silently corrupts a
stacked feature matrix (wrong halo, mis-cropped strips, shadowed parameters).
"""

from __future__ import annotations

import numpy as np
import pytest

from gems68 import blocks as B


def _synthetic(h=120, w=90):
    rng = np.random.default_rng(0)
    prev = rng.random((h, w)).astype(np.float32)
    # a couple of coherent lineaments so the orientation field is not noise
    prev[40:44, 10:80] += 3.0
    prev[70:74, 5:85] += 2.0
    cos2, sin2, coh = B.structure_tensor_orientation(prev, sigma=3.0)
    return prev, cos2, sin2, coh


def test_context_feature_names_match_block_shape():
    prev, cos2, sin2, _ = _synthetic()
    names = B.context_feature_names("p1")
    blk = B.build_context_block(prev, slice(30, 60), cos2, sin2)
    assert blk.shape == (len(names), 30, prev.shape[1])
    assert names[0] == "p1"
    assert names[-1] == "p1_iso8"
    assert len(set(names)) == len(names)


def test_context_block_is_cropped_to_requested_rows():
    prev, cos2, sin2, _ = _synthetic()
    blk = B.build_context_block(prev, slice(0, 10), cos2, sin2)
    assert blk.shape[1] == 10
    blk2 = B.build_context_block(prev, slice(110, 120), cos2, sin2)
    assert blk2.shape[1] == 10


def test_offset_sampling_uses_local_orientation():
    """A dot at +d along the local orientation must be recovered by the along-d
    channel, and must not appear in the across-d channel."""
    h = w = 61
    prev = np.zeros((h, w), np.float32)
    cos2 = np.ones((h, w), np.float32)      # orientation: pure +row gradient
    sin2 = np.zeros((h, w), np.float32)
    prev[30, 31] = 1.0                     # the "previous stage" peak
    rows = slice(30, 31)
    blk = B.build_context_block(prev, rows, cos2, sin2)
    names = B.context_feature_names("p1")
    i_along = names.index("p1_along1")
    i_across = names.index("p1_across1")
    # cos2 = +1, sin2 = 0 => the local gradient points along +column, so the
    # "along" sample one pixel to the right of (30, 30) must see the peak while
    # the "across" sample one pixel down must not.
    assert blk[i_along, 0, 30] == pytest.approx(1.0)
    assert blk[i_across, 0, 30] == pytest.approx(0.0)


def test_column_window_matches_full_width():
    prev, cos2, sin2, _ = _synthetic()
    full = B.build_context_block(prev, slice(20, 50), cos2, sin2)
    part = B.build_context_block(prev, slice(20, 50), cos2, sin2, cols=(10, 80))
    assert part.shape[2] == 70
    # Channels whose support is at most +/-16 px must match exactly.  The two
    # isotropic Gaussian smooths legitimately see the artificial slice boundary
    # (sigma 8 has weight out to ~24 px), so they are excluded from the exact
    # comparison and only checked for finiteness/shape.
    finite_support = [i for i, n in enumerate(B.context_feature_names("p1"))
                      if "iso" not in n]
    # Compare a 30-column interior that is >=20 px from either artificial window
    # edge, so no gather is clamped by the window boundary.
    np.testing.assert_allclose(full[finite_support, :, 30:60], part[finite_support, :, 20:50],
                               rtol=1e-6, atol=1e-7)
    assert np.isfinite(part).all()


def test_across_box_averages_perpendicular_not_along():
    h = w = 100
    p = np.zeros((h, w), np.float32)
    p[40:61, 50] = 1.0  # north/south strand, not east/west
    cos2 = np.ones_like(p)  # along=column, across=row
    sin2 = np.zeros_like(p)
    values = B.build_context_block(p, slice(50, 51), cos2, sin2)
    names = B.context_feature_names('p1')
    assert values[names.index('p1_boxalong4'), 0, 50] == pytest.approx(1 / 9)
    assert values[names.index('p1_boxacross4'), 0, 50] == pytest.approx(1)
    assert values[names.index('p1_boxacross4'), 0, 50] > values[names.index('p1_boxalong4'), 0, 50]


def test_gather_clamps_columns_without_row_wrap():
    p = np.zeros((100, 61), np.float32)
    p[51, 0] = 1.0  # a former flattened-index wrap would land on the next row
    cos2 = np.ones_like(p)
    sin2 = np.zeros_like(p)
    values = B.build_context_block(p, slice(50, 51), cos2, sin2)
    names = B.context_feature_names('p1')
    assert values[names.index('p1_along1'), 0, 60] == 0.0


def test_gaussian_is_invariant_at_interior_strip_boundary():
    rng = np.random.default_rng(5)
    p = rng.random((180, 70), dtype=np.float32)
    cos2 = np.ones_like(p)
    sin2 = np.zeros_like(p)
    full = B.build_context_block(p, slice(60, 120), cos2, sin2)
    for rows in (slice(60, 90), slice(90, 120)):
        part = B.build_context_block(p, rows, cos2, sin2)
        np.testing.assert_allclose(
            part[:, :, :], full[:, rows.start - 60:rows.stop - 60, :], atol=1e-6, rtol=1e-6
        )


def test_read_rows_halo_and_columns():
    stack = np.arange(2 * 200 * 90, dtype=np.float32).reshape(2, 200, 90)
    index = {"a": 0, "b": 1}
    out = B.read_rows(stack, index, slice(0, 5), 3, ("a", "b"))
    assert out.shape == (2, 11, 90)
    out2 = B.read_rows(stack, index, slice(0, 5), 3, ("a", "b"), cols=(4, 9))
    assert out2.shape == (2, 11, 5)
    np.testing.assert_allclose(out2[0, :, :], out[0, :, 4:9])


def test_core_rows_crops_halo():
    a = np.zeros((3, 11, 90), np.float32)
    a[:, 3:8, :] = 5.0
    cropped = B.core_rows(a, 3, 5)
    assert cropped.shape == (3, 5, 90)
    assert (cropped == 5.0).all()


def test_feature_set_sizes_are_pinned():
    assert len(B.STAGE1_CHANNELS) == 44
    assert len(B.STAGE2_RAW) == 21
    assert len(B.CATALOGUE_CONTEXT) == 11
    assert len(B.stage2_feature_names("p1")) == (
        len(B.context_feature_names("p1")) + 21 + 11)
    # no channel may appear twice across the stage-2 inputs
    all_names = B.stage2_feature_names("p1")
    assert len(set(all_names)) == len(all_names)


def test_catalogue_context_masks_the_target_block():
    """Distance to the nearest visible fault must ignore faults inside the block
    that is being predicted."""
    from gems68 import catalogue_context as CC

    h = w = 60
    cat = np.zeros((h, w), bool)
    cat[10, 10] = True          # inside the block that will be masked
    cat[50, 50] = True          # outside it
    footprint = np.ones((h, w), bool)
    cos2 = np.ones((h, w), np.float32)
    sin2 = np.zeros((h, w), np.float32)
    block = np.zeros((h, w), bool)
    block[0:30, 0:30] = True
    cc = CC.catalogue_context(cat & ~block, footprint, cos2, sin2)
    # the masked fault at (10,10) must no longer set d_cat there
    assert cc["d_cat"][10, 10] > 1.0
    assert cc["d_cat"][50, 50] == 0.0
    assert set(cc) == set(B.CATALOGUE_CONTEXT)


def test_structure_tensor_orientation_is_bounded():
    prev, cos2, sin2, coh = _synthetic()
    assert np.all(np.abs(cos2) <= 1.0 + 1e-6)
    assert np.all(np.abs(sin2) <= 1.0 + 1e-6)
    assert np.all((coh >= 0) & (coh <= 1 + 1e-6))
    # cos^2 + sin^2 == 1 for a unit doubled-angle vector
    np.testing.assert_allclose(cos2 ** 2 + sin2 ** 2, 1.0, atol=1e-5)
