#!/usr/bin/env python3
"""Read-only audit of the archived diagnostic all-finite October 12 variant.

The historical writer emitted a mislabelled zeros file with 7,113,308 NaNs.
A parallel 2026-10-20 repair regenerated it at main commit 06fc4f3;
this script is now intentionally READ ONLY and targets the archived paths.
All-finite format is diagnostic, NOT an organizer-approved alternative to the
published null/NaN exterior, and the candidate failed scientific/originality gates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "evidence/archives/2026-10-12"
NAN = ARCHIVE / "GEMSDOE68-AC-stack2-20261012T042050Z-nan.tif"
ZEROS = ARCHIVE / "GEMSDOE68-AC-stack2-20261012T042050Z-zeros.tif"


def main() -> int:
    import numpy as np
    import rasterio
    with rasterio.open(NAN) as source:
        a = source.read(1)
        shape, crs, transform = source.shape, source.crs, source.transform
    with rasterio.open(ZEROS) as source:
        b = source.read(1)
        assert source.shape == shape and source.crs == crs and source.transform == transform
    assert int(np.isnan(a).sum()) == 7_113_308
    assert np.isfinite(b).all() and b.min() >= 0 and b.max() <= 1
    assert np.array_equal(b > 0, a > 0)
    assert np.array_equal(b[np.isfinite(a)], a[np.isfinite(a)])
    result = {"historical_original_nan_outside": int(np.isnan(a).sum()),
              "diagnostic_finite": True, "positive_cells": int((b > 0).sum()),
              "nan_variant_sha256": hashlib.sha256(NAN.read_bytes()).hexdigest(),
              "diagnostic_sha256": hashlib.sha256(ZEROS.read_bytes()).hexdigest(),
              "competition_download_allowed": False,
              "portal_acceptance_confirmed": False}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
