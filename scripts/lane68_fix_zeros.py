#!/usr/bin/env python3
"""Rewrite the -zeros fallback so it is genuinely all-finite (IR-30, 2026-10-20 audit).

The shipped -zeros file was pixel-identical (NaN-aware) to the -nan file: it still
carried NaN outside the footprint, so it could not serve as the fallback for a
portal range check that mishandles NaN. This script replaces NaN with 0.0,
preserves CRS/shape/transform/dtype/tags, updates the receipt and emission
evidence hashes, and asserts: all-finite, values in [0,1], same >0 dot cells.
Run: /path/to/venv/bin/python scripts/lane68_fix_zeros.py
"""
import hashlib, json
from pathlib import Path
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
NAN = ROOT / "docs/downloads/GEMSDOE68-AC-stack2-20261012T042050Z-nan.tif"
ZEROS = ROOT / "docs/downloads/GEMSDOE68-AC-stack2-20261012T042050Z-zeros.tif"

with rasterio.open(NAN) as ds:
    a, prof = ds.read(1), ds.profile.copy()
assert np.isnan(a).sum() > 0
b = np.where(np.isnan(a), np.float32(0), a).astype("float32")
assert np.isfinite(b).all() and b.min() >= 0 and b.max() <= 1
assert np.array_equal(b > 0, a > 0) and np.array_equal(b[np.isfinite(a)], a[np.isfinite(a)])
prof.update(nodata=None)
with rasterio.open(ZEROS, "w", **prof) as ds:
    ds.write(b, 1)
sha = hashlib.sha256(ZEROS.read_bytes()).hexdigest()
receipt_p = Path(str(ZEROS) + ".receipt.json")
r = json.loads(receipt_p.read_text())
r.update({"sha256": sha, "nan_inside": 0, "nan_outside_count": 0, "all_finite": True,
          "note_on_variant": "Regenerated 2026-10-20 (IR-30): NaN outside footprint replaced by 0; inside-footprint pixels identical to -nan file."})
receipt_p.write_text(json.dumps(r, indent=1) + "\n")
emis_p = ROOT / "evidence/ac68_emission.json"
e = json.loads(emis_p.read_text())
e["files"]["zeros"].update({"sha256": sha, "bytes": ZEROS.stat().st_size})
emis_p.write_text(json.dumps(e, indent=1) + "\n")
print(json.dumps({"zeros_sha256": sha, "bytes": ZEROS.stat().st_size, "finite": int(np.isfinite(b).sum()), "gt0": int((b > 0).sum())}))
