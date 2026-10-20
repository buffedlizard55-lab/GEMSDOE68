#!/usr/bin/env python3
"""Regenerate docs/assets/dots_map.png for the CURRENT evidence/run_card.json.

Background: downsampled published catalogue (dim).  Overlay: emitted dots of
the card's primary file (bright), dilated for visibility at print size.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def block_max(a: np.ndarray, f: int) -> np.ndarray:
    h, w = a.shape
    a = a[: h // f * f, : w // f * f]
    return a.reshape(h // f, f, w // f, f).max(axis=(1, 3))


def main() -> int:
    card = json.loads((ROOT / "evidence/run_card.json").read_text())
    prim = ROOT / "docs/downloads" / card["files"]["primary"]["name"]
    with rasterio.open(prim) as s:
        dots = s.read(1) > 0
    with rasterio.open(ROOT / ".cache/gems_data/existing_faults.tif") as s:
        cat = s.read(1) == 1
    with rasterio.open(ROOT / ".cache/gems_data/sample_submission.tif") as s:
        fp = np.isfinite(s.read(1))
    f = 4
    img = np.zeros((*block_max(fp.astype(np.float32), f).shape, 3), np.float32)
    img[..., :] = 0.05
    img[block_max(fp.astype(np.float32), f) > 0] = [0.10, 0.13, 0.22]
    img[block_max(cat.astype(np.float32), f) > 0] = [0.45, 0.55, 0.75]
    d = block_max(ndi.binary_dilation(dots).astype(np.float32), f) > 0
    img[d] = [1.0, 0.75, 0.2]
    fig, ax = plt.subplots(figsize=(9, 10), dpi=130)
    ax.imshow(img, interpolation="nearest")
    ax.set_title(f"{prim.name}\n{int(dots.sum()):,} dots (amber) · published catalogue (pale blue) · 100 m grid, EPSG:32611",
                 color="#e8ecf5", fontsize=10)
    ax.set_axis_off()
    fig.patch.set_facecolor("#0b1020")
    out = ROOT / "docs/assets/dots_map.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", out, out.stat().st_size, "bytes; dots", int(dots.sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
