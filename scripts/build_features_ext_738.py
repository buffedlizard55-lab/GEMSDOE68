#!/usr/bin/env python3
"""Exp4 (AC68-N2) stage-1 matrix: cached X1 (26 cols) + 20 external channels.

External channels are decoded by the shared gems68.channel_stack (never a
private decoder): 11 lidar scarp aggregates + lidar_valid + 4 GeoDAWN
radiometric ranks + 4 GeoDAWN extension ranks.  NaNs inside the footprint are
filled with the in-footprint median (same policy as gems68.features._fill;
the lidar_valid flag preserves the missingness information).  The SGMC fault
derivative is deliberately NOT included (catalogue leak).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems68 import channel_stack as CS, features as F, grid as G  # noqa: E402

FEAT = ROOT / ".cache" / "features"
EXT_NAMES = list(CS.CHANNEL_NAMES[25:])          # 20 channels after 19 core + 6 derived
assert len(EXT_NAMES) == 20 and EXT_NAMES[0] == "ex_max" and "lidar_valid" in EXT_NAMES


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    t0 = time.time()
    g = G.load_grid()
    fp = g.footprint
    n = int(fp.sum())
    X1 = np.load(FEAT / "X1.npy", mmap_mode="r")
    assert X1.shape == (n, len(F.STAGE1_NAMES)), X1.shape
    names, stack, stack_fp = CS.load_channels(use_cache=True)
    assert np.array_equal(stack_fp, fp), "channel-stack footprint != official sample footprint"
    idx = {nm: i for i, nm in enumerate(names)}
    out_path = FEAT / "X1_ext.npy"
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                    shape=(n, len(F.STAGE1_NAMES) + len(EXT_NAMES)))
    out[:, : len(F.STAGE1_NAMES)] = X1
    stats = []
    for j, name in enumerate(EXT_NAMES):
        a = np.asarray(stack[idx[name]])[fp].astype(np.float64)
        finite = np.isfinite(a)
        med = float(np.nanmedian(a)) if finite.any() else 0.0
        stats.append(dict(channel=name, finite_in_footprint=int(finite.sum()),
                          coverage=float(finite.mean()), median_fill=med,
                          p05=float(np.nanpercentile(a, 5)) if finite.any() else None,
                          p95=float(np.nanpercentile(a, 95)) if finite.any() else None))
        col = np.where(finite, a, med).astype(np.float32)
        out[:, len(F.STAGE1_NAMES) + j] = col
        del a, col
        print(name, stats[-1], flush=True)
    out.flush()
    del out
    all_names = list(F.STAGE1_NAMES) + EXT_NAMES
    (FEAT / "stage1_ext_names.json").write_text(json.dumps(all_names, indent=1))
    receipt = dict(
        experiment="exp4 AC68-N2 stage-1 external enrichment",
        n_footprint=n, n_catalogue=int(g.catalogue.sum()),
        stage1_names=all_names, external_names=EXT_NAMES, channel_stats=stats,
        seconds=round(time.time() - t0, 1),
        source_sha256={
            "lidar_scarp_features_u8.tif": sha256_file(ROOT / ".cache/gems_data/external/lidar_scarp_features_u8.tif"),
            "geodawn_rad_u8.tif": sha256_file(ROOT / ".cache/gems_data/external/geodawn_rad_u8.tif"),
            "geodawn_extensions_u8.tif": sha256_file(ROOT / ".cache/gems_data/external/geodawn_extensions_u8.tif"),
        },
        excluded="derived_sgmc_faults_100m_u8.tif (catalogue derivative - leakage)",
        x1_ext_bytes=out_path.stat().st_size)
    (ROOT / "evidence" / "exp4_feature_receipt.json").write_text(json.dumps(receipt, indent=1))
    print("wrote", out_path, receipt["x1_ext_bytes"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
