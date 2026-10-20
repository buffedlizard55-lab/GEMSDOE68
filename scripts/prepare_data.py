#!/usr/bin/env python3
"""Prepare and leak-audit the complete 59-channel physical + GDR-1391 stack.

Runs after ``bash scripts/download_competition_data.sh``:
  1. Builds ``.cache/s68_stack/X_fp59.npy`` (5,167,373 in-footprint cells x 59
     strictly label-free channels).
  2. Runs the mandatory Protocol Rule 4 Leakage Canary on EVERY SINGLE ONE of
     the 59 channels against the withheld fault segments of ``gems68.holdout``
     (seed=20, 11,321 withheld positive pixels across 4 spatial quadrants).
  3. Writes ``evidence/s68_leakage_canary.json`` and fails closed (exit 2) if
     any single channel exceeds ``max(AUC, 1 - AUC) > 0.90``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems68 import grid as G68  # noqa: E402
from gems68.holdout import buffered_component_draw  # noqa: E402
from gems68.multiphysics_stack import build_or_load_stack  # noqa: E402


def main() -> int:
    t0 = time.time()
    print("[prepare_data] Building/verifying 59-channel multi-physics & GDR-1391 stack...", flush=True)
    names, mm, fp, meta = build_or_load_stack(force=False)
    print(f"[prepare_data] Stack ready: shape={mm.shape}, channels={len(names)} ({time.time()-t0:.1f}s)", flush=True)

    grid = G68.load_grid()
    folds, _ = buffered_component_draw(grid, seed=20)
    truth_all = np.zeros(fp.shape, bool)
    active = np.zeros(fp.shape, bool)
    for f in folds:
        truth_all |= f["truth"]
        active |= f["region"]
    active &= fp & ~folds[0]["masked_known"]
    rows_act = np.flatnonzero(active[fp])
    t_vec = truth_all[fp][rows_act]
    rng = np.random.default_rng(68)
    sub = rng.choice(rows_act, size=min(600_000, len(rows_act)), replace=False)
    sub = np.union1d(sub, rows_act[t_vec])
    y_true = truth_all[fp][sub].astype(np.int8)

    per_channel = {}
    flagged = {}
    max_sym_auc = 0.0
    max_ch = None
    for j, nm in enumerate(names):
        vals = np.asarray(mm[sub, j], dtype=np.float64)
        auc = float(roc_auc_score(y_true, vals))
        sym = max(auc, 1.0 - auc)
        is_flag = sym > 0.90
        per_channel[nm] = {"auc": auc, "symmetric_auc": sym, "flag": is_flag}
        if is_flag:
            flagged[nm] = per_channel[nm]
        if sym > max_sym_auc:
            max_sym_auc = sym
            max_ch = nm

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rule": "Protocol Rule 4: single-feature holdout AUC > 0.90 (symmetric max(AUC, 1-AUC) > 0.90) is leakage until proven otherwise",
        "withheld_positive_pixels": int(y_true.sum()),
        "evaluated_pixels": int(y_true.size),
        "n_channels": len(names),
        "max_symmetric_auc": max_sym_auc,
        "max_channel": max_ch,
        "flagged_channels_gt_0_90": flagged,
        "passed": len(flagged) == 0,
        "gdr_stats": meta["gdr_stats"],
        "per_channel": per_channel,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (ROOT / "evidence").mkdir(exist_ok=True)
    out_p = ROOT / "evidence" / "s68_leakage_canary.json"
    out_p.write_text(json.dumps(report, indent=2))
    print(f"[prepare_data] Leakage canary complete: {len(names)} channels checked, "
          f"max symmetric AUC = {max_sym_auc:.4f} ({max_ch}), flagged = {len(flagged)}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
