# GEMSDOE68 — auto-context stacked refinement lane for the DOE GEMS Prize

**Site (executive summary + download):** https://buffedlizard55-lab.github.io/GEMSDOE68/
**Standing brief (read first every session):** [`docs/PROMPT.md`](docs/PROMPT.md)
**Run card (the one JSON that states the verdict):** [`evidence/run_card.json`](evidence/run_card.json)

Core values carried through every decision here: **Maximize P(Win)** and **Own the Outcome**.

## What this repository is

One lane of a parallel run on the DrivenData #306 "DOE GEMS" fault-mapping prize: an
[auto-context](https://doi.org/10.1109/CVPR.2008.4587436) stack — a per-pixel LightGBM stage 1 on the
19 organiser bands plus gradient/tilt/scarp channels, then stages 2–3 that see the previous stage's
probability sampled ±1–15 px along and across local strike, multi-scale/anisotropic smooths, the
physics channels and catalogue-context (distance / strike / density of *visible* mapped faults) —
evaluated with nested, buffered, spatially blocked CV on a hide-and-recover holdout, and emitted
as binary dots with the mass discipline the 0.2778 champion ladder proved out.

Headline result (**HOLDOUT-DTI**, evaluator `gems57-pooled-hide-v2`, 11,321 withheld positives,
paired 20 km-block bootstrap; *not* a leaderboard score):

| budget | stage 1 | stage 3 | Δ | 95 % CI | Δ on pixels > 1 km from mapped faults | 95 % CI |
|---|---|---|---|---|---|---|
| 38,000 dots | 0.0816 | 0.1189 | +0.0373 | [0.026, 0.048] | +0.0146 | [0.004, 0.025] |

Both gates pass (> 2 SE), the lane-uniqueness check passes, the validator is green → the run card
verdict is in `evidence/run_card.json` and on the site banner. **Organizer-confirmed score: none.**

## Why the family's best file scored 0.2778 (short)

`h33-h33-2-b2` is a 37,654-dot *binary* raster: the last rung of a mass-pruning ladder on one
per-pixel field (0.1922 → 0.2477 → 0.2600 → 0.2708 → 0.2778). The metric
DTI = T/(0.2·(T+S−M)+0.8·|G|) charges 0.2 per emitted unit that is not within 300 m of an uncovered
*new* fault; deleting the field's tail and then the 100–200 m halo of the *published* catalogue paid
at each step. Pruning is nearly exhausted; what remains is **ranking the same mass better**, which is
what this lane measures. Full derivation with provenance tags:
[`knowledge/01_why_02778_and_can_we_beat_it.md`](knowledge/01_why_02778_and_can_we_beat_it.md).

## Reproduce (CPU, ≈ 45 min on 2 cores / 3 GB)

```bash
pip install -r requirements.txt
python scripts/restore_data.py            # sha256-verified restore from the family's GitHub mirror (gh auth needed)
python scripts/build_features.py          # 19 bands + 7 physics channels -> .cache/features
python scripts/run_autocontext.py         # experiment 1 (negative, leak record)   -> evidence/exp1_autocontext.json
python scripts/run_exp2.py                # experiment 2 (hide-draw labels, the result) -> evidence/exp2_autocontext_hidedraw.json
python scripts/build_submission.py        # final LOQO maps -> dots -> validated GeoTIFF + run card -> docs/downloads/
python scripts/build_site.py              # docs/*.html from evidence JSON
pytest -q
```

## Layout

```
docs/            GitHub Pages site; downloads/ holds the TIFF, ZIP, NaN variant and run card
evidence/        exp1, exp2, run_card JSON (every number on the site comes from here)
knowledge/       why-0.2778 analysis
registry/        data manifest + restore receipt, feature receipt, score ledger of the family
scripts/         restore_data, build_features, run_autocontext (exp1), run_exp2, build_submission, build_site
src/gems68/      grid, metric (template, verbatim), holdout (template subset), evaluate_holdout (template),
                 validate (template), features, autocontext, experiment, emission, uniqueness
tests/           metric oracle, emission, writer/validator, holdout invariants
```

## Limitations and access (honest list)

* **Network:** this sandbox reaches only GitHub/PyPI/npm. DrivenData (data, leaderboard, submission
  page), docs.nlr.gov, usgs.gov, nationalmap.gov, earthquake.usgs.gov and Dropbox are unreachable.
  Competition bytes were restored from the family's sha256-pinned GitHub mirror; the hashes match
  the lineage's organizer-download receipts but that is mirror consistency, not organizer authentication.
* **No organizer receipts:** every score in `registry/score_ledger.csv` is an owner report copied
  from the brief. Submitting and copying the receipt back is a human step.
* **Holdout ≠ live truth:** the holdout withholds *catalogue* components; the live truth is
  *unmapped* faults. The > 1 km gate and the ≤ 2 px catalogue exclusion mitigate but do not remove
  this (IR-68-PROXY-01).
* **Compute:** 2 CPU / 3 GB. LightGBM with 400k-row PU subsamples, 220 rounds. No GPU, so the
  "small CNN" alternative for stage 2 was not attempted.

## Next steps (for the next session)

1. **Submit** `docs/downloads/*-zeros.tif` if the selector agrees; record the receipt as
   ORGANIZER-CONFIRMED in the ledger. This is the only way to learn the live↔holdout relation for this lane.
2. **Budget curve on the live board:** the holdout optimum is 25–38k dots; the champion sat at 37.6k.
   If a slot is available, a 25k variant of the same surface is the cheapest second data point.
3. **Stage-2 CNN** (the brief's alternative) on a GPU machine; and a third hide-draw seed set to
   tighten the CI.
4. **H68-3 tilt-depth/topography coincidence** and **H68-4 K/eTh lineaments** can be validated with
   data already in the cache (`docs/hypotheses.md`); H68-2 and H68-5 need the external fetches listed there.
5. **Report IR-68-TEMPLATE-01** upstream (template `implementation_hashes()` hard-codes `spatial.py`).
