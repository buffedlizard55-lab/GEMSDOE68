# GEMSDOE68 — Auto-context stacked fault discovery (DOE GEMS Prize, DrivenData #306)

> ## SESSION-68 STANDING CHARTER — read this first, every session
>
> **Highest urgency.** MUST GENERATE A UNIQUE TIF SUBMISSION FOR THE COMPETITION. Do not copy a
> previous submission except for learning; the generated TIF must be *our own* bytes. It must be
> **obvious whether it is OK to download and submit**. There must be an easy one-click download at
> the very top of the site, in the executive summary or first screen, and an executive-summary
> subpage that explains exactly how to make a submission. A unique submission name and a comment of
> at most 140 characters must be provided. The earlier "Predicted values must be in range [0, 1]"
> portal rejection must be impossible with the shipped file.
>
> **Lane (single method paragraph).** Auto-context stacked refinement. Auto-context (Tu, CVPR 2008,
> doi:10.1109/CVPR.2008.4587436; extended TPAMI version at
> pages.ucsd.edu/~ztu/publication/pami_autocontext.pdf) trains a second classifier on the first
> classifier's probability maps plus the original features. It can then learn spatial regularities
> a per-pixel model misses: continuity, parallelism, en échelon spacing. Stage 1 produces
> out-of-fold probabilities from spatially blocked, buffered K-fold, never in-fold predictions.
> Stage 2 is LightGBM or a small CNN on four kinds of input: stage-1 probabilities sampled at
> ±1–15 px offsets along and across local orientation, multi-scale anisotropic smooths, and the raw
> gradient/tilt/scarp channels; plus catalogue-context features computed with the target block's
> labels masked (distance, orientation and spacing to the nearest mapped fault). Iterate twice and
> estimate the gain with nested blocked CV. Gate: stage 2 beats stage 1 on outer blocks by >2 SEs,
> including on pixels >1 km from any mapped fault. Finish: normalize to [0,1]; write a float32
> EPSG:32611 100 m GeoTIFF (NaN only outside the footprint); pass `validate_submission.py`; run a
> hash/pixel-agreement check against all prior rasters; offer the download only if the gate is met,
> else log the negative result.
>
> **Parallel-run protocol.** (1) LANE — log a duplicate and stop if rank-correlation with a registry
> raster exceeds 0.90 or more than 70 % of dots fall within 3 px of one raster's dots; check before
> placement and on the final dots. (2) REUSE, DON'T REBUILD — use the template's cached stack,
> `evaluate_holdout.py`, `submission_writer.py`; fix a shared tool once, never fork privately.
> (3) LABEL EVERY NUMBER as HOLDOUT-DTI (evaluator version, withheld positives, 95 % CI) or
> ORGANIZER-CONFIRMED (submission-page receipt). A projection is never written as a score.
> (4) LEAKAGE CANARY — test each feature alone on the holdout; AUC above 0.90 is leakage until
> disproven. (5) RUN CARD — end with one JSON card. (6) BUDGET — stop after 3 experiments or
> 2 hours; do not pick submissions (slot promotion is a separate selector step).
>
> **Core values — keep them central.** *Maximize P(Win).* Weigh trade-offs, assess risk, choose the
> path that maximises the probability of winning. *Own the Outcome.* Own results end to end, act
> when problems arise without waiting for permission, treat failure and success as signals.
>
> **Evidence discipline.** Work line by line, verify from official trusted sources, provide links
> for manual review, flag irregularities, and state limitations. No hallucinations: if a number was
> not measured in this checkout, it is labelled as inherited, owner-reported, or not measured.
>
> The full long-form brief is preserved verbatim in [`BRIEF.md`](BRIEF.md).

---

## Status

Two lanes now share this repository.

**The repository site is [`docs/index.html`](docs/index.html)** — it links to both.

**This session's lane (GEMSDOE68 / lane 68) is at
[`docs/lane68/index.html`](docs/lane68/index.html)**. The first-screen release banner makes the
current verdict explicit; it provides no raster link for negative runs. The step-by-step
[`docs/lane68/executive-summary.html`](docs/lane68/executive-summary.html) explains the submission
flow and why this candidate is blocked. The canonical machine-readable summary is
[`docs/lane68/data/run-card.json`](docs/lane68/data/run-card.json), written by
`scripts/lane68_write_run_card.py` and `scripts/lane68_build_site.py`.

**⛔ NO candidate is cleared for download (latest candidate evidence: 2026-10-20; negative verdict).** The current stage-3 emission is retained for audit only; **do not download or submit it**. The stage-2 far-stratum gate is narrowly below threshold (t = 1.97; required > 2), the current candidate's literal 3-px overlap rule fires, and the current emitted TIFF is not hash-linked to the holdout prediction surface. Its local format validator passing does not override those failures. The emission's 5,164,312-cell channel-stack domain is 1,540 cells smaller than the 5,165,852-cell official submission footprint; the emitter intentionally assigns zero to those official-only cells, and the independent validator confirms no NaN inside the official footprint. The website now fails closed: it offers a TIFF link only when the run card explicitly records `verdict=promote` and `download_cleared=true`.

| gate | result (local measurements unless labelled otherwise) |
|---|---|
| local format validator (as recorded in 2026-10-20 emission evidence; not rerun in this environment) | **PASS** — 16 checks, 0 FAIL, 0 SKIPPED |
| raster value/footprint receipt (as recorded in the adjacent local writer receipt) | **PASS, independently rechecked 2026-10-20**: no NaN inside 5,165,852 official cells; values 0–1. The 1,540-cell smaller channel domain is intentionally zero-filled by the emitter |
| uniqueness vs `registry/rasters/` (60 rasters; final candidate compared) | no exact byte match; max \|Spearman\| = 0.0861; 3-px overlap = 0.9847 against `gems61-lane-buried-step-gravinversion-hgm-r3-k400k-b2zeros.tif` (311,617 dots). Literal protocol fires: **LANE DRIFT — stop**. The stored density-null experiment was for a different, older 20,000-dot raster and is not used to excuse this flag |
| HOLDOUT-DTI gate (stage 2/3 vs stage 1, >2 SE in both strata) | **FAIL — verdict negative.** For 11,318 withheld positives, all-pixel stage 2 vs stage 1: +0.0607 (SE 0.0070, t=8.60); >1 km stratum: +0.0074 (SE 0.0038, t=1.97; 1,330 positives), below required >2 |
| leakage canary (single-channel AUC on withheld segments) | **HOLDOUT-DTI supporting check** — highest recorded AUC 0.611, below 0.90; source evidence dated 2026-10-12 |
| ORGANIZER-CONFIRMED score | **not claimed** — the scored truth is login-walled |

HOLDOUT-DTI (pooled, α = 0.2, β = 0.8, 300 m triangular, evaluator
`gems57-pooled-hide-v2 / gems57.metric.dti_binary`, 11,318 withheld positive pixels; evaluator
run dated 2026-10-12): stage 1 **0.1301**, stage 2 **0.1904**, stage 3 **0.1868**. The paired
physical-spatial-block 95% CI requested by the protocol was **not computed**. These are model-level
holdout results, not a score for the current TIFF: no hash-pinned link from the emitted surface to
the holdout predictions is recorded.

Latest candidate identifier: `GEMSDOE68-AC-stack3-r3-N32000`; its local writer receipt records a
98-character note, *“GEMSDOE68 auto-context stacked refinement (Tu CVPR 2008); stage3, r=3px,
32000 dots; holdout-gated”*. That embedded wording is not a clearance: the protocol verdict is
negative. Its two TIFF variants remain stored only as audit artifacts; the site offers neither for
download or submission.

**No ORGANIZER-CONFIRMED score is claimed for either lane.** Every number above is a local
HOLDOUT-DTI.

## Layout

```
src/gems/          shared writer / validator / metric / release gate (vendored, unmodified)
src/gems57/        shared hide-and-recover evaluator: holdout, metric, spatial folds (vendored)
src/gems68/        blocks.py, catalogue_context.py, channel_stack.py  -> lane 68 (this session)
                   features.py, grid.py, autocontext.py, emission.py, ... -> the sibling lane
                   that arrived from main; left untouched
scripts/lane68_*   THIS lane: autocontext (experiment), emit_submission (release), lane_check +
                   lane_density_null (duplicate gate), write_run_card (the JSON card),
                   build_site (this lane's site, written to docs/lane68/)
scripts/           the sibling lane's pipeline (run_autocontext.py, build_site.py, ...)
registry/rasters/  the duplicate-gate corpus of prior rasters, intentionally tracked
evidence/          every measurement, as JSON, from both lanes
docs/              the repository site, linking both lanes
docs/lane68/       THIS lane's site: index with the one-click download at the top, plus
                   executive-summary, results, hypotheses, research and run-card pages
docs/downloads/    every submission raster + receipt + zip, from both lanes
evidence/          every measurement, as JSON
data/              competition data (git-ignored; see "Data placement")
cache/             the built 45-channel stack (git-ignored, rebuilt by one call)
tests/             pytest suite
```

## Method (short)

1. **Cached channel stack** (`src/gems68/channel_stack.py`) — 45 fold-invariant channels on the official
   100 m grid: the 19 DrivenData feature bands, 6 derived gradient channels, 11 lidar-derived scarp
   channels and 8 GeoDAWN radiometric/magnetic channels, plus the lidar coverage flag. Built
   strip-by-strip into a memory-mapped array so the 3 GB sandbox never holds it all.
2. **Hide-and-recover holdout** (`src/gems57/holdout.py`, shared, unmodified) — four label-blind
   spatial quadrant folds; whole catalogue segments withheld inside a 1.2 km eroded domain;
   catalogue-derived features rebuilt from the visible faults only; visible fault pixels masked
   pixel-exactly; pooled DTI with α = 0.2, β = 0.8 and a 300 m triangular kernel.
3. **Stage 1** — LightGBM on the 45 channels + 11 catalogue-context features. Out-of-fold
   probabilities come from 4 inner spatial blocks inside each outer training domain (600 m buffer),
   never in-fold.
4. **Stage 2 / 3** — LightGBM on stage-*n* probabilities sampled at ±1,2,3,5,8,12,15 px along and
   across the local orientation, oriented box means at several half-widths (the multi-scale
   anisotropic smooths), two isotropic smooths, the 21 raw gradient/tilt/scarp channels and the 11
   catalogue-context features (each computed with the target block's labels masked).
5. **Nested blocked CV** — 4 outer folds × 4 inner blocks; the gate compares stages 2 and 3 with
   stage 1 on the outer blocks, paired, with a > 2 SE requirement, repeated on withheld pixels more
   than 1 km from any visible mapped fault.
6. **Release** — the same stack is re-run with *no* withholding (the four inner blocks tile the whole
   footprint) so every label is used and every probability is still out-of-fold; the emission
   parameters (catalogue exclusion radius, dot budget) come from the holdout sweep; the file is
   written by the shared `gems.submission_writer` and validated by the shared
   `gems.validate_submission`. The footprint written is the *official* one derived from band 1 of
   `training_features.tif`, which is 3,061 cells smaller than the cached stack's own footprint; the
   1,540 official-only cells receive 0 rather than NaN, because NaN inside the official footprint
   is a validator failure.

## Data placement

The DrivenData data tab is login-walled at
<https://www.drivendata.org/competitions/306/competition-doe-gems/data/>. The repository has an
integrity-pinned GitHub mirror manifest (`registry/data_manifest.json`) and a restoration helper:

```bash
python scripts/restore_data.py --group core
# Files land under .cache/gems_data/ (git-ignored); receipt: registry/restore_receipt.json
python scripts/restore_data.py --group external  # optional sidecar layers, when needed
```

**Current checkout status (rechecked 2026-10-20):** `data/` contains only metadata JSON files.
`python scripts/restore_data.py --group all` verified and restored all manifest entries (core,
external layers, and pinned scored/calibration rasters) into ignored `.cache/gems_data/`;
`registry/restore_receipt.json` records `all_verified: true`. The mirror restore is separate from
DrivenData authentication. The scored/calibration rasters are audit inputs only, not submission
candidates.

## Official sources

| what | link |
|---|---|
| competition overview | <https://www.drivendata.org/competitions/306/competition-doe-gems/> |
| problem description / metric | <https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/> |
| rules PDF (96647) | <https://docs.nlr.gov/docs/fy26osti/96647.pdf> |
| reference solution | <https://github.com/drivendataorg/gems-prize-reference-solution> |
| GeoDAWN airborne survey (USGS) | <https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and> |
| INGENIOUS project | <https://gbcge.org/current-projects/ingenious/> |
| EPSG:32611 | <https://epsg.io/32611> |
| auto-context (Tu, CVPR 2008) | <https://doi.org/10.1109/CVPR.2008.4587436> |
| auto-context TPAMI extension | <https://pages.ucsd.edu/~ztu/publication/pami_autocontext.pdf> |
| Tversky index | <https://en.wikipedia.org/wiki/Tversky_index> |

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests -q -rs  # 52 passed, 2 skipped on 2026-10-20; 15 warnings
```
