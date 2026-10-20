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

## Status — ✅ OK TO DOWNLOAD & SUBMIT (`VERDICT: PROMOTE`)

- **Primary GitHub Pages Site:** [`docs/index.html`](docs/index.html)
- **Executive Summary & Step-by-Step Submission Guide:** [`docs/executive-summary.html`](docs/executive-summary.html)
- **Canonical Run Card JSON:** [`evidence/run_card.json`](evidence/run_card.json) (mirrored at [`evidence/run-card-ac68.json`](evidence/run-card-ac68.json) and [`docs/lane68/data/run-card.json`](docs/lane68/data/run-card.json))
- **Recommended Primary GeoTIFF (All-Finite `[0.0, 1.0]`, 0 NaNs anywhere):**
  [`docs/downloads/gems68-autoctx-s3-scarp-ridge-35000dots-20261020T193800Z-e8956566-zeros.tif`](docs/downloads/gems68-autoctx-s3-scarp-ridge-35000dots-20261020T193800Z-e8956566-zeros.tif) (`145,480 bytes`, SHA-256 `eefc2885990d5f5ae0c943ab152ef69de6de1c6c722420e1dbcd6e4af634ecac`)
- **Compressed ZIP Archive:**
  [`docs/downloads/gems68-autoctx-s3-scarp-ridge-35000dots-20261020T193800Z-e8956566-zeros.zip`](docs/downloads/gems68-autoctx-s3-scarp-ridge-35000dots-20261020T193800Z-e8956566-zeros.zip) (`110,741 bytes`, SHA-256 `0b373e4548d7fb6dbc3ede485baf47dac90785e392ee1bf3505ef2966b2e6308`)
- **Template-Matched NaN-Outside Twin:**
  [`docs/downloads/gems68-autoctx-s3-scarp-ridge-35000dots-20261020T193800Z-e8956566-nan.tif`](docs/downloads/gems68-autoctx-s3-scarp-ridge-35000dots-20261020T193800Z-e8956566-nan.tif) (`217,144 bytes`, SHA-256 `faaad2cf0841f6d27de61d7048b0f16999eea80dd9479f24fef2f6065ef8f82f`)
- **Unique Submission Name:** `gems68-autoctx-s3-scarp-ridge-35000dots-20261020T193800Z-e8956566-zeros`
- **Short Submission Comment / Note (100 / 140 chars):**
  `GEMSDOE68 autoctx s3 scarp+ridge hide-draw; 35000 dots >2px off cat; HOLDOUT-DTI 0.1233 vs s1 0.0916`

| Gate | Result (`HOLDOUT-DTI` / Local Preflight Measurements) |
|---|---|
| Format & Range Validator (`gems68.validate` & `gems.validate_submission`) | **PASS** — single-band `float32`, `EPSG:32611`, `3730 × 3292`, 100 m pixel size, transform `(100, 0, 243350, 0, -100, 4508550)`, values strictly in `[0.0, 1.0]`, `0` NaN and `0` Inf across all `12,279,160` cells in `-zeros.tif` (`0` NaN inside `5,167,373` footprint cells in `-nan.tif`) |
| Holdout Gate — All Withheld Pixels (`HOLDOUT-DTI`, `gems57-pooled-hide-v2`, `11,321` withheld positives) | **PASS** — Stage 1 `0.0916` → Stage 2 `0.1231` → Stage 3 **`0.1233`** (`Δ = +0.0317`, 20 km spatial-cluster bootstrap 95% CI `[+0.0206, +0.0429]`, `z_boot = 5.58 SEs`, 4-outer-block paired `t = 4.13 SEs`, positive on all 4 quadrants) |
| Holdout Gate — `> 1 km` Far Pixels (`HOLDOUT-DTI`, `gems57-pooled-hide-v2`, `5,150` withheld positives) | **PASS** — Stage 1 `0.0773` → Stage 2 **`0.0996`** (`Δ = +0.0223`, 95% CI `[+0.0128, +0.0317]`, `z_boot = 4.38 SEs`, outer-block `t = 2.88 SEs`) → Stage 3 **`0.0958`** (`Δ = +0.0185`, 95% CI `[+0.0090, +0.0290]`, `z_boot = 3.45 SEs`, outer-block `t = 2.59 SEs`, positive on all 4 quadrants) |
| De-blobbing Resolution (`IR-68-BLOB-01`) | **PASS** — Dropping the 5 smooth regional seismicity/geodesy bands from Stage 1, replacing regional potentials with 1.5 km high-pass residuals, dropping `p_sm10`, and adding 1D ridge-vs-blob contrast reduced median dot distance to the mapped catalogue from `5,100 m` (Exp 2) to **`825 m`** (Exp 3 final) / **`1,118 m`** (Exp 3 CV) |
| Leakage Canary (Single-Feature Holdout ROC-AUC `< 0.90`) | **PASS** — All 44 Stage-1 features and all 69 Stage-2 features tested individually on withheld truth; max Stage-1 AUC = `0.6029` (`scarp_upface_max`), max Stage-2 AUC = `0.7451` (`cat_dens_1km`) |
| Uniqueness & Lane-Drift Gate vs. 81 Prior Rasters | **PASS** — Byte-unique SHA-256; max continuous-surface Spearman rank correlation = `0.1869 < 0.90`; max informative 3 px dot overlap = `0.5977 < 0.70` |
| `ORGANIZER-CONFIRMED` Score | **Not claimed** (`null` until uploaded to the DrivenData portal; champion reference is `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros` at `ORGANIZER-CONFIRMED` `0.2778`) |

## Layout

```
src/gems/          shared writer / validator / metric / release gate (vendored)
src/gems57/        shared hide-and-recover evaluator: holdout, metric, spatial folds (vendored)
src/gems68/        Auto-Context stacked refinement implementation (features.py, autocontext.py,
                   experiment.py, emission.py, uniqueness.py, validate.py, grid.py, holdout.py)
scripts/run_exp3.py          Experiment 3 nested blocked CV + leakage canary + gate evaluation
scripts/build_submission.py  Full-data LOQO Stage 1-3 build + 5x5 NMS emission + validator + run card
scripts/build_site.py        Builds docs/index.html, docs/executive-summary.html, and docs/lane68/
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

The competition files are login-walled at
<https://www.drivendata.org/competitions/306/competition-doe-gems/data/>. They are mirrored,
split and integrity-pinned in sibling repositories; `registry/data_manifest.json` records the exact
repo, ref, path and sha256 of each one. To place them locally:

```bash
mkdir -p data/bridge data/external
# training_features.tif (5 parts, repo buffedlizard55-lab/GEMSDOE @ c0c06ac8…)
cat data/bridge/gems-geodawn-numerical-features.tif.part-* > data/bridge/training_features.tif
# labels.tif / existing_faults.tif / sample_submission.tif
#   (repo buffedlizard55-lab/GEMSDOE24 @ 07345ea0…, path data/bridge/)
# data/external/* — lidar scarp, GeoDAWN radiometric and magnetic sidecars
```

Then verify the pins:

```bash
PYTHONPATH=src python - <<'PY'
import json, hashlib, pathlib
pins = json.loads(pathlib.Path('registry/data_manifest.json').read_text())
for f in pins['files']:
    p = pathlib.Path('data') / f['dest'].replace('training_features.tif', 'bridge/training_features.tif')
    p = pathlib.Path('data/bridge') / pathlib.Path(f['dest']).name
    if not p.exists():
        print('MISSING', f['dest']); continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    print('OK ' if h == f['sha256'] else 'MISMATCH ', f['dest'], h[:16])
PY
```

Measured in this checkout: all four core files match their pinned sha256 exactly.

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
PYTHONPATH=src python -m pytest tests -q     # 37 passed, 8 skipped (both lanes)
```
