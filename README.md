# GEMSDOE68 — auto-context stacked refinement for the DOE GEMS Prize Challenge

**Lane:** auto-context stacked refinement (Tu, *Contextual Classification for Semantic Image
Interpretation*, CVPR 2008, [doi:10.1109/CVPR.2008.4587436](https://doi.org/10.1109/CVPR.2008.4587436);
extended TPAMI version, <https://pages.ucsd.edu/~ztu/publication/pami_autocontext.pdf>).

**Competition:** [GEMS Prize Challenge — DOE](https://www.drivendata.org/competitions/306/competition-doe-gems/) ·
[problem description](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) ·
[about / resources](https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/) ·
[data tab (login-walled)](https://www.drivendata.org/competitions/306/competition-doe-gems/data/) ·
[leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/)

---

## ⬇ Download the submission

The current candidate GeoTIFF and its validator output are published on the GitHub Pages site:

* **Site:** <https://buffedlizard55-lab.github.io/GEMSDOE68/>
* **Executive summary / how to submit:** <https://buffedlizard55-lab.github.io/GEMSDOE68/executive-summary.html>
* **Local:** [`docs/downloads/`](docs/downloads/) — the `-zeros.tif` twin is the one to upload
  (all-finite, so it cannot trip the portal's *"Predicted values must be in range [0, 1]"*
  check); the `-nan.tif` twin matches `sample_submission.tif`'s masking byte-for-byte.

The site states, at the very top, whether the run gate was met and therefore whether it is
OK to spend a submission slot on the file. **Do not upload unless that banner is green.**

---

## Verified facts about this repository's inputs

Every hash below was computed from bytes on disk in this session and matches the pin
recorded in `buffedlizard55-lab/60GEMSDOE/registry/data_manifest.json`.

| file | bytes | sha256 | note |
| --- | ---: | --- | --- |
| `data/bridge/gems-geodawn-numerical-features.tif` | 418,912,844 | `4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5` | 19 float32 bands, nodata `-3.4028234663852886e+38` |
| `data/official/labels.tif` | 425,830 | `7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093` | int8 `{-1, 0, 1}`, nodata `-1` |
| `data/official/existing_faults.tif` | 425,830 | `7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093` | **byte-identical to `labels.tif`** |
| `data/official/sample_submission.tif` | 1,599,597 | `2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc` | float32, nodata `NaN` |

Grid (read directly with `rasterio`, `tests/test_grid.py`):

* 3730 × 3292 pixels, **EPSG:32611** (UTM zone 11N), 100 m pixels
* transform `(100, 0, 243350, 0, -100, 4508550)`; bounds 243350–572550 E, 4135550–4508550 N
* footprint = `isfinite(sample_submission)` = `labels != -1` = **5,167,373 px** (identical masks)
* catalogue positives = **60,988 px** (1.180 % of footprint), exactly the set of `1`s in
  `sample_submission.tif`

Nineteen source bands, read from the GeoTIFF band tags:
`mag_anom, rtp, tmi_hg, geod_2ndinv, iso_grav_anom_slope, tc, geod_shearrate, geod_dilaterate,
tmi_vg, deq_n100a15, iso_grav_anom_vg, det_elev, iso_grav_anom, tmi, depth_to_base_surf,
ieq_n100a15, cond_surf, iso_grav_anom_hg, det_elev_slope`.

## Metric

Distance-weighted Tversky index, `alpha = 0.2` (false positives), `beta = 0.8` (false
negatives), triangular kernel `k(d) = max(1 - d/R, 0)` with `R = 300 m = 3 px`:

```
TP_w = sum_{g in G} max_{x: d(x,g)<=R} p(x) * k(d(x,g))
FP_w = sum_{x: p(x)>0} p(x) * (1 - max_{g in G} k(d(x,g)))
FN_w = |G| - TP_w
DTI  = TP_w / (TP_w + 0.2*FP_w + 0.8*FN_w)
```

Pixels on already-mapped faults are masked out of the scored domain. Two consequences, both
proved in `tests/test_metric.py`:

* `1/DTI = alpha + alpha*(F/T) + beta*(K/T)` — so the denominator rises by exactly `alpha`
  for every unit of emitted mass, **wherever it lands**.
* a dot pays for itself iff its realised kernel weight exceeds `alpha * DTI`
  (= 0.0556 at DTI 0.2778), i.e. it must sit within ≈ 284 m of a hidden truth pixel.

## Layout

```
src/gems68/
  metric.py       official DTI + the two algebraic identities + the marginal bar
  data.py         grid, footprint, hash-pinned mirrors, memory-safe 47-channel feature stack
  autocontext.py  hide-and-recover folds, stage 1 (blocked buffered K-fold OOF),
                  stage 2 (auto-context), catalogue context, leakage canary support
  submission.py   writer (zeros/nan twins + zip), format validator, registry duplicate check
scripts/
  fetch_mirrors.sh   restore the competition rasters with no credentials (GitHub blob API)
  run_holdout.py     the gate: hide-and-recover holdout, stage 1 vs stage 2, canary
  build_submission.py  train, emit, write, validate, registry-check
tests/            pytest: metric vs brute force, identities, masking, grid, folds, writer
evidence/         JSON produced by each run
registry/rasters/ prior GEMSDOE submission rasters, for the duplicate check only
docs/             GitHub Pages site (index + executive summary + downloads)
```

## Run

```bash
bash scripts/fetch_mirrors.sh                       # restore the rasters (GitHub blob API)
python -m venv .venv && .venv/bin/pip install numpy scipy scikit-learn lightgbm rasterio pytest
PYTHONPATH=src .venv/bin/python -m pytest -q        # 9 metric/grid/fold tests
PYTHONPATH=src .venv/bin/python scripts/run_holdout.py --folds 4
PYTHONPATH=src .venv/bin/python scripts/build_submission.py --budget 40000
```

---

## The standing brief (re-read at the start of every session)

<details>
<summary><b>MUST GENERATE A UNIQUE TIF SUBMISSION FOR THE COMPETITION</b> … (full brief)</summary>

MUST GENERATE A UNIQUE TIF SUBMISSION FOR THE COMPETITION. DO NOT COPY A PREVIOUS SUBMISSION
UNLESS IT'S FOR LEARNING AND EDUCATION. BUT WE MUST GENERATE A UNIQUE TIF SUBMISSION. IT MUST
BE OBVIOUS WHETHER IT IS OK TO DOWNLOAD AND SUBMIT THE GENERATED TIF SUBMISSION.

There should be an easy to download submission tif file as described by the prompt. Read the
entire prompt.

Auto-context stacked refinement. Auto-context (Tu, CVPR 2008, doi:10.1109/CVPR.2008.4587436;
extended TPAMI version at pages.ucsd.edu/~ztu/publication/pami_autocontext.pdf) trains a
second classifier on the first classifier's probability maps plus the original features. It
can then learn spatial regularities a per-pixel model misses: continuity, parallelism,
en échelon spacing. Stage 1 produces out-of-fold probabilities from spatially blocked,
buffered K-fold, never in-fold predictions. Stage 2 is LightGBM or a small CNN on four kinds
of input. These are stage-1 probabilities sampled at ±1–15 px offsets along and across local
orientation, multi-scale anisotropic smooths, and the raw gradient/tilt/scarp channels. It
also gets catalogue-context features computed with the target block's labels masked
(distance, orientation and spacing to the nearest mapped fault). Iterate twice and estimate
the gain with nested blocked CV. Gate: stage 2 beats stage 1 on outer blocks by >2 SEs,
including on pixels >1 km from any mapped fault. Finish: normalize to [0,1]; write a float32
EPSG:32611 100 m GeoTIFF (NaN only outside the footprint); pass validate_submission.py; run a
hash/pixel-agreement check against all prior rasters; offer the download only if the gate is
met, else log the negative result.

PARALLEL-RUN PROTOCOL — read first. This session is one of several running from this same
prompt.

1. LANE. Your lane is the single method paragraph below. Stay inside it. If your raster's
   rank-correlation with any registry raster exceeds 0.90, or more than 70% of your dots fall
   within 3 px of one registry raster's dots, you have drifted into another lane: log it as a
   duplicate and stop. Check this on the surface before placement AND on the final dots.

2. REUSE, DON'T REBUILD. Use the template's cached feature stack, evaluate_holdout.py and
   submission_writer.py. Holdout = hide-and-recover: withhold whole fault segments with a
   buffer, derive every catalogue-based feature only from the visible faults, mask visible
   faults pixel-exactly, score pooled DTI (alpha 0.2, beta 0.8, 300 m triangular kernel). If a
   shared tool is wrong, fix it once in the template and report it; never keep a private fork.

3. LABEL EVERY NUMBER as HOLDOUT-DTI (evaluator version, number of withheld positives, 95%
   CI) or ORGANIZER-CONFIRMED (copied from a submission-page receipt). A projection is never
   written as a score.

4. LEAKAGE CANARY. Test each feature alone on the holdout before trusting any result. AUC
   above 0.90 means leakage until proven otherwise.

5. RUN CARD. End with one JSON card: hypothesis; mechanism; the named non-fault process that
   could mimic it; holdout DTI + CI; correlation/overlap vs registry; raster sha256; validator
   output (no NaN inside the footprint, values in [0,1], CRS/shape/transform match);
   submission name + note of at most 140 characters; verdict promote / negative. Negative
   results are deliverables.

6. BUDGET. Stop after 3 experiments or 2 hours. Do not pick submissions: promotion to a real
   slot is a separate selector step, within the weekly cap shown on the submission page.

…before implementing, generate 3–5 candidate geological hypotheses we haven't tried yet, each
naming: the specific layer(s) involved, the physical signature being targeted, why it should
catch a fault missing from the USGS/INGENIOUS catalogue rather than one already in it, and how
it differs from anything already implemented in this repo. Rank them by expected DTI
improvement and implementation cost. Validate the top candidate on our spatially-blocked
holdout set before touching a weekly submission slot.

Work line by line verifying from official verified trusted sources, provide links for manual
review. There should be no manual input, work on your own to complete tasks. Flag any
irregularities for review. No hallucinations. Verify no hallucinations.

Core values: **Maximize P(Win)** — in every decision weigh trade-offs, assess risk, choose the
path that maximises the probability that the project succeeds. **Own the Outcome** — own
results end to end, act without waiting for permission, treat failure and success as signals.

Run this task through multiple passes. Pass 1: implement completely and verify. Pass 2: review
for bugs, missing requirements, incorrect assumptions, edge cases; fix everything found.
Pass 3: re-check the entire implementation against the original request; improve accuracy,
reliability, completeness, code quality; fix remaining issues.

</details>

---

## Irregularities flagged for review

| id | what | evidence |
| --- | --- | --- |
| IR-68-01 | The brief says *"0.3195 is the highest score right now"* and elsewhere *"0.3774"*. The most recent **verified** public-leaderboard snapshot mirrored in `buffedlizard55-lab/60GEMSDOE/registry/leaderboard_history.jsonl` (observed 2026-10-11, 50 rows) has **rank 1 = `xiaofanhu` 0.4133**, rank 2 `JerryDataWorks` 0.3418, and `DARD` 0.3195 at rank 8. Both numbers in the brief are stale. **Unchecked here:** drivendata.org is not reachable from this sandbox, so the live board could not be re-read. | `registry/leaderboard_snapshot.json` |
| IR-68-02 | `labels.tif` and `existing_faults.tif` are **byte-identical** (one sha256). The training target available to us is therefore the mapped-fault catalogue, not an independent label set. Every holdout number in this repo is a **catalogue-proxy** DTI, labelled as such. | sha256 `7ba308cc…` for both files |
| IR-68-03 | No score quoted in this README or on the site is organizer-confirmed. `GEMSDOE32`'s 0.2778 is an **owner report**; the mirrored registry itself warns *"None of them has an organizer receipt linking those bytes to that row"* (`60GEMSDOE/registry/claims.json`, IR-32-SCORE-01). | `registry/claims.json` (upstream) |
| IR-68-04 | `raw.githubusercontent.com` is not reachable from this sandbox; the mirrored rasters must be fetched through `api.github.com` blob endpoints. `scripts/fetch_mirrors.sh` does this. | observed HTTP 302 → connection refused |
| IR-68-05 | The catalogue-proxy holdout is **anti-monotone** in emission budget while the live board is monotone (fewer dots → higher), because the organizer masks mapped-fault pixels and the proxy's truth *is* that catalogue. Documented upstream in `GEMSDOE32`. The emission budget is therefore a design choice, not a holdout optimum. | `docs/research/emission-budget.md` |
