# GEMSDOE68 — Auto-Context Stacked Refinement for Fault Discovery (DOE GEMS Prize, DrivenData #306)

## ✅ RELEASE VERDICT: `PROMOTE` — OK TO DOWNLOAD AND SUBMIT

Every gate required by the **Auto-context stacked refinement lane** (Tu, CVPR 2008, [doi:10.1109/CVPR.2008.4587436](https://doi.org/10.1109/CVPR.2008.4587436)) and the **Parallel-Run Protocol** is **PASSED**:

| Release Gate | Measured Result (`evidence/exp3_autocontext_strands.json` & `evidence/run_card.json`) | Status |
|---|---|---|
| **Primary Downloadable GeoTIFF (All-Finite `[0, 1]`, 0 NaNs)** | [`docs/downloads/gems68-autoctx-s3-strands-35000dots-20261020T210000Z-849a0f04-zeros.tif`](docs/downloads/gems68-autoctx-s3-strands-35000dots-20261020T210000Z-849a0f04-zeros.tif) (`345,388` bytes) · [`.zip`](docs/downloads/gems68-autoctx-s3-strands-35000dots-20261020T210000Z-849a0f04-zeros.zip) (`307,983` bytes) | **READY** |
| **Secondary NaN-Outside GeoTIFF Twin** | [`docs/downloads/gems68-autoctx-s3-strands-35000dots-20261020T210000Z-849a0f04-nan.tif`](docs/downloads/gems68-autoctx-s3-strands-35000dots-20261020T210000Z-849a0f04-nan.tif) (`419,642` bytes) · [`.zip`](docs/downloads/gems68-autoctx-s3-strands-35000dots-20261020T210000Z-849a0f04-nan.zip) (`334,495` bytes) | **READY** |
| **Submission Name** | `gems68-autoctx-s3-strands-35000dots-20261020T210000Z-849a0f04-zeros` | **UNIQUE** |
| **Submission Note (`<= 140` chars)** | `GEMS68 Tu08 auto-ctx s3 strands 35k dots; holdout DTI 0.1124 (+0.0182, t=11.1), >1km 0.0874 (+0.0102, t=2.53); sha8=849a0f04` (`124` chars) | **PASS** |
| **Primary `-zeros.tif` SHA-256** | `849a0f04d2ca8e644eb346ac30c3f3a467024de447b3e8e3e44eecf2f814ae35` | **VERIFIED** |
| **Secondary `-nan.tif` SHA-256** | `3f9553609ad7260f9e9a5520bde56ed4a728a9fa484e78fd59fe764101f8185d` | **VERIFIED** |
| **Format & `[0, 1]` Preflight (`validate.py`)** | `float32`, single band, `EPSG:32611`, `3730 x 3292` (100 m), transform `(100, 0, 243350, 0, -100, 4508550)`, `n_nan = 0` on `-zeros.tif`, values strictly in `[0, 1]`, `0` dots on mapped catalogue | **PASS (14/14 checks)** |
| **Holdout Gate — All Pixels (`11,321` withheld pos, 35k dots)** | Stage 1 `0.0942` → Stage 2 `0.1058` (`Δ +0.0116`, CI `[+0.0058, +0.0178]`, **`t = 6.47 SE`**) → Stage 3 **`0.1124`** (`Δ +0.0182`, CI `[+0.0110, +0.0259]`, **`t = 11.08 SE`**, 4/4 quadrants positive) (`HOLDOUT-DTI`, `gems57-pooled-hide-v2`) | **PASS (`> 2 SE` & `CI > 0`)** |
| **Holdout Gate — `> 1 km` Far Pixels (`4,310` withheld pos, 35k dots)** | Stage 1 `0.0772` → Stage 2 `0.0859` (`Δ +0.0087`, CI `[+0.0017, +0.0160]`, **`t = 2.90 SE`**) → Stage 3 **`0.0874`** (`Δ +0.0102`, CI `[+0.0012, +0.0187]`, **`t = 2.53 SE`**, 4/4 quadrants positive `[+0.0217, +0.0131, +0.0079, +0.0012]`) (`HOLDOUT-DTI`, `gems57-pooled-hide-v2`) | **PASS (`> 2 SE` & `CI > 0`)** |
| **Single-Feature Leakage Canary (`<= 0.90` AUC)** | Stage 1 max: `elev_asym = 0.5955 <= 0.90` (37 features); Stage 2/3 max: `cat_relay_15 = 0.7686 <= 0.90` (55 features) | **PASS** |
| **65-Raster Uniqueness Gate (`|rho| <= 0.90`, `<= 70%` 3-px overlap)** | Checked against all 55 registry rasters + 10 scored calibration rasters: max surface `|rho| = 0.1857 <= 0.90`; max dot `|rho| = 0.0462 <= 0.90`; max informative 3-px dot overlap = **`55.37% <= 70%`** (`gems10-h28-dotted-ridge`); only **`20.15%`** 3-px overlap with Exp 2 | **PASS** |
| **Eligible-Domain Density-Null Proof (`IR-68-UNIQ-02`)** | On space-filling `gems61...k400k-b2zeros.tif` (`311,617` dots, `99.38%` 3-px coverage of `d_cat > 2 px`), observed overlap `0.9843` is **`-0.0092` (`-21.4 SD`) below** random expectation (`0.9936`) | **PASS** |
| **Live Leaderboard Score** | **UNSCORED** (no number in this repo is a live projection; all holdout numbers are tagged `HOLDOUT-DTI`) | **DISCLOSED** |

---

## 1. GitHub Pages Website & Canonical Artifacts

- **Executive Summary & One-Click Download Page:** [`docs/index.html`](docs/index.html) and [`docs/executive-summary.html`](docs/executive-summary.html) (mirrored in [`docs/lane68/index.html`](docs/lane68/index.html))
- **Step-by-Step Submission Guide:** [`docs/how-to-submit.html`](docs/how-to-submit.html)
- **Experiments Ledger (`HOLDOUT-DTI` across Exp 1, Exp 2, Exp 3):** [`docs/experiments.html`](docs/experiments.html)
- **Ranked Geological Hypotheses (`H68-1` to `H68-5`):** [`docs/hypotheses.html`](docs/hypotheses.html) / [`docs/hypotheses.md`](docs/hypotheses.md)
- **PhD Analysis of `0.2778` (`h33-2-b2-zeros`):** [`docs/knowledge.html`](docs/knowledge.html) / [`knowledge/01_why_02778_and_can_we_beat_it.md`](knowledge/01_why_02778_and_can_we_beat_it.md)
- **Irregularities Ledger (`IR-68-BLOB-01`, `IR-68-UNIQ-02`, `IR-68-GATE-01` resolved):** [`docs/irregularities.html`](docs/irregularities.html) / [`docs/irregularities.md`](docs/irregularities.md)
- **Machine-Readable Run Card & Evidence JSONs:** [`evidence/run_card.json`](evidence/run_card.json), [`docs/runcard.json`](docs/runcard.json), [`evidence/exp3_autocontext_strands.json`](evidence/exp3_autocontext_strands.json), [`evidence/ac68_density_null.json`](evidence/ac68_density_null.json)

---

## 2. Method: Tu (2008) Auto-Context Strand Refinement (`exp3_autocontext_strands`)

1. **Stage 1 (37 Local Scarp, Footwall-Asymmetry & Geophysical Step Channels — Fixes `IR-68-BLOB-01`):**
   - Earlier Experiment 2 trained Stage 1 on raw basin-scale bands (`det_elev` gain `68,248`, `ieq_n100a15` `52,494`, `deq_n100a15` `46,843`, `geod_2ndinv` `42,534`, `geod_dilaterate` `41,902`, `cond_surf` `40,325`) whose `50–200 km` wavelengths acted as spatial lookup tables memorizing training-quadrant polygons (`IR-68-BLOB-01`).
   - Experiment 3 (`scripts/build_features.py` → `build_exp3_features()`) strips all smooth regional lookup bands and builds `X1_exp3.npy` (`5,167,373 × 37`) from purely local structural channels:
     - **H68-2 Footwall/Hanging-Wall Slope Asymmetry & Local Relief:** `elev_asym = |∇z(+500 m⊥) - ∇z(-500 m⊥)|` across local strike (#1 single-feature holdout AUC `0.5955`, #2 Stage-1 GBDT gain `37,957.9`), `elev_hp15` (#1 Stage-1 GBDT gain `39,846.1`), `elev_grad` (`17,265.7`), `elev_ridge_s2` (`14,422.5`), `elev_lap`, `elev_ridge_s1`, `elev_tilt`, `det_elev_slope`.
     - **H68-2 2 m Bare-Earth LiDAR Scarp Channels (`data/external/lidar_scarp_features_u8.tif`):** `lidar_lappos_max` (AUC `0.5949`), `lidar_relief` (`0.5927`), `lidar_lapneg_max` (`0.5892`), `lidar_step_max` (`0.5871`), `lidar_ex_max` (`0.5841`), `lidar_upface_max` (`0.5702`), `lidar_downface_max`, `lidar_cross_max`, `lidar_p95_2m`, `lidar_scarp_contrast = max(downface, upface) - cross_max`.
     - **H68-3 Basement-Step & Potential-Field Derivatives:** `db_grad = |∇(depth_to_base_surf)|` (#3 Stage-1 gain `28,089.5`), `db_hp15` (#4 gain `18,644.6`), `tc`, `tc_grad`, `cs_grad`, `cs_hp15`, `iso_grav_anom_hg/vg/slope`, `tmi_hg/vg`, `rtp_ridge_s2`, `grav_ridge_s2`.
     - **H68-4 High-Pass GeoDAWN Radiometric & Magnetic Extensions (`data/external/geodawn_extensions_u8.tif`):** `1.2 km` high-pass residuals `ext_ThK_hp`, `ext_UK_hp`, `ext_UTh_hp`, `ext_TMI_up150_hp`.
   - Out-of-fold Stage-1 probabilities are predicted across 4 spatially blocked quadrants with a `12 px` (`1.2 km`) buffer, never in-fold.
2. **Stage 2 & Stage 3 Auto-Context (Iterated Twice with Hide-Draw Labels & Damped Rank-Probability Stacking):**
   - Each stage `k in {2, 3}` takes 55 features across the four required families:
     - **(a) Oriented Stage-`(k-1)` probability offsets (`20` features):** sampled at `±1, 2, 4, 8, 15 px` along (`p_along±d`) and across (`p_across±d`) the local structure-tensor strike of `P^(k-1)`.
     - **(b) Multi-scale anisotropic smooths & ridge sharpness (`10` features, no `σ=10 px` blob smooth):** `p_self`, `p_sm1` (`σ=1.5`), `p_sm3` (`σ=3.5`), `p_along_short` (`±1..4 px`), `p_along_long` (`±8..15 px`), `p_along_mean`, `p_across_mean`, `p_aniso`, `p_ridge = p_self - p_across_mean`, `ori_coh`.
     - **(c) Raw local gradient/tilt/scarp channels (`17` features):** `elev_grad`, `elev_lap`, `elev_ridge_s1`, `elev_ridge_s2`, `elev_tilt`, `elev_asym`, `rtp_ridge_s2`, `grav_ridge_s2`, `tc_grad`, `db_grad`, and 7 2 m LiDAR scarp channels.
     - **(d) Masked catalogue-context features (`8` features):** `cat_dist`, `cat_strike_cos2`, `cat_strike_sin2`, `cat_align`, `cat_dens_1km`, `cat_dens_3km`, plus **directional along-strike fault-tip relays** `cat_relay_15` (`1.5 km` tip projection, single-feature holdout AUC `0.7686`, #2 Stage-2 gain `46,219.4`) and `cat_relay_30` (`3.0 km` tip projection).
   - Trained on 3 independent buffered component-hide draws (`seeds = 20, 21, 22`) with `2.5x` up-weighting of `> 1 km` positives and damped rank-probability sequential refinement (`α2 = 0.22, α3 = 0.35`).
   - Emits `35,000` 5×5-NMS local maxima outside a `2 px` (`200 m`) catalogue exclusion collar, placing **`20,178 / 35,000` dots (`57.7%`) `> 1 km` from any mapped fault**.

---

## 3. Reproducing the Full Pipeline & Running Tests

```bash
# 1. Restore all 23 sha256-pinned competition, external, and scored calibration rasters
.venv/bin/python scripts/restore_data.py --group all

# 2. Build cached feature matrices (.cache/features/X1.npy and .cache/features/X1_exp3.npy)
.venv/bin/python scripts/build_features.py

# 3. Run Experiment 3 nested blocked CV, canary, uniqueness gate, GeoTIFF emission & run cards
.venv/bin/python scripts/run_exp3.py

# 4. Run eligible-domain Monte Carlo density-null control (IR-68-UNIQ-02)
.venv/bin/python scripts/lane68_density_null.py

# 5. Rebuild all GitHub Pages HTML pages in docs/ and docs/lane68/
.venv/bin/python scripts/build_site.py

# 6. Run test suite
PYTHONPATH=src .venv/bin/pytest tests -v
```

---

## 4. Verbatim Session Prompt (Preserved as Starting Point)

> Review the repository (buffedlizard55-lab/GEMSDOE68) and work on the next steps from previous sessions.
>
> HIGHEST URGENCY:
> - MUST GENERATE A UNIQUE TIF SUBMISSION FOR THE COMPETITION. DO NOT COPY A PREVIOUS SUBMISSION UNLESS IT IS FOR LEARNING/EDUCATION.
> - IT MUST BE OBVIOUS ON THE WEBSITE WHETHER IT IS OK TO DOWNLOAD AND SUBMIT THE GENERATED TIF SUBMISSION.
> - THERE MUST BE AN EASY WAY TO DOWNLOAD THE TIF SUBMISSION AT THE VERY BEGINNING OF THE WEBSITE, IN THE EXECUTIVE SUMMARY OR FIRST SCREEN OF THE SITE, AND AN EXECUTIVE SUMMARY SUBPAGE THAT EXPLAINS HOW TO MAKE A SUBMISSION.
> - MUST PROVIDE A UNIQUE SUBMISSION NAME AND COMMENT OF MAX 140 CHARACTERS.
> - PREVIOUSLY WE DOWNLOADED A FILE FROM ONE OF THE REPOS AND SUBMITTED IT TO THE COMPETITION AND GOT THE FOLLOWING ERROR ON THE SUBMISSION FORM: "Predicted values must be in range [0, 1]".
>
> Core Values — Keep These Central:
> - Maximize P(Win): Weigh trade-offs, assess risk, and choose the path that maximizes our probability of winning.
> - Own the Outcome: Own results end to end, act when problems arise without waiting for permission, and treat failure and success as signals to keep improving.
>
> Single method paragraph (our lane):
> Auto-context stacked refinement. Auto-context (Tu, CVPR 2008, doi:10.1109/CVPR.2008.4587436; extended TPAMI version at pages.ucsd.edu/~ztu/publication/pami_autocontext.pdf) trains a second classifier on the first classifier's probability maps plus the original features — so it can learn spatial regularities a per-pixel model cannot see (continuity, parallelism, en-echelon spacing). Stage 1 produces out-of-fold probabilities from spatially-blocked, buffered K-fold (never in-fold predictions). Stage 2 (LightGBM or a small CNN) takes four input families: stage-1 probabilities sampled at +-1-15 px offsets along and across local orientation, multi-scale anisotropic smooths, raw gradient/tilt/scarp channels, and catalogue-context features computed with the target block's labels masked (distance, orientation, and spacing to the nearest mapped fault). Iterate twice and estimate gain with nested blocked CV. Gate: stage 2 beats stage 1 on outer blocks by >2 SEs, including on pixels >1 km from any mapped fault. Finish: normalize to [0,1]; write a float32 EPSG:32611 100 m GeoTIFF (NaN only outside the footprint); pass validate_submission.py; run a hash/pixel-agreement check against all prior rasters; offer the download only if the gate is met, else log the negative result.
>
> Parallel-run protocol (all 6 rules):
> 1. STAY IN YOUR LANE. Log a duplicate and abort if rank-correlation with any registry raster exceeds 0.90 or >70% of your dots fall within 3 px of one raster's dots (check on the surface before placement and on the final dots).
> 2. REUSE, DON'T REBUILD. Pull the cached feature stack, evaluate_holdout.py, and submission_writer.py from the shared template. If a shared tool has a bug, fix it once in the template and report it.
> 3. LABEL EVERY NUMBER. Every score is either HOLDOUT-DTI (with evaluator version, number of withheld positives, and 95% CI) or ORGANIZER-CONFIRMED (with the submission-page receipt). Never write a projection as a score.
> 4. RUN THE LEAKAGE CANARY. Test each feature alone on the holdout; any single feature with holdout AUC > 0.90 is leakage until proven otherwise.
> 5. END WITH ONE RUN CARD. Output a single JSON block: hypothesis, mechanism, named non-fault process that could mimic the signature, holdout DTI + CI, correlation/overlap with the registry, raster sha256, validator output, and verdict (promote / negative).
> 6. STOP ON BUDGET. Max 3 experiments or 2 hours. Do not pick which submissions to upload — a separate selector step promotes from the run cards to the weekly submission slots.
>
> The full verbatim prompt text is also archived in [`BRIEF.md`](BRIEF.md) and [`docs/PROMPT.md`](docs/PROMPT.md).
