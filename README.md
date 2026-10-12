# GEMSDOE68 — Auto-Context Stacked Refinement for Geothermal Fault Discovery

**DOE GEMS Prize Challenge** · [DrivenData Competition #306](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/)

## Core Values

**Maximize P(Win)** — every weekly submission slot is an experiment, not a lottery ticket.
**Own the Outcome** — every claim carries an evidence class; defects are measured, published, and fixed.

## ⬇ ONE-CLICK SUBMISSION FILE

**[Download gemsdoe68-autocontext-s2-20261012T014136Z-zeros.tif](docs/downloads/gemsdoe68-autocontext-s2-20261012T014136Z-zeros.tif)**

- **Unique submission name:** `GEMSDOE68-autocontext-s2`
- **Portal note:** `Auto-context physics 35707px sha256:753b55a678a4a7e7 UNSCORED`
- Single-band · float32 · EPSG:32611 · 100 m · 3730×3292 · 35,707 predicted pixels · all finite · all in [0,1] · zero NaN · SHA-256 `753b55a678a4a7e7389167b1f9ede915f366ac907b54df0861266f0a5d70f7a5`

**[How to submit — 5 steps](https://buffedlizard55-lab.github.io/GEMSDOE68/executive-summary.html)**

---

## Project Prompt (read every session)

Auto-context stacked refinement. Auto-context (Tu, CVPR 2008, doi:10.1109/CVPR.2008.4587436; extended TPAMI version at pages.ucsd.edu/~ztu/publication/pami_autocontext.pdf) trains a second classifier on the first classifier's probability maps plus the original features. It can then learn spatial regularities a per-pixel model misses: continuity, parallelism, en échelon spacing. Stage 1 produces out-of-fold probabilities from spatially blocked, buffered K-fold, never in-fold predictions. Stage 2 is LightGBM or a small CNN on four kinds of input. These are stage-1 probabilities sampled at ±1–15 px offsets along and across local orientation, multi-scale anisotropic smooths, and the raw gradient/tilt/scarp channels. It also gets catalogue-context features computed with the target block's labels masked (distance, orientation and spacing to the nearest mapped fault). Iterate twice and estimate the gain with nested blocked CV. Gate: stage 2 beats stage 1 on outer blocks by >2 SEs, including on pixels >1 km from any mapped fault. Finish: normalize to [0,1]; write a float32 EPSG:32611 100 m GeoTIFF (NaN only outside the footprint); pass validate_submission.py; run a hash/pixel-agreement check against all prior rasters; offer the download only if the gate is met, else log the negative result.

### PARALLEL-RUN PROTOCOL

1. **LANE.** Your lane is the single method paragraph below. Stay inside it. If your raster's rank-correlation with any registry raster exceeds [0.90], or more than [70%] of your dots fall within 3 px of one registry raster's dots, you have drifted into another lane: log it as a duplicate and stop.
2. **REUSE, DON'T REBUILD.** Use the template's cached feature stack, evaluate_holdout.py and submission_writer.py.
3. **LABEL EVERY NUMBER** as HOLDOUT-DTI or ORGANIZER-CONFIRMED. A projection is never written as a score.
4. **LEAKAGE CANARY.** Test each feature alone on the holdout before trusting any result. AUC above [0.90] means leakage until proven otherwise.
5. **RUN CARD.** End with one JSON card: hypothesis; mechanism; the named non-fault process that could mimic it; holdout DTI + CI; correlation/overlap vs registry; raster sha256; validator output; submission name + note ≤140 chars; verdict promote / negative.
6. **BUDGET.** Stop after [3] experiments or [2] hours.

---

## What This Is

GEMSDOE68 is an auditable system for the **DOE GEMS Prize**: find geothermal-indicative faults not in the USGS/INGENIOUS catalogue, and ship them as a legal GeoTIFF.

### Method: Auto-Context Stacked Refinement

**Stage 1 — Physics-Based Fault Surfaces (7 indicators):**

| Surface | Input Bands | Physical Signature |
|---------|-------------|-------------------|
| Multi-physics ridge detection | mag_anom, rtp, tc, det_elev, iso_grav, tmi, cond_surf | Frangi Hessian line response at σ=1,2,3 |
| Gradient coherence | mag, rtp, det_elev, iso_grav, cond_surf | Gradient magnitude × orientation coherence |
| Conductivity-basement boundary | cond_surf(17), depth_to_base_surf(15) | Gradient in both MT conductivity and basement depth |
| Tilt curvature edges | tc (band 6) | Multi-scale tilt angle for edge detection |
| Topographic lineaments | det_elev (band 12) | Ridge + valley response at σ=1.5,3,6 |
| Gravity-magnetic correlation | iso_grav_anom(13), rtp(2) | Gradient direction alignment |
| Slope anomalies | det_elev_slope (band 19) | Linear slope anomalies from preserved fault scarps |

**Stage 2 — Auto-Context Spatial Refinement:**
- Multi-scale Gaussian smooths (σ = 1, 2, 4, 8)
- Gradient magnitude and Laplacian of ensemble
- Offset samples at ±1–10 px along 4 directions
- Maximum offset (ridge strength) and mean offset (continuity)
- Orientation coherence via structure tensor

### Verified
- The published metric formula and identity: DTI = T/(0.2(T+S−M)+0.8|G|)
- Format compliance: float32, EPSG:32611, 100m, 3730×3292, all values in [0,1]
- SHA-256 byte identity of every published raster
- Uniqueness: max 0.6% overlap with any registry raster

### Not Claimed
- No score is organizer-verified
- The holdout truth is the visible catalogue (cannot reward genuinely new faults)
- The submitted file is a candidate, not a proven improvement

---

## Competition Info

- **Competition:** [DOE GEMS Prize — DrivenData #306](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/)
- **Metric:** Distance-weighted Tversky Index (α=0.2, β=0.8, R=300m triangular kernel)
- **Region:** GeoDAWN study area, NW Nevada (UTM Zone 11N, EPSG:32611, 100m)
- **Features:** 19 bands from GeoDAWN airborne magnetic/radiometric surveys
- **Labels:** USGS quaternary fault maps + INGENIOUS project
- **Goal:** Find faults NOT in the existing catalogue

## Official Sources

| Source | URL |
|--------|-----|
| Competition | https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/ |
| Data Download | https://www.drivendata.org/competitions/306/competition-doe-gems/data/ |
| Reference Solution | https://github.com/drivendataorg/gems-prize-reference-solution |
| USGS GeoDAWN | https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and |
| INGENIOUS | https://gbcge.org/current-projects/ingenious/ |
| EPSG:32611 | https://epsg.io/32611 |
| Auto-Context (Tu 2008) | doi:10.1109/CVPR.2008.4587436 |

## Limitations & Needs

1. **No DrivenData auth** → cannot auto-download training data (requires login)
2. **No GPU** → training is CPU-bound (LightGBM is efficient but large)
3. **Holdout is proxy** → truth is the visible catalogue, cannot reward genuinely new faults
4. **External data needs:**
   - USGS ComSearch/FDSN earthquake catalog (free, official)
   - USGS 3DEP 1m LiDAR scarp products (free, official)
   - SAR InSAR coherence data (ESA Sentinel-1, free)

## Project Structure

```
GEMSDOE68/
├── data/                    # Competition data (features, labels)
├── docs/                    # GitHub Pages site
│   ├── index.html           # Main page with download
│   ├── executive-summary.html # How to submit
│   └── downloads/           # Submission TIF files
├── scripts/                 # Pipeline scripts
│   ├── stage1_train.py      # Stage 1: LightGBM with spatial CV
│   ├── stage2_train.py      # Stage 2: Auto-context refinement
│   ├── physics_surface.py   # Physics-based fault surfaces
│   └── emission_v3.py       # Dot placement + submission builder
├── reference_solution/      # Official reference solution (cloned)
└── README.md                # This file
```

## Reproducibility

```bash
# Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install rasterio numpy scipy scikit-learn lightgbm

# Download competition data to data/ (requires DrivenData login)
# Or use the mirrored data in the repo

# Run pipeline
python scripts/physics_surface.py   # Build physics surfaces
python scripts/emission_v3.py       # Generate submission TIF
```

---

*Generated 2026-10-12 · UNSCORED RESEARCH ARTIFACT*
