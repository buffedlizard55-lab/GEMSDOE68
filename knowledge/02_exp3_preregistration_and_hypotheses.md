# GEMSDOE68 — Exp-3 preregistration + candidate geological hypotheses

*Written 2026-10-20 (this session), BEFORE any Exp-3 result was computed.
Lane: auto-context stacked refinement (Tu, CVPR 2008, doi:10.1109/CVPR.2008.4587436).
Instrument: the exp2 hide-draw nested blocked CV (`src/gems68/experiment.py::Lane`,
evaluator `gems57-pooled-hide-v2`, pooled DTI α=0.2 β=0.8, 300 m triangular kernel,
11,321 withheld positive pixels at eval seed 20). The lane68 `ac68_seed20` instrument
is NOT used for decisions: it emits the full dot budget in every outer fold, so its
headline DTI is not matched-mass comparable with exp2 (see IR log).*

## Why 0.2778 won, in one paragraph (measured lineage, knowledge/01)

The champion `h33-h33-2-b2` (OWNER-REPORTED 0.2778; bytes audited in this repo) is
37,654 binary dots: the end of a pruning ladder on one per-pixel field
(0.1922 @121k px → 0.2477 @60k → 0.2600 @44k → 0.2708 after removing the d=1 px
catalogue halo → 0.2778 after removing d=2 px). DTI charges 0.2 per emitted unit
earning no 300 m kernel credit, so (a) mass discipline and (b) staying >2 px off the
published catalogue paid ~+0.086 without any new geology. What remains is *ranking*:
at ~38k dots, more kernel credit per dot toward genuinely unmapped faults. The
brief's leaderboard figures (0.3195, then 0.3774) are OWNER-REPORTED in the prompt,
not re-fetched (the DrivenData leaderboard is outside this sandbox's allowlist);
they are targets, not measurements made here.

## Candidate hypotheses NOT yet tried in this repo (ranked)

### H68-A (TOP — Exp-3 primary): ridge-crest emission on the auto-context surface
- **Layers:** stage-2/3 probability map + its structure-tensor orientation field
  (`features.orientation_field`, σ_grad 1.5, σ_win 4); raw channels already inside
  stages 1–3 (`det_elev`, `rtp`, `tmi_hg`, gravity, geodesy, MT + 7 physics channels).
- **Physical signature:** a fault is a *line*, so probability mass should sit on the
  across-strike crest of the stacked-evidence surface. Emission candidates = pixels
  that are local maxima across local strike (±1, ±2 px), then greedy ≥2.8 px spacing
  (the champion ladder's winning thinning scale), >2 px off the visible catalogue.
- **Why it should catch catalogue-missing faults:** mappers catalogue high-relief
  range-front scarps; blind/buried normal faults under alluvium give weak per-pixel
  responses in every single band but a coherent linear crest once evidence is stacked
  and spatially corroborated by auto-context. Isotropic 5×5-NMS emission (all prior
  candidates here) spends dots filling the *area* of regional probability highs
  (repo irregularity IR-68-BLOB-01: median dot 5.1 km from catalogue, blob fill);
  crest emission concentrates the same budget on the line itself, which is the only
  place the 300 m kernel pays.
- **Difference from anything implemented:** no prior run in this repo constrained
  emission to orientation crests or enforced exact greedy min-spacing; exp1/exp2/
  lane68 all used isotropic 5×5 maxima + top-k.
- **Expected DTI effect / cost:** moderate gain at matched mass if the blob-fill
  diagnosis is right; cost = emission-only change on cached outer maps (no retrain
  beyond reproducing exp2 maps). Named non-fault mimic: erosional escarpment /
  shoreline-terrace crests and bedding-parallel ridges produce identical crests.

### H68-B: en-échelon stepover gap completion (Exp-3 does NOT implement; next)
- Layers: stage-2/3 map sampled along strike at ±7/±15 px (already stage-k inputs).
- Signature: min(p_along+15, p_along−15) high while p_self moderate = relay/stepover
  gap between collinear segments; emit bridge dots there. Unmapped because stepover
  relays often lack a through-going scarp. Not tried as an explicit emission rule.
  Cost: emission-only, but needs its own preregistered holdout pass.

### H68-C: far-specialist stage-2 reweighting
- Layers: catalogue-context distance (`cat_dist`) used as training sample weight.
- Signature: upweight withheld training positives >1 km from the visible catalogue so
  stage 2 stops trading far recovery for near-catalogue density cues. Targets isolated
  intra-basin faults. Cost: full stage-2/3 retrain. Do only if Exp-3's far gate fails.

### H68-D (Exp-3 secondary variant V3): cross-physics crest concordance
- Layers: Hessian ridge channels `elev_ridge_s2`, `rtp_ridge_s2`, `grav_ridge_s2`.
- Signature: rank crest candidates by rank(surface) × rank(max |ridge| support):
  a buried basement fault offsets magnetic/gravity basement with no surface scarp, so
  the probability crest should coincide with an independent potential-field ridge.
  Sibling lanes used potential-field ridges as standalone fields (OWNER-REPORTED
  0.05–0.26); none used them as a concordance ranker on an auto-context crest.

### H68-E: 1 m lidar micro-scarp crests — NOT viable this session
- Would need USGS 3DEP 1 m DEMs (free, official: https://www.usgs.gov/3d-elevation-program),
  but this sandbox's outbound allowlist (github/npm/pypi only) cannot fetch them and no
  lidar sidecar is pinned in `registry/data_manifest.json` for the exp2 lane. Named and
  parked; not proposed as viable until the tiles are mirrored and hash-pinned.

## Exp-3 design (single experiment; emission variants on one set of outer maps)

Training: identical to exp2 (same `Lane`, seeds, LightGBM params, 3 stages, hide-draw
train seeds 20/21/22, eval seed 20). Outer maps cached to `.cache/exp3/`.
Replication check: variant V0 must reproduce exp2's committed gate table within
±0.005 DTI at 25k/38k (LightGBM/BLAS nondeterminism tolerance); if it does not, Exp-3
is void and the discrepancy is logged instead of a candidate shipped.

Emission arms (all: binary 1.0, exclude ≤2 px from the *visible* catalogue, per-fold
area quota, budgets **25,000 and 38,000 only** — preregistered to limit fishing):
- `s1_V0`, `s2_V0`, `s3_V0` — exp2 rule (5×5 local maxima, top-k) = control / current best.
- `s2_V2`, `s3_V2` — spacing-only control: all allowed pixels, greedy ≥2.8 px by value.
- `s2_V1`, `s3_V1` — H68-A: across-strike crest candidates, greedy ≥2.8 px by value.
- `s3_V3` — H68-D: crest candidates, greedy ≥2.8 px by rank-product with ridge support.

Scopes (all HOLDOUT-DTI, evaluator `gems57-pooled-hide-v2`):
- `all`: full fold domains (11,321 withheld px).
- `far`: exp2-comparable stratum — scoring domain restricted to >10 px (1 km) from any
  masked-known catalogue pixel (dots outside the stratum are ignored by the evaluator;
  disclosed, identical treatment for every arm).
- `farfull` (sensitivity): truth restricted to far pixels but *all* emitted dots in the
  region count as FP mass (the lane68-style far definition).

Leakage canary: single-feature AUCs recomputed exactly as exp2; any AUC (or 1−AUC)
> 0.90 flags the run.

## Decision rule (binding)

Promote at most ONE arm, chosen as: among {s3_V1, s2_V1, s3_V3, s3_V2} at the
38,000 budget (25,000 reported for context; live mass anchored to the champion
ladder's 37,654), the arm with the highest `all` DTI that satisfies ALL of:
1. paired 20 km-block bootstrap 95 % CI of Δ vs `s1_V0` entirely > 0 in `all` AND `far`;
2. paired 95 % CI of Δ vs `s3_V0` (current holdout best) entirely > 0 in `all`;
3. point Δ vs `s1_V0` > 0 in `farfull`;
4. canary clean; final raster passes the shared validator on re-read bytes;
5. lane check vs every raster in `registry/rasters/rasters/`: sha256 distinct,
   |Spearman| < 0.90, and dot-overlap-within-3 px reported literally for every raster
   plus a density-null for any raster over 0.70 (a literal breach that the null shows
   is forced by that raster's own density is disclosed, not hidden; a breach vs any
   non-degenerate raster = duplicate = negative).

If no arm satisfies 1–5: verdict **negative**, no download offered, results logged.
Selection among the four preregistered arms on one holdout carries mild selection
bias; it is disclosed here rather than eliminated. No ORGANIZER-CONFIRMED score is
claimed for anything in this repo; promotion to a weekly slot remains the selector's
separate step.
