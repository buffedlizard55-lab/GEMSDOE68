# GEMSDOE68 — Exp-4 preregistration (stratified spacing), 2026-10-20

*Written after Exp-3's negative result was read and diagnosed, BEFORE any Exp-4
number was computed. This is experiment 2 of this session's 3-experiment budget.
Exp-3's verdict (`evidence/exp3_ridgecrest.json`: replication OK, canary clean,
chosen arm = none) is not revised by anything here.*

## What Exp-3 measured (HOLDOUT-DTI, `gems57-pooled-hide-v2`, 11,321 withheld px)

At 38,000 dots, `all` scope: s1_V0 0.0829 · s3_V0 (previous best) 0.1204 ·
s3_V2 (greedy ≥2.8 px spacing) **0.1665** · s3_V1 (crest+spacing) 0.1632 ·
s2_V1 0.1583 · s3_V3 (concordance) 0.1248. Spacing is a large matched-mass win
(Δ vs s3_V0 CI [0.035, 0.059]); the crest restriction adds nothing on top of
spacing (−0.003); concordance ranking hurts.

But every candidate arm failed the preregistered far gates at 38k: e.g. s3_V2
far Δ vs s1 CI [−0.012, 0.022], farfull Δ −0.0175. **Diagnosis (measured from
the cached outer maps):** greedy-by-value spacing collapsed far allocation —
fraction of emitted dots >10 px (1 km) from the visible catalogue fell from
0.52 (s3_V0) to 0.22 (s3_V2) / 0.27 (s2_V1), vs 0.71 for s1_V0, 0.814 far-zone
share of allowed area, 0.427 far share of withheld truth pixels, and the
champion's measured 0.698 (>10 px from the full catalogue). Mechanism: the
5×5-NMS prefilter in V0 is an implicit spatial diversifier (one candidate per
local peak, peaks exist in diffuse far areas too); raw-value greedy packs the
extensive near-catalogue highs at 2.8 px pitch and never reaches far values.

## Exp-4 hypothesis (H68-G)

Restoring allocation diversification *explicitly* — zone quotas proportional to
allowed area, the same area-quota principle the instrument already uses across
folds — while keeping exact ≥2.8 px spacing, will retain most of the all-scope
spacing gain and recover far-stratum performance. The champion's measured
geometry (69.8 % of dots >1 km from the catalogue at the live-optimal rung,
OWNER-REPORTED ladder) is the external anchor; the zone split itself is *not*
tuned: near zone = ≤10 px from the **visible** catalogue, far zone = >10 px,
quota ∝ zone allowed-area inside each fold region (far ≈ 81 % of dots).

## Arms (same cached exp3 outer maps, same budgets 25k/38k, decision at 38k)

- `s2_V6`, `s3_V6`: stratified greedy spacing (candidates = all allowed pixels).
- `s2_V7`, `s3_V7`: stratified greedy spacing on across-strike crest candidates.
Controls `s1_V0`, `s3_V0`, `s3_V2` recomputed in the same run for paired terms.
Single ranked pass per region: descending surface value, accept iff ≥2.8 px
from every accepted dot (global across zones) and the candidate's zone quota
is unexhausted. Deterministic (raster-order tie-break).

Scopes identical to Exp-3: `all`, `far` (domain-restricted, exp2-comparable),
`farfull` (far truth, all mass counts). Canary: Exp-3's (same maps/features).

## Decision rule (identical gates to Exp-3; no goalpost movement)

Promote at most one of {s3_V6, s2_V6, s3_V7, s2_V7} at 38,000 dots — the
highest `all` DTI satisfying ALL of: (1) paired 95 % CI of Δ vs s1_V0 entirely
> 0 in `all` AND `far`; (2) paired 95 % CI of Δ vs s3_V0 entirely > 0 in `all`;
(3) point Δ vs s1_V0 > 0 in `farfull`; (4) Exp-3 replication + canary stand;
(5) final-raster validator + lane checks (as before). If none: verdict
**negative**, no download, and this session stops experimenting (the third
experiment slot is left unused rather than spent fishing).

Named risk: area-proportional allocation may *over*-spend far (81 % of dots
for 42.7 % of holdout truth pixels), sacrificing the all-scope gain. That is
the tension the experiment measures; both outcomes are publishable here.
