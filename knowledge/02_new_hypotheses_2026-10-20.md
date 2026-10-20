# New candidate hypotheses, 2026-10-20 session (GEMSDOE68, auto-context lane)

Five candidates not yet *tested* in this repository. Each names the layer(s), the physical
signature, why it could catch a fault **missing** from the USGS/INGENIOUS catalogue rather than
one already in it, how it differs from what this repo already implements, the external-data
requirement (checked, below), and a qualitative ranking by expected HOLDOUT-DTI gain per unit
implementation cost. The ranking is a judgement, not a measurement; no number here is a score.
Only **AC68-N1** is run in this session (`evidence/exp3_preregistration.json` was written before
the run). The rest are a queue, in order.

Why these and not more of the same: across the family ledger (`registry/score_ledger.csv`,
OWNER-REPORT, no organiser receipts held here), purely geometric detectors (Euler, worms,
ridges) cluster at 0.03–0.13, per-pixel supervised/geophysical fields at 0.15–0.19, and the same
fields *dotted at disciplined mass* at 0.25–0.28. The remaining headroom is ranking quality at
≈38 k dots (`knowledge/01_why_02778_and_can_we_beat_it.md`): the marginal-credit bar at DTI 0.28
is `0.2·DTI ≈ 0.056`. Both auto-context experiments in this repo already show the all-pixel gain
is real (stage 2/3 − stage 1 = +0.035…+0.061, bootstrap CIs excluding 0) and the **far (>1 km)
stratum is the weak point** — which is exactly the stratum closest to the live task of finding
*unmapped* faults. All five candidates attack that weakness from different sides.

| rank | id | layers | physical signature | why it catches an *off-catalogue* fault | difference vs this repo | data needed (checked 2026-10-20) | expected gain | cost | named mimic |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **AC68-N1 — catalogue-context-free (far-robust) auto-context** | 19 organiser bands + 7 physics channels + stage-(k−1) probability offsets/smooths (no catalogue-derived inputs at stage ≥2) | continuity, parallelism and en-échelon arrangement of geophysical/topographic evidence itself, with the "distance to nearest mapped fault" shortcut removed | the live truth is by definition absent from the catalogue; the organiser states the known-fault set "is not complete and may even contain some inaccurate data" (problem description, fetched 2026-10-20). A proximity-keyed model spends mass in the ≤2 px catalogue halo that the family ladder proved is FP tax, and is blind >1 km from mapped traces, where buried/isolated strands live | both implemented lanes feed 5–11 catalogue-context features into stage ≥2; the ablation has never been run | **none new** — core rasters restored hash-verified this session (`registry/data_manifest.json`) | medium on the far stratum; small-positive to neutral overall; the live-relevant direction | **low** (one switch in `gems68.experiment.Lane`, same nested CV) | range-front erosional escarpments / alluvial terraces producing the same gradient/ridge responses and being connected by continuity |
| 2 | **AC68-N2 — external-channel stage-1 enrichment under the whole-component instrument** | + lidar scarp u8 (incl. downface/upface/step → *signed polarity asymmetry*), GeoDAWN radiometric/magnetic extension u8 ranks (USGS data release doi:10.5066/P93LGLVQ) | a normal fault drops the hanging wall: the steep face and footwall facet sit on opposite sides of the trace, so scarp *asymmetry* is signed; erosional scarps are near-symmetric. Radiometric rank anomalies add a topography-independent view | the catalogue maps traces, not damage zones or alteration; a young buried trace can retain a polarity signal in 2 m lidar aggregates while having no mapped line | the lane-68 stack used these channels only under the flawed 12-px-chunk / per-fold-budget instrument, so their effect on the valid instrument is **unmeasured**; the sibling X1 (19+7) never had them. Note: the u8 sidecars are percentile *ranks*, not physical units — ratios of ranks must not be called K/Th ratios | pinned in `registry/data_manifest.json` (group `external`, sha256-pinned); restore mechanism proven this session on `core`, but the `external` fetch hit a GitHub API rate limit (HTTP 403, 2026-10-20 19:06 UTC) and must be retried next session before use | medium | medium (rebuild feature cache, new canary pass, memory budget) | fluvial terrace risers (also one-sided); lithologic K contrasts in volcanic flows |
| 3 | **AC68-N3 — footwall-facet slope-step asymmetry (core-only)** | `det_elev`, `det_elev_slope`, local strike from the structure tensor | Basin-and-Range range-front faults bound a steep planar footwall facet against a gentle hanging-wall fan: a *one-sided slope step* across a 500–800 m window perpendicular to local strike, facet aspect facing the basin | where young alluvium buries the trace, the catalogue stops but the facet/piedmont boundary continues — the classic "piedmont fault" the catalogue under-maps | every scarp channel in the repo is a symmetric magnitude (gradient, Laplacian, Hessian ridge); none is a signed across-strike slope-step | **none new** (core bands only) | low–medium | low (one derived channel + canary) | pediment edges and lake-shoreline (Lahontan) benches, which are also one-sided steps |
| 4 | **AC68-N4 — tilt-depth basement-step coincidence** | `rtp`, `tmi_hg`, `tmi_vg`, `tc` + `det_elev` | Salem et al. (2007, *The Leading Edge* 26:1502–1505, doi:10.1190/1.2821934): over a vertical contact the tilt angle is 0° at the edge and ±45° at horizontal distances equal to source depth, so the 0°→±45° distance estimates depth. A fault offsetting magnetic basement makes a tilt-depth *step* collocated (≤300 m) with a topographic step | faults buried under basin fill have no scarp but do offset basement; the surface-mapped catalogue cannot see them | prior family "Euler" lanes clustered deconvolution solutions; nobody here has used tilt-depth *steps* or required coincidence with topography | **none new** (bands already in the core stack); the method reference was verified this session | low–medium (high only where basement is shallow/magnetic) | medium (contour/distance machinery at 100 m, careful with RTP derivatives) | intrusive contacts and dike swarms — also magnetic edges with depth steps |
| 5 | **AC68-N5 — en-échelon periodicity of the probability field** | stage-1 probability map + `tmi_hg` | Riedel shears in a releasing bend repeat at a quasi-regular spacing (~1–3 km, set by brittle-layer thickness); signature = an along-strike autocorrelation peak of the stage-1 field at 10–30 px lags | individual short Riedels are below catalogue mapping resolution, but their *spacing* is a regional statistic that survives 100 m aggregation | stage-2 offset sampling stops at 15 px and is pointwise; no periodicity statistic exists anywhere in the registry | **none new** | low (offsets already capture much of it) | medium | regularly spaced drainages / yardang-like erosional lineations and survey-line artefacts (GeoDAWN flight-line spacing) |

**Explicitly not proposed as viable today:** anything needing the 1 m DEM tiles
(`1m_DEM_links.csv`, USGS 3DEP) or the USGS ANSS catalogue API for hypocentre depths. Those are
free, official sources —
<https://www.usgs.gov/3d-elevation-program> and
<https://earthquake.usgs.gov/fdsnws/event/1/> — but neither host is reachable from this sandbox's
allow-list, and exact tile coverage/alignment to the competition footprint has not been verified
in this checkout. They remain next-session items (registry hypotheses LIDAR-01 / H5), not inputs
to any decision made here.

**Validation order (binding):** N1 first (this session). N2–N5 may only be promoted after beating
the then-current holdout best on the *same* whole-component instrument, with the leakage canary
(AUC > 0.90 = leakage until disproven) and both gate strata reported. A submission slot is never
spent on an unvalidated candidate.

---

## Outcomes (2026-10-20, measured; HOLDOUT-DTI, evaluator `gems57-pooled-hide-v2`, 11,321 withheld positives)

- **N1 (Exp3, catalogue-context-free) — NEGATIVE for promotion.** Passed its own-stage bootstrap
  gates only at 15 k/25 k (25 k stage 3: all 0.0659→0.0770, far 0.0498→0.0667) but loses
  0.03–0.04 all-pixel DTI to Exp2 at matched budgets and fails bootstrap at 38 k/50 k.
  Catalogue context is load-bearing, not merely leakage (`cat_dist` = 41.3 % of stage-2 gain,
  25.1 % of stage-3 gain in Exp2). `evidence/exp3_autocontext_ccfree.json`.
- **N2 (Exp4, external stage-1 channels) — promote-with-caveats (38 k, stage 2).** All
  0.0828→0.1189 (Δ95% CI [+0.0258, +0.0475]); far 0.0644→0.0756 ([+0.0007, +0.0218]); beats Exp2
  point estimates at matched budget/stage (+0.0024 all, +0.0045 far) but the n=4 per-fold far
  t = 1.53 < 2. External channels improved stage 1 most at low mass (15 k: +0.0093 vs Exp2
  stage 1) and only modestly at 38 k (+0.0012); lidar Laplacian/step columns top the stage-1
  canary (AUC ≤ 0.595, clean). `evidence/exp4_autocontext_ext.json`, `evidence/exp4_decision.json`.
- N3–N5 untested. Suggested next: N3 (one-sided slope-step) as additional stage-1 channels on
  top of the Exp4 stack, and the IR-68-BLOB-01 anti-regional/coherence-gated emission ablation —
  each as its own preregistered single-change experiment against the Exp4 numbers above.
