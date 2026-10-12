# Candidate hypotheses not yet tried in the GEMSDOE family (ranked)

Each entry names the layer(s), the physical signature, why it should catch a fault **missing** from
the USGS/INGENIOUS catalogue rather than one already in it, how it differs from what the 60+
sibling sites implemented (`registry/score_ledger.csv`), and the data needed. Ranking is by expected
HOLDOUT-DTI gain per unit of implementation cost — a judgement, not a measurement, except where a
measurement is cited. Only H68-1 is in this lane; it was validated on the blocked holdout
(`evidence/exp1_autocontext.json`) before any download was offered.

| rank | id | layers | signature | why off-catalogue | novelty vs registry | data | exp. gain | cost |
|---|---|---|---|---|---|---|---|---|
| 1 | **H68-1 Auto-context continuity/parallelism** | all 19 bands + det_elev Hessian ridge/tilt/gradient + stage-1 probability map | stage-1 probability sampled ±1–15 px *along and across local strike*, anisotropy index, multi-scale smooths, visible-catalogue distance/strike/density | unmapped faults are mostly **continuations, splays and parallel strands** of mapped ones (compilation gaps, step-overs, antithetic strands); a per-pixel model cannot see "on-strike between two strong responses", a context stack can | no sibling used orientation-sampled probability context with nested blocked CV; prior "two-stage" sites (h46, h53) were cascade filters on wide masks | none new | medium (+0.01–0.03 holdout at matched mass if continuity is learnable; **measured below**) | low–medium (done) |
| 2 | H68-2 Footwall-dip asymmetry of range fronts | det_elev, det_elev_slope, 1 m LiDAR (3DEP) where available | Basin-and-Range normal faults sit at the base of a **steep, planar footwall facet** with a gentle hanging-wall fan: a one-sided slope step with facet aspect opposite to basin dip. Signature = signed slope asymmetry across a 300–800 m window oriented perpendicular to local strike | the catalogue stops where young alluvium buries the trace; the *facet/piedmont boundary* continues and marks the buried range-front fault ("piedmont fault" class) | sibling scarp filters were symmetric ridge/curvature responses, not footwall/hanging-wall asymmetry | 1 m DEM links (competition-provided, USGS 3DEP, free) — obtainable from apps.nationalmap.gov but **not reachable from this sandbox** | medium | medium |
| 3 | H68-3 Magnetic-source edge *depth-consistency* with topography | rtp, tmi_hg, tmi_vg, tc (tilt), det_elev | tilt-depth: for a vertical contact the horizontal distance between the tilt-angle 0° and ±45° contours equals source depth (Salem et al. 2007, The Leading Edge 26:1502). A fault that offsets basement produces a tilt-depth **step** collocated with a topographic step; coincidence of the two within 300 m while the catalogue is absent is the target | buried faults under basin fill have no surface scarp but do offset magnetic basement; the catalogue is surface-mapped | sibling "Euler" lanes (h8, h38, h45) used Euler deconvolution clusters, not tilt-depth coincidence with topography | none new | low–medium | medium |
| 4 | H68-4 Radiometric K/Th alteration lineaments | GeoDAWN radiometrics (K, eTh, eU; mirrored as `external/geodawn_rad_u8.tif`) | hydrothermal potassic alteration and K-leaching along fluid-conducting faults produce **linear K/eTh anomalies** elongated along strike; target = oriented line response in K/eTh ratio coherent with the local structural strike field | records *fluid pathways*, which is what the geothermal panel maps; independent of topography so immune to burial | h15 "conj_alteration_mag" used magnetics, r11f/r12 used radiometric *fusion* with scarps (not oriented K/eTh ratio lineaments) | GeoDAWN rad (USGS, public domain; mirrored) | low–medium | low |
| 5 | H68-5 Post-2020 surface-rupture context (Monte Cristo Range) | ComCat relocated seismicity, published rupture maps | the Mw 6.5 2020 rupture lies inside the footprint on largely unmapped parts of the Candelaria system (Koehler et al. 2021, doi:10.1785/0220200371) | a ruptured-but-unmapped fault is the definition of hidden truth | h60-1 was proposed in GEMSDOE32 but never built | USGS ComCat (free, official; **not reachable from this sandbox**), NBMG rupture map | high locally, tiny area | high (external fetch) |

**Validation status.** H68-1 is validated in `evidence/exp1_autocontext.json` (HOLDOUT-DTI,
evaluator `gems57-pooled-hide-v2`, 11,321 withheld positives, paired 95 % CI) — see the run card in
`evidence/run_card.json` and the executive summary for the verdict. H68-2/3/4 can be validated with
data already mirrored (H68-4) or already in the stack (H68-3); H68-2 and H68-5 need external fetches
from hosts this sandbox cannot reach (apps.nationalmap.gov, earthquake.usgs.gov) and are therefore
proposed as *next-session* items, not as viable-today.

**Named non-fault processes that could mimic each signature** (for the run card): H68-1 — any
linear feature the stage-1 model already likes (roads, canals, shoreline terraces, lithologic
contacts) is *reinforced* by continuity; H68-2 — fluvial terrace risers and pediment edges;
H68-3 — intrusive contacts and dike swarms; H68-4 — lithologic K contrasts in volcanic flows and
playa evaporites; H68-5 — none (direct observation), but positional error of the rupture map.
