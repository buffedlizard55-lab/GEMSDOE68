# Hypothesis backlog (ranked; H68-1 tested, H68-2–5 proposed)

Each entry names the layers, physical signature, off-catalogue rationale, novelty, data, and effort.
The ranking is qualitative expected value, not a score projection. H68-1 is the implemented
auto-context lane, not an untried idea; its historical model-level HOLDOUT-DTI has no paired 95% CI,
and the latest emitted TIFF is not hash-linked to the held-out prediction surface. The release
verdict is negative and no raster is cleared for download.

| rank | id | layers | signature | why off-catalogue | novelty vs registry | data | exp. gain | cost |
|---|---|---|---|---|---|---|---|---|
| reference | **H68-1 Auto-context continuity/parallelism (implemented)** | all 19 bands + det_elev Hessian ridge/tilt/gradient + stage-1 probability map | stage-1 probability sampled ±1–15 px *along and across local strike*, anisotropy index, multi-scale smooths, visible-catalogue distance/strike/density | unmapped faults may continue, splay, or parallel mapped ones; contextual probabilities test that mechanism | orientation-sampled probability context with blocked CV differs from earlier cascade filters | no new data | HOLDOUT-DTI stage 2 0.1904 vs stage 1 0.1301 (11,318 withheld positives; CI not computed; not a TIFF score) | implemented; release negative |
| 2 | H68-2 Footwall-dip asymmetry of range fronts | det_elev, det_elev_slope, 1 m LiDAR (3DEP) where available | Basin-and-Range normal faults sit at the base of a **steep, planar footwall facet** with a gentle hanging-wall fan: a one-sided slope step with facet aspect opposite to basin dip. Signature = signed slope asymmetry across a 300–800 m window oriented perpendicular to local strike | the catalogue stops where young alluvium buries the trace; the *facet/piedmont boundary* continues and marks the buried range-front fault ("piedmont fault" class) | sibling scarp filters were symmetric ridge/curvature responses, not footwall/hanging-wall asymmetry | 1 m DEM links (competition-provided, USGS 3DEP, free) — obtainable from apps.nationalmap.gov but **not reachable from this sandbox** | medium | medium |
| 3 | H68-3 Magnetic-source edge *depth-consistency* with topography | rtp, tmi_hg, tmi_vg, tc (tilt), det_elev | tilt-depth: for a vertical contact the horizontal distance between the tilt-angle 0° and ±45° contours equals source depth (Salem et al. 2007, The Leading Edge 26:1502). A fault that offsets basement produces a tilt-depth **step** collocated with a topographic step; coincidence of the two within 300 m while the catalogue is absent is the target | buried faults under basin fill have no surface scarp but do offset magnetic basement; the catalogue is surface-mapped | sibling "Euler" lanes (h8, h38, h45) used Euler deconvolution clusters, not tilt-depth coincidence with topography | none new | low–medium | medium |
| 4 | H68-4 Radiometric K/Th alteration lineaments | GeoDAWN radiometrics (K, eTh, eU; mirrored as `external/geodawn_rad_u8.tif`) | hydrothermal potassic alteration and K-leaching along fluid-conducting faults produce **linear K/eTh anomalies** elongated along strike; target = oriented line response in K/eTh ratio coherent with the local structural strike field | records *fluid pathways*, which is what the geothermal panel maps; independent of topography so immune to burial | h15 "conj_alteration_mag" used magnetics, r11f/r12 used radiometric *fusion* with scarps (not oriented K/eTh ratio lineaments) | GeoDAWN rad (USGS, public domain; mirrored) | low–medium | low |
| 5 | H68-5 Post-2020 surface-rupture context (Monte Cristo Range) | ComCat relocated seismicity, published rupture maps | the Mw 6.5 2020 rupture lies inside the footprint on largely unmapped parts of the Candelaria system (Koehler et al. 2021, doi:10.1785/0220200371) | a ruptured-but-unmapped fault is the definition of hidden truth | h60-1 was proposed in GEMSDOE32 but never built | USGS ComCat (free, official; **not reachable from this sandbox**), NBMG rupture map | high locally, tiny area | high (external fetch) |

**Validation status, checked 2026-10-20.** H68-1 historical model-level evidence is in
`evidence/ac68_seed20.json` (HOLDOUT-DTI evaluator `gems57-pooled-hide-v2`, 11,318 withheld positives).
The paired physical-spatial-block 95% CI was **not computed**. Stage-2 all-pixel vs stage-1 is
+0.0607 (SE 0.0070; t=8.60), but the >1 km stratum is +0.0074 (SE 0.0038; t=1.97), below the
required >2. This is not a live score or a measurement of the emitted TIFF. The exact emitted file
also trips the final dot-overlap rule; no file is cleared. H68-3 source bands are in core data;
H68-4 GeoDAWN radiometric sidecars are restored from the pinned mirror but need a separate audit.
Neither H68-3 nor H68-4 has been experimentally validated here. H68-2 requires USGS 3DEP 1 m DEM
coverage and H68-5 requires USGS ComCat / rupture products; live retrieval was not tested in this
turn, so obtainability is unconfirmed. These are backlog proposals, not authorized changes to the
single auto-context lane.

**Named non-fault processes that could mimic each signature** (for the run card): H68-1 — any
linear feature the stage-1 model already likes (roads, canals, shoreline terraces, lithologic
contacts) is *reinforced* by continuity; H68-2 — fluvial terrace risers and pediment edges;
H68-3 — intrusive contacts and dike swarms; H68-4 — lithologic K contrasts in volcanic flows and
playa evaporites; H68-5 — none (direct observation), but positional error of the rupture map.
