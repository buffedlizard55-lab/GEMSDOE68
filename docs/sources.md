# Sources (for manual review)

Status legend: **verified-here** = fetched/recomputed in this session's sandbox (which can reach only
github.com, api.github.com, codeload.github.com, registry.npmjs.org, pypi.org, files.pythonhosted.org);
**verified-by-lineage** = recorded as verified with date in a sibling repository's `registry/sources.json`
(GEMSDOE32, 2026-10-04) and reused; **listed** = cited in the brief, not independently fetched.

| source | url | status | used for |
|---|---|---|---|
| DOE GEMS Prize overview | https://www.drivendata.org/competitions/306/competition-doe-gems/ | verified-by-lineage | rules, slots |
| Problem description, metric, submission format | https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/ | verified-by-lineage (worked example TP 3.00 / FP 1.89 / FN 2.00 → 0.60 reproduced here as 0.6027) | `src/gems68/metric.py` |
| About / resources | https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/ | listed | — |
| Data page (login-walled) | https://www.drivendata.org/competitions/306/competition-doe-gems/data/ | listed; not accessible | — |
| Public leaderboard | https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/ | listed; brief quotes 0.3774 / 0.3195 | context only |
| Rules PDF NLR 96647 | https://docs.nlr.gov/docs/fy26osti/96647.pdf | listed (mirrored by lineage, sha256 50d854b1e0239fe6…) | submission rules |
| Reference solution (U-Net, Tversky loss) | https://github.com/drivendataorg/gems-prize-reference-solution | verified-by-lineage | baseline design |
| GeoDAWN airborne magnetic & radiometric surveys | https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and | listed | band provenance (mag_anom, rtp, tmi*, tc) |
| INGENIOUS (GBCGE) | https://gbcge.org/current-projects/ingenious/ | listed | catalogue provenance |
| EPSG:32611 | https://epsg.io/32611 | listed | CRS |
| Tversky index | https://en.wikipedia.org/wiki/Tversky_index | listed | metric background |
| GDR submission 1391 | https://gdr.openei.org/submissions/1391 | listed | external data index |
| Auto-context, Tu CVPR 2008 | https://doi.org/10.1109/CVPR.2008.4587436 | listed (method citation from the brief) | lane method |
| Auto-context, Tu & Bai TPAMI 2010 | https://pages.ucsd.edu/~ztu/publication/pami_autocontext.pdf | listed | lane method |
| Koehler et al. 2021, 2020 M6.5 Monte Cristo Range earthquake, SRL 92(2A) | https://doi.org/10.1785/0220200371 | verified-by-lineage | H68-5 |
| Salem et al. 2007, tilt-depth method, The Leading Edge 26(12):1502 | https://doi.org/10.1190/1.2821934 | listed | H68-3 |
| USGS 3DEP / National Map (1 m DEM links provided by competition) | https://apps.nationalmap.gov/downloader/ | listed; host unreachable from sandbox | H68-2 |
| USGS ComCat | https://earthquake.usgs.gov/fdsnws/event/1/ | listed; host unreachable from sandbox | H68-5 |
| Competition rasters, GitHub mirror (sha256-pinned) | https://github.com/buffedlizard55-lab/GEMSDOE (training_features parts) · https://github.com/buffedlizard55-lab/GEMSDOE24/tree/07345ea0604953d7efb858d9cfbc21e20c7aca0b/data/bridge | verified-here: sha256 4371c82e… / 7ba308cc… / 2176d08e… match `registry/data_manifest.json` | all modelling |
| Champion raster h33-h33-2-b2 (0.2778 owner-report) | https://github.com/buffedlizard55-lab/GEMSDOE32/blob/main/docs/downloads/gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif | verified-here: sha256 c55bafc470054e8271dcb89347a17e07fefe50de6af6e6ba6c4b169ef7ab6fa9 matches its audit JSON | uniqueness registry, analysis |
| Shared template (metric, holdout, evaluator, validator) | https://github.com/buffedlizard55-lab/GEMSDOE65/tree/main/lane65-sarprop/src/gems65_sarprop/template | verified-here: copied with provenance headers; `metric.py` byte-identical apart from header | rule 2 reuse |
| Champion-ladder analysis (why 0.2778) | https://github.com/buffedlizard55-lab/GEMSDOE65/blob/main/docs/research/why-02778-and-what-beats-it-2026-10-11.md · https://github.com/buffedlizard55-lab/GEMSDOE32/blob/main/knowledge/01_why_0260_and_the_path_to_03195.md | verified-here (read; ladder subset relations not recomputed here) | knowledge/01 |

**Irregularity to note:** the brief states the "current leaderboard GEMSDOE high score" as 0.3774 in one
place and 0.3195 in another, and the lineage's 2026-10-11 note quotes a #1 of 0.4133. None can be
re-fetched from this sandbox; treat all three as unverified here.
