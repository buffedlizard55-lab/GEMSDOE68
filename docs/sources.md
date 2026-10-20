# Sources (for manual review)

Status legend: **verified-here** = fetched/recomputed in this session's sandbox (which can reach only
github.com, api.github.com, codeload.github.com, registry.npmjs.org, pypi.org, files.pythonhosted.org);
**verified-by-lineage** = recorded as verified with date in a sibling repository's `registry/sources.json`
(GEMSDOE32, 2026-10-04) and reused; **listed** = cited in the brief, not independently fetched.

| source | url | status | used for |
|---|---|---|---|
| DOE GEMS Prize overview | https://www.drivendata.org/competitions/306/competition-doe-gems/ | verified-by-lineage | rules, slots |
| Problem description, metric, submission format | https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/ | **verified-here 2026-10-20** (α=0.2, β=0.8, R=300 m; Final Prize Round re-scores on expanded expert labels; worked example reproduced as 0.6027) | `src/gems68/metric.py`, knowledge/03 |
| About / resources | https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/ | listed | — |
| Data page (login-walled) | https://www.drivendata.org/competitions/306/competition-doe-gems/data/ | listed; not accessible | — |
| Public leaderboard | https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/ | listed; brief quotes 0.3774 / 0.3195 | context only |
| Rules PDF NLR 96647 | https://docs.nlr.gov/docs/fy26osti/96647.pdf | listed (mirrored by lineage, sha256 50d854b1e0239fe6…) | submission rules |
| Reference solution (U-Net, Monte-Carlo CV notebook) | https://github.com/drivendataorg/gems-prize-reference-solution | **verified-here 2026-10-20** (single commit aebe92f, 2026-06-16) | organiser baseline design |
| GeoDAWN airborne magnetic & radiometric surveys | https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and | **verified-here 2026-10-20**: data release doi:10.5066/P93LGLVQ (Glen & Earney 2024), CC0 1.0; 149,030 line-km / 51,857 km², flown 2021-11→2022-11 | band provenance (mag_anom, rtp, tmi*, tc) + Exp4 radiometric/extension percentile-rank channels |
| INGENIOUS (GBCGE) | https://gbcge.org/current-projects/ingenious/ | listed | catalogue provenance |
| EPSG:32611 | https://epsg.io/32611 | listed | CRS |
| Tversky index | https://en.wikipedia.org/wiki/Tversky_index | listed | metric background |
| GDR submission 1391 | https://gdr.openei.org/submissions/1391 | listed | external data index |
| Auto-context, Tu CVPR 2008 | https://doi.org/10.1109/CVPR.2008.4587436 | **verified-here 2026-10-20** (DOI checked by web search) | lane method |
| Auto-context, Tu & Bai TPAMI 2010 | https://pages.ucsd.edu/~ztu/publication/pami_autocontext.pdf | listed | lane method |
| Koehler et al. 2021, 2020 M6.5 Monte Cristo Range earthquake, SRL 92(2A) | https://doi.org/10.1785/0220200371 | verified-by-lineage | H68-5 |
| Salem et al. 2007, tilt-depth method, The Leading Edge 26(12):1502 | https://doi.org/10.1190/1.2821934 | **verified-here 2026-10-20** (DOI checked by web search) | H68-3 / N4 |
| USGS 3DEP / National Map (1 m DEM links provided by competition) | https://apps.nationalmap.gov/downloader/ | host unreachable from sandbox; 2 m lidar scarp aggregates used via hash-pinned mirror + sidecar (75.3% footprint coverage measured 2026-10-20) | H68-2, Exp4 stage-1 channels |
| USGS ComCat | https://earthquake.usgs.gov/fdsnws/event/1/ | listed; host unreachable from sandbox | H68-5 |
| External-channel sidecars (quantisation JSONs for lidar/GeoDAWN u8 products) | https://github.com/buffedlizard55-lab/GEMSDOE24/tree/07345ea0604953d7efb858d9cfbc21e20c7aca0b/data/external | **verified-here 2026-10-20**: fetched from the pinned commit; receipt `evidence/restore68_external_sidecars.json`; product sha256 in sidecars match `registry/data_manifest.json` | Exp4 channel decode |
| GEMSDOE32 site (champion provenance check) | https://buffedlizard55-lab.github.io/GEMSDOE32/docs/index.html | **verified-here 2026-10-20**: page calls h33-2-b2 UNSCORED, "projected 0.2747" a model, no organiser score exists there | knowledge/03 attribution audit |
| Competition rasters, GitHub mirror (sha256-pinned) | https://github.com/buffedlizard55-lab/GEMSDOE (training_features parts) · https://github.com/buffedlizard55-lab/GEMSDOE24/tree/07345ea0604953d7efb858d9cfbc21e20c7aca0b/data/bridge | verified-here: sha256 4371c82e… / 7ba308cc… / 2176d08e… match `registry/data_manifest.json` | all modelling |
| Champion raster h33-h33-2-b2 (0.2778 owner-report) | https://github.com/buffedlizard55-lab/GEMSDOE32/blob/main/docs/downloads/gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif | verified-here: sha256 c55bafc470054e8271dcb89347a17e07fefe50de6af6e6ba6c4b169ef7ab6fa9 matches its audit JSON | uniqueness registry, analysis |
| Shared template (metric, holdout, evaluator, validator) | https://github.com/buffedlizard55-lab/GEMSDOE65/tree/main/lane65-sarprop/src/gems65_sarprop/template | verified-here: copied with provenance headers; `metric.py` byte-identical apart from header | rule 2 reuse |
| Champion-ladder analysis (why 0.2778) | https://github.com/buffedlizard55-lab/GEMSDOE65/blob/main/docs/research/why-02778-and-what-beats-it-2026-10-11.md · https://github.com/buffedlizard55-lab/GEMSDOE32/blob/main/knowledge/01_why_0260_and_the_path_to_03195.md | verified-here (read; ladder subset relations not recomputed here) | knowledge/01 |

**Irregularity to note:** the brief states the "current leaderboard GEMSDOE high score" as 0.3774 in one
place and 0.3195 in another, and the lineage's 2026-10-11 note quotes a #1 of 0.4133. None can be
re-fetched from this sandbox; treat all three as unverified here.
