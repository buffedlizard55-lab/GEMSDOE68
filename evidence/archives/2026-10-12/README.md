# Historical candidate bytes — NOT CLEARED FOR SUBMISSION

Moved unchanged from `docs/downloads/` on 2026-10-20 after a release-gate audit. These are evidence files, **not download recommendations**.

* `GEMSDOE68-AC-stack2-…-nan.tif` has a valid *local* GeoTIFF format check, but its own `evidence/run-card-ac68.json` reports a failed >1 km two-SE gate (t=1.97) **and** a 99.48% dot overlap with a prior registry raster (70% stop threshold). A density-null analysis cannot override the literal stop rule.
* Its `-zeros.tif` twin is **not actually all-finite**. Re-reading both raster bands gives 7,113,308 NaNs each; only their file bytes/tags differ. `src/gems/submission_writer.py` previously overwrote the proposed zero exterior with NaNs. Both were withdrawn, not re-labelled as accepted by the portal.
* The other sibling artifacts have incompatible/incomplete holdout and/or registry evidence. No archive artifact is selected for a competition slot. The related `*.zip` and receipts have the same historical status.

A new candidate is generated independently by `scripts/lane68_strict_experiment.py` and `scripts/lane68_m1_candidate.py`, then measured against these historical rasters and `registry/rasters/`. The official challenge score is available only through the organizer's submission receipt; none is claimed here.
