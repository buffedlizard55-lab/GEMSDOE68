# GEMSDOE68 · auto-context research for the DOE GEMS Prize

> **SUBMISSION STATUS — NOT CLEARED. DO NOT SUBMIT OR DOWNLOAD AN OLD TIFF.**
> The former "promote-with-caveats" decision was incorrect: the historical >1 km
> gate measured 1.97 SE, below the required >2 SE, and final-dot overlap with a
> registry raster was 0.9948 (>0.70 stop). The claimed "zeros" fallback also had
> 7,113,308 NaNs in its raster band; the DrivenData portal's “[0,1]” rejection
> cannot be dismissed by that filename. Historical downloads were withdrawn from
> GitHub Pages and retained for audit under `evidence/archives/2026-10-12/`.
> The corrected, preregistered H68-M1 holdout is **negative** (2026-10-20):
> all withheld Δ +0.00050, SE 0.00426, t = 0.12; >1 km Δ +0.00106,
> SE 0.00275, t = 0.39 (each requires >2). Of 644 feature/fold canaries,
> 132 had non-finite inputs and cannot be cleared; the highest measurable
> directional AUC was 0.670 (none observed >0.90). Scientific gates failed
> **before final raster training**. No new TIFF was emitted, offered or sent
> to DrivenData, and no organizer submission receipt exists.

The live decision and step-by-step participant guide will be in
[`docs/index.html`](docs/index.html) and
[`docs/executive-summary.html`](docs/executive-summary.html). A green format
check alone cannot authorize a weekly contest slot. If any gate fails, **log
and stop** rather than serving a TIF. The repository has not submitted to
DrivenData.

---

# Standing brief (verbatim from the session prompt — read this first every session)

> Review the repo.
>
> THE FOLLOWING IS THE HIGHEST URGENCY AND MUST BE FOLLOWED!
>
> MUST GENERATE A UNIQUE TIF SUBMISSION FOR THE COMPETITION. DO NOT COPY A PREVIOUS SUBMISSION UNLESS IT'S FOR LEARNING AND EDUCATION. BUT WE MUST GENERATE A UNIQUE TIF SUBMISSION. IT MUST BE OBVIOUS WHETHER IT IS OK TO DOWNLOAD AND SUBMIT THE GENERATED TIF SUBMISSION.
>
> There should be an easy to download submission tif file as described by the prompt. Read the entire prompt.
>
> **Lane: Auto-context stacked refinement.** Auto-context (Tu, CVPR 2008, doi:10.1109/CVPR.2008.4587436; extended TPAMI version at pages.ucsd.edu/~ztu/publication/pami_autocontext.pdf) trains a second classifier on the first classifier's probability maps plus the original features. It can then learn spatial regularities a per-pixel model misses: continuity, parallelism, en échelon spacing. Stage 1 produces out-of-fold probabilities from spatially blocked, buffered K-fold, never in-fold predictions. Stage 2 is LightGBM or a small CNN on four kinds of input. These are stage-1 probabilities sampled at ±1–15 px offsets along and across local orientation, multi-scale anisotropic smooths, and the raw gradient/tilt/scarp channels. It also gets catalogue-context features computed with the target block's labels masked (distance, orientation and spacing to the nearest mapped fault). Iterate twice and estimate the gain with nested blocked CV. Gate: stage 2 beats stage 1 on outer blocks by >2 SEs, including on pixels >1 km from any mapped fault. Finish: normalize to [0,1]; write a float32 EPSG:32611 100 m GeoTIFF (NaN only outside the footprint); pass validate_submission.py; run a hash/pixel-agreement check against all prior rasters; offer the download only if the gate is met, else log the negative result.
>
> **PARALLEL-RUN PROTOCOL — read first.** This session is one of several running from this same prompt.
> 1. LANE. Your lane is the single method paragraph above. Stay inside it. If your raster's rank-correlation with any registry raster exceeds [0.90], or more than [70%] of your dots fall within 3 px of one registry raster's dots, you have drifted into another lane: log it as a duplicate and stop. Check this on the surface before placement AND on the final dots.
> 2. REUSE, DON'T REBUILD. Use the template's cached feature stack, evaluate_holdout.py and submission_writer.py. Holdout = hide-and-recover: withhold whole fault segments with a buffer, derive every catalogue-based feature only from the visible faults, mask visible faults pixel-exactly, score pooled DTI (alpha 0.2, beta 0.8, 300 m triangular kernel). If a shared tool is wrong, fix it once in the template and report it; never keep a private fork.
> 3. LABEL EVERY NUMBER as HOLDOUT-DTI (evaluator version, number of withheld positives, 95% CI) or ORGANIZER-CONFIRMED (copied from a submission-page receipt). A projection is never written as a score.
> 4. LEAKAGE CANARY. Test each feature alone on the holdout before trusting any result. AUC above [0.90] means leakage until proven otherwise.
> 5. RUN CARD. End with one JSON card: hypothesis; mechanism; the named non-fault process that could mimic it; holdout DTI + CI; correlation/overlap vs registry; raster sha256; validator output (no NaN inside the footprint, values in [0,1], CRS/shape/transform match); submission name + note of at most 140 characters; verdict promote / negative. Negative results are deliverables.
> 6. BUDGET. Stop after [3] experiments or [2] hours. Do not pick submissions: promotion to a real slot is a separate selector step, within the weekly cap shown on the submission page.
>
> WE NEED TO STUDY, ANALYZE, AND UNDERSTAND THE HIGHEST SCORE FROM THE GEMDOE SITE WHERE THE SUBMISSION TIF IS DOWNLOADED FROM WHICH IS THE FOLLOWING: https://buffedlizard55-lab.github.io/GEMSDOE32/docs/index.html — h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros: 0.2778. Why and how did this get the highest score and are we able to generate a submission that scores higher than 0.2778?
>
> Answer the question using PhD level experience, knowledge, and judgement. Then use the answer to generate a unique TIF submission into the competition. Must be unique submission unlike any within the GEMSDOE sites. Verify working line by line no hallucinations.
>
> Current competition leaderboard GEMSDOE high score: 0.3774 (as given in the brief; 0.3195 quoted later in the same brief). Leaderboard: https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/
>
> Before implementing, generate 3–5 candidate geological hypotheses we haven't tried yet, each naming: the specific layer(s) involved, the physical signature being targeted, why it should catch a fault missing from the USGS/INGENIOUS catalogue rather than one already in it, and how it differs from anything already implemented. Rank them by expected DTI improvement and implementation cost. Validate the top candidate on our spatially-blocked holdout set before touching a weekly submission slot. If a candidate can't be validated without new external data, name the specific free, official source needed and check it's obtainable before proposing the idea as viable.
>
> Work line by line verifying from official verified trusted sources, provide links for manual review. There should be no manual input, work on your own to complete tasks. Flag any irregularities for review. No hallucinations.
>
> We need to focus on being able to generate a submission into the competition. The site should be able to generate a TIF file that is required for submission. It should be as easy as download to click a File to submit into the competition. This needs to be in the executive summary or the very beginning of the site. A previous download returned "Predicted values must be in range [0, 1]" on the submission form. Also we need to give it a unique name and a short note (e.g. "clustering with k=25"). Create an executive summary subpage that explains exactly how to make a submission into the contest.
>
> Competition: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/ · rules PDF https://docs.nlr.gov/docs/fy26osti/96647.pdf · reference solution https://github.com/drivendataorg/gems-prize-reference-solution · GeoDAWN https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and · INGENIOUS https://gbcge.org/current-projects/ingenious/ · EPSG:32611 https://epsg.io/32611 · Tversky index https://en.wikipedia.org/wiki/Tversky_index · GDR 1391 https://gdr.openei.org/submissions/1391
>
> Core values to keep as focal point: **Maximize P(Win)** and **Own the Outcome**.
>
> Run this task through multiple passes (implement+verify; review for bugs/missing requirements/edge cases; re-check against the original request). Create a pull request and merge it onto main. Make suggestions for what work still needs to be done and any limitations.

The full list of sibling sites and their owner-reported scores is kept in `registry/score_ledger.csv`.

---

## Current scientific plan and reproducibility

1. **Preregister first.** The four ranked geological ideas, physical mechanisms,
   non-fault mimics and expected relative cost are at
   [`docs/research/2026-10-20-autocontext-hypotheses.md`](docs/research/2026-10-20-autocontext-hypotheses.md).
   H68-M1 tests a magnetic-edge tangent (not the previously used topographic
   gradient direction), with stage-1 OOF probabilities; later hypotheses are not
   run unless the three-experiment/two-hour limit and preregistration permit it.
2. **Inputs.** The challenge's official [data download](https://www.drivendata.org/competitions/306/competition-doe-gems/data/)
   requires a participant login. `registry/data_manifest.json` pins the owner
   GitHub mirrors at immutable refs and SHA-256, including three owner-derived
   external-channel sidecars. These hashes verify *integrity only*, not that
   the files are independently authenticated by the organizer. No secrets are
   required to reproduce a local holdout. Free USGS GeoDAWN has [DOI
   10.5066/P93LGLVQ](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and);
   only locally pinned derivatives in the manifest are used for this fit.
3. **Restore / test / fit.** Use a Python environment with the dependencies in
   `requirements.txt` (this workspace uses `.venv`). From the repository root:

   ```sh
   python scripts/restore_data.py --group all --bridge
   PYTHONPATH=src python -m pytest -q tests
   PYTHONPATH=src python scripts/lane68_strict_experiment.py --out evidence/ac68_m1_holdout.json
   # ONLY if the preregistered gate is green: private candidate + surface/dot scans;
   # this command refuses to train a final field on a negative result:
   PYTHONPATH=src python scripts/lane68_m1_candidate.py
   # Whether the result is negative or positive, write one auditable card/site:
   python scripts/lane68_strict_run_card.py
   python scripts/build_site_verified.py
   python scripts/check_site_release.py
   ```

   The restore command downloads pinned owner mirrors, verifies all byte counts
   and SHA-256s, and creates the `data/bridge/` + `data/external/` local links.
   Do not treat `registry/restore_receipt.json` as a DrivenData receipt. The
   common 45-channel stack is rebuilt in a git-ignored cache if needed; 19
   competition bands plus derived gradients, LiDAR scarp and GeoDAWN features.
4. **Protocol.** The same fixed 38,000-dot rule emits stage 1/2/3 predictions
   for four *whole-component*, spatially buffered outer holdouts. Four blocked
   inner folds per outer block build OOF probabilities and the masked-catalogue
   contextual channels. Shared `gems57.evaluate_holdout` pools distance-weighted
   Tversky counts (alpha 0.2, beta 0.8, 300 m triangle), with spatial-block
   bootstrap 95% CIs. Promotion needs stage 2 minus stage 1 >2 SE on *both*
   all withheld labels and those >1 km from mapped faults; every individual
   feature canary AUC <=0.90. A local surrogate is not a private score.
5. **Originality / artifact.** If scientific gates pass, test the continuous
   probability field **before placing dots** and the final raster separately
   against all accessible prior registry submissions and archived artifacts:
   exact file hash / decoded agreement, absolute Spearman >0.90 and >70% of
   emitted dots within 3 px of any one prior dot raster are literal stops.
   The accessible corpus is not a certificate of all organizer submissions.
   Only a candidate clearing originality, a strict local single-band float32
   EPSG:32611 100 m grid check with NaNs *only outside* the official footprint,
   independent provenance/review and a separate weekly-slot selector may be
   copied into `docs/downloads/`. No upload is made from the repository.
6. **Portal range bug.** The shared `gems.submission_writer` now supports an
   explicit zero-exterior all-finite *diagnostic* variant. The default is the
   published-spec NaN exterior, and the strict `gems.validate_submission`
   requires NaN outside. The optional `--outside-policy nan-or-zero` checks
   range for diagnosing portal errors, **not** organizer acceptance or
   specification conformity. Never call the old mislabeled zeros file a fix.
7. **Fail-closed Pages.** `scripts/build_site.py` and
   `scripts/lane68_build_site.py` delegate to `scripts/build_site_verified.py`.
   A run card cannot expose a download unless a selector, the shared release
   gate, provenance, validator, file SHA and uniqueness all pass. The audit
   trail remains in `evidence/` and `registry/`, not in a public download.

## Why owner-reported H33 0.2778 is plausible, not a benchmark prediction

The user and family ledger report `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros`
at **0.2778**, but [the owner H33 page](https://buffedlizard55-lab.github.io/GEMSDOE32/docs/index.html)
does not supply an authenticated organizer receipt for it. Label this
**OWNER-REPORTED**, not **ORGANIZER-CONFIRMED**. The [official distance-weighted
Tversky objective](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/)
rewards covered fault mass while penalizing redundant/false lines and especially
missed truth (beta 0.8). H33's sparse, pruned line-placement design is a
reasonable way to trade recall for false positives; the historical number
cannot establish which component caused its success, or that our lane would
score above 0.2778 on private expert labels. See
[`knowledge/01_why_02778_and_can_we_beat_it.md`](knowledge/01_why_02778_and_can_we_beat_it.md)
for the mass-accounting derivation and limitations. The [public
leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/)
was observed at 0.4413 top score on 2026-10-20, a dated table observation about
another participant, **not** a receipt for any raster in this repository.

## Sources, red flags, and next review

The [dated official/author/owner source ledger](registry/sources_2026-10-20.json)
links each claim to a page. [The research site](docs/research.html) separates
physical rationale from validation, and [the audit page](docs/irregularities.html)
flags leakage, the mislabeled zeros twin, the literal duplicate, inherited
holdout issues and the login-gated source provenance. Three iterative review passes found and regression-tested shared-tool bugs,
then checked the final run card, full test suite (54 passed, 7 skipped), guarded
candidate builder, format preflight (no new candidate to validate) and every
Pages front door (zero public TIFF/ZIP links). The first scored attempt was
quarantined because the emitter's tie handling violated NMS; the corrected
H68-M1 is the **only** canonical experiment result. See
[`evidence/run-card-2026-10-20.json`](evidence/run-card-2026-10-20.json),
[`evidence/ac68_m1_holdout.json`](evidence/ac68_m1_holdout.json) and
[`evidence/archives/2026-10-20/README.md`](evidence/archives/2026-10-20/README.md).
These are three passes by one agent, **not** two independent scientific reviewers. Official [rules PDF](https://docs.nlr.gov/docs/fy26osti/96647.pdf):
three submissions per week, eligibility and disclosure requirements.

**Open limitations:** access to original authenticated competition downloads,
full organizer-wide prior-submission registry, portal upload acceptance and
private/final score requires authorized participant/organizer evidence.
We do not ask for credentials or burn a contest slot for a local hypothesis.
