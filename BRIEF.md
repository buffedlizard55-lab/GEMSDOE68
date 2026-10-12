# GEMSDOE68 — the standing brief

This file preserves the brief verbatim, so that any later session can re-read it without
reconstructing it from the run log.

## 1. SESSION-68 STANDING CHARTER (verbatim)

> **Highest urgency.** MUST GENERATE A UNIQUE TIF SUBMISSION FOR THE COMPETITION. Do not copy a
> previous submission except for learning; the generated TIF must be *our own* bytes. It must be
> **obvious whether it is OK to download and submit**. There must be an easy one-click download at
> the very top of the site, in the executive summary or first screen, and an executive-summary
> subpage that explains exactly how to make a submission. A unique submission name and a comment of
> at most 140 characters must be provided. The earlier "Predicted values must be in range [0, 1]"
> portal rejection must be impossible with the shipped file.
>
> **Lane (single method paragraph).** Auto-context stacked refinement. Auto-context (Tu, CVPR 2008,
> doi:10.1109/CVPR.2008.4587436; extended TPAMI version at
> pages.ucsd.edu/~ztu/publication/pami_autocontext.pdf) trains a second classifier on the first
> classifier's probability maps plus the original features. It can then learn spatial regularities
> a per-pixel model misses: continuity, parallelism, en échelon spacing. Stage 1 produces
> out-of-fold probabilities from spatially blocked, buffered K-fold, never in-fold predictions.
> Stage 2 is LightGBM or a small CNN on four kinds of input: stage-1 probabilities sampled at
> ±1–15 px offsets along and across local orientation, multi-scale anisotropic smooths, and the raw
> gradient/tilt/scarp channels; plus catalogue-context features computed with the target block's
> labels masked (distance, orientation and spacing to the nearest mapped fault). Iterate twice and
> estimate the gain with nested blocked CV. Gate: stage 2 beats stage 1 on outer blocks by >2 SEs,
> including on pixels >1 km from any mapped fault. Finish: normalize to [0,1]; write a float32
> EPSG:32611 100 m GeoTIFF (NaN only outside the footprint); pass `validate_submission.py`; run a
> hash/pixel-agreement check against all prior rasters; offer the download only if the gate is met,
> else log the negative result.
>
> **Parallel-run protocol.** (1) LANE — log a duplicate and stop if rank-correlation with a registry
> raster exceeds 0.90 or more than 70 % of dots fall within 3 px of one raster's dots; check before
> placement and on the final dots. (2) REUSE, DON'T REBUILD — use the template's cached stack,
> `evaluate_holdout.py`, `submission_writer.py`; fix a shared tool once, never fork privately.
> (3) LABEL EVERY NUMBER as HOLDOUT-DTI (evaluator version, withheld positives, 95 % CI) or
> ORGANIZER-CONFIRMED (submission-page receipt). A projection is never written as a score.
> (4) LEAKAGE CANARY — test each feature alone on the holdout; AUC above 0.90 is leakage until
> disproven. (5) RUN CARD — end with one JSON card. (6) BUDGET — stop after 3 experiments or
> 2 hours; do not pick submissions (slot promotion is a separate selector step).
>
> **Core values — keep them central.** *Maximize P(Win).* Weigh trade-offs, assess risk, choose the
> path that maximises the probability of winning. *Own the Outcome.* Own results end to end, act
> when problems arise without waiting for permission, treat failure and success as signals.
>
> **Evidence discipline.** Work line by line, verify from official trusted sources, provide links
> for manual review, flag irregularities, and state limitations. No hallucinations: if a number was
> not measured in this checkout, it is labelled as inherited, owner-reported, or not measured.

## 2. Additional standing instructions recorded during the session

These are the operational instructions the owner issued while the work was in progress. They are
recorded here as standing instructions, in the order they were given.

1. **Autonomous end to end.** Do not stop to ask for confirmation part-way through. Finish the
   session's next steps from any prior session first, then this session's, then report.
2. **Three passes.** Work the task in three passes rather than one, so that a first pass discovers
   the shape of the problem before later passes commit to detail.
3. **Arena Core Values stay focal.** Maximize P(Win) and Own the Outcome are decision criteria,
   not decoration: when a trade-off appears, weigh it and choose the path that maximizes the
   probability of winning, and own the result end to end.
4. **Site content.** The GitHub Pages site must contain, at minimum: an index page with a
   one-click download of the submission at the very top; an executive-summary subpage with the
   submission name, the ≤ 140-character note, the step-by-step "how to submit" instructions and a
   second download link; a results page whose every number traces to a dated JSON file in
   `evidence/`; a candidate-geological-hypotheses page that names the layers, the physical
   signature, why the hypothesis could catch an off-catalogue fault rather than one already in the
   catalogue, and how it differs from what this repository already implements; a research page
   listing the sources actually consulted; and a run-card page pointing at the single JSON card.
5. **Targets.** Beat the current valid holdout best of **0.3195**, and aim at the family best of
   **0.2778**. Both are *targets*, not scores; neither is an ORGANIZER-CONFIRMED figure.
6. **Finite, unique bytes.** The shipped raster must be our own bytes, byte-distinct from every
   raster in `registry/rasters/`, and must pass `gems.validate_submission` with no FAIL and no
   SKIPPED check. NaN may appear only outside the footprint.
7. **A static current feed.** The site must be buildable from evidence alone with no network access
   at build time, so that the published page can never drift from the measurements.
8. **Open a pull request and merge it** on the session branch
   `arena/ecb0c13a-gemsdoe68`. The pull request is opened after the submission raster, the evidence
   files and the site exist, because those are the point of the pull request.
9. **No label unmeasured numbers.** If a quantity was not measured in this checkout, it is written
   as "not measured", "inherited" or "owner-reported", never as a bare figure.

## 3. What is *not* in scope

- Choosing which submission gets the competition slot. Slot promotion is a separate selector step
  and is explicitly out of scope for this session.
- Claiming an organiser score. The competition's scoring truth is login-walled in this checkout.
- Rebuilding shared tools privately. `src/gems/` and `src/gems57/` are vendored and unmodified;
  a fix to a shared tool is made once, in place, and pinned by a test.
