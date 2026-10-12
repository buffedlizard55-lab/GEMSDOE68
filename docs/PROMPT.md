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
