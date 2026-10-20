# Why `h33-h33-2-b2` scored 0.2778, and whether this lane can beat it

*GEMSDOE68, 2026-10-12. Tags: **[official]** = competition page / rules PDF; **[owner-report]** = the
group's own ledger (no organizer receipt is held in this repository); **[measured-here]** = recomputed
in this session from restored bytes (`registry/restore_receipt.json`); **[model]** = arithmetic on the
above. Nothing below is written as a score unless it is tagged.*

## 1. The metric is a budget problem with a first-order condition

**[official]** DTI = TP_w / (TP_w + 0.2·FP_w + 0.8·FN_w), triangular kernel k(d) = max(1 − d/300 m, 0),
with TP_w = Σ_g max_x p(x)·k(d(x,g)), FP_w = Σ_x p(x)·[1 − max_g k], FN_w = |G| − TP_w
(DrivenData problem description, page 967; the page's worked example TP_w = 3.00, FP_w = 1.89,
FN_w = 2.00 → 0.60 is reproduced by `src/gems68/metric.py` as 0.6027 **[measured-here]**).

Writing T = TP_w, S = Σp (emitted mass) and M = Σ_x p(x)·max_g k (self-credit of the emitted mass):

```
DTI = T / ( 0.2·(T + S − M) + 0.8·|G| )                                   (★)
one more non-redundant unit of mass raises DTI  ⟺  its kernel credit k > 0.2·DTI   (★★)
scaling every value by λ<1 lowers DTI  ⇒  for a fixed pixel set, value 1.0 is optimal
```

(★★) is the whole story of the leaderboard: every emitted pixel that is more than ≈283 m from an
*uncovered* hidden-truth pixel costs 0.2 units of denominator and earns nothing. At DTI 0.2778 the
break-even credit is 0.0556.

## 2. What the champion file actually is

**[measured-here]** `gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif`
(sha256 `c55bafc470054e8271dcb89347a17e07fefe50de6af6e6ba6c4b169ef7ab6fa9`, restored from
GEMSDOE32 `docs/downloads/`): float32, EPSG:32611, 3730×3292, 100 m, **binary** — 37,654 pixels at
1.0, everything else 0.0 (zeros outside the footprint, no NaN). 0 dots on the catalogue; every dot
≥ 3 px (300 m) from the nearest mapped fault.

**[owner-report]** It is the last rung of a nested mass-pruning ladder on one field, H19-5
("topo-geophys multi-line corroborated", 19GEMSDOE):

| rung | dots | live score | what changed |
|---|---:|---:|---|
| h19-5 solid | 121,131 | 0.1922 | — |
| d1.5 dotted | 60,069 | 0.2477 | thinned to ~1.5 px spacing |
| d2.8 dotted | 44,090 | 0.2600 | thinned to ~2.8 px spacing |
| h27-4 | 40,199 | 0.2708 | removed dots at exactly d = 1 px from catalogue |
| **h33-2-b2** | **37,654** | **0.2778** | removed dots at exactly d = 2 px from catalogue |

Three independent facts follow from (★) and the ladder, and they are the design constraints for
anything that wants to beat it:

1. **Mass discipline is worth ~+0.07.** Going from a 121k-pixel ribbon to 44k dots of the *same*
   field raised the score 0.19 → 0.26 purely by deleting pixels whose realised credit was below
   the 0.2·DTI bar. The organizer-provided U-Net baseline and most of the sibling sites lose here,
   not on geology.
2. **The published catalogue's 100–200 m halo is worth less than nothing.** Both halo removals
   *raised* the score. This is consistent with the organizer masking mapped faults out of scoring
   (the truth is *new* faults), so dots hugging the mapped trace are pure FP tax.
3. **The remaining prize is ranking, not thrift.** Under the lineage's calibration
   (|G| ≈ 12k truth px **[model]**) the champion's dots earn a mean credit ≈ 0.135, i.e. ≈ 2.4× the
   bar; further pruning paid only +0.004–0.007 per rung. To reach the quoted leaders (0.32–0.38
   **[official per the brief, not re-fetched here]**) the top ~38k pixels of the surface must carry
   ~15–40 % more kernel credit toward real unmapped faults *at the same mass*. That is a
   **field/ranking problem**.

## 3. Why auto-context is the right lever for a ranking problem

Every scored sibling field is per-pixel: a pixel's score depends on its own 19 bands (plus hand
built ridge/scarp filters). Faults are not per-pixel objects. They are **continuous**, **parallel
to their neighbours** (Basin-and-Range normal-fault sets strike NNE and repeat at 10–20 km), and
**en échelon** (right-stepping segments with ≈ 1–3 km step-overs, e.g. the 2020 Monte Cristo Range
rupture, Koehler et al. 2021, doi:10.1785/0220200371). A per-pixel model has no way to know that a
weak pixel sitting exactly on the strike-line between two strong responses is far more likely to be
fault than an equally weak isolated one.

Auto-context (Tu 2008; Tu & Bai, TPAMI 2010) is the minimal machinery that learns this: the stage-2
classifier sees the stage-1 probability **sampled along and across the local strike at 1–15 px**,
multi-scale smooths, the raw gradient/tilt/scarp channels, and catalogue-context (distance, strike,
density of *visible* faults). Its gain over stage 1 is exactly the "continuity / parallelism /
spacing" information. If the gain is real, it shows up as higher credit at matched mass — the only
quantity (★) rewards.

**Distinct from previous two-stage attempts in the registry** (h46-twostageAB 0.0715;
h53-twostage 0.1047 **[owner-report]**): those were cascade/AB-filter designs on wide masks, not
orientation-sampled probability-context stacks with nested blocked CV and matched-mass emission.
Their low scores are consistent with mass indiscipline (rule 1 above), not with a verdict on
auto-context.

## 4. The honest limits of the local instrument

* The holdout truth is **withheld catalogue components** (11,321 px over 4 quadrants
  **[measured-here]**, seed 20). The live truth is *unmapped* faults. A method that merely learns
  "near the catalogue" will look good locally and fail live (the halo lesson). This is why the gate
  is also evaluated on pixels **> 1 km from any visible mapped fault**, and why every candidate is
  emitted with the ≤ 2 px catalogue exclusion that the ladder proved out.
* Catalogue-context features will trip the leakage canary almost by construction (AUC of
  distance-to-visible-fault vs withheld components). They are retained *only* if the > 1 km gate
  still passes, and the run card reports the canary verbatim.
* No number in this repository is a live score. Promotion to a real submission slot is a separate
  selector step.

## 5. Decision rule carried into the run card

Promote only if, at a budget inside 25k–50k dots: HOLDOUT-DTI(stage 2 or 3) − HOLDOUT-DTI(stage 1)
has a paired 95 % block-bootstrap CI entirely above 0 on **both** the full domain and the > 1 km
domain; rank-correlation with every registry raster < 0.90; < 70 % of dots within 3 px of any one
registry raster's dots; validator all-green; sha256 recorded. Otherwise log the negative result.
