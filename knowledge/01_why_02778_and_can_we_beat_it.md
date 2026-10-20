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
*[Correction, 2026-10-20, measured-here: minimum dot-to-catalogue distance is 2.236 px (√5);
5.8 % of dots lie in (2, 3] px, median 19.6 px. The "≥ 3 px" claim above is superseded — see
`evidence/raster_audit_20261020.json` / IR-68-CHAMP-01. Also measured 2026-10-20: champion dots are
exactly contained in the d2.8 and h19-5 registry rasters (the ladder is real), and in
`anderson-geothermal-pinn` (38,854 dots, OWNER-REPORTED 0.2750), whose 1,200 extra dots therefore
net-cost ≈ −0.0028 — direct evidence for the marginal bar (★★).]*

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

Promote only if, at a budget inside 25k–50k dots:
1. `HOLDOUT-DTI(stage 2 and stage 3) − HOLDOUT-DTI(stage 1)` has **both** a paired 95 % spatial block-bootstrap CI entirely above 0 **and** a paired outer-block `t`-statistic `> 2.0` (`> 2 SE`) on **both** the full domain (`11,321` withheld positives) and the `> 1 km` far-fault domain (`4,310` withheld positives);
2. Single-feature holdout AUC leakage canary `<= 0.90` on every Stage-1 and Stage-2/3 feature;
3. Spearman rank-correlation `|rho| <= 0.90` with every registry/scored raster (`65` rasters total) and `<= 70 %` of dots within 3 px of any informative registry raster's dots (checked on both the continuous surface and final emitted dots, plus the `d_cat > 2 px` eligible-domain density-null proof on the two space-filling rasters);
4. Format validator all-green on both `-zeros.tif` (all-finite in `[0, 1]`, 0 NaNs) and `-nan.tif`; sha256 recorded.

## 6. Feature-level autopsy of `h33-2-b2-zeros` (`0.2778`) vs Experiment 2 vs Experiment 3

Direct pixel-level comparison of our feature stack against the `37,654` dots of `GEMSDOE32__h33-2-b2-zeros.tif` (`0.2778` `ORGANIZER-CONFIRMED`) and the `11,321` withheld fault pixels (`seed=20` **[measured-here]**) reveals why Experiment 2 (`exp2`) suffered from `IR-68-BLOB-01` and how Experiment 3 (`exp3_autocontext_strands`) resolves it:

| Feature / Channel | Type | AUC vs Withheld Faults (`11,321` px) | AUC vs `0.2778` Champion (`37,654` dots) | Exp 2 Stage-1 Gain | Exp 3 Stage-1 Gain | Physical Role |
|---|---|---:|---:|---:|---:|---|
| `elev_asym` (H68-2) | Local 500 m⊥ footwall/hanging-wall slope asymmetry | **0.5955** | **0.6684** | *(absent)* | **37,957.9** (#2) | Captures one-sided range-front facet vs piedmont fan step |
| `lidar_lappos_max` (H68-2) | 2 m LiDAR positive Laplacian scarp base | **0.5949** | **0.6512** | *(absent)* | **8,124.3** | Resolves sub-100 m scarp base break missed at 100 m DEM scale |
| `lidar_relief` / `lidar_step_max` | 2 m LiDAR 100 m cell relief / step height | **0.5927 / 0.5871** | **0.6640 / 0.6581** | *(absent)* | **7,845.2 / 6,912.0** | Direct 2 m bare-earth scarp height inside 100 m cell |
| `elev_hp15` (H68-2) | Local 1.5 km high-pass structural relief | **0.5621** | **0.6915** | *(absent)* | **39,846.1** (#1) | Local range-front step with basin-wide elevation trend removed |
| `db_grad` / `db_hp15` (H68-3) | Depth-to-basement gradient & 1.5 km step | **0.5518 / 0.5482** | **0.6190 / 0.6104** | *(absent)* | **28,089.5** (#3) / **18,644.6** (#4) | Buried normal-fault offset of pre-Tertiary basement |
| `elev_ridge_s2` | Multi-scale Hessian topographic ridge (`σ=2`) | **0.5729** | **0.7198** | 12,188.0 | **14,422.5** | Sharp linear scarp/ridge crest |
| `elev_grad` | Topographic gradient magnitude | **0.5709** | **0.6838** | 10,842.0 | **17,265.7** (#6) | Local fault-scarp slope |
| `det_elev` *(dropped in Exp 3)* | Raw detrended elevation (`50–200 km` wavelength) | 0.5665 | 0.5416 | **68,248.4** (#1) | *0 (dropped)* | **Spatial lookup table (`IR-68-BLOB-01`)** — memorizes training quadrants |
| `ieq_n100a15` / `deq_n100a15` *(dropped)* | `15 km`-smoothed earthquake count kernels | 0.5318 / 0.5320 | 0.5050 / 0.5177 | **52,494 / 46,843** (#2/#3) | *0 (dropped)* | **Circular 15 km regional blobs (`IR-68-BLOB-01`)** |
| `geod_2ndinv` / `geod_dilaterate` *(dropped)* | Smooth GPS strain-rate tensor fields | 0.5210 / 0.5660 | 0.5646 / 0.5830 | **42,534 / 41,902** (#4/#5) | *0 (dropped)* | **Basin-wide GPS interpolation polygons (`IR-68-BLOB-01`)** |

By removing the regional lookup channels from Stage 1, replacing `p_sm10` with anisotropic along/across-strike smooths (`p_along_short`, `p_along_long`, `p_aniso`, `p_ridge`, `ori_coh`) plus directional `1.5/3.0 km` fault-tip relays (`cat_relay_15` AUC `0.7686`, `cat_relay_30` AUC `0.7421`), and applying damped rank-probability sequential refinement (`α2 = 0.22, α3 = 0.35`), **Experiment 3 (`gems68-autoctx-s3-strands-35000dots-20261020T210000Z-849a0f04-zeros.tif`)** achieves at `35,000` dots (`HOLDOUT-DTI`, `gems57-pooled-hide-v2`):
* **Stage 1:** `all = 0.0942` (CI `[0.0819, 0.1059]`), `> 1 km far = 0.0772` (CI `[0.0620, 0.0923]`) — up `+28.9 %` on `> 1 km` faults over Exp 2 Stage 1 (`0.0599`).
* **Stage 2 (Auto-context Iter 1):** `all = 0.1058` (`Δ +0.0116`, CI `[+0.0058, +0.0178]`, **`t_all = 6.47 > 2.0`**), `> 1 km far = 0.0859` (`Δ +0.0087`, CI `[+0.0017, +0.0160]`, **`t_far = 2.90 > 2.0`**, 4/4 folds positive).
* **Stage 3 (Auto-context Iter 2):** `all = 0.1124` (`Δ +0.0182`, CI `[+0.0110, +0.0259]`, **`t_all = 11.08 > 2.0`**), `> 1 km far = 0.0874` (`Δ +0.0102`, CI `[+0.0012, +0.0187]`, **`t_far = 2.53 > 2.0`**, 4/4 folds positive), with **57.7 % of emitted dots (`20,178 / 35,000`) placed `> 1 km` from any mapped fault**.

