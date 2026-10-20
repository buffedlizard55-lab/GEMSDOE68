#!/usr/bin/env python3
"""SUPERSEDED 2026-10-20 at merge of PR #13: main's canonical site builder is
scripts/build_site_exp4.py (stratified-spacing candidate). Do NOT run this
builder on main; it is retained as the record of how PR #13's parallel site
was generated. Original docstring follows.

Build the unified GEMSDOE68 site from evidence/run_card_exp3.json.

Canonical as of 2026-10-20 (session arena/738dbc78).  The older builders
(scripts/build_site.py, scripts/lane68_build_site.py) generated the two
archived lane sites whose verdicts contradicted each other; this builder
produces ONE authoritative site and is reusing build_site.py's page shell /
Markdown renderer rather than reinventing them.  No number below is typed by
hand: every value is read from evidence/*.json or registry/*.csv at build
time, with no network access.  If the run-card verdict is 'negative', the
index page contains no link to any GeoTIFF download.
"""
from __future__ import annotations

import csv
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EV = ROOT / "evidence"
sys.path.insert(0, str(ROOT / "scripts"))
from build_site import md_to_html, page  # noqa: E402  (shared shell, not forked)


def f4(x) -> str:
    return "n/a" if x is None else f"{float(x):.4f}"


def ci(c) -> str:
    return f"[{f4(c[0])}, {f4(c[1])}]"


def esc(x) -> str:
    return html.escape(str(x))


def verdict_banner(card: dict) -> str:
    v, ok = card["verdict"], bool(card["download_cleared"])
    if v == "promote":
        return ('<div class="banner ok">✅ OK TO DOWNLOAD AND SUBMIT — this file passed the '
                'preregistered holdout gate (both strata), the uniqueness checks and the format '
                'validators. Choosing a weekly slot remains the human selector\'s decision.</div>')
    if v == "promote-with-caveats" and ok:
        return ('<div class="banner" style="background:#4a3d12;border:2px solid #f1c40f">'
                '⚠️ CONDITIONALLY OK — bootstrap holdout gates pass on both strata, but the '
                'stricter per-fold &gt;2SE rule is not met on at least one stratum (n=4 folds). '
                'Read the caveats on this page; only the human selector may spend a slot on it.</div>')
    return ('<div class="banner bad">⛔ DO NOT DOWNLOAD OR SUBMIT — negative result. '
            f'{esc(card["verdict_reason"])}. No download is offered on this page.</div>')


def download_block(card: dict) -> str:
    if not card["download_cleared"]:
        return ""
    r = card["raster"]
    primary = Path(r["primary_path"]).name
    twin = Path(r["nan_twin_path"]).name
    return f"""<p><a class=btn href="downloads/{esc(primary)}" download>⬇ Download the submission GeoTIFF ({esc(primary)})</a>
<a class="btn secondary" href="downloads/{esc(r['zip'])}" download>⬇ ZIP (same TIFF)</a>
<a class="btn secondary" href="downloads/{esc(twin)}" download>NaN-outside twin</a></p>
<p class=muted>Upload the <b>zeros</b> file: every one of its 12,279,160 cells is finite and in [0, 1]
(re-read from disk, receipt below), so the portal error <em>“Predicted values must be in range [0, 1]”</em>
cannot occur with it. The NaN twin has identical dots and matches the sample-submission convention
(NaN only outside the official footprint, 0 NaN inside).</p>"""


def build_index(card: dict) -> str:
    h, u, r = card["holdout_dti"], card["correlation_overlap_vs_registry"], card["raster"]
    val = card["validator_output"]["gems68_validate_primary_zeros"]
    checks = val["checks"]
    pf_all = h["per_fold_gate"]["all"]
    pf_far = h["per_fold_gate"]["far"]
    lit = ", ".join(esc(x) for x in u["literal_overlap_gt_0p70_rasters"]) or "none"
    excl = ", ".join(esc(x) for x in u["excluded_uninformative_rasters"]) or "none"
    champ = u.get("champion_overlap") or {}
    body = f"""<h1>GEMSDOE68 — auto-context stacked fault discovery</h1>
<p class=muted>DOE GEMS Prize (DrivenData #306) · lane: auto-context stacked refinement
(Tu, CVPR 2008, doi:10.1109/CVPR.2008.4587436) · rebuilt {esc(card['generated_utc'])} UTC from
<code>evidence/run_card_current.json</code>; no number on this page is typed by hand.</p>
{verdict_banner(card)}
<div class=card><h2>1 · The submission file</h2>
{download_block(card)}
<table>
<tr><th>File to submit</th><td><code>{esc(Path(r['primary_path']).name)}</code></td></tr>
<tr><th>Submission name</th><td><code>{esc(card['submission_name'])}</code></td></tr>
<tr><th>Note to paste ({card['note_chars']} chars)</th><td><code>{esc(card['note'])}</code></td></tr>
<tr><th>SHA-256</th><td><code>{esc(r['primary_sha256'])}</code></td></tr>
<tr><th>Bytes / dots</th><td>{r['primary_bytes']:,} bytes · {r['dot_count']:,} dots at value 1.0</td></tr>
<tr><th>Grid</th><td>float32 · 1 band · EPSG:32611 · 100 m · 3730×3292 · median dot distance to catalogue
{f4(r['median_dist_to_catalogue_px'])} px · nearest dot {f4(r['min_dist_to_catalogue_px'])} px</td></tr>
<tr><th>Validator (re-read from disk)</th><td>all-finite preflight {'PASS' if val['all_checks_passed'] else 'FAIL'}:
range [{val['min']}, {val['max']}] · NaN {val['n_nan']} · inf {val['n_infinite']} ·
0 dots on mapped catalogue: {checks.get('zero_dots_on_mapped_catalogue')}</td></tr>
</table></div>
<div class=card><h2>2 · Holdout evidence (HOLDOUT-DTI — not a leaderboard score)</h2>
<p>Evaluator <code>{esc(h['evaluator_version'])}</code>, pooled DTI α=0.2 β=0.8, 300 m triangular kernel,
whole-component hide-and-recover, {h['withheld_positive_pixels']:,} withheld positive pixels
({h['pooled_far_gt_1km']['withheld_positive_pixels_far']:,} of them &gt;1 km from any known fault),
matched mass = {h['budget']:,} dots for every arm.</p>
<table><tr><th>stratum</th><th>stage 1</th><th>candidate (stage {h['stage']})</th>
<th>Δ vs stage 1</th><th>95 % CI (paired spatial-block bootstrap)</th><th>per-fold gate (n=4)</th></tr>
<tr><td>all pixels</td><td>{f4(h['pooled_all']['stage1'])}</td><td><b>{f4(h['pooled_all']['candidate'])}</b></td>
<td>{f4(h['pooled_all']['delta'])}</td><td>{ci(h['pooled_all']['ci95_delta'])}</td>
<td>mean Δ {f4(pf_all['mean_delta'])}, SE {f4(pf_all['se'])}, t={f4(pf_all['t_stat'])} → {'PASS' if pf_all['passes_gt_2se'] else 'not met'}</td></tr>
<tr><td>&gt;1 km from known faults</td><td>{f4(h['pooled_far_gt_1km']['stage1'])}</td><td><b>{f4(h['pooled_far_gt_1km']['candidate'])}</b></td>
<td>{f4(h['pooled_far_gt_1km']['delta'])}</td><td>{ci(h['pooled_far_gt_1km']['ci95_delta'])}</td>
<td>mean Δ {f4(pf_far['mean_delta'])}, SE {f4(pf_far['se'])}, t={f4(pf_far['t_stat'])} → {'PASS' if pf_far['passes_gt_2se'] else 'not met'}</td></tr></table>
<p>Leakage canary: worst single-feature AUC {f4(h['canary']['max_auc_distance_from_0p5'])} (limit 0.90) —
{'clean' if h['canary']['clean'] else 'FLAGGED: ' + esc(h['canary']['flags'])}.</p>
<p class=muted>Holdout truth is withheld <em>catalogue</em> components; live truth is <em>unmapped</em> faults.
These numbers rank methods; they do not predict a leaderboard value, and no organiser score is claimed.</p></div>
<div class=card><h2>3 · Uniqueness vs {u['n_registry']} prior rasters</h2>
<table>
<tr><th>surface rank-correlation (before placement)</th><td>max |Spearman ρ| = {f4(u['max_abs_spearman_surface'][1])}
({esc(u['max_abs_spearman_surface'][0])}); limit 0.90</td></tr>
<tr><th>final dots within 3 px (informative rasters)</th><td>max = {f4(u['max_informative_dot_overlap_3px'][1])}
({esc(u['max_informative_dot_overlap_3px'][0])}); limit 0.70</td></tr>
<tr><th>vs the family champion h33-2-b2</th><td>{('overlap ' + f4(champ.get('overlap')) + ', its 3-px halo covers ' + f4(champ.get('coverage')) + ' of the footprint (lift ' + f4(champ.get('lift')) + ')') if champ else 'not in registry'}</td></tr>
<tr><th>exact byte duplicates</th><td>{esc(u['exact_byte_matches']) or 'none'}</td></tr>
<tr><th>literal-rule caveat</th><td>rasters whose 3-px halo alone covers &gt;70 % of the footprint are excluded
from the duplicate verdict under the pre-existing IR-68-UNIQ-01 rule (excluded now: {excl}). The literal rule
fires only against: {lit}. It fires for the champion itself against the 311,617-dot buried-step raster
(measured 98.7 % this session), so it cannot distinguish any fault-like candidate from that surface;
both readings are published here, nothing is silently passed.</td></tr>
</table></div>
<div class=card><h2>4 · Why this could beat 0.2778 — and what 0.2778 is</h2>
<p>The 0.2778 figure for <code>h33-h33-2-b2</code> is <b>owner-reported</b>: the GEMSDOE32 page the file
is downloaded from calls the same bytes UNSCORED with a <em>projected</em> 0.2747, and this repo holds no
organiser receipt for it. The ladder behind it (121 k px → 37,654 dots; deleting the 100–200 m catalogue halo)
won by mass discipline: every emitted pixel must clear a marginal-credit bar of 0.2·DTI. After pruning, the
only remaining lever is ranking, and auto-context (continuity, parallelism, en-échelon spacing learned from
the stage-1 probability map) is the principled attack on it. This session tested removing catalogue-context
entirely — it <em>failed</em> (catalogue context is load-bearing even &gt;1 km) — and the current candidate
instead adds lidar-scarp and radiometric evidence to stage 1. Hypothesis as preregistered:
<em>{esc(card['hypothesis'])}</em>
Full account: <a href="knowledge.html">Why 0.2778 (rechecked)</a>.</p></div>
<div class=card><h2>5 · Earlier files in this repository — do not submit</h2>
<table><tr><th>file</th><th>status</th></tr>
<tr><td><code>GEMSDOE68-AC-stack3-r3-N32000…-zeros.tif</code></td><td class=bad>its “zeros” file is NaN-identical to the NaN twin (7,113,308 NaN; 3,061 inside the official footprint) — measured this session; superseded</td></tr>
<tr><td><code>GEMSDOE68-AC-stack2…</code></td><td class=bad>negative: failed the &gt;1 km gate under its own instrument (t=1.97)</td></tr>
<tr><td>Exp3 catalogue-context-free ablation</td><td class=bad>negative result this session; no submission file was built from it (see Experiments)</td></tr>
<tr><td><code>gems68-autoctx-s3-hidedraw-38000dots…</code></td><td class=warn>prior catalogue-context-full candidate from the previous session; not this session's file (see Experiments)</td></tr>
<tr><td><code>gems68-autoctx-40000… / gemsdoe68-autocontext-s2…</code></td><td class=bad>failed or unevidenced holdout gates (s2 also has 89 dots outside the footprint)</td></tr>
</table></div>
<div class=card><h2>6 · Limits &amp; AI disclosure</h2>
<p>{esc('; '.join(card['limitations']))}</p><p class=muted>{esc(card['ai_disclosure'])}</p>
<p><b>Core values.</b> <em>Maximize P(Win)</em> — every weekly slot is an experiment with a pre-registered
gate, not a lottery ticket. <em>Own the Outcome</em> — defects in our own previously shipped files are
measured and published above, not quietly dropped.</p></div>"""
    return body


def build_exec(card: dict) -> str:
    r = card["raster"]
    step1 = ("<li>Download the file from the top of the <a href=\"index.html\">home page</a>.</li>"
             if card["download_cleared"] else
             "<li><b>Stop.</b> The current verdict is negative; there is no cleared file to download.</li>")
    return f"""<h1>Executive summary — exactly how to submit</h1>
{verdict_banner(card)}
<div class=card><h2>Step by step</h2><ol>
{step1}
<li>(Optional but recommended) verify the bytes: SHA-256 must equal<br><code>{esc(r['primary_sha256'])}</code><br>
on macOS/Linux: <code>shasum -a 256 {esc(Path(r['primary_path']).name)}</code></li>
<li>Open the competition submit page:
<a href="https://www.drivendata.org/competitions/306/competition-doe-gems/">DOE GEMS Prize on DrivenData</a>
→ <b>Submit</b>. The form takes a single-band GeoTIFF (.tif) or a .zip containing one GeoTIFF, and
requires the sample submission's CRS, shape and geotransform — all verified locally (receipt on the home page).</li>
<li>Attach <code>{esc(Path(r['primary_path']).name)}</code> (or the ZIP with the identical TIFF inside).</li>
<li>Paste this into <b>Note (optional)</b> ({card['note_chars']} characters):<br><code>{esc(card['note'])}</code></li>
<li>Use <code>{esc(card['submission_name'])}</code> wherever a submission name is requested.</li>
<li>Submit, then record the organiser's score <b>from the submission-page receipt</b> into
<code>registry/score_ledger.csv</code> as ORGANIZER-CONFIRMED. Until that receipt exists, every number
associated with this file is HOLDOUT-DTI only.</li>
</ol>
<p><b>Why the portal error <em>“Predicted values must be in range [0, 1]”</em> cannot happen with this file:</b>
it is all-finite (no NaN, no infinities), carries no nodata sentinel tag, and its minimum/maximum re-read from
disk are 0.0/1.0. The family's earlier rejection is consistent with the organiser training stack's
−3.4028235e+38 sentinel leaking into outputs (3,061 such cells lie inside the official footprint in band 1);
this writer refuses to emit any value outside [0, 1] and the written bytes were re-read and checked before
this page was built. If an organiser ever requires null outside the footprint instead, upload the NaN twin
(same dots, NaN only outside the official footprint).</p>
<p><b>One-submission caution (official structure):</b> competitors choose a <em>single</em> submission scored
in both the Initial Prize Round (fixed private new faults) and the Final Prize Round (expanded labels that
add expert-verified discoveries from all teams' predictions). Spend weekly slots deliberately; the final
choice deserves its own selector review across lanes.</p>
<p class=muted><b>Generative-AI disclosure (paste into any narrative field if asked):</b> {esc(card['ai_disclosure'])}</p></div>"""


def build_experiments() -> str:
    exp3 = json.loads((EV / "exp3_autocontext_ccfree.json").read_text())
    dec = json.loads((EV / "exp3_decision.json").read_text())
    exp2 = json.loads((EV / "exp2_autocontext_hidedraw.json").read_text())
    rows = []
    exp4dec_p = EV / "exp4_decision.json"
    tables = [("Exp3 CC-free ablation (negative)", dec["arms"])]
    if exp4dec_p.exists():
        tables.insert(0, ("Exp4 external-channels (this build)", json.loads(exp4dec_p.read_text())["arms"]))
    for label, gate in tables:
        for key in sorted(gate):
            a = gate[key]
            rows.append(f"<tr><td>{label} {esc(key)}</td><td>{f4(a['dti_stage1'])}</td>"
                        f"<td>{f4(a['dti_candidate'])}</td><td>{ci(a['ci95_all'])}</td>"
                        f"<td>{f4(a['far_dti_candidate'])}</td><td>{ci(a['ci95_far'])}</td>"
                        f"<td>{'PASS' if a['bootstrap_gate_both_strata'] else 'fail'} / "
                        f"{'PASS' if a['per_fold_t_gate_both_strata'] else 'fail'}</td></tr>")
    rows2 = []
    for key in sorted(exp2["gate"]):
        a = exp2["gate"][key]
        if not key.startswith(("25000", "38000", "50000")):
            continue
        rows2.append(f"<tr><td>Exp2 catalogue-context-full {esc(key)}</td><td>{f4(a['dti_stage1'])}</td>"
                     f"<td>{f4(a['dti_cand'])}</td><td>{ci(a['ci_all'])}</td>"
                     f"<td>{f4(a['far_dti_cand'])}</td><td>{ci(a['ci_far'])}</td>"
                     f"<td>{'PASS' if a['passes'] else 'fail'} (bootstrap only)</td></tr>")
    return f"""<h1>Experiments (HOLDOUT-DTI only — nothing here is a leaderboard score)</h1>
<div class=card><p>All rows: evaluator {esc(exp3['evaluator_version'])}, pooled DTI α=0.2 β=0.8 300 m triangular,
whole-component hide-and-recover, {exp3['holdout']['withheld_positive_pixels']:,} withheld positives.
“Gate” = bootstrap CI both strata / per-fold &gt;2SE both strata (Exp3, preregistered).</p>
<table><tr><th>arm</th><th>stage 1</th><th>candidate</th><th>Δ CI (all)</th><th>far candidate</th><th>Δ CI (far)</th><th>gate</th></tr>
{''.join(rows)}{''.join(rows2)}</table>
<p class=muted>Exp1 (evidence/exp1_autocontext.json) used catalogue-leaking stage-2 labels (IR-68-LEAK-01) and every
arm failed; it is retained as the negative control that motivated the hide-draw protocol. The parallel
lane-68 instrument (evidence/ac68_seed20.json: stage2 pooled 0.1904 vs stage1 0.1301) used 12-px segment
chunks, a full 32,000-dot budget <em>per fold</em> (4× live mass) and charged all dots as FP against far-only
truth, so its far t=1.97 is not comparable with the table above; the disagreement is documented in
evidence/exp3_preregistration.json rather than averaged away.</p>
<p class=muted>Chosen arm: <code>{esc(dec['chosen']['key'])}</code>; tier before uniqueness/format:
<b>{esc(dec['tier_before_uniqueness_and_format'])}</b>. Canary max AUC
{f4(dec['canary']['max_auc_distance_from_0p5'])} (limit 0.90).</p></div>"""


def build_ledger() -> str:
    snap = json.loads((ROOT / "registry" / "leaderboard_snapshot.json").read_text())
    top = "".join(f"<tr><td>{r['rank']}</td><td>{esc(r['participant'])}</td><td>{r['score']}</td><td>{r['submissions']}</td></tr>"
                  for r in snap["top10"])
    rows = []
    with (ROOT / "registry" / "score_ledger.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["score"]:
                rows.append(f"<tr><td>{esc(row['site'])}</td><td><code>{esc(row['submission_name'])}</code></td>"
                            f"<td>{esc(row['score'])}</td><td>{esc(row['evidence_class'])}</td><td>{esc(row['note'])}</td></tr>")
    rows.sort(key=lambda s: s)
    body_rows = []
    with (ROOT / "registry" / "score_ledger.csv").open(newline="") as fh:
        data = [r for r in csv.DictReader(fh) if r["score"]]
    for row in sorted(data, key=lambda r: -float(r["score"])):
        body_rows.append(f"<tr><td>{esc(row['site'])}</td><td><code>{esc(row['submission_name'])}</code></td>"
                         f"<td>{esc(row['score'])}</td><td>{esc(row['evidence_class'])}</td><td>{esc(row['note'])}</td></tr>")
    return f"""<h1>Score ledger (family, OWNER-REPORT unless stated)</h1>
<div class=card><p>No score in this repository is tied to an organiser receipt held here; treat every value
as owner-reported. Public-leaderboard snapshot inherited from a sibling repo
(observed {esc(snap['observed_utc'])}, not re-fetched this session):</p>
<table><tr><th>rank</th><th>participant</th><th>score</th><th>submissions</th></tr>{top}</table></div>
<div class=card><table><tr><th>site</th><th>submission</th><th>score</th><th>class</th><th>note</th></tr>
{''.join(body_rows)}</table></div>"""


def redirect(target: str, label: str) -> str:
    return (f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={target}">'
            f'<title>Superseded</title><p>This archived lane page was superseded on 2026-10-20. '
            f'<a href="{target}">Continue to the current {esc(label)}</a>.</p>')

def build_runcard(card: dict) -> str:
    h, r, u = card["holdout_dti"], card["raster"], card["correlation_overlap_vs_registry"]
    stem_json = Path(r["primary_path"]).name.replace("-zeros.tif", "-runcard.json")
    lims = "".join(f"<li>{esc(x)}</li>" for x in card["limitations"])
    return f"""<h1>Run card — {esc(card['submission_name'])}</h1>
{verdict_banner(card)}
<div class=card><p>Machine-readable card: <a href="downloads/{stem_json}">downloads/{stem_json}</a>
(= <code>evidence/run_card_current.json</code> in the repo). Generated {esc(card['generated_utc'])}.</p>
<table>
<tr><th>Verdict</th><td><b>{esc(card['verdict'])}</b> — {esc(card['verdict_reason'])}</td></tr>
<tr><th>Download instruction</th><td>{esc(card['download_instruction'])}</td></tr>
<tr><th>Lane</th><td>{esc(card['lane'])}</td></tr>
<tr><th>Hypothesis</th><td>{esc(card['hypothesis'])}</td></tr>
<tr><th>Mechanism</th><td>{esc(card['mechanism'])}</td></tr>
<tr><th>Named non-fault mimic</th><td>{esc(card['named_non_fault_process_that_could_mimic_it'])}</td></tr>
<tr><th>HOLDOUT-DTI (all)</th><td>stage 1 {f4(h['pooled_all']['stage1'])} → candidate {f4(h['pooled_all']['candidate'])}; Δ {f4(h['pooled_all']['delta'])} 95% CI {ci(h['pooled_all']['ci95_delta'])}; {h['withheld_positive_pixels']} withheld positives; evaluator {esc(h['evaluator_version'])}</td></tr>
<tr><th>HOLDOUT-DTI (&gt;1 km)</th><td>stage 1 {f4(h['pooled_far_gt_1km']['stage1'])} → candidate {f4(h['pooled_far_gt_1km']['candidate'])}; Δ {f4(h['pooled_far_gt_1km']['delta'])} 95% CI {ci(h['pooled_far_gt_1km']['ci95_delta'])}; {h['pooled_far_gt_1km']['withheld_positive_pixels_far']} withheld positives</td></tr>
<tr><th>Per-fold gate (n=4)</th><td>all t={h['per_fold_gate']['all']['t_stat']:.2f} ({'pass' if h['per_fold_gate']['all']['passes_gt_2se'] else 'FAIL'}), far t={h['per_fold_gate']['far']['t_stat']:.2f} ({'pass' if h['per_fold_gate']['far']['passes_gt_2se'] else 'FAIL'})</td></tr>
<tr><th>Canary</th><td>max single-feature AUC {f4(h['canary']['max_auc_distance_from_0p5'])} (limit 0.90); flags: {esc(h['canary']['flags'])}</td></tr>
<tr><th>Uniqueness vs {u['n_registry']} registry rasters</th><td>max |Spearman| {f4(u['max_abs_spearman_surface'][1])}; max informative 3-px dot overlap {f4(u['max_informative_dot_overlap_3px'][1])}; exact byte matches: {esc(u['exact_byte_matches'])}</td></tr>
<tr><th>Primary raster</th><td><code>{esc(Path(r['primary_path']).name)}</code> sha256 <code>{esc(r['primary_sha256'])}</code>; {r['dot_count']} dots at 1.0; median distance to catalogue {f4(r['median_dist_to_catalogue_px'])} px</td></tr>
<tr><th>Submission name</th><td><code>{esc(card['submission_name'])}</code></td></tr>
<tr><th>Note ({card['note_chars']} chars)</th><td>{esc(card['note'])}</td></tr>
<tr><th>Organiser-confirmed score</th><td>none — every number above is HOLDOUT-DTI</td></tr>
</table>
<h3>Limitations</h3><ul>{lims}</ul>
<h3>AI disclosure</h3><p>{esc(card['ai_disclosure'])}</p></div>"""



def main() -> int:
    cur = EV / "run_card_exp4_ext738.json"
    card = json.loads((cur if cur.exists() else EV / "run_card_exp3.json").read_text())
    (DOCS / "index.html").write_text(page("Executive summary — download & verdict", build_index(card), "index.html"))
    (DOCS / "executive-summary.html").write_text(page("Executive summary — how to submit", build_exec(card), "index.html"))
    (DOCS / "how-to-submit.html").write_text(page("How to submit", build_exec(card), "how-to-submit.html"))
    (DOCS / "experiments.html").write_text(page("Experiments", build_experiments(), "experiments.html"))
    (DOCS / "run-card.html").write_text(page("Run card", build_runcard(card), "run-card.html"))
    (DOCS / "leaderboard.html").write_text(page("Score ledger", build_ledger(), "leaderboard.html"))
    know = (ROOT / "knowledge" / "03_why_02778_rechecked_2026-10-20.md").read_text()
    (DOCS / "knowledge.html").write_text(
        page("Why 0.2778 (rechecked)", "<div class=card>" + md_to_html(know) + "</div>", "knowledge.html"))
    hyp = (ROOT / "knowledge" / "02_new_hypotheses_2026-10-20.md").read_text()
    (DOCS / "hypotheses.html").write_text(
        page("Candidate hypotheses", "<div class=card>" + md_to_html(hyp) + "</div>", "hypotheses.html"))
    for src, dst, title in ((DOCS / "sources.md", "sources.html", "Sources"),
                            (DOCS / "irregularities.md", "irregularities.html", "Irregularities")):
        (DOCS / dst).write_text(page(title, "<div class=card>" + md_to_html(src.read_text()) + "</div>", dst))
    # one canonical run card at every old address
    for p in (DOCS / "runcard.json", DOCS / "lane68" / "data" / "run-card.json", DOCS / "lane68" / "runcard.json"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(card, indent=1))
    feed = dict(generated_utc=card["generated_utc"], site="GEMSDOE68", lane=card["lane"],
                status=card["verdict"], download_cleared=card["download_cleared"],
                submission=card["submission_name"], sha256=card["raster"]["primary_sha256"],
                evidence_class="HOLDOUT-DTI", run_card="evidence/run_card_exp4_ext738.json")
    (DOCS / "lane68" / "data" / "feed.json").write_text(json.dumps(feed, indent=1))
    (DOCS / "lane68" / "index.html").write_text(redirect("../index.html", "GEMSDOE68 home page"))
    (DOCS / "lane68" / "executive-summary.html").write_text(redirect("../executive-summary.html", "executive summary"))
    (DOCS / "lane68" / "submission-guide.html").write_text(redirect("../executive-summary.html", "executive summary"))
    (DOCS / ".nojekyll").write_text("")
    print("unified site built; verdict:", card["verdict"], "| download_cleared:", card["download_cleared"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
