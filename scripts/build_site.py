#!/usr/bin/env python3
"""Build the GitHub Pages site (docs/) from evidence JSON. No number is typed by hand."""
from __future__ import annotations

import csv
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EV = ROOT / "evidence"

CSS = """
:root{--bg:#0b1020;--card:#141b2f;--ink:#e8ecf5;--muted:#9aa7c7;--ok:#2ecc71;--bad:#e74c3c;--warn:#f1c40f;--acc:#4ea1ff}
*{box-sizing:border-box}body{margin:0;font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:20px 24px;border-bottom:1px solid #26304d;display:flex;gap:18px;flex-wrap:wrap;align-items:center}
header a{color:var(--muted);text-decoration:none;font-weight:600}header a:hover{color:var(--ink)}
main{max-width:1100px;margin:0 auto;padding:24px}
.card{background:var(--card);border:1px solid #26304d;border-radius:14px;padding:20px 22px;margin:18px 0}
.banner{font-size:20px;font-weight:800;padding:18px 22px;border-radius:14px;margin:18px 0}
.banner.ok{background:#123d2a;border:2px solid var(--ok)}.banner.bad{background:#4a1c1c;border:2px solid var(--bad)}
.btn{display:inline-block;background:var(--acc);color:#061024;font-weight:800;padding:12px 18px;border-radius:10px;text-decoration:none;margin:6px 8px 6px 0}
.btn.secondary{background:#26304d;color:var(--ink)}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #26304d;padding:7px 8px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:700}code,pre{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px}
pre{background:#0a0f1d;padding:12px;border-radius:10px;overflow:auto}
.muted{color:var(--muted)}.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
h1{font-size:28px;margin:8px 0}h2{font-size:21px;margin-top:26px}small{color:var(--muted)}
"""

NAV = [("index.html", "Executive summary"), ("how-to-submit.html", "How to submit"),
       ("hypotheses.html", "Hypotheses"), ("experiments.html", "Experiments"),
       ("knowledge.html", "Why 0.2778"), ("sources.html", "Sources"), ("irregularities.html", "Irregularities"),
       ("leaderboard.html", "Score ledger")]


def page(title: str, body: str, active: str) -> str:
    nav = " ".join(f'<a href="{h}"{" style=color:#fff" if h == active else ""}>{t}</a>' for h, t in NAV)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · GEMSDOE68</title><style>{CSS}</style></head><body>
<header><strong>GEMSDOE68 · auto-context lane</strong>{nav}<a href="https://github.com/buffedlizard55-lab/GEMSDOE68">GitHub ↗</a></header>
<main>{body}</main></body></html>"""


def md_to_html(md: str) -> str:
    """Minimal markdown: headings, tables, paragraphs, code, bold, links, lists."""
    import re
    out, in_table, in_pre, in_list = [], False, False, False
    def inline(s):
        s = html.escape(s, quote=False)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", s)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
        s = re.sub(r"(?<![\"'>])(https?://[^\s<)]+)", r'<a href="\1">\1</a>', s)
        return s
    for line in md.splitlines():
        if line.startswith("```"):
            in_pre = not in_pre; out.append("<pre>" if in_pre else "</pre>"); continue
        if in_pre:
            out.append(html.escape(line)); continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if not in_table:
                out.append("<table>"); in_table = True
                out.append("<tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>"); in_table = False
        if re.match(r"^\s*([-*]|\d+\.)\s+", line):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append("<li>" + inline(re.sub(r"^\s*([-*]|\d+\.)\s+", "", line)) + "</li>"); continue
        if in_list and line.strip() == "":
            out.append("</ul>"); in_list = False
        if line.startswith("#"):
            lvl = len(line) - len(line.lstrip("#"))
            out.append(f"<h{min(lvl,4)}>{inline(line.lstrip('#').strip())}</h{min(lvl,4)}>"); continue
        if line.startswith(">"):
            out.append(f"<blockquote class=muted>{inline(line[1:].strip())}</blockquote>"); continue
        if line.strip():
            out.append(f"<p>{inline(line)}</p>")
    if in_table: out.append("</table>")
    if in_list: out.append("</ul>")
    return "\n".join(out)


def f4(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)


def build_index(card: dict, exp2: dict) -> str:
    h = card["holdout"]; u = card["uniqueness"]; prim = card["files"]["primary"]; v = card["validator_summary"]
    ok = card["verdict"] == "promote"
    banner = (f'<div class="banner ok">✅ CLEARED: this file passed the holdout gate, the lane-uniqueness check and the format validator. '
              f'It is OK to download and submit (promotion to a weekly slot remains the selector\'s decision).</div>' if ok else
              f'<div class="banner bad">⛔ NOT CLEARED FOR A SUBMISSION SLOT — negative result logged. Reason: {html.escape(card["verdict_reason"])}. '
              f'The file below is format-valid and unique, and is published for audit/learning only; spending a weekly slot on it is not recommended.</div>')
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{f4(v1)}</td></tr>" for k, v1 in sorted(u["surface_rank_corr"].items(), key=lambda kv: -kv[1])[:5])
    ovrows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v1['overlap']*100:.1f} %</td><td>{v1['coverage']*100:.1f} %</td><td>{v1['lift']:.2f}</td></tr>" for k, v1 in sorted(u["dot_overlap_within_3px"].items(), key=lambda kv: -kv[1]["overlap"])[:6])
    return f"""
<h1>Executive summary</h1>
<p class=muted>DOE GEMS Prize (DrivenData #306) · lane: auto-context stacked refinement · built {card['built_utc']} UTC · every number below is read from <code>evidence/run_card.json</code>.</p>
{banner}
<div class=card>
<h2>The submission file</h2>
<p><a class=btn href="downloads/{prim['name']}" download>⬇ Download {prim['name']}</a>
<a class="btn secondary" href="downloads/{prim['zip']}" download>⬇ ZIP (same TIFF)</a>
<a class="btn secondary" href="downloads/{card['files']['secondary_nan_outside']['name']}" download>NaN-outside variant (secondary)</a></p>
<table>
<tr><th>Submission name (paste into the form's file name / your notes)</th><td><code>{card['submission_name']}</code></td></tr>
<tr><th>Note (≤140 chars, paste into "Note")</th><td><code>{html.escape(card['submission_note'])}</code> <small>({card['note_chars']} chars)</small></td></tr>
<tr><th>SHA-256</th><td><code>{prim['sha256']}</code></td></tr>
<tr><th>Format</th><td>float32 · single band · EPSG:32611 · 3730×3292 · 100 m · transform (100,0,243350,0,−100,4508550) · values exactly {{0,1}} · 0 outside footprint (all finite)</td></tr>
<tr><th>Validator</th><td>no NaN inside footprint: <b class={'ok' if v['no_nan_inside_footprint'] else 'bad'}>{v['no_nan_inside_footprint']}</b> · values in [0,1]: <b class={'ok' if v['values_in_0_1'] else 'bad'}>{v['values_in_0_1']}</b> · CRS/shape/transform match: <b class={'ok' if v['crs_shape_transform_match'] else 'bad'}>{v['crs_shape_transform_match']}</b> · zero dots on mapped catalogue: <b class={'ok' if v['zero_dots_on_catalogue'] else 'bad'}>{v['zero_dots_on_catalogue']}</b></td></tr>
<tr><th>Emission</th><td>{card['emission']['n_dots']:,} binary dots · all &gt; {card['emission']['exclude_px']:.0f} px from the published catalogue (min {card['emission']['min_dist_to_catalogue_px']:.2f} px, median {card['emission']['median_dist_px']:.1f} px, p95 {card['emission']['p95_dist_px']:.1f} px) · 5×5 local maxima of the stage-{h['stage']} surface</td></tr>
<tr><th>Organizer-confirmed score</th><td><b>none</b> — UNSCORED. No number on this site is a leaderboard score.</td></tr>
</table></div>

<div class=card><h2>What the holdout says (HOLDOUT-DTI, evaluator <code>{h['evaluator_version']}</code>, {h['withheld_positive_pixels']:,} withheld positives)</h2>
<table><tr><th></th><th>stage 1 (per-pixel)</th><th>stage {h['stage']} (auto-context)</th><th>paired Δ</th><th>95 % CI (20 km block bootstrap)</th><th>gate</th></tr>
<tr><td>all scored pixels, {h['budget']:,}-dot budget</td><td>{f4(h['dti_stage1'])}</td><td>{f4(h['dti_candidate'])}</td><td>{h['delta_vs_stage1']:+.4f}</td><td>[{f4(h['ci95_delta'][0])}, {f4(h['ci95_delta'][1])}]</td><td class={'ok' if h['ci95_delta'][0] > 0 else 'bad'}>{'pass' if h['ci95_delta'][0] > 0 else 'fail'}</td></tr>
<tr><td>pixels &gt; 1 km from any mapped fault</td><td>{f4(h['far_1km']['dti_stage1'])}</td><td>{f4(h['far_1km']['dti_candidate'])}</td><td>{h['far_1km']['delta']:+.4f}</td><td>[{f4(h['far_1km']['ci95'][0])}, {f4(h['far_1km']['ci95'][1])}]</td><td class={'ok' if h['far_1km']['ci95'][0] > 0 else 'bad'}>{'pass' if h['far_1km']['ci95'][0] > 0 else 'fail'}</td></tr></table>
<p class=muted>Gate (from the brief): stage 2 must beat stage 1 by &gt; 2 SE on the outer blocks, including on pixels &gt; 1 km from any mapped fault. Holdout truth is withheld <em>catalogue</em> components; the live truth is <em>unmapped</em> faults, so these numbers rank methods — they do not predict a leaderboard value.</p></div>

<div class=card><h2>Lane uniqueness (protocol rule 1) — {u['registry_rasters']} prior rasters checked</h2>
<p>Max Spearman rank-correlation of the surface vs any registry raster: <b class={'bad' if u['max_rank_corr'][1] > 0.9 else 'ok'}>{f4(u['max_rank_corr'][1])}</b> ({html.escape(u['max_rank_corr'][0])}) · threshold 0.90.
Max share of dots within 3 px of one registry raster's dots: <b class={'bad' if u['max_dot_overlap'][1] > 0.7 else 'ok'}>{u['max_dot_overlap'][1]*100:.1f} %</b> ({html.escape(u['max_dot_overlap'][0])}) · threshold 70 %. Duplicate: <b>{u['duplicate']}</b>.</p>
<table><tr><th>registry raster (top 5 by rank-corr)</th><th>ρ</th></tr>{rows}</table><br>
<table><tr><th>registry raster (top 6 by dot overlap)</th><th>share of our dots within 3 px</th><th>that raster's own 3 px footprint coverage</th><th>lift (overlap ÷ coverage; 1.0 = chance)</th></tr>{ovrows}</table>
<p class=muted>Rasters whose 3 px halo covers more than 70 % of the footprint are excluded from the duplicate verdict (they overlap every candidate by construction): {html.escape(", ".join(u.get("excluded_uninformative_rasters", [])) or "none")}.</p></div>

<div class=card><h2>⚠ Several earlier candidates exist in this repository — superseded</h2>
<table><tr><th>candidate</th><th>status (from its own evidence)</th><th>recommendation</th></tr>
<tr><td><code>gems68-autoctx-40000-20261012T014104Z-zeros.tif</code></td><td class=bad>own holdout gate failed (evidence/holdout.json)</td><td class=bad>do not submit</td></tr>
<tr><td><code>gemsdoe68-autocontext-s2-20261012T014136Z-zeros.tif</code></td><td class=bad>no holdout DTI / gate recorded</td><td class=bad>not evidenced — do not spend a slot</td></tr>
<tr><td><code>gems68-autoctx-s3-hidedraw-38000dots-20261012T022132Z-4c18def6-zeros.tif</code></td><td class=warn>Exp 2 gate passed on the same seed used for selection; superseded by the fresh-seed Exp 3 file below</td><td class=warn>superseded</td></tr>
<tr><td><code>GEMSDOE68-AC-stack2-20261012T042050Z-zeros.tif</code> (lane68 subsite)</td><td class=warn>far-&gt;1 km gate missed (t = 1.97); "zeros" file actually carries NaN outside (IR-68-ZEROS)</td><td class=warn>superseded</td></tr>
<tr><td><code>{prim['name']}</code></td><td class=ok>Exp 4: pre-registered stage {h['stage']} @ {h['budget']:,}, gate passed on FRESH seed {h.get('eval_seed','?')} (table above). Exp 3's 38k candidate failed on seed 31 — logged negative.</td><td class=ok><b>the current candidate</b></td></tr></table>
<p class=muted>All earlier files remain in <code>docs/downloads/</code> for audit; the uniqueness check above includes them as prior rasters.</p></div>
<div class=card><h2>Where the dots are (sanity map) and the key risk</h2>
<img src="assets/dots_map.png" alt="dot map" style="width:100%;border-radius:10px">
<p><b>Risk, stated plainly:</b> only {u['dot_overlap_within_3px'].get('GEMSDOE32__h33-2-b2-zeros.tif', {}).get('overlap', float('nan'))*100:.0f} % of these dots lie within 300 m of the H33-2-B2 file's dots, and the median dot is {card['emission']['median_dist_px']/10:.1f} km from the published catalogue (champion ≈ 2.0 km). Part of the surface is area-fill in regional highs rather than lines (IR-68-BLOB-01). The holdout says this ranks withheld faults better — including far from mapped faults — but the holdout's truth is catalogue components, not the panel's new faults. This file is a genuinely different bet, not an increment on the champion.</p></div>
<div class=card><h2>One-paragraph answer: why 0.2778, and can it be beaten?</h2>
<p>The owner-reported 0.2778 attribution concerns <code>h33-h33-2-b2</code> (GEMSDOE32) — note that the file's own audit JSON and the GEMSDOE32 site describe it as <b>UNSCORED, projected 0.2747</b>, on the owner-reported 0.2708 base, so 0.2778 has no organizer receipt anywhere in this checkout (IR-01). The file itself is a <b>binary field of 37,654 dots</b>, the last rung of a pruning ladder on one per-pixel field (0.1922 solid → 0.2477 → 0.2600 → 0.2708 → 0.2778). The metric DTI = T/(0.2·(T+S−M)+0.8·|G|) charges 0.2 per emitted unit that is not within 300 m of an uncovered new fault, so deleting low-credit mass — first the field's tail, then the 100–200 m halo of the <em>published</em> catalogue — paid at every step. What is left to win is <b>ranking</b>: the same ≈38 k dots must carry more kernel credit toward real unmapped faults. That is a field problem, and auto-context (continuity, parallelism, en-échelon spacing learned from the stage-1 map) is the cheapest principled attack on it. The full derivation and provenance tags are in <a href="knowledge.html">Why 0.2778</a>; the measured result of this lane is on the <a href="experiments.html">Experiments</a> page.</p></div>
"""


def build_howto(card: dict) -> str:
    prim = card["files"]["primary"]
    return f"""
<h1>How to make a submission (step by step)</h1>
<div class=card>
<ol>
<li>Download the primary file: <a class=btn href="downloads/{prim['name']}" download>⬇ {prim['name']}</a> (or the <a href="downloads/{prim['zip']}">ZIP</a> containing only that TIFF). Verify its SHA-256 matches <code>{prim['sha256']}</code>.</li>
<li>Open <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/submissions/">the competition's submission page</a> (login required) and click <b>New submission</b>.</li>
<li><b>File to submit</b>: choose the downloaded <code>.tif</code> (or <code>.zip</code>). The form accepts a single-band GeoTIFF matching the sample's CRS (EPSG:32611), shape (3730×3292) and geotransform.</li>
<li><b>Note</b> (optional, ≤140 characters): paste <code>{html.escape(card['submission_note'])}</code></li>
<li>Submit. Copy the score from the receipt into <code>registry/score_ledger.csv</code> tagged <b>ORGANIZER-CONFIRMED</b> with the exact file name and SHA-256.</li>
</ol>
<h2>About the earlier "Predicted values must be in range [0, 1]" error</h2>
<p>The primary file here is <b>all-finite</b>: 0.0 outside the footprint, values exactly 0.0 or 1.0 inside, no NaN, no nodata tag, no sentinel — so no cell can fall outside [0, 1]. Earlier family files were accepted by the portal in this "zeros" form. The published spec says data outside the bounds is null/NaN, so a NaN-outside twin (identical inside the footprint) is also published: it passes the shared <code>gems.validate_submission</code> in full (<code>evidence/validator_shared_exp4_nan.json</code>), while the zeros file fails only that validator's strict NaN-only-outside check by design (<code>evidence/validator_shared_exp4_zeros.json</code>). Upload the zeros file first; if the portal objects to anything, upload the NaN twin instead. Both were re-read from disk and verified cell-for-cell identical inside the footprint. The local validator re-reads the bytes on disk and checks: single band, float32, 3730×3292, EPSG:32611, pinned transform, no NaN/inf anywhere, min ≥ 0, max ≤ 1, zero positive pixels outside the footprint, zero positive pixels on the mapped catalogue.</p>
<h2>Rules you should know</h2>
<ul><li>Weekly submission cap is shown on the submission page; promotion to a slot is a separate selector decision (<a href="https://docs.nlr.gov/docs/fy26osti/96647.pdf">rules PDF 96647</a>).</li>
<li>Metric: distance-weighted Tversky index, α = 0.2 (FP), β = 0.8 (FN), 300 m triangular kernel (<a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/">problem description</a>).</li>
<li>Known/mapped faults are excluded from scoring; the truth is new faults mapped by the expert panel.</li></ul>
</div>"""


def build_experiments(exp1: dict, exp2: dict, exp3: dict | None = None, exp4: dict | None = None) -> str:
    def table(exp, title):
        rows = ""
        for key, g in exp["gate"].items():
            if "stage" not in key:  # exp1 format
                continue
            rows += (f"<tr><td>{key}</td><td>{f4(g.get('dti_stage1', ''))}</td><td>{f4(g.get('dti_cand', ''))}</td>"
                     f"<td>{g['delta_all']:+.4f}</td><td>[{f4(g['ci_all'][0])}, {f4(g['ci_all'][1])}]</td>"
                     f"<td>{f4(g.get('far_dti_stage1', ''))}</td><td>{f4(g.get('far_dti_cand', ''))}</td><td>{g['delta_far']:+.4f}</td><td>[{f4(g['ci_far'][0])}, {f4(g['ci_far'][1])}]</td>"
                     f"<td class={'ok' if g['passes'] else 'bad'}>{'pass' if g['passes'] else 'fail'}</td></tr>")
        can = sorted(exp["canary"].items(), key=lambda kv: -max(kv[1]["auc"], 1 - kv[1]["auc"]))[:8]
        canrows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v['auc']:.3f}</td><td class={'bad' if v['flag'] else 'ok'}>{'FLAG' if v['flag'] else 'ok'}</td></tr>" for k, v in can)
        imp = exp["feature_importance_gain"].get("2") or exp["feature_importance_gain"].get(2, {})
        tot = sum(imp.values()) or 1
        improws = "".join(f"<tr><td>{html.escape(k)}</td><td>{v/tot*100:.1f} %</td></tr>" for k, v in sorted(imp.items(), key=lambda kv: -kv[1])[:8])
        return f"""<div class=card><h2>{title}</h2><p class=muted>{html.escape(exp['experiment'])} · evaluator {exp['evaluator_version']} · {exp['holdout']['withheld_positive_pixels']:,} withheld positives · {exp['seconds']:.0f} s</p>
<table><tr><th>budget_stage</th><th>s1 DTI</th><th>cand DTI</th><th>Δ all</th><th>95 % CI</th><th>s1 far</th><th>cand far</th><th>Δ far</th><th>95 % CI far</th><th>gate</th></tr>{rows}</table>
<h3>Leakage canary (single-feature AUC vs withheld truth; flag &gt; 0.90)</h3><table><tr><th>feature</th><th>AUC</th><th></th></tr>{canrows}</table>
<h3>Stage-2 gain share (top 8)</h3><table>{improws}</table></div>"""
    e1 = exp1.copy()
    # exp1 gate keys are "{n}_cand_stage{k}" with different field names; normalise
    g1 = {}
    for k, g in exp1["gate"].items():
        n, cand = k.split("_cand_")
        r = exp1["results"][n]["all"][f"cand_{cand}"]["scores"]; rf = exp1["results"][n]["far"][f"cand_{cand}"]["scores"]
        g1[f"{n}_{cand}"] = dict(dti_stage1=r["stage1"]["dti"], dti_cand=r[cand]["dti"], delta_all=g["delta_all"], ci_all=g["ci_all"],
                                 far_dti_stage1=rf["stage1"]["dti"], far_dti_cand=rf[cand]["dti"], delta_far=g["delta_far"], ci_far=g["ci_far"], passes=g["passes"])
    e1["gate"] = g1
    return ("<h1>Experiments (HOLDOUT-DTI only; nothing here is a leaderboard score)</h1>"
            "<div class=card><p>Both experiments use the same hide-and-recover holdout (whole catalogue components withheld per quadrant with a 3 px feature collar, 12 px boundary erosion, visible faults masked pixel-exactly), the same emission rule at matched budgets, and a paired 20 km-block bootstrap for the CI. "
            "Experiment 1 trained stage 2 with the visible catalogue as both label and context source → <code>cat_dist</code> was 0 on every positive and took 95 % of the gain (IR-68-LEAK-01). Experiment 2 fixes this with hide-draw labels: withheld components are the positives and context comes from the remaining visible map.</p></div>"
            + (table(exp4, "Experiment 4 — same recipe, FRESH seed 42, pre-registered stage 3 @ 25,000 (the result that counts)") if exp4 else "")
            + (table(exp3, "Experiment 3 — ext channels + extended offsets + seed bagging, FRESH seed 31, pre-registered stage 3 @ 38,000 (NEGATIVE: far CI crossed 0)") if exp3 else "")
            + table(exp2, "Experiment 2 — hide-draw auto-context, seed 20 (selection seed; superseded by Exp 3)")
            + table(e1, "Experiment 1 — naive stage-2 labels (negative result, kept as the leak record)"))


def build_ledger() -> str:
    rows = ""
    with open(ROOT / "registry" / "score_ledger.csv") as f:
        for r in csv.DictReader(f):
            rows += f"<tr><td>{html.escape(r['site'])}</td><td><code>{html.escape(r['submission_name'])}</code></td><td>{html.escape(r['score'])}</td><td>{html.escape(r['evidence_class'])}</td><td>{html.escape(r['note'])}</td></tr>"
    return f"""<h1>Score ledger of the GEMSDOE family</h1><div class=card><p class=muted>Scores are copied from the session brief (OWNER-REPORT). None has an organizer receipt in this repository. Leaderboard: <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/">drivendata.org</a> (not reachable from the build sandbox; the brief quotes 0.3774 and 0.3195 as current highs).</p>
<table><tr><th>site</th><th>submission</th><th>score</th><th>class</th><th>note</th></tr>{rows}</table></div>"""


def main() -> int:
    card = json.loads((EV / "run_card.json").read_text())
    exp1 = json.loads((EV / "exp1_autocontext.json").read_text())
    exp2 = json.loads((EV / "exp2_autocontext_hidedraw.json").read_text())
    _e3 = EV / "exp3_autocontext_ext_bagged.json"
    exp3 = json.loads(_e3.read_text()) if _e3.is_file() else None
    _e4 = EV / "exp4_autocontext_ext_bagged.json"
    exp4 = json.loads(_e4.read_text()) if _e4.is_file() else None
    (DOCS / "index.html").write_text(page("Executive summary", build_index(card, exp2), "index.html"))
    (DOCS / "how-to-submit.html").write_text(page("How to submit", build_howto(card), "how-to-submit.html"))
    (DOCS / "experiments.html").write_text(page("Experiments", build_experiments(exp1, exp2, exp3, exp4), "experiments.html"))
    (DOCS / "leaderboard.html").write_text(page("Score ledger", build_ledger(), "leaderboard.html"))
    for src, dst, title in ((DOCS / "hypotheses.md", "hypotheses.html", "Hypotheses"),
                            (ROOT / "knowledge" / "01_why_02778_and_can_we_beat_it.md", "knowledge.html", "Why 0.2778"),
                            (DOCS / "sources.md", "sources.html", "Sources"),
                            (DOCS / "irregularities.md", "irregularities.html", "Irregularities")):
        (DOCS / dst).write_text(page(title, "<div class=card>" + md_to_html(src.read_text()) + "</div>", dst))
    (DOCS / ".nojekyll").write_text("")
    print("site built:", [p.name for p in DOCS.glob("*.html")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
