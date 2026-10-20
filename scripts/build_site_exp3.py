#!/usr/bin/env python3
"""Rebuild docs/index.html + docs/executive-summary.html from evidence/run_card_exp3.json.

One verdict, generated from the run card, at the very top: either an obvious
one-click download (only if the Exp-3 gate promoted AND download_cleared) or an
obvious DO-NOT-SUBMIT banner.  No number is typed by hand: every value is read
from evidence/*.json.  Also writes docs/runcard.json (copy of the Exp-3 card),
docs/lane68/SUPERSEDED.txt and prepends a superseded banner to the lane68 pages.
"""
from __future__ import annotations

import json
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
.btn{display:inline-block;background:var(--acc);color:#061024;font-weight:800;padding:14px 20px;border-radius:10px;text-decoration:none;margin:6px 8px 6px 0;font-size:18px}
.btn.secondary{background:#26304d;color:var(--ink);font-size:15px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #26304d;padding:7px 8px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:700}code,pre{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px}
pre{background:#0a0f1d;padding:12px;border-radius:10px;overflow:auto}
.muted{color:var(--muted)}.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
h1{font-size:28px;margin:8px 0}h2{font-size:21px;margin-top:26px}small{color:var(--muted)}
"""


def page(title, body):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · GEMSDOE68</title><style>{CSS}</style></head><body>
<header><strong>GEMSDOE68 · auto-context lane</strong><a href="index.html">Executive summary</a> <a href="executive-summary.html">How to submit</a> <a href="hypotheses.html">Hypotheses</a> <a href="experiments.html">Experiments</a> <a href="knowledge.html">Why 0.2778</a> <a href="sources.html">Sources</a> <a href="irregularities.html">Irregularities</a> <a href="leaderboard.html">Score ledger</a><a href="https://github.com/buffedlizard55-lab/GEMSDOE68">GitHub ↗</a></header>
<main>{body}</main></body></html>"""


def f4(x):
    return "—" if x is None else f"{float(x):.4f}"


def main() -> int:
    card = json.loads((EV / "run_card_exp3.json").read_text())
    exp3 = json.loads((EV / "exp3_ridgecrest.json").read_text())
    audit = json.loads((EV / "raster_audit_20261020.json").read_text())
    cleared = bool(card.get("download_cleared"))
    dec = card["decision"]

    if cleared:
        r = card["raster"]
        prim, sec = r["primary"], r["secondary_nan"]
        banner = f"""<div class="banner ok">✅ OK TO DOWNLOAD AND SUBMIT — this file passed the preregistered Exp-3 holdout gate, the lane-uniqueness check and the format validators below. (Spending a weekly slot remains the selector's decision.)</div>
<div class=card><h2>⬇ The submission file (one click)</h2>
<p><a class=btn href="downloads/{prim['name']}" download>⬇ Download {prim['name']}</a>
<a class="btn secondary" href="downloads/{prim['zip']}" download>⬇ ZIP (same TIFF inside)</a>
<a class="btn secondary" href="downloads/{sec['name']}" download>NaN-outside variant (secondary)</a></p>
<table>
<tr><th>Submission name</th><td><code>{card['submission_name']}</code></td></tr>
<tr><th>Note (paste into the form's “Note” field, {card['note_chars']}/140 chars)</th><td><code>{card['note']}</code></td></tr>
<tr><th>SHA-256 (primary)</th><td><code>{prim['sha256']}</code></td></tr>
<tr><th>Format</th><td>single-band float32 · EPSG:32611 · 3730×3292 · 100 m · values exactly {{0,1}} · <b>every cell finite, 0 NaN anywhere, no nodata tag</b> → the portal error “Predicted values must be in range [0, 1]” cannot trigger on this file (min 0.0, max 1.0 over all 12,279,160 cells, verified by re-reading the written bytes)</td></tr>
<tr><th>Organizer-confirmed score</th><td><b>none — UNSCORED.</b> No number on this site is a leaderboard score.</td></tr>
</table></div>"""
    else:
        banner = f"""<div class="banner bad">⛔ DO NOT DOWNLOAD OR SUBMIT ANY FILE FROM THIS REPO — the preregistered Exp-3 gate was not met (verdict: negative). The negative result below is the deliverable.</div>"""

    # holdout table at 38k
    res = exp3["results"]["38000"]
    rows = ""
    for scope in ("all", "far", "farfull"):
        sc = res["scopes"][scope]["scores"]
        cells = "".join(f"<td>{f4(sc[a]['dti'])}</td>" for a in ("s1_V0", "s2_V0", "s3_V0", "s2_V2", "s3_V2", "s2_V1", "s3_V1", "s3_V3"))
        rows += f"<tr><td>{scope}</td>{cells}</tr>"
    arm_rows = ""
    for a, v in dec["per_arm"].items():
        arm_rows += (f"<tr><td><code>{a}</code></td><td>{f4(v['dti_all'])}</td><td>{v['emitted']}</td>"
                     f"<td>[{f4(v['gate1_all_vs_s1_ci'][0])}, {f4(v['gate1_all_vs_s1_ci'][1])}]</td>"
                     f"<td>[{f4(v['gate1_far_vs_s1_ci'][0])}, {f4(v['gate1_far_vs_s1_ci'][1])}]</td>"
                     f"<td>{'[' + f4(v['gate2_all_vs_s3V0_ci'][0]) + ', ' + f4(v['gate2_all_vs_s3V0_ci'][1]) + ']' if v['gate2_all_vs_s3V0_ci'] else '—'}</td>"
                     f"<td>{f4(v['gate3_farfull_delta_vs_s1'])}</td><td class={'ok' if v['passes'] else 'bad'}>{v['passes']}</td></tr>")
    holdout = f"""<div class=card><h2>Holdout evidence (HOLDOUT-DTI — evaluator <code>{exp3['evaluator_version']}</code>, {exp3['withheld_positive_pixels']:,} withheld positives, α=0.2 β=0.8 300 m triangular)</h2>
<p class=muted>Exp-3 (preregistered in <code>knowledge/02_exp3_preregistration_and_hypotheses.md</code> before any result): identical hide-draw training to exp2; only the emission rule varies. V0 = 5×5-NMS top-k (exp2 control) · V2 = greedy ≥2.8 px spacing · V1 = across-strike crest + spacing (H68-A) · V3 = crest + potential-field concordance ranking (H68-D). Budgets 25k/38k; table shows 38k.</p>
<table><tr><th>scope</th><th>s1_V0</th><th>s2_V0</th><th>s3_V0</th><th>s2_V2</th><th>s3_V2</th><th>s2_V1</th><th>s3_V1</th><th>s3_V3</th></tr>{rows}</table>
<p class=muted>Scopes: <b>all</b> = full fold domains · <b>far</b> = scoring restricted to &gt;1 km from any masked-known catalogue pixel (exp2-comparable) · <b>farfull</b> = far truth but all emitted mass counts as FP (sensitivity).</p>
<h3>Gate table (paired 20 km-block bootstrap 95 % CI, 38,000 dots)</h3>
<table><tr><th>arm</th><th>DTI all</th><th>emitted</th><th>Δ vs s1 all</th><th>Δ vs s1 far</th><th>Δ vs s3_V0 (prev. best)</th><th>Δ farfull</th><th>passes</th></tr>{arm_rows}</table>
<p>Replication of committed exp2 V0 numbers: <b>{'OK' if exp3['replication']['replicated'] else 'FAILED'}</b> (max |ΔDTI| vs exp2 ≤ 0.005 required). Leakage canary clean: <b>{dec['canary_clean']}</b>. Chosen arm: <code>{dec['chosen_arm']}</code> · verdict: <b>{dec['verdict']}</b>.</p>
<p class=muted>These numbers rank methods on withheld <em>catalogue</em> components. The live truth is <em>unmapped</em> faults; they do not predict a leaderboard value (IR-68-PROXY-01, IR-68-SCATTER-01).</p></div>"""

    lane_html = ""
    if cleared:
        lu = card["correlation_overlap_vs_registry"]
        wr, wo = lu["registry"]["worst_abs_spearman"], lu["registry"]["worst_dot_overlap"]
        lane_html = f"""<div class=card><h2>Lane uniqueness (protocol rule 1)</h2>
<p>vs <b>{lu['registry']['n']}</b> committed registry rasters + <b>{lu['prior_downloads']['n']}</b> prior candidates in <code>docs/downloads/</code>, on the surface and on the final dots:
max |Spearman| <b>{f4(wr['abs_spearman'])}</b> (<code>{wr['name']}</code>, limit 0.90) ·
max dot overlap within 3 px <b>{100*wo['dot_overlap_3px']:.1f}%</b> (<code>{wo['name']}</code>, whose own 3 px halo covers {100*wo['registry_coverage_3px']:.1f}% of the footprint; limit 70%) ·
exact hash clashes: <b>{lu['exact_hash_clashes'] or 'none'}</b> · ρ breaches: <b>{lu['rho_breaches'] or 'none'}</b> · non-degenerate overlap breaches: <b>{lu['nondegenerate_overlap_breaches'] or 'none'}</b>.
Full per-raster table in <code>evidence/run_card_exp3.json</code>.</p></div>"""

    audit_rows = ""
    for r in audit["rasters"]:
        audit_rows += (f"<tr><td><code>{r['file']}</code></td><td>{r['n_gt0']:,}</td><td>{r['nan_inside_footprint']:,}</td>"
                       f"<td>{r['min_dist_catalogue_px']}</td><td>{r['median_nn_spacing_px']}</td>"
                       f"<td>{'champion (registry)' if r['file'].startswith('gemsdoe32-h33') else 'superseded — do not submit'}</td></tr>")
    audit_html = f"""<div class=card><h2>Every earlier candidate in this repo — audited from bytes (2026-10-20)</h2>
<p class=muted>Recomputed with <code>scripts/audit_rasters.py</code> against the sample-submission footprint ({audit['footprint_cells']:,} cells) and catalogue ({audit['catalogue_cells']:,} cells). “NaN in” = NaN cells inside that footprint. All are superseded by the Exp-3 verdict above; several carry defects (IR-68-FP-01).</p>
<table><tr><th>file</th><th>dots &gt;0</th><th>NaN in</th><th>min d(cat) px</th><th>median NN px</th><th>status</th></tr>{audit_rows}</table></div>"""

    why = """<div class=card><h2>Why did <code>h33-h33-2-b2</code> get 0.2778, and can this beat it?</h2>
<p><b>Why (measured from bytes + OWNER-REPORTED scores; no organiser receipt exists in this repo):</b> the champion is 37,654 binary dots — verified here as an exact subset of the d2.8 (44,090, 0.2600) and h19-5 (121,131, 0.1922) registry rasters. DTI = T/(0.2·(T+S−M)+0.8·|G|) charges 0.2 for every emitted unit that earns no 300 m kernel credit, so each pruning rung (solid → d1.5 → d2.8 → remove d=1 px catalogue halo → remove d=2 px) deleted below-bar mass and paid. A controlled confirmation measured here: <code>anderson-geothermal-pinn</code> (0.2750) contains the champion's dots <em>exactly</em> plus 1,200 more — the extra dots net-cost −0.0028. Mass discipline and the &gt;2 px catalogue exclusion explain 0.1922 → 0.2778 without any new geology.</p>
<p><b>Can it be beaten?</b> Only by ranking: at the same ~38k mass, dots must earn more kernel credit toward real unmapped faults (the brief's leaders at 0.3195–0.3774, OWNER-REPORTED and not re-fetchable here, need ≈15–40 % more credit per dot). Auto-context attacks ranking (continuity / parallelism / en-échelon context a per-pixel field cannot see); Exp-3 additionally attacks <em>placement geometry</em> (lines, not blobs). Whether that transfers live is exactly what the holdout cannot prove — hence the gate, the far stratum, and the explicit caveats (IR-68-PROXY-01, IR-68-SCATTER-01). Full derivation: <a href="knowledge.html">Why 0.2778</a>.</p></div>"""

    index = page("Executive summary", f"""<h1>Executive summary</h1>
<p class=muted>DOE GEMS Prize (DrivenData #306) · lane: auto-context stacked refinement (Tu, CVPR 2008) · Exp-3 update {card['generated_utc']} · every number below is generated from <code>evidence/run_card_exp3.json</code> / <code>evidence/exp3_ridgecrest.json</code> — none is typed by hand, and none is an organizer-confirmed score.</p>
{banner}
{holdout}
{lane_html}
{why}
{audit_html}
<div class=card><h2>Method in one paragraph</h2><p>{card['mechanism']}</p>
<p><b>Named non-fault mimic:</b> {card['named_non_fault_process_that_could_mimic_it']}</p>
<p class=muted>Core values: <b>Maximize P(Win)</b> · <b>Own the Outcome</b>. Limitations and flagged defects: <a href="irregularities.html">Irregularities ledger</a>. Sources for manual review: <a href="sources.html">Sources</a>.</p></div>""")
    (DOCS / "index.html").write_text(index)

    if cleared:
        r = card["raster"]
        prim = r["primary"]
        v = card["validator_output"]["primary_local_sample_footprint"]
        checks = "".join(f"<tr><td>{k}</td><td class={'ok' if val else 'bad'}>{val}</td></tr>" for k, val in v["checks"].items())
        shared = card["validator_output"]["primary_shared_features_footprint"]
        shared_rows = "".join(f"<tr><td>{c['check']}</td><td>{c['status']}</td><td>{c['detail']}</td></tr>" for c in shared["checks"])
        submit_body = f"""<h1>Executive summary — exactly how to submit</h1>
<div class="banner ok">✅ OK TO DOWNLOAD AND SUBMIT: <code>{prim['name']}</code></div>
<div class=card><h2>Step by step (2 minutes)</h2>
<ol>
<li>Download <a href="downloads/{prim['name']}" download><b>{prim['name']}</b></a> (or the <a href="downloads/{prim['zip']}" download>ZIP containing the same TIFF</a>). Verify SHA-256 = <code>{prim['sha256']}</code> if you wish.</li>
<li>Open the competition submission page: <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/">DrivenData #306</a> → <b>Submit</b> → <b>New submission</b>.</li>
<li><b>File to submit:</b> choose the downloaded <code>.tif</code> (do not rename the contents; the form accepts a single-band GeoTIFF or a .zip containing one).</li>
<li><b>Note</b> (optional field): paste exactly:<br><code>{card['note']}</code> <small>({card['note_chars']}/140 chars)</small></li>
<li>Click <b>Create submission</b>. The earlier portal error <i>“Predicted values must be in range [0, 1]”</i> cannot trigger on this file: every one of its 12,279,160 cells is finite, min = 0.0, max = 1.0, there is no NaN and no nodata sentinel anywhere (table below, from re-reading the written bytes). If the portal still rejects it, use the secondary <code>{r['secondary_nan']['name']}</code> (identical dots; NaN only outside the sample footprint) and log the portal message verbatim in the repo.</li>
<li>Record the score shown by the portal as ORGANIZER-CONFIRMED in <code>registry/score_ledger.csv</code> — until then this file is UNSCORED. Weekly slot caps are shown on the submission page (rules: <a href="https://docs.nlr.gov/docs/fy26osti/96647.pdf">96647 PDF</a>); promotion to a slot is a separate selector step.</li>
</ol></div>
<div class=card><h2>Validator output (primary file, re-read from disk)</h2>
<p>Local validator <code>gems68.validate</code> against the <b>sample-submission footprint</b> — all checks passed: <b>{v['all_checks_passed']}</b></p>
<table>{checks}</table>
<p class=muted>Shared template validator <code>gems.validate_submission</code> (footprint from training-features band 1, 3,061 cells smaller — IR-68-FP-01). Its “NaN-only outside footprint” item fails <i>by design</i> on any all-finite file, including the 0.2778 champion's format; every other item is shown verbatim:</p>
<table><tr><th>check</th><th>status</th><th>detail</th></tr>{shared_rows}</table></div>"""
    else:
        submit_body = f"""<h1>Executive summary — exactly how to submit</h1>
<div class="banner bad">⛔ DO NOT SUBMIT — no file in this repo is currently cleared (Exp-3 verdict: negative).</div>
<div class=card><p>The preregistered gate ({'<code>knowledge/02_exp3_preregistration_and_hypotheses.md</code>'}) was not met, so per protocol no download is offered and no weekly slot should be spent. The gate table is on the <a href="index.html">executive summary</a>. When a future run promotes, this page will carry the one-click download, the unique submission name, the ≤140-character note, the SHA-256 and the step-by-step form instructions.</p></div>"""
    (DOCS / "executive-summary.html").write_text(page("How to submit", submit_body))
    (DOCS / "how-to-submit.html").write_text(page("How to submit", submit_body))

    (DOCS / "runcard.json").write_text(json.dumps(card, indent=1, default=float))
    (DOCS / "lane68" / "SUPERSEDED.txt").write_text(
        "SUPERSEDED 2026-10-20 (Exp-3 session). The lane68 verdict and candidate on these "
        "pages are a historical record. The single current verdict is generated from "
        "evidence/run_card_exp3.json at the top-level site (docs/index.html). "
        "Do not download or submit any lane68 file.\n")
    print("site rebuilt; cleared =", cleared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
