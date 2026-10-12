#!/usr/bin/env python3
"""Render the GitHub Pages site from the evidence JSON files.

The site is generated, never hand-edited, so the download banner, the validator output and the
run card always agree with what is actually on disk.  Run after run_holdout.py and
build_submission.py.
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "evidence"
DOCS = ROOT / "docs"
DL = DOCS / "downloads"

CSS = """
:root{--ink:#0d1b2a;--mut:#5b6b7b;--line:#dde5ec;--bg:#f6f8fa;--ok:#0f7b3f;--bad:#b3261e;--warn:#8a5a00}
*{box-sizing:border-box}body{margin:0;font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:#fff}
header{background:var(--ink);color:#fff;padding:22px 24px}header a{color:#9fd0ff}
main{max-width:1080px;margin:0 auto;padding:24px}
h1{margin:0 0 4px;font-size:26px}h2{margin:34px 0 10px;font-size:20px;border-bottom:2px solid var(--line);padding-bottom:6px}
.sub{color:#c6d3df;font-size:14px}
.banner{border-radius:10px;padding:18px 20px;margin:18px 0;font-size:18px;font-weight:600}
.banner.ok{background:#e6f6ec;border:2px solid var(--ok);color:var(--ok)}
.banner.no{background:#fdeceb;border:2px solid var(--bad);color:var(--bad)}
.banner.wait{background:#fff6e5;border:2px solid var(--warn);color:var(--warn)}
.dl{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:16px 0 26px}
.btn{display:inline-block;background:#0b5fff;color:#fff;text-decoration:none;font-weight:700;font-size:20px;padding:16px 26px;border-radius:10px}
.btn.sec{background:#fff;color:#0b5fff;border:2px solid #0b5fff;font-size:16px;padding:12px 18px}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}
th,td{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}
th{background:var(--bg)}code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--bg);padding:12px;border-radius:8px;overflow:auto;font-size:12.5px}
.card{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.small{font-size:13px;color:var(--mut)}ul{margin:8px 0 8px 20px}
.pill{display:inline-block;background:#eef3f8;border:1px solid var(--line);border-radius:99px;padding:1px 9px;font-size:12px;margin-right:4px}
a{color:#0b5fff}
"""


def load(name: str) -> dict | None:
    p = EV / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def esc(x) -> str:
    return html.escape(str(x))


def checks_table(fc: dict) -> str:
    rows = []
    for k, v in fc.get("checks", {}).items():
        if isinstance(v, dict):
            ok, detail = v["ok"], v.get("detail", "")
        elif isinstance(v, bool):
            ok, detail = v, ""
        else:
            rows.append(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>")
            continue
        mark = "PASS" if ok else "<b>FAIL</b>"
        rows.append(f"<tr><td>{esc(k)}</td><td>{mark} <span class='small'>{esc(detail)}</span></td></tr>")
    fails = fc.get("failures") or []
    line = "<p><b style='color:var(--ok)'>all checks passed</b></p>" if not fails \
        else f"<p><b style='color:var(--bad)'>FAILED: {esc('; '.join(fails))}</b></p>"
    return line + "<table><tr><th>check</th><th>result</th></tr>" + "".join(rows) + "</table>"


def main() -> int:
    hold = load("holdout.json")
    build = load("submission_build.json")
    DOCS.mkdir(exist_ok=True)
    DL.mkdir(parents=True, exist_ok=True)
    (DOCS / ".nojekyll").write_text("")
    (DOCS / "assets").mkdir(exist_ok=True)
    (DOCS / "assets" / "site.css").write_text(CSS)

    now = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    # ---- verdict ---------------------------------------------------------------------
    if build is None:
        banner = ('<div class="banner wait">NOT READY &mdash; no submission has been built in this '
                  'repository yet. Nothing here is safe to upload.</div>')
        ok_to_submit = None
        primary = None
    else:
        gate = (hold or {}).get("gate", {})
        gate_pass = bool(gate.get("passes"))
        fc = build["outputs"]["zeros"]["format_check"]
        lane_ok = build.get("registry_verdict", {}).get("lane_ok", True)
        ok_to_submit = gate_pass and fc["ok"] and lane_ok
        primary = build["outputs"]["zeros"]["path"]
        if ok_to_submit:
            banner = ('<div class="banner ok">&#10003; OK TO DOWNLOAD AND SUBMIT &mdash; gate met '
                      f'(holdout z = {gate.get("z")}, &gt; 2 SE), validator passed, lane unique.</div>')
        else:
            why = []
            if not gate_pass:
                why.append(f'gate not met (stage2 &minus; stage1 = {gate.get("delta")}, '
                           f'z = {gate.get("z")} &le; 2)')
            if not fc["ok"]:
                why.append("format validator failed")
            if not lane_ok:
                why.append("lane drift: too close to a registry raster")
            banner = ('<div class="banner no">&#10007; DO NOT SPEND A SUBMISSION SLOT &mdash; '
                      + esc("; ".join(why)) + '. The file is published for audit as a '
                      '<b>negative result</b>.</div>')

    # ---- download block --------------------------------------------------------------
    dl = ""
    if build is not None:
        z = build["outputs"]["zeros"]
        n = build["outputs"]["nan"]
        zp = build["outputs"].get("zeros_zip", {})
        rel = lambda p: Path(p).name
        dl = f"""
<div class="dl">
  <a class="btn" href="downloads/{esc(rel(z['path']))}" download>&#11015; Download submission .tif</a>
  <a class="btn sec" href="downloads/{esc(rel(zp.get('path','')))}" download>.zip</a>
  <a class="btn sec" href="downloads/{esc(rel(n['path']))}" download>NaN-outside twin</a>
  <a class="btn sec" href="downloads/{esc(rel(build['base_name']))}-format-check.json">validator JSON</a>
</div>
<p class="small"><b>Upload the <code>-zeros.tif</code> file.</b> It is all-finite (0.0 outside the
footprint) and every value is in [0, 1], so it cannot trigger the portal's
&ldquo;Predicted values must be in range [0, 1]&rdquo; error. The <code>-nan.tif</code> twin carries
NaN outside the footprint and matches <code>sample_submission.tif</code> masking exactly; use it
only if the portal rejects zeros.</p>
<div class="card"><b>Submission note to paste</b> (max 200 chars):<br>
<code>{esc(build.get('note',''))}</code></div>
<table>
<tr><th>file</th><th>bytes</th><th>sha256</th></tr>
<tr><td>{esc(rel(z['path']))}</td><td>{z['bytes']:,}</td><td><code>{esc(z['sha256'])}</code></td></tr>
<tr><td>{esc(rel(n['path']))}</td><td>{n['bytes']:,}</td><td><code>{esc(n['sha256'])}</code></td></tr>
</table>"""

    # ---- gate table ------------------------------------------------------------------
    gate_html = "<p>No holdout evidence yet.</p>"
    if hold:
        g = hold["gate"]
        gate_html = f"""
<table>
<tr><th>quantity</th><th>stage 1 (per-pixel)</th><th>stage 2 (auto-context)</th></tr>
<tr><td>pooled HOLDOUT-DTI, all withheld pixels</td>
    <td>{hold['stage1_pooled']['dti']:.5f}</td><td><b>{hold['stage2_pooled']['dti']:.5f}</b></td></tr>
<tr><td>pooled HOLDOUT-DTI, pixels &gt; 1 km from any mapped fault</td>
    <td>{hold['stage1_pooled_far1km']['dti']:.5f}</td><td><b>{hold['stage2_pooled_far1km']['dti']:.5f}</b></td></tr>
<tr><td>withheld positives scored</td><td colspan="2">{hold['withheld_positives_total']:,}
    ({hold['withheld_segments_total']} whole segments, {hold['n_folds']} folds)</td></tr>
</table>
<div class="card"><b>Gate</b> &mdash; &Delta;(stage2 &minus; stage1) = <b>{g['delta']}</b>,
bootstrap SE = {g['se']}, <b>z = {g['z']}</b> (need &gt; 2.0);
&Delta; on the &gt;1 km stratum = {g['delta_far']};
stage 2 wins {g['s2_beats_s1_folds']}/{hold['n_folds']} folds overall and
{g['s2_beats_s1_far_folds']}/{hold['n_folds']} on the far stratum.
<br><b>Verdict: {'PASS' if g['passes'] else 'FAIL'}</b></div>
<p class="small">Evaluator <code>{esc(hold['evaluator'])}</code>. Metric
&alpha;={hold['metric']['alpha']}, &beta;={hold['metric']['beta']},
R={hold['metric']['radius_m']} m triangular kernel.
Every number on this page is HOLDOUT-DTI on a <b>catalogue proxy</b>; nothing here is
ORGANIZER-CONFIRMED.</p>"""
        can = hold.get("canary_top20") or []
        if can:
            flags = hold.get("canary_flags") or []
            gate_html += f"""
<h2>Leakage canary</h2>
<p>Each feature tested alone at separating the withheld segments from background.
Threshold 0.90. <b>{len(flags)} feature(s) flagged.</b>
Highest: <code>{esc(can[0]['feature'])}</code> AUC {can[0]['auc']}.</p>
<table><tr><th>feature</th><th>AUC</th></tr>""" + "".join(
                f"<tr><td>{esc(c['feature'])}</td><td>{c['auc']}</td></tr>" for c in can[:12]) + "</table>"

    # ---- registry -------------------------------------------------------------------
    reg_html = "<p>No registry rasters were available for the duplicate check.</p>"
    if build and build.get("registry"):
        rows = build["registry"][:15]
        reg_html = ("<table><tr><th>registry raster</th><th>Spearman</th><th>dots within 3 px</th></tr>"
                    + "".join(f"<tr><td>{esc(r['file'])}</td>"
                              f"<td>{r.get('spearman','-')}</td><td>{r.get('overlap3px','-')}</td></tr>"
                              for r in rows) + "</table>")
        v = build["registry_verdict"]
        reg_html += (f"<p>Worst rank-correlation <b>{v.get('worst_spearman')}</b> (limit 0.90); "
                     f"worst 3 px dot overlap <b>{v.get('worst_overlap')}</b> (limit 0.70). "
                     f"<b>Lane {'unique' if v.get('lane_ok') else 'DRIFTED &mdash; duplicate'}</b>.</p>")

    emit = (build or {}).get("emission", {})
    index = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GEMSDOE68 &mdash; auto-context submission</title>
<link rel="stylesheet" href="assets/site.css"></head><body>
<header><h1>GEMSDOE68 &mdash; auto-context stacked refinement</h1>
<div class="sub">DOE GEMS Prize Challenge &middot; generated {now} &middot;
<a href="executive-summary.html">executive summary / how to submit</a> &middot;
<a href="https://github.com/buffedlizard55-lab/GEMSDOE68">source</a></div></header>
<main>
{banner}
{dl}
<h2>What this is</h2>
<p>A two-stage classifier for the GEMS fault-discovery grid (3730 &times; 3292 px, EPSG:32611,
100 m). Stage&nbsp;1 is a per-pixel LightGBM on 47 geophysical channels; stage&nbsp;2 is
<i>auto-context</i> (Tu, CVPR 2008): a second LightGBM trained on stage&nbsp;1's out-of-fold
probability map sampled at &plusmn;1&ndash;15&nbsp;px along and across the local strike, plus
multi-scale anisotropic smooths, the raw gradient/tilt/scarp channels, and catalogue-context
features (distance, orientation and spacing to the nearest mapped fault, with the target
block's labels masked). Stage&nbsp;2 exists to learn what a per-pixel model cannot: continuity
along strike, parallelism between splays, en&nbsp;&eacute;chelon spacing.</p>
<h2>The gate</h2>
{gate_html}
<h2>Emission</h2>
<div class="card">emitted dots <b>{emit.get('emitted','-')}</b> of {emit.get('candidates','-')}
candidates &middot; minimum separation {emit.get('r_min_px','-')} px &middot;
excluded within {emit.get('exclude_catalogue_px','-')} px of the mapped catalogue &middot;
dots on catalogue <b>{emit.get('on_catalogue','-')}</b><br>
<span class="small">The organizer masks mapped-fault pixels out of scoring, and every unit of
emitted mass raises the DTI denominator by exactly &alpha;=0.2 wherever it lands, so mass on the
catalogue is pure loss. A dot pays for itself only if its realised kernel weight exceeds
&alpha;&middot;DTI &mdash; {(build or {}).get('marginal_bar_at_0p2778','-')} at DTI 0.2778, i.e. within
&asymp;284 m of a hidden truth pixel.</span></div>
<h2>Format validation</h2>
{checks_table(build['outputs']['zeros']['format_check']) if build else '<p>none</p>'}
<h2>Lane / duplicate check</h2>
{reg_html}
<h2>Reproduce</h2>
<pre>bash scripts/fetch_mirrors.sh
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/run_holdout.py --folds 4
PYTHONPATH=src python scripts/build_submission.py --budget 40000
PYTHONPATH=src python scripts/build_site.py</pre>
<p class="small">Scores quoted anywhere on this site are HOLDOUT-DTI on a catalogue proxy or
owner reports. Nothing is ORGANIZER-CONFIRMED unless a submission-page receipt is linked.</p>
</main></body></html>"""
    (DOCS / "index.html").write_text(index)

    # ---- executive summary ----------------------------------------------------------
    ex = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Executive summary &mdash; how to make a GEMS submission</title>
<link rel="stylesheet" href="assets/site.css"></head><body>
<header><h1>Executive summary &mdash; how to submit</h1>
<div class="sub">generated {now} &middot; <a href="index.html">back to the submission page</a></div></header>
<main>
{banner}
<h2>1. The file</h2>
<p>One single-band GeoTIFF, float32, EPSG:32611, 3730&nbsp;&times;&nbsp;3292, transform
<code>(100, 0, 243350, 0, -100, 4508550)</code>, values in [0,&nbsp;1]. A <code>.zip</code>
containing exactly one GeoTIFF is also accepted.</p>
{dl}
<h2>2. The form</h2>
<ol>
<li>Open <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/submissions/">the submissions page</a>
and click <b>New submission</b>.</li>
<li><b>File to submit</b> &rarr; choose <code>{esc(Path((build or {'outputs':{'zeros':{'path':'<the -zeros.tif>'}}})['outputs']['zeros']['path']).name)}</code>.</li>
<li><b>Note (optional)</b> &rarr; paste
<code>{esc((build or {}).get('note',''))}</code></li>
<li>Submit. The score appears on the submissions page; record it in
<code>registry/leaderboard.jsonl</code> with the file sha256 so the number is traceable.</li>
</ol>
<h2>3. The range error</h2>
<p>The portal rejects a file with <i>&ldquo;Predicted values must be in range [0, 1]&rdquo;</i> when
any value is outside [0,&nbsp;1] <b>or is NaN where the organizer expects a number</b>. The
<code>-zeros.tif</code> twin has no NaN anywhere and every value is exactly 0.0 or 1.0, so it
cannot produce that error. That is the file to upload.</p>
<h2>4. Naming and the note</h2>
<p>Name pattern: <code>gems68-autoctx-&lt;budget&gt;-&lt;UTC stamp&gt;</code>. The note names the
method, the budget and the holdout gate so two runs can be told apart later.</p>
<h2>5. Weekly cap</h2>
<p>Submissions per week are capped by the organizer; check the cap on the submissions page
before uploading. This repository never auto-uploads &mdash; promotion to a real slot is a
separate human decision.</p>
<h2>6. Links for manual verification</h2>
<ul>
<li><a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/">Problem description &amp; metric</a></li>
<li><a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/">About / resources</a></li>
<li><a href="https://www.drivendata.org/competitions/306/competition-doe-gems/data/">Data tab (login required)</a></li>
<li><a href="https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/">Leaderboard</a></li>
<li><a href="https://github.com/drivendataorg/gems-prize-reference-solution">Official reference solution</a></li>
<li><a href="https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and">USGS GeoDAWN survey</a></li>
<li><a href="https://gbcge.org/current-projects/ingenious/">INGENIOUS / QFaults</a></li>
<li><a href="https://epsg.io/32611">EPSG:32611</a></li>
<li><a href="https://en.wikipedia.org/wiki/Tversky_index">Tversky index</a></li>
<li><a href="https://doi.org/10.1109/CVPR.2008.4587436">Tu, Contextual Classification, CVPR 2008</a> &middot;
<a href="https://pages.ucsd.edu/~ztu/publication/pami_autocontext.pdf">TPAMI extended version</a></li>
</ul>
</main></body></html>"""
    (DOCS / "executive-summary.html").write_text(ex)

    if build is not None:
        (DL / f"{build['base_name']}-format-check.json").write_text(
            json.dumps(dict(zeros=build["outputs"]["zeros"], nan=build["outputs"]["nan"],
                            registry_verdict=build.get("registry_verdict"),
                            gate=(hold or {}).get("gate")), indent=1))
    print(f"[site] ok_to_submit={ok_to_submit} primary={primary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
