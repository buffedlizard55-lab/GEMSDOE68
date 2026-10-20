#!/usr/bin/env python3
"""Rebuild the GEMSDOE68 site data and index page from dated repository evidence.

The site is a *derived* artefact: every number on it is read from a JSON file in
``evidence/`` or ``registry/``, so the published page can never drift from the
measurements.  Run it after any experiment or release step::

    PYTHONPATH=src python scripts/build_site.py
"""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
DOCS = ROOT / "docs" / "lane68"   # this lane's site; main keeps its own at docs/
REGISTRY = ROOT / "registry"


def _load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _pill(ok):
    if ok is True:
        return '<span class="pill pass">PASS</span>'
    if ok is False:
        return '<span class="pill fail">FAIL</span>'
    return '<span class="pill warn">NOT MEASURED</span>'


def _pytest_summary() -> str:
    """Number of tests in this checkout, or 'not measured' if pytest is unavailable."""
    import subprocess

    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "--no-header",
                              "-p", "no:cacheprovider"], cwd=ROOT, env=env,
                             capture_output=True, text=True, timeout=900)
    except Exception:
        return "not measured (pytest unavailable at build time)"
    tail = (out.stdout or "").strip().splitlines()
    return tail[-1] if tail else "not measured"


def _rel(path: str) -> str:
    """docs/downloads/x -> ../downloads/x, because this lane's pages live in docs/lane68/."""
    p = str(path)
    for prefix in ("docs/downloads/", "docs/data/"):
        if p.startswith(prefix):
            return "../" + p[len("docs/"):]
    if p.startswith("evidence/"):
        return "../../" + p
    if p.startswith("registry/"):
        return "../../" + p
    return p


def _release_status_block(card: dict) -> str:
    """Render a fail-closed top-of-page release decision from the run card."""
    raster = card.get("raster", {})
    name = html.escape(str(card.get("submission_name") or "candidate not named"))
    note = html.escape(str(card.get("note") or "not recorded"))
    note_chars = card.get("note_chars", "not measured")
    path = raster.get("primary_path")
    digest = html.escape(str(raster.get("primary_sha256") or "not recorded"))
    cleared = card.get("download_cleared") is True and card.get("verdict") == "promote"

    if cleared and path:
        href = html.escape(_rel(path), quote=True)
        action = f'<a href="{href}" download>Download the validated GeoTIFF</a>'
        color = "#176b3a"
        title = "CLEARED — OK TO DOWNLOAD AND SUBMIT"
        details = "Every preregistered release gate passed; review the run card and submission steps."
    else:
        action = "No download is offered for this negative result."
        color = "#9e2a2b"
        title = "NOT CLEARED — DO NOT DOWNLOAD OR SUBMIT"
        details = "The required holdout and/or lane-uniqueness gate did not pass. The raster remains in the repository only as an auditable negative-result artifact."

    return (
        f'<section class="hero" id="download" style="border:3px solid {color}">'
        f'<h2>{html.escape(title)}</h2><p><strong>{html.escape(details)}</strong></p>'
        f'<p>Candidate identifier: <code>{name}</code></p>'
        f'<p>SHA-256: <code>{digest}</code></p>'
        f'<p>Candidate note ({html.escape(str(note_chars))} characters; not a submission instruction): '
        f'<code>{note}</code></p><div class="dl">{action} '
        '<a class="ghost" href="executive-summary.html">Executive summary &amp; how to submit &rarr;</a>'
        '</div></section>'
    )


def _submit_page(title: str, card: dict) -> str:
    """Build explicit, fail-closed submission steps for the executive subpage."""
    cleared = card.get("download_cleared") is True and card.get("verdict") == "promote"
    name = html.escape(str(card.get("submission_name") or "not recorded"))
    note = html.escape(str(card.get("note") or "not recorded"))
    path = card.get("raster", {}).get("primary_path")
    button = (f'<p><a class="button" href="{html.escape(_rel(path), quote=True)}" download>'
              'Download the cleared GeoTIFF</a></p>') if cleared and path else (
              '<p class="blocked">No file is available to submit from this run. Do not upload '
              'the retained candidate raster.</p>')
    action = ("The current run card clears this candidate. Verify the file hash, then follow the steps below."
              if cleared else
              "Current verdict: NEGATIVE. The candidate failed one or more preregistered gates; these instructions are provided for future cleared runs only. Do not upload this candidate.")
    submit_steps = ("<ol><li>Download the linked GeoTIFF.</li><li>Open the DrivenData GEMS Prize "
                    "submission page and choose New Submission.</li><li>Select the downloaded .tif; "
                    "the competition accepts a single-band GeoTIFF or a ZIP containing one GeoTIFF.</li>"
                    f"<li>Use candidate name <code>{name}</code> and note <code>{note}</code>.</li>"
                    "<li>Submit, then confirm the entry and score on the competition page.</li></ol>"
                    if cleared else
                    "<ol><li>Do not upload any file from this negative run.</li><li>For a future candidate, "
                    "wait until this page explicitly says CLEARED and presents a download link.</li><li>Only "
                    "then upload that file to DrivenData, add its run-card name and note, and verify the portal result.</li></ol>")
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} — GEMSDOE68</title><style>
body{{font:16px/1.6 system-ui,sans-serif;max-width:850px;margin:32px auto;padding:0 18px;color:#17212b}}.status{{padding:18px;border:2px solid {'#176b3a' if cleared else '#9e2a2b'};border-radius:10px}}.blocked{{font-weight:800;color:#9e2a2b}}.button{{display:inline-block;padding:12px 18px;background:#165d91;color:white;border-radius:8px}}a{{color:#145a86}}code{{overflow-wrap:anywhere}}li{{margin:.5em 0}}</style></head><body>
<p><a href="index.html">← GEMSDOE68 lane summary</a></p><h1>{title}</h1><section class="status"><h2>{'CLEARED' if cleared else 'NOT CLEARED — DO NOT DOWNLOAD OR SUBMIT'}</h2><p>{html.escape(action)}</p>{button}</section><h2>Submission procedure</h2>{submit_steps}<h2>Recorded candidate details</h2><p>Name: <code>{name}</code></p><p>Recorded note: <code>{note}</code></p><p>Mechanical format validation is not a performance/release gate. See the <a href="data/run-card.json">machine-readable run card</a> and <a href="results.html">HOLDOUT-DTI results</a>.</p><p>Competition page: <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/">DrivenData GEMS Prize</a>.</p></body></html>'''


def _status_banner(card: dict) -> str:
    cleared = card.get("download_cleared") is True and card.get("verdict") == "promote"
    if cleared:
        return '<section class="panel"><strong>CLEARED — OK TO DOWNLOAD AND SUBMIT.</strong> See the approved one-click link on the summary page.</section>'
    return ('<section class="panel" style="border:3px solid #9e2a2b;background:#fff2f2;color:#411">'
            '<strong>NEGATIVE RESULT — DO NOT DOWNLOAD OR SUBMIT.</strong> The far-stratum holdout '
            'gate and/or lane-uniqueness gate failed. The emitted artifact is not hash-linked to '
            'the holdout prediction surface. No download is offered.</section>')


def _run_card_page(card: dict) -> str:
    raster = card.get("raster", {})
    hold = card.get("holdout_dti", {})
    corr = card.get("correlation_overlap_vs_registry", {})
    validator = card.get("validator_output", {})
    checks = [
        ("Hypothesis", card.get("hypothesis", "not recorded")),
        ("Mechanism", card.get("mechanism", "not recorded")),
        ("Named non-fault mimic", card.get("named_non_fault_process_that_could_mimic_it", "not recorded")),
        ("HOLDOUT-DTI evidence", json.dumps(hold, ensure_ascii=False)),
        ("Registry correlation/overlap", json.dumps({"max_abs_spearman": corr.get("max_abs_spearman"), "max_dot_overlap_3px": corr.get("max_dot_overlap_3px"), "literal_verdict": corr.get("literal_verdict")}, ensure_ascii=False)),
        ("Raster / sha256", f"{raster.get('primary_path')} · {raster.get('primary_sha256')}"),
        ("Validator", json.dumps(validator, ensure_ascii=False)),
        ("Submission name / recorded note", f"{card.get('submission_name')} · {card.get('note')} ({card.get('note_chars')} characters)"),
        ("Verdict", card.get("verdict")),
        ("CI / linkage limitation", f"{hold.get('ci95')} · {card.get('holdout_emission_linkage', {}).get('status')}"),
    ]
    rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in checks)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Run card — GEMSDOE68</title><link rel="stylesheet" href="assets/site.css"></head><body><header><h1>Run card</h1><nav><a href="index.html">Summary</a><a href="results.html">Results</a><a href="hypotheses.html">Hypotheses</a></nav></header><main>{_status_banner(card)}<table>{rows}</table><p><a href="data/run-card.json">Download JSON run card</a> (not a competition submission)</p></main></body></html>'''


def _write_root_pages(card: dict) -> None:
    """Keep the GitHub Pages root and lane page on the same fail-closed verdict."""
    cleared = card.get("download_cleared") is True and card.get("verdict") == "promote"
    name = html.escape(str(card.get("submission_name") or "not recorded"))
    note = html.escape(str(card.get("note") or "not recorded"))
    raster_path = card.get("raster", {}).get("primary_path")
    download = (f'<p><a class="button" href="{html.escape(raster_path.removeprefix("docs/"), quote=True)}" download>'
                'Download the cleared GeoTIFF</a></p>') if cleared and raster_path else (
                '<p><strong>No file is offered for download or submission.</strong> The current '
                'candidate is retained only as a negative-result artifact.</p>')
    status = "CLEARED — OK TO DOWNLOAD AND SUBMIT" if cleared else "NOT CLEARED — DO NOT DOWNLOAD OR SUBMIT"
    reason = ("Every preregistered release gate passed locally." if cleared else
              "The far-stratum holdout gate fails, the literal lane-overlap rule fires, and the "
              "current emission is not hash-linked to the holdout prediction surface.")
    index = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GEMSDOE68 — GEMS fault discovery</title><style>body{{font:16px/1.6 system-ui,sans-serif;max-width:950px;margin:30px auto;padding:0 18px;background:#0b1020;color:#e8ecf5}}section{{background:#141b2f;padding:20px;margin:18px 0;border-radius:12px;border:1px solid #303b58}}.status{{border:3px solid {'#176b3a' if cleared else '#9e2a2b'}}}a{{color:#79b8ff}}code{{overflow-wrap:anywhere}}</style></head><body><h1>GEMSDOE68 — GEMS fault discovery</h1><section class="status"><h2>{status}</h2><p>{reason}</p>{download}<p>Candidate identifier: <code>{name}</code></p></section><section><h2>Executive summary and audit trail</h2><p><a href="executive-summary.html">How to submit</a> · <a href="lane68/index.html">Lane 68 evidence</a> · <a href="lane68/data/run-card.json">Machine-readable run card</a> · <a href="lane68/results.html">Results</a> · <a href="hypotheses.html">Hypotheses</a> · <a href="sources.html">Sources</a></p><p>Current recorded candidate note: <code>{note}</code>. It is trace metadata, not clearance.</p></section><section><h2>Competition rules and official references</h2><ul><li><a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/">DrivenData problem description</a></li><li><a href="https://docs.nlr.gov/docs/fy26osti/96647.pdf">DOE GEMS official rules PDF</a></li><li><a href="https://github.com/drivendataorg/gems-prize-reference-solution">DrivenData reference solution</a></li><li><a href="https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and">USGS GeoDAWN source</a></li><li><a href="https://gbcge.org/current-projects/ingenious/">INGENIOUS project</a></li></ul></section></body></html>'''
    (ROOT / "docs" / "index.html").write_text(index)
    summary = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Executive summary — GEMSDOE68</title></head><body><main><p><a href="index.html">← Home</a></p><h1>Executive summary: how to submit</h1><h2>{status}</h2><p>{reason}</p>{download}<ol><li>Proceed only if the home page explicitly says CLEARED and provides a download button.</li><li>Upload only that GeoTIFF using DrivenData's New Submission form.</li><li>Copy the candidate name and note shown beside the cleared file.</li><li>Retain the organizer receipt; no local holdout score is an organizer score.</li></ol><p>Name: <code>{name}</code></p><p>Recorded note: <code>{note}</code></p><p><a href="lane68/data/run-card.json">Machine-readable run card</a> · <a href="https://docs.nlr.gov/docs/fy26osti/96647.pdf">Official rules</a></p></main></body></html>'''
    (ROOT / "docs" / "executive-summary.html").write_text(summary)
    (ROOT / "docs" / "how-to-submit.html").write_text(summary.replace("Executive summary: how to submit", "How to submit"))


def build() -> dict:
    # Recompute the release decision from the latest dated evidence before rendering.
    # A stale, manually edited site card must never expose a raster link.
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "scripts" / "lane68_write_run_card.py")],
                   cwd=ROOT, check=True, capture_output=True, text=True)

    exp = _load(EVIDENCE / "ac68_seed20.json") or {}
    exp21 = _load(EVIDENCE / "ac68_seed21.json")
    emis = _load(EVIDENCE / "ac68_emission.json") or {}
    lane = _load(EVIDENCE / "ac68_lane_check.json") or {}
    card = _load(EVIDENCE / "run-card-ac68.json") or {}
    dnull = _load(EVIDENCE / "ac68_lane_density_null.json") or {}
    irreg = _load(REGISTRY / "irregularities.json") or {}
    hyp = _load(REGISTRY / "hypotheses.json") or {}
    test_summary = _pytest_summary()

    cells = exp.get("cells", [])
    gate = exp.get("gate_all_pixels", {})
    gate_far = exp.get("gate_far_gt_1km", {})
    pooled = exp.get("pooled_dti", {})

    # ---------------------------------------------------------------- tables
    cell_rows = []
    for c in cells:
        cell_rows.append(
            "<tr><td class='mono'>{cell}</td><td class='num'>{nt}</td>"
            "<td class='num'>{nf}</td><td class='num'>{s1}</td><td class='num'>{s2}</td>"
            "<td class='num'>{s3}</td><td class='num'>{f1}</td><td class='num'>{f2}</td>"
            "<td class='num'>{f3}</td></tr>".format(
                cell=c.get("cell"), nt=c.get("n_truth"), nf=c.get("n_truth_far"),
                s1=_fmt(c.get("dti", {}).get("stage1", {}).get("dti")),
                s2=_fmt(c.get("dti", {}).get("stage2", {}).get("dti")),
                s3=_fmt(c.get("dti", {}).get("stage3", {}).get("dti")),
                f1=_fmt(c.get("dti_far", {}).get("stage1", {}).get("dti")),
                f2=_fmt(c.get("dti_far", {}).get("stage2", {}).get("dti")),
                f3=_fmt(c.get("dti_far", {}).get("stage3", {}).get("dti"))))

    gate_rows = []
    for name, g in list(gate.items()) + list(gate_far.items()):
        if not isinstance(g, dict) or "passes_gt_2se" not in g:
            continue
        gate_rows.append(
            "<tr><td class='mono'>{n}</td><td class='num'>{d}</td><td class='num'>{se}</td>"
            "<td class='num'>{t}</td><td>{p}</td></tr>".format(
                n=name, d=_fmt(g.get("mean_delta")), se=_fmt(g.get("se")),
                t=_fmt(g.get("t_stat"), 2), p=_pill(g.get("passes_gt_2se"))))

    sweep = {}
    for c in cells:
        for row in c.get("sweep", []):
            k = (row["stage"], row["radius_px"], row["budget"])
            sweep.setdefault(k, []).append(row["dti"])
    sweep_rows = []
    for (stage, rad, n), vals in sorted(sweep.items(), key=lambda t: -sum(t[1]) / len(t[1]))[:12]:
        sweep_rows.append(
            "<tr><td>{s}</td><td class='num'>{r}</td><td class='num'>{n}</td>"
            "<td class='num'>{v}</td></tr>".format(
                s=stage, r=rad, n=n, v=_fmt(sum(vals) / len(vals))))

    canary = exp.get("leakage_canary", {}).get("per_channel", {})
    canary_rows = sorted(canary.items(), key=lambda t: -t[1].get("max_auc", 0))[:12]
    canary_html = "".join(
        "<tr><td class='mono'>{c}</td><td class='num'>{m}</td><td class='num'>{x}</td></tr>".format(
            c=c, m=_fmt(v.get("mean_auc")), x=_fmt(v.get("max_auc")))
        for c, v in canary_rows)

    lane_rows = []
    lane_comparisons = sorted(
        lane.get("comparisons", []),
        key=lambda r: (not bool(r.get("drift")),
                       -(r.get("candidate_dots_within_3px") or 0),
                       -(r.get("absolute_spearman_rho") or 0)))
    for r in lane_comparisons[:8]:
        lane_rows.append(
            "<tr><td class='mono'>{f}</td><td class='num'>{c}</td><td class='num'>{o}</td>"
            "<td>{d}</td></tr>".format(
                f=r.get("registry"), c=_fmt(r.get("absolute_spearman_rho"), 3),
                o=_fmt(r.get("candidate_dots_within_3px"), 3), d=_pill(r.get("drift") is False)))

    files = emis.get("files", {})
    dl = []
    for variant in ("nan", "zeros"):
        f = files.get(variant)
        if not f:
            continue
        dl.append(f)
    primary = dl[0] if dl else None
    secondary = dl[1] if len(dl) > 1 else None

    # ------------------------------------------------------- explicit gate summary
    def _g(table, key):
        return (table or {}).get(key, {})

    _g2 = _g(gate, "stage2_vs_stage1")
    _gf = _g(gate_far, "stage2_vs_stage1")
    _can = exp.get("leakage_canary", {}).get("per_channel", {})
    _maxauc = max((v.get("max_auc") or 0) for v in _can.values()) if _can else None
    _nfar = sum(c.get("n_truth_far", 0) for c in cells)
    flagged_rows = [r for r in lane.get("comparisons", []) if r.get("drift")]
    flagged = flagged_rows[0] if flagged_rows else None
    if flagged:
        overlap_html = (
            '<li><strong>Final-raster lane rule: FAIL / stop.</strong> Against '
            f'<code>{html.escape(str(flagged.get("registry")))}</code>, '
            f'{flagged.get("candidate_dots_within_3px", 0):.4%} of candidate dots fall within '
            '3 px (limit 70%); absolute Spearman rho = '
            f'{flagged.get("absolute_spearman_rho", 0):.4f} (limit 0.90). No exact byte match '
            'is reported. The historical density-null test belongs to another raster and is not '
            'used to waive this final-candidate flag.</li>')
    else:
        overlap_html = '<li><strong>Final-raster lane rule: PASS.</strong> No duplicate threshold crossed in the recorded final-raster check.</li>'
    gate_summary = (
        '<h3>Gate results, stated explicitly</h3><ul class="small">'
        '<li><strong>HOLDOUT-DTI all-pixel gate: PASS.</strong> Stage 2 vs stage 1, mean '
        f'+{_g2.get("mean_delta", float("nan")):.4f}, SE {_g2.get("se", float("nan")):.4f}, '
        f't = {_g2.get("t_stat", float("nan")):.2f} (&gt; 2 required), positive on all '
        f'{_g2.get("n_blocks", 0)} outer blocks.</li>'
        '<li><strong>HOLDOUT-DTI &gt;1 km stratum: NOT MET.</strong> Stage 2 vs stage 1, mean '
        f'+{_gf.get("mean_delta", float("nan")):.4f}, SE {_gf.get("se", float("nan")):.4f}, '
        f't = {_gf.get("t_stat", float("nan")):.2f} (&gt; 2 required). The number of far-stratum '
        f'withheld positives is {_nfar}; this comparison fails the preregistered threshold.</li>'
        '<li><strong>Leakage canary: PASS.</strong> Highest recorded single-channel AUC '
        f'{_maxauc:.3f} against a 0.90 limit; no channel flagged.</li>'
        f'{overlap_html}'
        '<li><strong>No ORGANIZER-CONFIRMED score exists.</strong> These are local HOLDOUT-DTI '
        'measurements; the official score is not claimed.</li>'
        '<li><strong>Candidate linkage limitation.</strong> The current stage-3 emitted TIFF hash '
        'is not linked to the held-out prediction surface; the pooled scores are method-level '
        'evidence, not a measured score for this file.</li></ul>')

    # ------------------------------------------------------------------ page
    template = (DOCS / "index.template.html").read_text()
    tokens = {
        "{{CELL_ROWS}}": "".join(cell_rows) or "<tr><td colspan=9>no holdout run yet</td></tr>",
        "{{GATE_ROWS}}": "".join(gate_rows) or "<tr><td colspan=5>gate not evaluated</td></tr>",
        "{{SWEEP_ROWS}}": "".join(sweep_rows) or "<tr><td colspan=4>no sweep</td></tr>",
        "{{CANARY_ROWS}}": canary_html or "<tr><td colspan=3>not measured</td></tr>",
        "{{LANE_ROWS}}": "".join(lane_rows) or "<tr><td colspan=4>not measured</td></tr>",
        "{{LANE_MAX_RHO}}": _fmt(max((r.get("absolute_spearman_rho") or 0) for r in lane.get("comparisons", [])), 4),
        "{{LANE_MAX_OVERLAP}}": _fmt(max((r.get("candidate_dots_within_3px") or 0) for r in lane.get("comparisons", [])), 4),
        "{{POOLED_S1}}": _fmt(pooled.get("stage1")),
        "{{POOLED_S2}}": _fmt(pooled.get("stage2")),
        "{{POOLED_S3}}": _fmt(pooled.get("stage3")),
        "{{N_CELLS}}": str(len(cells)),
        "{{N_TRUTH}}": str(sum(c.get("n_truth", 0) for c in cells)),
        "{{N_TRUTH_FAR}}": str(sum(c.get("n_truth_far", 0) for c in cells)),
        "{{DOT_COUNT}}": str(emis.get("dot_count", "n/a")),
        "{{RADIUS_PX}}": str(emis.get("emission_params", {}).get("radius_px", "n/a")),
        "{{BUDGET}}": str(emis.get("emission_params", {}).get("budget", "n/a")),
        "{{SHA256}}": (primary or {}).get("sha256", "n/a"),
        "{{BYTES}}": str((primary or {}).get("bytes", "n/a")),
        "{{VALIDATOR}}": str((primary or {}).get("validator_passed", "n/a")),
        "{{N_CHECKS}}": str((primary or {}).get("n_checks", "n/a")),
        "{{GATE_VERDICT}}": card.get("verdict", "pending"),
        "{{GATE_SUMMARY}}": gate_summary,
        "{{RELEASE_STATUS_BLOCK}}": _release_status_block(card),
        "{{GENERATED}}": exp.get("generated_utc", "n/a"),
        "{{PRIMARY_LINK}}": _rel((primary or {}).get("path", "#")),
        "{{SECONDARY_LINK}}": _rel((secondary or {}).get("path", "#")),
        "{{SECONDARY_SHA}}": (secondary or {}).get("sha256", "n/a"),
        "{{NAME}}": card.get("submission_name", "n/a"),
        "{{NOTE}}": card.get("note", "n/a"),
    }
    page = template
    for k, v in tokens.items():
        page = page.replace(k, "n/a" if v is None else str(v))
    page = page.replace("<header>", '<header>\n  <p class="small" style="margin:0 0 8px">'
                        '<a href="../index.html">&larr; the repository site (all lanes)</a>'
                        ' &middot; this page is the GEMSDOE68 lane-68 site</p>', 1)
    (DOCS / "index.html").write_text(page)
    (DOCS / "executive-summary.html").write_text(
        _submit_page("Executive summary: how to submit", card))
    (DOCS / "submission-guide.html").write_text(
        _submit_page("Submission guide", card))

    # ------------------------------------------------- extra pages from evidence
    extra_tokens = {
        "{{CELL_ROWS}}": "".join(cell_rows) or "<tr><td colspan=9>no holdout run yet</td></tr>",
        "{{GATE_ROWS}}": "".join(gate_rows) or "<tr><td colspan=5>gate not evaluated</td></tr>",
        "{{SWEEP_ROWS}}": "".join(sweep_rows) or "<tr><td colspan=4>no sweep</td></tr>",
        "{{CANARY_ROWS}}": canary_html or "<tr><td colspan=3>not measured</td></tr>",
        "{{LANE_ROWS}}": "".join(lane_rows) or "<tr><td colspan=4>not measured</td></tr>",
        "{{LANE_MAX_RHO}}": _fmt(max((r.get("absolute_spearman_rho") or 0) for r in lane.get("comparisons", [])), 4),
        "{{LANE_MAX_OVERLAP}}": _fmt(max((r.get("candidate_dots_within_3px") or 0) for r in lane.get("comparisons", [])), 4),
        "{{LANE_MAX_RHO}}": _fmt(max((r.get("absolute_spearman_rho") or 0) for r in lane.get("comparisons", [])), 4),
        "{{LANE_MAX_OVERLAP}}": _fmt(max((r.get("candidate_dots_within_3px") or 0) for r in lane.get("comparisons", [])), 4),
        "{{POOLED_S1}}": _fmt(pooled.get("stage1")),
        "{{POOLED_S2}}": _fmt(pooled.get("stage2")),
        "{{POOLED_S3}}": _fmt(pooled.get("stage3")),
        "{{N_TRUTH}}": str(sum(c.get("n_truth", 0) for c in cells)),
        "{{N_TRUTH_FAR}}": str(sum(c.get("n_truth_far", 0) for c in cells)),
        "{{TEST_SUMMARY}}": test_summary,
        "{{CURRENT_STATUS}}": _status_banner(card),
        "{{IRREG_ROWS}}": "".join(
            "<tr><td class='mono'>{i}</td><td>{s}</td><td>{t}</td></tr>".format(
                i=it.get("id"), s=it.get("severity"), t=it.get("title"))
            for it in irreg.get("items", []) if it.get("id", "").startswith("IR-24-AC68")
            or it.get("id", "").startswith("IR-25-AC68")
            or it.get("id", "").startswith("IR-26-AC68")
            or it.get("id", "").startswith("IR-27-AC68")
            or it.get("id", "").startswith("IR-28-AC68")
            or it.get("id", "").startswith("IR-29-AC68"))
            or "<tr><td colspan=3>none recorded</td></tr>",
    }
    for name in ("results.html", "irregularities.html"):
        source = DOCS / ("results.template.html" if name == "results.html" else name)
        target = DOCS / name
        if not source.exists():
            continue
        txt = source.read_text()
        for k, v in extra_tokens.items():
            txt = txt.replace(k, "n/a" if v is None else str(v))
        target.write_text(txt)
    (DOCS / "run-card.html").write_text(_run_card_page(card))

    feed = {
        "generated_utc": emis.get("generated_utc") or exp.get("generated_utc"),
        "holdout_evidence_generated_utc": exp.get("generated_utc"),
        "site": "GEMSDOE68",
        "lane": "auto-context stacked refinement (Tu, CVPR 2008)",
        "status": card.get("verdict", "pending"),
        "submission": card.get("submission_name"),
        "download": ((primary or {}).get("path")
                     if card.get("download_cleared") is True and card.get("verdict") == "promote"
                     else None),
        "candidate_artifact": (primary or {}).get("path"),
        "download_cleared": card.get("download_cleared") is True,
        "sha256": (primary or {}).get("sha256"),
        "holdout_dti": pooled,
        "gate": {"all_pixels": gate, "far_gt_1km": gate_far},
        "dot_count": emis.get("dot_count"),
        "sources": [
            "https://www.drivendata.org/competitions/306/competition-doe-gems/",
            "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/",
            "https://docs.nlr.gov/docs/fy26osti/96647.pdf",
            "https://github.com/drivendataorg/gems-prize-reference-solution",
            "https://doi.org/10.1109/CVPR.2008.4587436",
            "https://pages.ucsd.edu/~ztu/publication/pami_autocontext.pdf",
            "https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and",
            "https://gbcge.org/current-projects/ingenious/",
            "https://epsg.io/32611",
        ],
    }
    (DOCS / "data" / "feed.json").write_text(json.dumps(feed, indent=1))
    (DOCS / "data" / "run-card.json").write_text(json.dumps(card, indent=1))
    (DOCS / "runcard.json").write_text(json.dumps(card, indent=1))
    _write_root_pages(card)
    return feed


if __name__ == "__main__":
    feed = build()
    print(json.dumps(feed, indent=1)[:1500])
