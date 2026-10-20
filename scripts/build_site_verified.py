#!/usr/bin/env python3
"""Fail-closed GitHub Pages builder: never link a candidate on an unreviewed run card.

Only the final selector may authorize publication; an automated experiment is
insufficient. Build all front doors from one run card and archive old URLs as
non-download redirects. This deliberately does not call the legacy builders.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
sys.path.insert(0, str(ROOT / "src"))
CARD_PATH = ROOT / "evidence" / "run-card-2026-10-20.json"
SOURCES = ROOT / "registry" / "sources_2026-10-20.json"
SNAPSHOT = ROOT / "registry" / "leaderboard_snapshot_2026-10-20.json"


def h(value: object) -> str:
    return html.escape(str(value), quote=True)


def site_release_allowed(card: dict, root: Path = ROOT) -> bool:
    """Document gate, independent decision, pinned file, and offline checks required.

    No booleans supplied by a stale historical card are sufficient: the file
    itself must reside in docs/downloads and hash to the approved run card.
    Never copy a candidate to that directory automatically.
    """
    if card.get("verdict") != "promote" or card.get("selector_status") != "approved" or \
            card.get("download_allowed") is not True:
        return False
    gate = card.get("paired_gate") or {}
    leak = card.get("leakage_canary") or {}
    unique = card.get("uniqueness") or {}
    fmt = card.get("format") or {}
    if not all((gate.get("all", {}).get("pass") is True,
                gate.get("far_gt_1km", {}).get("pass") is True,
                leak.get("status") == "PASS",
                isinstance(leak.get("max_auc_measurable"), (int, float)) and
                leak["max_auc_measurable"] <= .90 and
                leak.get("count_non_finite") == 0 and
                leak.get("count_over_0_90") == 0 and
                leak.get("full_feature_auc_list_in_report") is True,
                unique.get("surface", {}).get("status") == "PASS",
                unique.get("final_dots", {}).get("status") == "PASS",
                fmt.get("strict_validator_passed") is True,
                fmt.get("outside_fill") == "nan",
                card.get("input_provenance_verified") is True)):
        return False
    raster = card.get("raster") or {}
    relative = raster.get("path")
    if not isinstance(relative, str) or not relative.startswith("docs/downloads/") \
            or not relative.endswith(".tif") or Path(relative).name != relative.split("/")[-1]:
        return False
    file = (root / relative).resolve()
    if not file.is_relative_to((root / "docs" / "downloads").resolve()) or not file.is_file():
        return False
    expected = raster.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64 or \
            hashlib.sha256(file.read_bytes()).hexdigest() != expected:
        return False
    if not (isinstance(card.get("note"), str) and len(card["note"]) <= 140 and
            isinstance(card.get("name"), str) and card["name"].strip()):
        return False
    # The stricter shared release gate checks certified *complete* registry,
    # per-feature leakage, tied surface hash, provenance and independent reviews.
    # This Pages wrapper may not override it, even with a forged local card flag.
    from gems.release_gate import assess_release
    evidence = card.get("release_evidence") or {}
    review = assess_release(file, evidence.get("shared_card") or {},
                            evidence.get("strict_validation") or {},
                            evidence.get("surface_audit") or {},
                            evidence.get("final_audit") or {},
                            evidence.get("surface_sha256"))
    return review["cleared_for_review"] is True


def shell(title: str, active: str, body: str, *, path_prefix: str = "") -> str:
    nav = (("index.html", "Decision"), ("executive-summary.html", "How it works"),
           ("results.html", "Holdout results"), ("hypotheses.html", "Hypotheses"),
           ("research.html", "Sources & methods"), ("irregularities.html", "Audit"),
           ("leaderboard.html", "Score ledger"))
    links = "".join(f'<a href="{path_prefix}{dest}" class="{("active" if dest == active else "")}">{h(label)}</a>'
                    for dest, label in nav)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="GEMSDOE68 research and fail-closed GEMS Prize submission decision.">
<title>{h(title)} · GEMSDOE68</title><link rel="stylesheet" href="{path_prefix}assets/site.css"></head>
<body><a class="skip" href="#main">Skip to content</a><header class="top"><div class="wrap bar"><a class="brand" href="{path_prefix}index.html">GEMS<span>DOE68</span><small>Research run · 20 Oct 2026</small></a><nav aria-label="Main navigation">{links}</nav></div></header>
<main id="main" class="wrap">{body}</main><footer><div class="wrap">Experimental fault mapping, not a competition receipt. <a href="{path_prefix}data/run-card.json">Read the machine-readable run card</a> · <a href="https://github.com/buffedlizard55-lab/GEMSDOE68/blob/arena/fd107b38-gemsdoe68/README.md">Source & standing prompt</a> · Data: owner GitHub mirrors; organizer authentication unavailable.</div></footer>
</body></html>\n'''


def gate_text(gate: dict) -> str:
    if not gate:
        return "Not measured"
    delta, se, t = (gate.get(k) for k in ("delta_mean", "se", "t_stat"))
    if any(v is None for v in (delta, se, t)):
        return "Not measured"
    result = f'{"PASS" if gate.get("pass") is True else "FAIL"} · Δ {float(delta):+.5f} / SE {float(se):.5f} = {float(t):.2f} (&gt;2 required)'
    ci = gate.get("bootstrap_paired_delta_ci95")
    if isinstance(ci, list) and len(ci) == 2:
        result += f' · paired 95% CI [{float(ci[0]):+.5f}, {float(ci[1]):+.5f}]'
    return result


def metric_cell(row: dict) -> str:
    if not row or row.get("value") is None:
        return "Not measured"
    ci = row.get("ci95")
    interval = f' · 95% CI [{float(ci[0]):.4f}, {float(ci[1]):.4f}]' if ci else " · CI unavailable"
    return f'{float(row["value"]):.4f}{interval}'


def build(card: dict, sources: dict, snapshot: dict, ledger: list[dict], root: Path = ROOT) -> dict[str, str]:
    release = site_release_allowed(card, root)
    if not release:
        # Neither the presence of a TIFF nor a positive internal model comparison
        # can grant clearance; the selector and provenance gate are separate.
        badge = '<span class="badge danger">NOT CLEARED · DO NOT SUBMIT</span>'
        verdict = "No public TIFF download. Do not reuse a historical file."
        next_step = "The preregistered experiment produced a negative or incomplete release decision. No organizer slot was used."
    else:
        badge = '<span class="badge good">REVIEWED · DOWNLOAD CLEARED</span>'
        verdict = "One selector-approved TIFF is available below."
        next_step = '<a class="button" href="' + h(card["raster"]["path"].removeprefix("docs/")) + '">Download the approved GeoTIFF</a>'
    stem = f'<section class="hero">{badge}<p class="eyebrow">GEMS Prize · fault prediction · research decision</p><h1>{h(verdict)}</h1><p class="lead">{h(next_step) if not release else next_step}</p><div class="meta">H68-M1 · {h(card.get("created_utc", "not yet completed"))} · {h(card.get("evidence_class", "not yet measured"))}</div></section>'
    withheld = card.get("withheld_positive_pixels")
    n_far = card.get("withheld_positive_pixels_far_gt_1km")
    paired = card.get("paired_gate") or {}
    stage = card.get("stage_dti") or {}
    stage_far = card.get("stage_dti_far") or {}
    canary = card.get("leakage_canary") or {}
    uniqueness = card.get("uniqueness") or {}
    fmt = card.get("format") or {}
    raster = card.get("raster") or {}
    metrics = f'''<div class="cards">
      <div class="card"><span class="eyebrow">All withheld</span><strong>{gate_text(paired.get('all') or {})}</strong><small>Stage 2 minus stage 1, four buffered outer blocks; must exceed 2 standard errors.</small></div>
      <div class="card"><span class="eyebrow">More than 1 km from mapped faults</span><strong>{gate_text(paired.get('far_gt_1km') or {})}</strong><small>Independent hard gate; a positive trend alone is not a pass.</small></div>
      <div class="card"><span class="eyebrow">Single-feature leakage</span><strong>{h(canary.get('status', 'NOT MEASURED'))} · max measurable AUC {f'{canary["max_auc_measurable"]:.3f}' if isinstance(canary.get('max_auc_measurable'), (float, int)) else 'N/A'}</strong><small>{h(canary.get('count_non_finite', 'pending'))} feature/fold AUCs unavailable due to non-finite inputs. No measured AUC &gt;0.90 does not clear missing values.</small></div>
    </div>'''
    blockers = card.get("restrictions") or ["No positive run card and release review recorded."]
    blocker_html = "".join(f"<li>{h(reason)}</li>" for reason in blockers)
    summary = f'''{stem}<div class="section-heading"><p class="eyebrow">01 / Decision first</p><h2>What the evidence actually permits</h2></div>
    {metrics}<div class="two"><section class="panel"><h3>Release gate</h3><ul class="checklist">{blocker_html}</ul>
    <p class="muted">If any gate fails, there is no approved file, even if an older local validator passed. <strong>HOLDOUT-DTI is not an organizer score.</strong></p></section>
    <section class="panel"><h3>Fast path for the participant</h3><ol><li>Read the red/green decision banner above; red means stop.</li><li>Only on a future green decision: compare filename and SHA-256 with the run card.</li><li>Sign in to the official <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/submissions/">DrivenData submissions page</a>, upload the single GeoTIFF and the ≤140-character note.</li><li>Keep the organizer receipt. A local report cannot confirm acceptance or private score.</li></ol><a class="text-link" href="executive-summary.html">Detailed how-to →</a></section></div>
    <section class="panel band"><p class="eyebrow">02 / Evidence chain</p><h2>Four outer holdouts; zero contest slots</h2>
    <p>Experiment <strong>{h(card.get('experiment_id','H68-M1'))}</strong> used {h(card.get('outer_blocks', '4'))} spatial outer blocks and {h(card.get('inner_blocks_per_outer', '4'))} buffered inner blocks, with {h(withheld if withheld is not None else 'pending')} withheld positive cells ({h(n_far if n_far is not None else 'pending')} in the &gt;1 km stratum). Evaluator <code>{h(card.get('evaluator_version','not yet measured'))}</code>; 300 m triangular distance weighting, α = 0.2, β = 0.8. This is <strong>local hide-and-recover</strong>, not access to the private expert labels.</p>
    <p>Stage 1: {metric_cell(stage.get('stage1') or {})}; stage 2: {metric_cell(stage.get('stage2') or {})}; stage 3: {metric_cell(stage.get('stage3') or {})}. Intervals are spatial-block bootstrap 95% CIs, <em>not</em> organizer-confirmed scores. <a href="results.html">Protocol and per-fold results →</a></p></section>
    <section class="panel"><p class="eyebrow">03 / Honesty checkpoint</p><h2>Why H33 could do well; can this beat 0.2778?</h2><p>The prompt supplies <strong>0.2778 as an owner-reported H33 score</strong>; this session found a pinned owner-mirror file hash but no organizer submission receipt tying the claimed score to those bytes. A sparse, pruned set of high-mass lines is consistent with the official DTI geometry: true coverage increases the numerator; extra length adds false-positive mass while β = 0.8 makes uncovered truth costly. This is an <em>explanation of design incentives</em>, not a causal ablation of H33 or a prediction that this lane will beat it. A new outer-fold gain, even if positive, cannot establish private-prize performance.</p><a class="text-link" href="research.html">Metric derivation & cited sources →</a></section>'''
    files = {"index.html": shell("Submission decision", "index.html", summary)}
    how = f'''{stem}<div class="section-heading"><p class="eyebrow">Step-by-step</p><h2>How to submit — only after a green decision</h2></div>
    <section class="panel"><ol class="big-list"><li><strong>Stop now if this page is red.</strong> A valid raster is not necessarily a good or unique submission. Historical binaries have been withdrawn from the public download directory.</li><li><strong>On green only:</strong> download the one named TIFF shown on the decision page. The run card must give its exact SHA-256, a unique submission name and a note of no more than 140 characters. Do not substitute a historical “zeros” file: the October-12 one was subsequently repaired to be all-finite but failed scientific/originality gates; the newer parallel-main stage3 “zeros” file still contains 7,113,308 NaNs and also failed both gates.</li><li><strong>Verify it locally:</strong> <code>sha256sum file.tif</code>; then run <code>PYTHONPATH=src .venv/bin/python -m gems.validate_submission file.tif --reference data/bridge/sample_submission.tif --features data/bridge/training_features.tif</code>. This proves local format only, not portal acceptance.</li><li><strong>Upload yourself</strong> via the <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/submissions/">official submissions page</a> using a registered, eligible account. Check the current three/week quota. The organizer, not this site, assigns a score and receipt.</li><li><strong>If the portal says “[0,1]” error:</strong> do not claim that relabelling a TIFF fixes it. The original October-12 “zeros” artifact contained NaNs before an archival repair; the October-20 stage3 “zeros” artifact still does. The shared writer can now emit a genuine all-finite zero-exterior <em>diagnostic</em> variant, but the published spec says null/NaN exterior and acceptance of zero exterior is unconfirmed. Obtain organizer guidance and revalidate before using a slot.</li></ol></section>
    <section class="panel"><h3>Current file and note</h3><dl class="kv"><dt>Approved download</dt><dd>{h(raster.get('path') or 'NONE — do not submit')}</dd><dt>Submission name</dt><dd>{h(card.get('name') or 'NOT ASSIGNED')}</dd><dt>≤140-character note</dt><dd>{h(card.get('note') or 'NOT ASSIGNED')}</dd><dt>Current selector decision</dt><dd>{h(card.get('selector_status','not requested'))}</dd><dt>Verified organizer receipt</dt><dd>NONE</dd></dl></section>'''
    files["executive-summary.html"] = shell("Executive summary / how to", "executive-summary.html", how)
    files["how-to-submit.html"] = files["executive-summary.html"]
    holdout_path = card.get("holdout_source") or "not created"
    uniq_rows = "".join(f"<tr><th>{h(label)}</th><td>{h((uniqueness.get(key) or {}).get('status','NOT RUN'))}</td><td>{h((uniqueness.get(key) or {}).get('worst_raster','—'))}</td></tr>" for label,key in (("Pre-placement surface", "surface"), ("Final dots", "final_dots")))
    results = f'''{stem}<div class="section-heading"><p class="eyebrow">Measured locally, never organizer-confirmed</p><h2>Blocked hide-and-recover results</h2></div>
    <section class="panel"><p>Preregistered <a href="research/2026-10-20-autocontext-hypotheses.md">H68-M1</a>. A continuous stage-1 OOF prior trains two iterations of a LightGBM auto-context model with magnetic-edge-aligned ±1–15-pixel line offsets, across/along anisotropic context, original gradients, tilt and scarp channels, and catalogue context buffered away from truth. Four held-out geographic quadrants and four interior blocks per outer fold. Stage 1 and both refinements are compared at the <strong>same fixed 38,000-dot global budget, apportioned by fold footprint</strong>; withheld outer labels never enter the model training features. The feature context excludes whole withheld components plus a 3-pixel collar. The scoring mask excludes remaining visible known faults while retaining withheld positives.</p><p>The evaluator version <code>{h(card.get('evaluator_version','pending'))}</code> uses 300 m support, α = 0.2, β = 0.8, pooled TP/FP/FN, paired spatial-block bootstrap 95% CIs. Withheld positives: {h(withheld if withheld is not None else 'pending')} total; {h(n_far if n_far is not None else 'pending')} &gt;1 km from the remaining visible mapped faults. The far stratum simulates an off-visible-catalogue gap but still consists of previously mapped catalogue faults; it cannot certify discovery of the private new-fault truth. Run source: <code>{h(holdout_path)}</code>.</p>
    <div class="table-wrap"><table><caption>HOLDOUT-DTI, not a DrivenData score</caption><thead><tr><th>Measure</th><th>Stage 1</th><th>Stage 2</th><th>Stage 3</th></tr></thead><tbody><tr><th>Pooled on all withheld</th><td>{metric_cell(stage.get('stage1') or {})}</td><td>{metric_cell(stage.get('stage2') or {})}</td><td>{metric_cell(stage.get('stage3') or {})}</td></tr><tr><th>Pooled on &gt;1 km stratum</th><td>{metric_cell(stage_far.get('stage1') or {})}</td><td>{metric_cell(stage_far.get('stage2') or {})}</td><td>{metric_cell(stage_far.get('stage3') or {})}</td></tr><tr><th>Stage 2 − stage 1</th><td colspan="3">{gate_text(paired.get('all') or {})}</td></tr><tr><th>Same comparison, &gt;1 km</th><td colspan="3">{gate_text(paired.get('far_gt_1km') or {})}</td></tr></tbody></table></div>
    <p class="muted">Per-fold details and per-feature canaries are in <a href="data/run-card.json">the run card</a> and the source experiment JSON. The &gt;2SE gate uses four paired outer-fold scores; the displayed 95% CIs use resampled 20 km physical clusters, conditional on fitted models. They are different uncertainty summaries, neither a p-value nor a private score.</p></section>
    <section class="panel"><h3>Literal originality & format gate</h3><div class="table-wrap"><table><thead><tr><th>Comparison</th><th>Status</th><th>Worst prior raster</th></tr></thead><tbody>{uniq_rows}</tbody></table></div><p>Screen byte hash and exact pixels, |Spearman| &gt; 0.90 on the continuous pre-placement field and the emitted file, and emitted candidate-dot overlap &gt; 0.70 within 3 pixels. A dense-reference null argument does not cancel the 0.70 stop rule. Historical failure: 99.48% overlap for October-12 stack2 with <code>gems61-lane-buried-step-gravinversion-hgm-r3-k400k-b2zeros.tif</code>. Parallel-main October-20 stage3 (32,000 dots) also drifted: 98.465625% against that prior and far paired t=1.88 &lt;2; it was archived, not offered as an alternative. Accessible corpus is <em>not</em> certified as all organizer submissions.</p><p>New candidate: {h(raster.get('path') or 'none emitted; release stopped')}. Strict format validator: {h(fmt.get('status','NOT RUN'))}. Input mirrors: hash-pinned owner repository only; no organizer authentication.</p></section>'''
    files["results.html"] = shell("Holdout and uniqueness", "results.html", results)
    hypotheses = f'''{stem}<div class="section-heading"><p class="eyebrow">Ranked before fitting</p><h2>Geological hypotheses, not post-hoc score stories</h2></div>
    <section class="panel"><p>Pre-registered before H68-M1 started: <a href="research/2026-10-20-autocontext-hypotheses.md">dated hypotheses and stop rules</a>. Magnetic boundaries may mark lithology, acquisition artefacts, or cultural features, not just faults; each claim is a testable physical conjecture. Competing approaches are lower-priority research ideas, <strong>not further experiments run here</strong>.</p><div class="table-wrap"><table><thead><tr><th>Rank</th><th>Idea</th><th>Geological discriminant</th><th>Main falsifier</th></tr></thead><tbody>
    <tr><td>1 · H68-M1</td><td>Magnetic-edge-tangent auto-context</td><td>Faults juxtapose contrasting magnetization; a line-continuation prior from field boundaries and stage-1 probabilities could recover partially masked fault traces without simply drawing radial halos.</td><td>No &gt;2 SE paired gain on both all withheld and &gt;1 km.</td></tr>
    <tr><td>2 · H68-M2</td><td>Gravity × magnetic stage-2 orientation agreement</td><td>Co-oriented gravity and magnetic gradients may corroborate covered basement boundaries; igneous contacts can mimic displacement.</td><td>No outer-fold gain when acquisition streaks and intrusions are separated.</td></tr>
    <tr><td>3 · H68-M3</td><td>Signed LiDAR scarp-side polarity in oriented context</td><td>Asymmetric across-fault scarps may bridge mapped gaps; erosional range fronts and alluvial-fan terraces mimic this signal.</td><td>No coverage-stratified far-fold gain or a leakage canary &gt;0.90.</td></tr>
    <tr><td>4 · H68-M4</td><td>Radiometric contrast × along-strike probability</td><td>K/Th/U spatial contrasts may extend a weak line beyond mapped traces when LiDAR is unavailable; rock-type and soil changes are mimics.</td><td>No far-fold gain after land-cover and lithology controls.</td></tr></tbody></table></div></section>
    <section class="panel"><h3>Constraints for every hypothesis</h3><p>Physical consistency is not ground truth. Anisotropy must be computed from <em>geophysical</em> edges, not labels. Mask all hidden fault components plus a feature buffer. No tuning to the four outer blocks; three experiments or two hours maximum, and a failed gate means stop.</p></section>'''
    files["hypotheses.html"] = shell("Research hypotheses", "hypotheses.html", hypotheses)
    rows = "".join(f'<tr><td>{h(item["id"])}</td><td><a href="{h(item["url"])}">{h(item["kind"])}</a></td><td>{h(item["verified_claim"])}</td></tr>' for item in sources["sources"])
    research = f'''{stem}<div class="section-heading"><p class="eyebrow">Citation ledger · accessed {h(sources['observed_utc_date'])}</p><h2>Source-by-source verification</h2></div>
    <section class="panel"><p><strong>Problem:</strong> the prize target is <em>fault pixels relevant to geothermal exploration</em>, not a training set of geothermal vents. The competition masks its private new expert fault lines. Owner mirrors restore inputs under pinned hashes but cannot replace authenticated organizer downloads. This site makes no statement about an undisclosed 2026 final label set.</p><p><strong>Why the geometry matters:</strong> the official kernel gives a predicted dot partial credit only within 300 m of a truth pixel. The distance-weighted Tversky index is <code>TP / (TP + 0.2·FP + 0.8·FN)</code>, with TP/FP/FN derived from bidirectional nearest-neighbour kernel coverage, not simply a pixelwise classification rate. An elongated 1-pixel line can cover more unique truth than a thick binary mask, but excessive lines spend FP mass. This is why H33-style pruning is plausible—<em>not</em> proof it scored 0.2778 or that this model will improve it.</p><p><strong>Physics:</strong> GeoDAWN airborne magnetics and radiometrics and reprocessed LiDAR scarps can reveal structural discontinuities; a magnetic edge alone may be a lithologic contact, survey boundary or human infrastructure. Directional offsets and smooths encode this uncertainty; blocked, buffered withheld faults test extrapolation rather than memorizing the labelled catalogue. <a href="https://github.com/buffedlizard55-lab/GEMSDOE68/blob/arena/fd107b38-gemsdoe68/knowledge/01_why_02778_and_can_we_beat_it.md">Read the mass/pruning derivation</a>.</p><div class="table-wrap"><table><thead><tr><th>ID</th><th>Trust class / source link</th><th>What was actually checked</th></tr></thead><tbody>{rows}</tbody></table></div><p class="muted">The numbered source ledger is also archived at <code>registry/sources_2026-10-20.json</code>. Official challenge and government sources support rules/data provenance; author papers support method motivation, not this lane's skill. Link-only external data is not used unless locally hash-pinned with documented rights.</p></section>'''
    files["research.html"] = shell("Official sources and methodology", "research.html", research)
    issues = f'''{stem}<div class="section-heading"><p class="eyebrow">Failure-first review · three passes, one agent</p><h2>Irregularities and unresolved risks</h2></div>
    <section class="panel"><ol class="issues"><li><strong>IR-68-01 · historical off-range gate failed:</strong> +0.0074055 / 0.00376155 = 1.9687, not &gt;2. The older page called this “promote-with-caveats”; that wording and the old download were withdrawn. All-pixel holdout +0.0606585 / 0.0070493 did pass but cannot substitute for the far stratum.</li><li><strong>IR-68-02 · literal overlap stop:</strong> 0.9948 of 20,000 old candidate dots lay within 3 pixels of one dense prior registry raster, exceeding 0.70. A density-null control is not a rule exemption.</li><li><strong>IR-68-03 · misleading “zeros” variants:</strong> the original October-12 band had 7,113,308 NaNs, like its “nan” twin. Parallel PR #5 repaired that old archived fallback to genuinely all-finite (diagnostic only); later PR #6's stage3 “zeros” band still contains 7,113,308 NaNs. The shared writer now has an explicit <code>outside_fill='zero'</code> diagnostic path and separate local validator policy; no portal acceptance is claimed.</li><li><strong>IR-68-04 · source & score provenance:</strong> the owner H33 page states no organizer receipt there; the 0.2778 in the prompt remains owner-reported. Original competition rasters require login; restored files are SHA-pinned mirrors only. The public leaderboard number is a dated table observation, not a per-raster score.</li><li><strong>IR-68-05 · buffered holdout defects:</strong> previous across-strike means actually sampled along-strike; offset indexing wrapped at raster edges; Gaussian strip halo was too narrow; old outer test labels also leaked into training. Corrections and regression tests precede H68-M1; old performance estimates remain historical.</li><li><strong>IR-68-06 · shared evaluator integration:</strong> the first preregistered attempt stopped before any scored fold because a <code>fold.get</code> default accessed a missing <code>visible</code> key even when <code>masked_known</code> existed. Repaired in the shared evaluator, unit-tested, and re-ran exactly the same hypothesis/parameters. Not counted as positive evidence.</li><li><strong>IR-68-07 · shared NMS plateau bug:</strong> the historical 5×5 maximum filter passed every tied LightGBM plateau pixel, then chose an arbitrary top-k; adjacent dots could violate the claimed spacing. Fixed deterministic greedy suppression in <code>gems68.emission</code> with a regression test <em>before inspecting the first scored attempt</em>. That provisional attempt is not used for promotion; the same preregistered experiment was rerun with the fixed emitter. The corrected stage-2 t-statistics were 0.12 overall and 0.39 &gt;1 km, both below 2. The 644 stored canary attempts had no measured AUC &gt;0.90, but 132 LiDAR-derived feature/fold occurrences were non-finite, so the individual-feature safety audit is incomplete.</li><li><strong>IR-68-08 · parallel-main stage3 candidate withdrawn:</strong> PR #6 emitted a 32,000-dot stage3 raster and offered a download despite inherited &gt;1 km stage3 t=1.88 (&gt;2 required) and a measured 0.98465625 dot overlap (0.70 stop). Both file variants and local receipts were archived under <code>evidence/archives/2026-10-20/</code>; the old card is now negative. The method was not rerun on fresh blocked folds; no organizer score is claimed.</li><li><strong>IR-68-09 · current limitation:</strong> available prior rasters are not a proven complete organizer-wide registry; original downloadable source data and current portal eligibility/quota have not been authenticated. A local GeoTIFF preflight can be green while the upload portal still rejects the file.</li></ol><p>Pass A: baseline assets/sources/duplicates and leakage review. Pass B: shared stacking, geometry and evaluation unit tests + preregistration. Pass C: inspected fresh evidence, the run card, the stopped candidate builder, validator non-applicability, and every Pages/no-download URL. These are iterative passes by one agent, <strong>not two independent scientific leads</strong>.</p></section>'''
    files["irregularities.html"] = shell("Irregularities and review", "irregularities.html", issues)
    score_rows = "".join(f'<tr><td>{h(row.get("site",""))}</td><td>{h(row.get("submission_name",""))}</td><td>{h(row.get("score","—") or "—")}</td><td>{h(row.get("evidence_class", "NOT VERIFIED"))}</td><td>{h(row.get("note",""))}</td></tr>' for row in ledger)
    board = f'''{stem}<div class="section-heading"><p class="eyebrow">Snapshot, not a live API</p><h2>Public leaderboard and owner score claims</h2></div>
    <section class="panel"><h3>Highest public row observed on {h(snapshot['observed_utc_date'])}</h3><p><a href="{h(snapshot['url'])}">DrivenData public leaderboard</a> showed participant <strong>{h(snapshot['top_public']['participant'])}</strong> at <strong>{float(snapshot['top_public']['displayed_score']):.4f}</strong>. This is a dated public observation, not evidence that a GEMSDOE raster was uploaded or scored. <span id="stale" class="badge caution">Snapshot date: {h(snapshot['observed_utc_date'])} · NOT LIVE</span></p><p>The initial high values 0.3774/0.3195 are conflicting user-supplied snapshots, not current verified leader values. No score for H68-M1 was submitted; <strong>no ORGANIZER-CONFIRMED receipt</strong> is in this repo. We cannot poll a login-gated team's submissions from a public static site.</p></section>
    <section class="panel"><h3>Family ledger (self/owner report, NOT verified receipts)</h3><p>Historical owner-reported values, listed for audit not ranking. Any number without a matching organizer receipt is not a verified score; do not infer a new lane's expected prize value from them. Source: <code>registry/score_ledger.csv</code>.</p><div class="table-wrap"><table><thead><tr><th>Site</th><th>Submission name</th><th>Reported score</th><th>Evidence class</th><th>Source note</th></tr></thead><tbody>{score_rows}</tbody></table></div></section>
    <script>const d=new Date('{h(snapshot['observed_utc_date'])}T00:00:00Z');if(Date.now()-d.getTime()>86400000){{document.getElementById('stale').textContent='STALE PUBLIC SNAPSHOT — visit DrivenData for current leaderboard';}}</script>'''
    files["leaderboard.html"] = shell("Score evidence ledger", "leaderboard.html", board)
    # Legacy URLs must no longer display historically favourable copy or an
    # old download link. A static self-contained warning (not JS-only redirect).
    for alias, target in (("lane68/index.html", "index.html"),
                          ("lane68/executive-summary.html", "executive-summary.html"),
                          ("lane68/submission-guide.html", "executive-summary.html"),
                          ("lane68/results.html", "results.html"),
                          ("lane68/research.html", "research.html"),
                          ("lane68/hypotheses.html", "hypotheses.html"),
                          ("lane68/run-card.html", "results.html"),
                          ("lane68/irregularities.html", "irregularities.html"),
                          ("experiments.html", "results.html"),
                          ("knowledge.html", "research.html"),
                          ("sources.html", "research.html")):
        prefix = "../" if alias.startswith("lane68/") else ""
        body = f'<section class="hero"><span class="badge danger">HISTORICAL PAGE WITHDRAWN · DO NOT SUBMIT</span><h1>Read the current release decision</h1><p>Older content and downloads here were invalidated by strict holdout or originality checks.</p><a class="button outline" href="{prefix}{target}">Current audited page →</a></section>'
        files[alias] = shell("Historical page withdrawn", "", body, path_prefix=prefix)
    return files


def main() -> int:
    if not CARD_PATH.is_file():
        raise SystemExit(f"missing run card {CARD_PATH}; no site built")
    card = json.loads(CARD_PATH.read_text())
    sources = json.loads(SOURCES.read_text())
    snapshot = json.loads(SNAPSHOT.read_text())
    with (ROOT / "registry" / "score_ledger.csv").open(newline="") as file:
        ledger = list(csv.DictReader(file))
    files = build(card, sources, snapshot, ledger)
    for name, contents in files.items():
        path = DOCS / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    feed = {"evidence_class": "HOLDOUT-DTI (not organizer receipt)",
            "run_card_created_utc": card.get("created_utc"),
            "verdict": card.get("verdict", "negative"),
            "download_allowed": site_release_allowed(card),
            "leaderboard_observed_utc_date": snapshot["observed_utc_date"],
            "leaderboard_is_live": False,
            "leaderboard_url": snapshot["url"],
            "public_top": snapshot["top_public"],
            "organizer_confirmed_this_lane": None,
            "message": "No submission slot or organizer receipt. Check official public leaderboard for current standings."}
    data = DOCS / "data"
    data.mkdir(exist_ok=True)
    card_json = json.dumps(card, indent=2, allow_nan=False) + "\n"
    feed_json = json.dumps(feed, indent=2, allow_nan=False) + "\n"
    (data / "run-card.json").write_text(card_json)
    (data / "feed.json").write_text(feed_json)
    # Every old feed/run-card URL must carry the same decision, not a stale
    # "promote-with-caveats" from a previous raster.
    lane_data = DOCS / "lane68"
    (lane_data / "run-card.json").write_text(card_json)
    (lane_data / "feed.json").write_text(feed_json)
    (DOCS / "runcard.json").write_text(card_json)
    (lane_data / "runcard.json").write_text(card_json)
    (lane_data / "data" / "run-card.json").write_text(card_json)
    (lane_data / "data" / "feed.json").write_text(feed_json)
    print(f"wrote {len(files)} fail-closed pages; download_allowed={feed['download_allowed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
