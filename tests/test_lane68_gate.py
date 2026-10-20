"""Gate-logic tests (IR-31): download only if EVERY preregistered condition passes."""
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("l68rc", pathlib.Path(__file__).parents[1] / "scripts/lane68_write_run_card.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_current_evidence_is_negative():
    # stage2 far stratum t=1.97 and a drift flag must block promotion
    assert m.decide_verdict(True, False, [], True, []) == ("negative", False)
    assert m.decide_verdict(True, True, [], True, ["dense.tif"]) == ("negative", False)

def test_only_full_pass_promotes():
    assert m.decide_verdict(True, True, [], True, []) == ("promote", True)
    assert m.decide_verdict(True, True, ["leaky"], True, []) == ("negative", False)
    assert m.decide_verdict(True, True, [], False, []) == ("negative", False)


def test_exp3_promoted_run_card_and_submission_artifacts():
    import hashlib
    import json
    ROOT = pathlib.Path(__file__).parents[1]
    rc_paths = [
        ROOT / "evidence" / "run_card.json",
        ROOT / "docs" / "runcard.json",
        ROOT / "docs" / "lane68" / "data" / "run-card.json",
        ROOT / "docs" / "lane68" / "runcard.json",
        ROOT / "data" / "run_card.json",
    ]
    ref_text = rc_paths[0].read_text()
    for p in rc_paths[1:]:
        assert p.read_text() == ref_text, f"{p} is out of sync with evidence/run_card.json"

    rc = json.loads(ref_text)
    assert rc["verdict"] == "promote"
    assert rc["ok_to_download_and_submit"] is True

    # Holdout gate: Stage 2 and Stage 3 both beat Stage 1 by > 2 SE and 95% CI > 0 on all and >1 km far
    h = rc["holdout_dti"]
    assert h["evidence_class"] == "HOLDOUT-DTI"
    assert h["evaluator_version"] == "gems57-pooled-hide-v2"
    assert h["withheld_positives_all"] == 11321
    assert h["withheld_positives_far_gt_1km"] == 4310
    for key in ("stage2_all", "stage3_all", "stage2_far_gt_1km", "stage3_far_gt_1km"):
        s = h[key]
        assert s["delta_vs_stage1"] > 0.0
        assert s["delta_ci95"][0] > 0.0
        assert s["outer_block_t_stat"] > 2.0
        assert all(d > 0.0 for d in s["per_fold_delta"])

    # Leakage canary <= 0.90
    can = rc["leakage_canary"]
    assert can["passed"] is True
    assert can["max_stage1_auc"] <= 0.90
    assert can["max_stage2_auc"] <= 0.90

    # 65+ raster uniqueness check (62 in registry/rasters/rasters + 10 in .cache/gems_data/scored = 72 paths, 65 unique names)
    u = rc["registry_comparison"]
    assert u["n_rasters_checked"] >= 65
    assert u["duplicate"] is False
    assert abs(u["max_rank_corr_surface"]["rho"]) <= 0.90
    assert abs(u["max_rank_corr_dots"]["rho"]) <= 0.90
    assert u["max_dot_overlap_3px_informative"]["overlap"] <= 0.70
    assert u["max_surface_overlap_3px_informative"]["overlap"] <= 0.70

    # Submission artifacts on disk & SHA-256 verification
    sub = rc["submission"]
    assert len(sub["submission_note"]) <= 140
    zeros_tif = ROOT / "docs" / sub["primary_zeros_tif"]
    zeros_zip = ROOT / "docs" / sub["primary_zeros_zip"]
    nan_tif = ROOT / "docs" / sub["secondary_nan_tif"]
    nan_zip = ROOT / "docs" / sub["secondary_nan_zip"]
    for p in (zeros_tif, zeros_zip, nan_tif, nan_zip):
        assert p.is_file(), f"Missing artifact {p}"
    assert hashlib.sha256(zeros_tif.read_bytes()).hexdigest() == sub["sha256_zeros_tif"]
    assert hashlib.sha256(zeros_zip.read_bytes()).hexdigest() == sub["sha256_zeros_zip"]
    assert hashlib.sha256(nan_tif.read_bytes()).hexdigest() == sub["sha256_nan_tif"]
    assert hashlib.sha256(nan_zip.read_bytes()).hexdigest() == sub["sha256_nan_zip"]
    assert sub["validator_zeros"]["all_checks_passed"] is True
    assert sub["validator_zeros"]["n_nan"] == 0

    # Verify website pages link to the promoted primary GeoTIFF
    for html_rel in ("docs/index.html", "docs/executive-summary.html", "docs/how-to-submit.html", "docs/lane68/index.html", "README.md"):
        txt = (ROOT / html_rel).read_text()
        assert zeros_tif.name in txt, f"{html_rel} does not reference {zeros_tif.name}"
        assert sub["sha256_zeros_tif"] in txt, f"{html_rel} does not reference {sub['sha256_zeros_tif']}"

