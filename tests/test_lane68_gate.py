"""Gate-logic tests (IR-31): download only if EVERY preregistered condition passes."""
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("l68rc", pathlib.Path(__file__).parents[1] / "scripts/lane68_write_run_card.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_current_evidence_is_negative():
    # stage2 far stratum t=1.97 and a drift flag must block promotion
    assert m.decide_verdict(True, False, [], True, []) == ("negative", False)
    assert m.decide_verdict(True, True, [], True, ["dense.tif"]) == ("negative", False)

def test_only_full_pass_promotes():
    # A score alone is insufficient: the exact holdout raster, pre-placement
    # uniqueness check, and paired CI must also be present and pass.
    provenance = {"holdout_linked": True, "surface_check_passed": True, "ci_available": True}
    assert m.decide_verdict(True, True, [], True, [], **provenance) == ("promote", True)
    assert m.decide_verdict(True, True, ["leaky"], True, [], **provenance) == ("negative", False)
    assert m.decide_verdict(True, True, [], False, [], **provenance) == ("negative", False)
    assert m.decide_verdict(True, True, [], True, []) == ("negative", False)
