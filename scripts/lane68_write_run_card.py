#!/usr/bin/env python3
"""Compatibility command for the strict, negative-by-default H68-M1 run card.

An old renderer offered "promote-with-caveats" despite the far and originality
stops. Calling this command now writes the canonical evidence-based card only;
publication requires the independent shared release and selector gates.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lane68_strict_run_card import main  # noqa: E402


def decide_verdict(all_pass: bool, far_pass: bool, flagged_features: list[str],
                   validator_pass: bool, drift_rows: list[str]) -> tuple[str, bool]:
    """Historical IR-31 test compatibility; NEVER authorizes a public download.

    This small pure helper ensures a missed far gate or literal drift cannot be
    represented as a positive result. The real site gate is stricter and checks
    pinned bytes, provenance, registry scope and independent review as well.
    """
    complete = bool(all_pass and far_pass and not flagged_features and
                    validator_pass and not drift_rows)
    return ("promote" if complete else "negative", complete)


if __name__ == "__main__":
    raise SystemExit(main())
