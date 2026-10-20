#!/usr/bin/env python3
"""Compatibility command for the current strict, negative-by-default run card.

The prior implementation declared "promote-with-caveats" after a failed far
holdout and duplicate gate. It has been retired, not used for release.
"""
from lane68_strict_run_card import main

if __name__ == "__main__":
    raise SystemExit(main())
