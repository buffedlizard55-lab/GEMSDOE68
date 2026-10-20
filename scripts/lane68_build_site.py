#!/usr/bin/env python3
"""Compatibility entry point; the old site renderer was not release-gated.

Keep existing research commands working while routing *all* public pages through
the same fail-closed 2026-10-20 run card builder.
"""
from build_site_verified import main

if __name__ == "__main__":
    raise SystemExit(main())
