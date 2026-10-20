#!/usr/bin/env python3
"""Compatibility entry point; fail-closed release renderer for GitHub Pages.

Historical experiments remain in evidence/archives, not a public TIFF link.
This command no longer promotes a file from an old run card.
"""
from build_site_verified import main

if __name__ == "__main__":
    raise SystemExit(main())
