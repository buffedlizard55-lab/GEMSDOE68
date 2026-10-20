#!/usr/bin/env python3
"""CI preflight: every public surface must agree with the canonical run card."""
from __future__ import annotations

import json
import re
from pathlib import Path

from build_site_verified import CARD_PATH, DOCS, ROOT, site_release_allowed


def main() -> int:
    if not CARD_PATH.is_file():
        raise SystemExit(f"no current run card: {CARD_PATH}")
    card = json.loads(CARD_PATH.read_text())
    released = site_release_allowed(card)
    tif = sorted((DOCS / "downloads").glob("*.tif"))
    zipfiles = sorted((DOCS / "downloads").glob("*.zip"))
    if not released and (tif or zipfiles):
        raise SystemExit(f"negative run card but public binaries remain: {tif + zipfiles}")
    if released and (len(tif) != 1 or zipfiles or
                     (ROOT / card["raster"]["path"]).resolve() != tif[0].resolve()):
        raise SystemExit("approved run card must map to exactly one file, without ambiguous ZIPs")
    for name in ("data/run-card.json", "runcard.json", "lane68/runcard.json",
                 "lane68/run-card.json", "lane68/data/run-card.json"):
        alias = json.loads((DOCS / name).read_text())
        if alias != card:
            raise SystemExit(f"stale public run-card alias: {name}")
    for name in ("data/feed.json", "lane68/feed.json", "lane68/data/feed.json"):
        feed = json.loads((DOCS / name).read_text())
        if feed.get("download_allowed") is not released or feed.get("verdict") != card["verdict"]:
            raise SystemExit(f"stale public feed: {name}")
    for page in DOCS.rglob("*.html"):
        text = page.read_text()
        relative = page.relative_to(DOCS)
        if not released and re.search(r'(?i)href=["\'][^"\']*\bdownloads/[^"\']*\.(?:tif|tiff|zip)(?:["\']|\?)', text):
            raise SystemExit(f"forbidden download link on negative page: {relative}")
        if not released and relative in (Path("index.html"), Path("executive-summary.html"),
                                           Path("lane68/index.html"), Path("lane68/executive-summary.html")) \
                and "DO NOT SUBMIT" not in text:
            raise SystemExit(f"missing prominent stop verdict on {relative}")
    print(f"Pages verified: verdict={card['verdict']}, download_allowed={released}, "
          f"public_tif={len(tif)}, public_zip={len(zipfiles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
