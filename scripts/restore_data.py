#!/usr/bin/env python3
"""Restore and SHA-256 verify the competition rasters from the group's GitHub mirrors.

Every file in registry/data_manifest.json is pinned to a repo + commit + SHA-256.
The hashes were computed from the organizer-distributed bytes by earlier sessions
(GEMSDOE24 / GEMSDOE32); a mismatch here aborts the restore.  Nothing in this script
authenticates against DrivenData - it only proves mirror consistency.

Usage:  python scripts/restore_data.py [--group core|external|scored|all]
Files land in .cache/gems_data/ (git-ignored, excluded from snapshots).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".cache" / "gems_data"
MANIFEST = ROOT / "registry" / "data_manifest.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(repo: str, ref: str, path: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        with tmp.open("wb") as out:
            subprocess.run(
                ["gh", "api", f"repos/{repo}/contents/{path}?ref={ref}",
                 "-H", "Accept: application/vnd.github.raw"],
                stdout=out, check=True,
            )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def restore_entry(entry: dict) -> dict:
    dest = DATA / entry["dest"]
    base = {"id": entry["id"], "dest": entry["dest"], "group": entry["group"]}
    if dest.exists() and sha256_file(dest) == entry["sha256"]:
        return {**base, "status": "present", "bytes": dest.stat().st_size, "sha256": entry["sha256"]}
    if "parts" in entry:
        tmp = dest.with_suffix(".assembling")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as out:
            for part in entry["parts"]:
                p = DATA / "raw_parts" / Path(part).name
                fetch(entry["repo"], entry["ref"], part, p)
                with p.open("rb") as src:
                    shutil.copyfileobj(src, out, 1 << 20)
                p.unlink()
        got = sha256_file(tmp)
        if got != entry["sha256"]:
            tmp.unlink()
            return {**base, "status": "HASH_MISMATCH", "expected": entry["sha256"], "got": got}
        tmp.replace(dest)
    else:
        fetch(entry["repo"], entry["ref"], entry["path"], dest)
        got = sha256_file(dest)
        if got != entry["sha256"]:
            dest.unlink()
            return {**base, "status": "HASH_MISMATCH", "expected": entry["sha256"], "got": got}
    return {**base, "status": "restored", "bytes": dest.stat().st_size, "sha256": entry["sha256"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", choices=["core", "external", "scored", "all"])
    args = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    DATA.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in manifest["files"]:
        if args.group != "all" and entry["group"] != args.group:
            continue
        r = restore_entry(entry)
        results.append(r)
        print(f"{r['status']:14s} {r['id']:45s} {r.get('bytes', '')}", flush=True)
    receipt = {"storage_root": str(DATA), "files": results,
               "all_verified": all(r["status"] in ("present", "restored") for r in results)}
    (ROOT / "registry" / "restore_receipt.json").write_text(json.dumps(receipt, indent=2))
    return 0 if receipt["all_verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
