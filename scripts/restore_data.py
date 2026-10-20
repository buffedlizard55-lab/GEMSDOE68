#!/usr/bin/env python3
"""Restore and SHA-256 verify the competition rasters from the group's GitHub mirrors.

Every file in registry/data_manifest.json is pinned to an owner-mirror repo,
commit and SHA-256. A mismatch aborts the restore. The hashes are NOT
independently authenticated against organizer-distributed bytes; this proves
only consistency with the named mirrors.

Usage:  python scripts/restore_data.py [--group core|external|scored|all] [--bridge]
Files land in .cache/gems_data/ (git-ignored). --bridge verifies and links all
inputs to data/bridge/ and data/external/ as used by the cached stack.
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


def bridge_inputs(manifest: dict, data_root: Path = DATA, repo_root: Path = ROOT) -> list[str]:
    """Symlink verified local mirrors into the paths used by the cached stack.

    Never download a credential-protected file or trust an unverified destination.
    The owner mirror is a reproducible *integrity* source, not an organizer receipt.
    """
    needed = {"training_features.tif", "labels.tif", "existing_faults.tif",
              "sample_submission.tif", "external/lidar_scarp_features_u8.tif",
              "external/lidar_scarp_features.json", "external/geodawn_rad_u8.tif",
              "external/geodawn_rad.json", "external/geodawn_extensions_u8.tif",
              "external/geodawn_extensions.json"}
    entries = {e["dest"]: e for e in manifest["files"] if e["dest"] in needed}
    if set(entries) != needed:
        raise ValueError(f"missing pinned bridge entries: {sorted(needed - set(entries))}")
    # Verify *every* dependency before changing any symlink.
    for key, entry in entries.items():
        source = data_root / key
        if not source.is_file() or source.stat().st_size != entry["bytes"] or \
                sha256_file(source) != entry["sha256"]:
            raise ValueError(f"unrestored/unverified bridge input: {key}")
    out = []
    for key in sorted(needed):
        source = (data_root / key).resolve()
        path = repo_root / "data" / ("bridge" if not key.startswith("external/")
                                     else "external") / Path(key).name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            if path.resolve() != source:
                raise ValueError(f"refuse to replace unexpected symlink: {path}")
        elif path.exists():
            if sha256_file(path) != entries[key]["sha256"]:
                raise ValueError(f"refuse to replace different local file: {path}")
            out.append(str(path.relative_to(repo_root)))
            continue
        else:
            path.symlink_to(source)
        out.append(str(path.relative_to(repo_root)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", choices=["core", "external", "scored", "all"])
    ap.add_argument("--bridge", action="store_true",
                    help="after hash verification, link core + external inputs into data/bridge and data/external")
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
               "all_verified": all(r["status"] in ("present", "restored") for r in results),
               "provenance": "owner GitHub mirrors verified by pin only; NOT independently organizer-authenticated"}
    if receipt["all_verified"] and args.bridge:
        try:
            receipt["bridged"] = bridge_inputs(manifest)
        except ValueError as exc:
            receipt["all_verified"] = False
            receipt["bridge_error"] = str(exc)
            print(f"bridge failed: {exc}", file=sys.stderr)
    (ROOT / "registry" / "restore_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if receipt["all_verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
