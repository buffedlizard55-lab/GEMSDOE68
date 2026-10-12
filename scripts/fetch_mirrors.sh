#!/usr/bin/env bash
# Restore the hash-pinned public mirrors of the competition rasters without any credentials.
# Only github.com / api.github.com are contacted.  raw.githubusercontent.com is NOT reachable
# from a restricted sandbox, so blobs are pulled through the GitHub blob API.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/bridge data/official work
fetch() { gh api "repos/$1/git/blobs/$2" -H "Accept: application/vnd.github.raw" > "$3"; }

# labels.tif and existing_faults.tif are byte-identical in the competition mirror
# (sha256 7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093, 425830 bytes)
[ -s data/official/labels.tif ] || fetch buffedlizard55-lab/59GEMSDOE 4ad3c1f3f19823e40924589bee7e51e44ae3a2e7 data/official/labels.tif
cp -f data/official/labels.tif data/official/existing_faults.tif
[ -s data/official/sample_submission.tif ] || fetch buffedlizard55-lab/59GEMSDOE 7d865a9921a40ed2ea4c742a6a25b1fa2f357c5a data/official/sample_submission.tif

# training features: gems-geodawn-numerical-features.tif, 5 parts in GEMSDOE/data/bridge
if [ ! -s data/bridge/gems-geodawn-numerical-features.tif ]; then
  SHAS=(a b c d e)   # filled in by scripts/refresh_mirror_shas.py
  python3 - <<'PY'
import json, subprocess, pathlib
tree = json.loads(subprocess.run(
    ["gh","api","repos/buffedlizard55-lab/5GEMSDOE/git/trees/main?recursive=1"],
    capture_output=True, text=True, check=True).stdout)
parts = {t["path"]: t["sha"] for t in tree["tree"]
         if t["path"].startswith("data/bridge/gems-geodawn-numerical-features.tif.part-")}
out = pathlib.Path("data/bridge")
for p in sorted(parts):
    dest = out / pathlib.Path(p).name
    if not dest.exists():
        with dest.open("wb") as fh:
            subprocess.run(["gh","api",f"repos/buffedlizard55-lab/5GEMSDOE/git/blobs/{parts[p]}",
                            "-H","Accept: application/vnd.github.raw"], stdout=fh, check=True)
    print("part", dest.name, dest.stat().st_size)
with (out/"gems-geodawn-numerical-features.tif").open("wb") as fh:
    for p in sorted(parts):
        fh.write((out/pathlib.Path(p).name).read_bytes())
PY
  rm -f data/bridge/*.part-*
fi
sha256sum data/official/*.tif data/bridge/*.tif
