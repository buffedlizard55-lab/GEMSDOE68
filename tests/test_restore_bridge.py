"""Offline tests for the owner-mirror bridge: a hash mismatch must never pass."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from restore_data import bridge_inputs  # noqa: E402


NEEDED = ("training_features.tif", "labels.tif", "existing_faults.tif",
          "sample_submission.tif", "external/lidar_scarp_features_u8.tif",
          "external/lidar_scarp_features.json", "external/geodawn_rad_u8.tif",
          "external/geodawn_rad.json", "external/geodawn_extensions_u8.tif",
          "external/geodawn_extensions.json")


def _fixture(tmp_path):
    repo = tmp_path / "repo"
    cache = tmp_path / "cache"
    records = []
    for name in NEEDED:
        path = cache / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"test-pin:{name}".encode()
        path.write_bytes(content)
        records.append({"dest": name, "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest()})
    return repo, cache, {"files": records}


def test_bridge_uses_only_pinned_bytes_and_is_idempotent(tmp_path):
    repo, cache, manifest = _fixture(tmp_path)
    linked = bridge_inputs(manifest, data_root=cache, repo_root=repo)
    assert len(linked) == len(NEEDED)
    assert (repo / "data/bridge/training_features.tif").resolve() == (cache / "training_features.tif").resolve()
    assert (repo / "data/external/geodawn_rad.json").read_bytes() == (cache / "external/geodawn_rad.json").read_bytes()
    assert bridge_inputs(manifest, data_root=cache, repo_root=repo) == linked


def test_bridge_fails_closed_before_creating_links_on_tamper(tmp_path):
    repo, cache, manifest = _fixture(tmp_path)
    (cache / "external/geodawn_rad.json").write_text("tampered")
    with pytest.raises(ValueError, match="unrestored/unverified"):
        bridge_inputs(manifest, data_root=cache, repo_root=repo)
    assert not (repo / "data").exists()


def test_bridge_does_not_replace_unexpected_file(tmp_path):
    repo, cache, manifest = _fixture(tmp_path)
    conflict = repo / "data/bridge/training_features.tif"
    conflict.parent.mkdir(parents=True, exist_ok=True)
    conflict.write_text("not the mirror")
    with pytest.raises(ValueError, match="refuse to replace"):
        bridge_inputs(manifest, data_root=cache, repo_root=repo)
    assert conflict.read_text() == "not the mirror"
