"""Manifests record config, code revision, environment and command outputs."""

from __future__ import annotations

import json

from ._helpers import tiny_config
from tools.experiments.alkane_hotspots.manifest import build_manifest, write_manifest


def test_manifest_contents_and_history(tmp_path) -> None:
    loaded = tiny_config(tmp_path)
    manifest = build_manifest(loaded, "generate", {"outputs": {"n": 1}})
    assert manifest["config_hash"] == loaded.config.config_hash()
    assert manifest["generation_hash"] == loaded.config.generation_hash()
    assert manifest["smoke_only"] is True and manifest["outputs"] == {"n": 1}
    assert set(manifest["environment"]) >= {"python", "packages", "git"}
    assert manifest["config"]["name"] == "tiny" and len(manifest["config_sha256"]) == 64
    path = write_manifest(loaded, "generate", {"cp2k": {"version": "2026.2", "settings_hash": "abc"}})
    written = json.loads(path.read_text())
    assert written["cp2k"]["version"] == "2026.2"
    write_manifest(loaded, "status")
    history = (loaded.artifacts_root / "manifests" / "history.jsonl").read_text().splitlines()
    assert [json.loads(line)["command"] for line in history] == ["generate", "status"]
