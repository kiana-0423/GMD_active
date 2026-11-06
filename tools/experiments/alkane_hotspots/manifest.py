"""Provenance manifests: configuration, code revision, environment, CP2K version and settings.

Every command writes ``<artifacts_root>/manifests/<command>.json`` (and appends a line to
``manifests/history.jsonl``) so that each artifact can be traced to the exact configuration, code
revision and program versions that produced it.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from gmd_active_learning.contracts.ids import file_sha256
from tools.experiments.alkane_hotspots.config import LoadedConfig

LOGGER = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_revision(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(status) if status is not None else None}


def package_versions(names: tuple[str, ...] = ("numpy", "scipy", "ase", "pydantic", "PyYAML", "torch")) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_info(root: Path) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
        "git": git_revision(root),
    }


def build_manifest(loaded: LoadedConfig, command: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    config = loaded.config
    config_path = loaded.path
    relative = None
    if config_path is not None:
        try:
            relative = str(config_path.resolve().relative_to(loaded.repo_root))
        except ValueError:
            relative = str(config_path)
    return {
        "command": command,
        "created_at": utc_now(),
        "config_name": config.name,
        "config_path": relative,
        "config_sha256": None if config_path is None else file_sha256(config_path),
        "config_hash": config.config_hash(),
        "generation_hash": config.generation_hash(),
        "smoke_only": config.cp2k.smoke_only,
        "artifacts_root": str(loaded.artifacts_root),
        "environment": environment_info(loaded.repo_root),
        "config": config.model_dump(mode="json"),
        **(extra or {}),
    }


def write_manifest(loaded: LoadedConfig, command: str, extra: dict[str, Any] | None = None) -> Path:
    manifest = build_manifest(loaded, command, extra)
    directory = loaded.artifacts_root / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{command}.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)
    with (directory / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({k: manifest[k] for k in ("command", "created_at", "config_hash")} | {"git": manifest["environment"]["git"]}, default=str) + "\n")
    LOGGER.info("manifest written to %s", path)
    return path


__all__ = ["build_manifest", "environment_info", "git_revision", "utc_now", "write_manifest"]
