"""Packaging metadata and ignore rules (ledger OPS-005, OPS-007)."""

from __future__ import annotations

import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_metadata_extras_and_entry_point() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["scripts"]["gmd-al"] == "gmd_active_learning.cli:main"
    assert {"ruff>=0.6", "mypy>=1.10"} <= set(project["optional-dependencies"]["dev"])
    assert any(dep.startswith("mkdocs") for dep in project["optional-dependencies"]["docs"])
    assert not any("hotspot" in dep or "lammps" in dep for dep in project["dependencies"])
    assert "MIT License" in (ROOT / "NOTICE").read_text() and "96263b6" in (ROOT / "NOTICE").read_text()


@pytest.mark.slow
def test_wheel_builds_from_the_committed_tree_without_the_reference(tmp_path: Path) -> None:
    export = tmp_path / "export"
    export.mkdir()
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", "HEAD"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(export)], input=archive, check=True)
    assert not (export / "ref").exists()
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--quiet", "--no-deps", "--no-build-isolation", "-w", str(tmp_path / "dist"), str(export)],
                   check=True, capture_output=True)
    wheel = next((tmp_path / "dist").glob("gmd_active_learning-*.whl"))
    names = zipfile.ZipFile(wheel).namelist()
    assert "gmd_active_learning/workflow/controller.py" in names
    assert not any(name.startswith(("ref/", "tests/", "hotspot_al/")) for name in names)


def _ignored(path: str) -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", path]).returncode == 0


def test_runtime_outputs_are_ignored_but_fixtures_and_sources_are_not() -> None:
    for path in ("runtime/x", "runs/a/state", "image.sif", "job.o12345", "slurm-77.out", "benchmark.json", "cp2k_jobs/t/in.inp",
                 "src/gmd_active_learning/__pycache__/x.pyc", "ref/lammps_monitor/README.md"):
        assert _ignored(path), path
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.extxyz", "*.pt", "*.lammpstrj"], check=True, capture_output=True, text=True).stdout.split()
    assert tracked, "fixtures with these suffixes are tracked"
    for path in [*tracked, "tests/fixtures/new.extxyz", "examples/data/new.pt", "tools/runtime/new_tool.py"]:
        assert not _ignored(path), path
