"""Shared fixtures for CP2K adapter tests; registers the ``cp2k`` marker for real-CP2K tests."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from gmd_active_learning.contracts.regions import ExtractedRegion

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "cp2k: runs the real CP2K binary (skipped when it is not installed)")


@pytest.fixture()
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture()
def load_region():
    def _load(name: str) -> ExtractedRegion:
        return ExtractedRegion.from_dict(json.loads((FIXTURES / "structures" / "regions" / f"{name}.json").read_text()))

    return _load


@pytest.fixture()
def cp2k_bin() -> str:
    candidate = os.environ.get("CP2K_BIN") or shutil.which("cp2k.psmp") or shutil.which("cp2k.popt")
    if not candidate:
        pytest.skip("CP2K binary not found (set CP2K_BIN or put cp2k.psmp/cp2k.popt on PATH)")
    return candidate
