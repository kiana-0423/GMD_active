"""Fixtures for GMD adapter tests: a fake GMD executable (same CLI and file formats)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture()
def fake_gmd(tmp_path: Path) -> Path:
    executable = tmp_path / "bin" / "gmd"
    executable.parent.mkdir()
    executable.write_text(f"#!/bin/sh\nexec {sys.executable} {HERE / 'fake_gmd.py'} \"$@\"\n")
    executable.chmod(0o755)
    return executable


@pytest.fixture()
def water(tmp_path: Path) -> Path:
    path = tmp_path / "water.xyz"
    path.write_text("3\n8.0 8.0 8.0\nO 1.0 1.0 1.0\nH 1.9 1.0 1.0\nH 1.0 1.9 1.0\n")
    return path
