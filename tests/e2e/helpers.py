"""Shared helpers for the end-to-end tests: the scripted fake CP2K of tests/labeling/conftest.py
(loaded by path, not duplicated) and the fake GMD executable of tests/adapters/gmd."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
FAKE_GMD_SCRIPT = TESTS / "adapters" / "gmd" / "fake_gmd.py"


def fake_cp2k_class():  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("_labeling_fake_cp2k", TESTS / "labeling" / "conftest.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.FakeCP2K


def fake_gmd_executable(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    executable = directory / "gmd"
    executable.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE_GMD_SCRIPT} \"$@\"\n")
    executable.chmod(0o755)
    return executable
