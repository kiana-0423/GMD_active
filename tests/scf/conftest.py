"""Shared configuration for the SCF / ML-SCF test suite.

* Puts the repository root on ``sys.path`` so the optional tooling under
  ``tools/scf`` (not an installed package) and the test helpers under
  ``tests/scf/fakes`` are importable.
* Registers the ``cp2k`` marker: tests that launch a real CP2K binary. They are
  skipped unless a CP2K executable is found (``$GMD_SCF_CP2K`` / ``$CP2K_BIN`` or
  ``cp2k.psmp`` on PATH).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    config.addinivalue_line("markers", "cp2k: launches a real CP2K binary (skipped when none is installed)")
    config.addinivalue_line("markers", "torch: needs PyTorch (skipped when torch is not importable)")
