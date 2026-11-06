"""Alkane hotspot demonstration (PHAL in GMD_active): structures, CP2K labels, fragment fidelity,
and (part beta) the hotspot model study. Run as ``python -m tools.experiments.alkane_hotspots``.

This is demo code in the source tree, not part of the installed package. When ``gmd_active_learning``
is not installed, the checkout's ``src/`` directory (located relative to this file) is put on
``sys.path`` so the commands work from a plain clone.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

if importlib.util.find_spec("gmd_active_learning") is None:  # pragma: no cover - depends on the environment
    _SRC = Path(__file__).resolve().parents[3] / "src"
    if (_SRC / "gmd_active_learning").is_dir():
        sys.path.insert(0, str(_SRC))
