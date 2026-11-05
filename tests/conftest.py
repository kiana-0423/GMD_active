"""Shared pytest configuration: makes ``tests/`` importable for helper packages (e.g. ``fakes``)."""

from __future__ import annotations

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
