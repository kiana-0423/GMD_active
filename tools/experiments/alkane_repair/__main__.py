"""``python -m tools.experiments.alkane_repair <command> --benchmark <yaml>``."""

from __future__ import annotations

import sys

from tools.experiments.alkane_repair.cli import main

sys.exit(main())
