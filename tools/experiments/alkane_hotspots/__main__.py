"""``python -m tools.experiments.alkane_hotspots <command> --config <yaml>``."""

from __future__ import annotations

import sys

from tools.experiments.alkane_hotspots.cli import main

sys.exit(main())
