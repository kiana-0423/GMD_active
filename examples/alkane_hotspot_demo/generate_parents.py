"""Generate the alkane parents (deterministic) for a demo configuration.

Usage (from the GMD_active root):

    python examples/alkane_hotspot_demo/generate_parents.py --config configs/demos/alkane_hotspots/pilot.yaml

Thin wrapper around ``python -m tools.experiments.alkane_hotspots generate``; artifacts go to the
configuration's ``artifacts_root`` (default ``../alkane_artifacts/<name>``).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiments.alkane_hotspots.cli import main as cli_main  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifacts-root", default=None)
    args = parser.parse_args(argv)
    extra = ["--artifacts-root", args.artifacts_root] if args.artifacts_root else []
    return cli_main(["generate", "--config", args.config, *extra])


if __name__ == "__main__":
    raise SystemExit(main())
