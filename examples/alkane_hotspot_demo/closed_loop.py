"""Closed loop + frozen-model evaluation + recovery demonstration of the alkane demo (part gamma).

Usage (from the GMD_active root; real GMD, GMD_SGT and CP2K within the YAML budgets, resumable):

    python examples/alkane_hotspot_demo/closed_loop.py --closed-loop configs/demos/alkane_hotspots/closed_loop_smoke.yaml
    python examples/alkane_hotspot_demo/closed_loop.py --closed-loop configs/demos/alkane_hotspots/closed_loop_pilot.yaml --steps run frozen-eval

``--steps`` selects any of ``run`` (closed-loop-run), ``frozen-eval`` and ``recovery`` (recovery-demo), in
that order. The demo configuration (``--config``) is the closed-loop YAML's ``base_config``.
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

COMMANDS = {"run": "closed-loop-run", "frozen-eval": "frozen-eval", "recovery": "recovery-demo"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--closed-loop", required=True)
    parser.add_argument("--steps", nargs="+", choices=list(COMMANDS), default=list(COMMANDS))
    parser.add_argument("--max-wall-s", type=float, default=None, help="closed-loop-run: stop waiting for labels after this (resumable)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    import yaml

    closed_loop = Path(args.closed_loop)
    base = yaml.safe_load(closed_loop.read_text())["base_config"]
    config = str(closed_loop.parent / base)
    for step in [name for name in COMMANDS if name in args.steps]:
        extra = ["--max-wall-s", str(args.max_wall_s)] if step == "run" and args.max_wall_s is not None else []
        code = cli_main(["--log-level", args.log_level, COMMANDS[step], "--config", config, "--closed-loop", str(closed_loop), *extra])
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
