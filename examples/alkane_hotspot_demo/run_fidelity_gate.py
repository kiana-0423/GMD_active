"""Fragment-fidelity gate: prepare -> CP2K labels -> report, for the calibration or final_eval split.

Usage (from the GMD_active root; runs real CP2K within the YAML budgets, resumable):

    python examples/alkane_hotspot_demo/run_fidelity_gate.py --config configs/demos/alkane_hotspots/smoke.yaml
    python examples/alkane_hotspot_demo/run_fidelity_gate.py --config configs/demos/alkane_hotspots/pilot.yaml --split calibration
    python examples/alkane_hotspot_demo/run_fidelity_gate.py --config configs/demos/alkane_hotspots/pilot.yaml --split final_eval

``--prepare-only`` stops before any CP2K job (prints the number of calculations and the status/cost
estimate). final_eval requires the frozen calibration selection.
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
    parser.add_argument("--split", choices=["calibration", "final_eval"], default="calibration")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-wall-s", type=float, default=None)
    args = parser.parse_args(argv)
    common = ["--config", args.config, *(["--artifacts-root", args.artifacts_root] if args.artifacts_root else [])]
    from tools.experiments.alkane_hotspots.config import load_config

    parents = load_config(args.config, artifacts_root=args.artifacts_root).artifacts_root / "parents.jsonl"
    if not parents.exists():
        code = cli_main(["generate", *common])
        if code:
            return code
    code = cli_main(["fidelity-prepare", *common, "--split", args.split])
    if code:
        return code
    if args.prepare_only:
        return cli_main(["status", *common])
    run = ["fidelity-run", *common, "--split", args.split, *(["--max-wall-s", str(args.max_wall_s)] if args.max_wall_s else [])]
    code = cli_main(run)
    if code:
        return code
    code = cli_main(["fidelity-report", *common, "--split", args.split])
    if code:
        return code
    return cli_main(["status", *common])


if __name__ == "__main__":
    raise SystemExit(main())
