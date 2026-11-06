# Ported from PHAL (examples/01_monitor_lammps_dump.py) at 96263b6; MIT License, see NOTICE.
"""Monitor a trajectory with the atom-wise staged OOD monitor.

    python examples/01_monitor_trajectory.py                              # legacy LAMMPS dump (examples/data)
    python examples/01_monitor_trajectory.py examples/data/gmd_water_frames.extxyz
    python examples/01_monitor_trajectory.py run.lammpstrj --type-map 1=C,2=H --config my_run.yaml

Accepts GMD frame files and legacy LAMMPS custom dumps (converted by
``gmd_active_learning.importers``), so historical PHAL trajectories can be re-monitored. PHAL's
version assembled the metrics by hand; here the ``OnlineMonitorDriver`` runs the configured
``monitor``/``events``/``hotspots`` sections (defaults unless ``--config`` overrides them).
Prints one line per frame: step, stage, max score, triggered, action, reasons.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _example_utils import DATA_DIR, add_common_arguments, iter_trajectory, load_sections, require_file

from gmd_active_learning.monitors import OnlineMonitorDriver
from gmd_active_learning.utils.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser, default_trajectory=DATA_DIR / "legacy_ethane_kick.lammpstrj")
    parser.add_argument("--timestep-fs", type=float, default=None, help="MD time step of a LAMMPS dump (sets Frame.time_fs)")
    args = parser.parse_args(argv)
    require_file(parser, args.trajectory, "trajectory")
    configure_logging("WARNING")

    sections = load_sections(args.config)
    driver = OnlineMonitorDriver(sections.get("monitor"), sections.get("events"), sections.get("hotspots"))
    frames = iter_trajectory(args.trajectory, type_map=args.type_map, timestep_fs=args.timestep_fs)
    summary = driver.run(frames)
    print("step stage max_score triggered action reasons")
    for result in summary.results:
        print(result.step, result.stage.value, f"{result.max_score:.4f}", result.triggered, result.action.value,
              ",".join(result.trigger_reasons) or "-")
    print(f"processed_frames={len(summary.results)} stop_reason={summary.stop_reason} errors={summary.errors}")
    return 0 if summary.errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
