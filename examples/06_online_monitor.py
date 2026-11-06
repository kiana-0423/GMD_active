# Ported from PHAL (examples/06_online_monitor.py) at 96263b6; MIT License, see NOTICE.
"""Online monitoring of GMD frames: replay a GMD frame file, or monitor a live GMD run.

    python examples/06_online_monitor.py                                  # replay examples/data (no binaries)
    python examples/06_online_monitor.py --event-dir events/
    python examples/06_online_monitor.py --live --gmd-executable ../GMD/build-torch/gmd \\
        --model model.pt --structure examples/data/water2.xyz --steps 20 --work-dir gmd_run

Replay mode feeds a frame file GMD wrote (``frame_file`` directive, ``gmd-frame-v1``) through the
``OnlineMonitorDriver``. Live mode starts GMD (built with ``-DGMD_ENABLE_TORCH=ON``) on a GMD_SGT
TorchScript export through the GMD adapter's ``GMDFrameSource`` and monitors each flushed frame
while the process runs; a STOP action ends the run cooperatively. Events are written to
``--event-dir`` when given. PHAL's example never called ``start()`` on its LAMMPS controller and
hung with real LAMMPS, and used a zero-force fake MLIP; here there is no fake: replay uses the
forces GMD recorded, live mode needs the real binary and model and fails clearly without them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from _example_utils import DATA_DIR, DEFAULT_TYPE_MAP, iter_trajectory, load_sections, require_file

from gmd_active_learning.contracts import ModelArtifact, UNITS
from gmd_active_learning.contracts.ids import file_sha256
from gmd_active_learning.events import EventStore
from gmd_active_learning.monitors import OnlineMonitorDriver
from gmd_active_learning.utils.logging import configure_logging


def live_source(args: argparse.Namespace, sections: dict[str, Any]) -> Any:
    from gmd_active_learning.adapters.gmd.engine import GMDConfig, GMDEngine, GMDFrameSource, structure_species
    from gmd_active_learning.workflow.protocols import SegmentRequest

    md = dict(sections.get("md") or {})
    # GMD runs inside the work directory, so every path handed to it must be absolute.
    md.update({"executable": str(args.gmd_executable.resolve()), "initial_structure": str(args.structure.resolve())})
    md.setdefault("frame_interval", 1)
    engine = GMDEngine(GMDConfig.model_validate(md))  # raises CapabilityUnavailableError without a binary
    digest = file_sha256(args.model)
    model = ModelArtifact(
        model_id=f"example-{digest[:12]}", checkpoint_path=str(args.model), checkpoint_sha256=digest,
        export_path=str(args.model.resolve()), export_sha256=digest, dataset_version=None, parent_model_id=None,
        species=tuple(sorted(set(structure_species(args.structure)))), cutoff=None, units=dict(UNITS),
        metrics={}, train_mode="external", backend="gmd_sgt", created_at="example",
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    request = SegmentRequest(run_id="example-live", segment_index=0, work_dir=args.work_dir.resolve(), model=model,
                             n_steps=args.steps, restart_from=None, previous_model=None)
    return GMDFrameSource(engine, request)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames", nargs="?", type=Path, default=DATA_DIR / "gmd_water_frames.extxyz", help="GMD frame file or legacy LAMMPS dump (replay mode)")
    parser.add_argument("--type-map", default=DEFAULT_TYPE_MAP, help="LAMMPS type map for dumps, e.g. 1=C,3=H")
    parser.add_argument("--config", type=Path, default=None, help="YAML with monitor/events/hotspots/md sections")
    parser.add_argument("--event-dir", type=Path, default=None, help="write finished events here (EventStore)")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--live", action="store_true", help="run GMD instead of replaying a frame file")
    parser.add_argument("--gmd-executable", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None, help="GMD_SGT TorchScript export (live mode)")
    parser.add_argument("--structure", type=Path, default=DATA_DIR / "water2.xyz", help="GMD xyz input (live mode)")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--work-dir", type=Path, default=Path("gmd_example_run"))
    args = parser.parse_args(argv)
    configure_logging("WARNING")
    sections = load_sections(args.config)

    if args.live:
        if args.gmd_executable is None or args.model is None:
            parser.error("--live needs --gmd-executable and --model")
        require_file(parser, args.model, "model export")
        require_file(parser, args.structure, "structure")
        source: Any = live_source(args, sections)
    else:
        require_file(parser, args.frames, "frame file")
        source = iter_trajectory(args.frames, type_map=args.type_map, run_id="example-replay")

    driver = OnlineMonitorDriver(
        sections.get("monitor"), sections.get("events"), sections.get("hotspots"),
        on_event=EventStore(args.event_dir) if args.event_dir else None,
    )
    try:
        summary = driver.run(source, max_frames=args.max_frames)
    finally:
        if hasattr(source, "close"):
            source.close()
    triggered = [result.step for result in summary.results if result.triggered]
    print(f"processed_frames={len(summary.results)} stop_reason={summary.stop_reason} errors={summary.errors}")
    print(f"steps={[result.step for result in summary.results]}")
    print(f"triggered_steps={triggered} events={driver.stats['events']}")
    return 0 if summary.errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
