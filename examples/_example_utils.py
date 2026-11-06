"""Shared helpers for the numbered examples (not part of the package API).

Examples run as scripts (``python examples/01_monitor_trajectory.py``), so this module is
imported from the script directory. Inputs default to the small files in ``examples/data``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from gmd_active_learning.contracts import ExtractedRegion, Frame
from gmd_active_learning.extraction import CappingConfig, EmbeddingConfig, ExtractionConfig, extract_and_prepare
from gmd_active_learning.hotspots import hotspots_from_result
from gmd_active_learning.monitors import OnlineMonitorDriver
from gmd_active_learning.utils.yaml import load_yaml

EXAMPLES_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLES_DIR / "data"
LAMMPS_SUFFIXES = {".lammpstrj", ".dump", ".lammpsdump"}
# Type map of the legacy toy ethane dump (LAMMPS types 1 = C, 3 = H; type 2 is unused).
DEFAULT_TYPE_MAP = "1=C,3=H"


def parse_type_map(text: str | None) -> dict[int, str] | None:
    """``"1=C,3=H"`` -> ``{1: "C", 3: "H"}``; empty -> ``None`` (dumps with an ``element`` column)."""

    if not text:
        return None
    mapping: dict[int, str] = {}
    for item in text.split(","):
        key, _, symbol = item.partition("=")
        if not symbol:
            raise argparse.ArgumentTypeError(f"type map entries look like 1=C, got {item!r}")
        mapping[int(key)] = symbol.strip()
    return mapping


def iter_trajectory(path: Path, *, type_map: str | None = None, run_id: str = "example", timestep_fs: float | None = None) -> Iterator[Frame]:
    """Frames from a GMD frame file (extxyz) or a legacy LAMMPS custom dump."""

    if path.suffix.lower() in LAMMPS_SUFFIXES:
        from gmd_active_learning.importers import iter_lammps_dump

        yield from iter_lammps_dump(path, type_map=parse_type_map(type_map), timestep_fs=timestep_fs, run_id=run_id)
        return
    from gmd_active_learning.adapters.gmd.engine import iter_frames

    yield from iter_frames(path, run_id)


def load_sections(path: Path | None) -> dict[str, Any]:
    """Section mappings from a YAML file (a full ``gmd-al`` run config or just some sections)."""

    return load_yaml(path) if path is not None else {}


def add_common_arguments(parser: argparse.ArgumentParser, *, default_trajectory: Path) -> None:
    parser.add_argument("trajectory", nargs="?", type=Path, default=default_trajectory,
                        help="GMD frame file (.extxyz) or legacy LAMMPS dump (.lammpstrj)")
    parser.add_argument("--type-map", default=DEFAULT_TYPE_MAP, help="LAMMPS type map for dumps, e.g. 1=C,3=H")
    parser.add_argument("--config", type=Path, default=None, help="YAML with the sections to override (defaults otherwise)")


def require_file(parser: argparse.ArgumentParser, path: Path, what: str) -> None:
    if not path.is_file():
        parser.error(f"{what} does not exist: {path}")


def first_frame(parser: argparse.ArgumentParser, path: Path, type_map: str | None) -> Frame:
    for frame in iter_trajectory(path, type_map=type_map):
        return frame
    parser.error(f"{path} contains no frames")
    raise AssertionError  # pragma: no cover - parser.error exits


def add_hotspot_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--atom-id", type=int, default=None, help="use this stable atom ID as the hotspot instead of monitoring")
    parser.add_argument("--frame", type=int, default=0, help="frame index for --atom-id")
    parser.add_argument("--mode", choices=["cluster", "slab", "graph", "block"], default=None, help="override extraction.mode")


def select_hotspots(args: argparse.Namespace, sections: dict[str, Any]) -> tuple[Frame | None, list[list[int]], str | None]:
    """(frame, hotspot groups of stable atom IDs, event id)."""

    if args.atom_id is not None:
        for index, frame in enumerate(iter_trajectory(args.trajectory, type_map=args.type_map)):
            if index == args.frame:
                frame.index_of([args.atom_id])
                return frame, [[int(args.atom_id)]], None
        return None, [], None
    driver = OnlineMonitorDriver(sections.get("monitor"), sections.get("events"), sections.get("hotspots"))
    for frame in iter_trajectory(args.trajectory, type_map=args.type_map):
        result, _ = driver.process(frame)
        if result.triggered:
            hotspots = hotspots_from_result(frame, result, sections.get("hotspots"))
            return frame, [hotspot.atom_ids for hotspot in hotspots], result.metadata.get("event_id")
    return None, [], None


def extract_from_trajectory(args: argparse.Namespace) -> tuple[Frame | None, list[ExtractedRegion]]:
    sections = load_sections(args.config)
    extraction = ExtractionConfig.from_mapping(sections.get("extraction"))
    if args.mode:
        extraction = extraction.model_copy(update={"mode": args.mode})
    frame, groups, event_id = select_hotspots(args, sections)
    if frame is None or not groups:
        return frame, []
    regions = extract_and_prepare(
        frame,
        groups,
        extraction=extraction,
        capping=CappingConfig.from_mapping(sections.get("capping")),
        embedding=EmbeddingConfig.from_mapping(sections.get("embedding")),
        event_id=event_id,
    )
    return frame, regions
