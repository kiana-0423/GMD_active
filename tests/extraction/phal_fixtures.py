"""Helpers to replay PHAL golden region fixtures (tests/fixtures/phal_reference)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from ase.io import read

from gmd_active_learning.contracts import Frame
from gmd_active_learning.extraction import CappingConfig, ExtractionConfig

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "phal_reference"
SCENARIOS = ("ethane_box", "polyethylene_chains", "pt111_co_slab")
MODES = ("cluster", "slab", "graph", "block")

#: input_frames.extxyz stores positions with 8 decimals; PHAL ran on full-precision positions.
FIXTURE_POSITION_ATOL = 5.0e-8


@lru_cache(maxsize=None)
def phal_config(scenario: str) -> dict[str, Any]:
    return json.loads((ROOT / scenario / "config.json").read_text())


@lru_cache(maxsize=None)
def _frames(scenario: str) -> dict[int, Any]:
    return {int(atoms.info["step"]): atoms for atoms in read(ROOT / scenario / "input_frames.extxyz", ":")}


def frame_at(scenario: str, step: int) -> Frame:
    """Atom IDs are the 0-based fixture indices (the input carries no ids)."""

    return Frame.from_atoms(_frames(scenario)[step].copy(), run_id=scenario, step=step)


def extraction_config(scenario: str, mode: str, **overrides: Any) -> ExtractionConfig:
    section = json.loads(json.dumps(phal_config(scenario)["extraction"]))
    section["block"].pop("label_region", None)  # schema key PHAL never read
    section["mode"] = mode
    section.update(overrides)
    return ExtractionConfig.from_mapping(section)


def capping_config(scenario: str) -> CappingConfig:
    return CappingConfig.from_mapping(phal_config(scenario)["h_capping"])


@lru_cache(maxsize=None)
def region_records(scenario: str, event: str, mode: str) -> list[dict[str, Any]]:
    return json.loads((ROOT / scenario / "regions" / event / f"{mode}.json").read_text())


def events(scenario: str) -> list[str]:
    return sorted(path.name for path in (ROOT / scenario / "regions").iterdir())


def all_cases() -> list[tuple[str, str, str, int]]:
    cases = []
    for scenario in SCENARIOS:
        for event in events(scenario):
            for mode in MODES:
                for index, _ in enumerate(region_records(scenario, event, mode)):
                    cases.append((scenario, event, mode, index))
    return cases


def hotspots_for(scenario: str, event: str, mode: str, index: int) -> list[int]:
    """The hotspot atoms PHAL extracted around, in PHAL's order.

    Cluster regions always contain their hotspot atoms (in input order), so the cluster record of
    the same event/hotspot is the source for cluster, slab and graph; block mode stores its
    anomalous atoms in metadata.
    """

    if mode == "block":
        return list(region_records(scenario, event, mode)[index]["raw_region"]["metadata"]["ood_atom_indices"])
    cluster = region_records(scenario, event, "cluster")[index]["raw_region"]
    return [cluster["original_indices"][local] for local in cluster["hotspot_indices"]]
