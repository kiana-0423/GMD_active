"""Helpers to replay the PHAL golden fixtures (tests/fixtures/phal_reference) through GMD_active."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read

from gmd_active_learning.contracts.frame import Frame
from gmd_active_learning.monitors.lj_local import LocalLJEstimator, LocalLJFit
from gmd_active_learning.utils.neighbors import NeighborPairs

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "phal_reference"
SCENARIOS = ("ethane_box", "polyethylene_chains", "pt111_co_slab")


@dataclass
class Scenario:
    name: str
    config: dict[str, Any]
    frames: list[Frame]
    results: list[dict[str, Any]]
    events: list[dict[str, Any]]
    arrays: dict[str, np.ndarray]


def load_scenario(name: str) -> Scenario:
    root = FIXTURES / name
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    committee_size = len(config["allegro"]["model_paths"])
    frames = []
    for atoms in read(root / "input_frames.extxyz", index=":"):
        committee = np.stack([atoms.arrays[f"committee_forces_{member}"] for member in range(committee_size)])
        frames.append(
            Frame(
                run_id=f"phal-{name}",
                step=int(atoms.info["step"]),
                atom_ids=np.arange(len(atoms)),
                numbers=atoms.get_atomic_numbers(),
                positions=atoms.get_positions(),
                cell=atoms.cell.array.copy(),
                pbc=tuple(bool(flag) for flag in atoms.pbc),
                forces=atoms.get_forces(),
                time_fs=float(atoms.info["time"]),
                committee_forces=committee,
            )
        )
    with np.load(root / "monitor_arrays.npz") as data:
        arrays = {key: data[key] for key in data.files}
    return Scenario(
        name=name,
        config=config,
        frames=frames,
        results=json.loads((root / "monitor_results.json").read_text(encoding="utf-8")),
        events=json.loads((root / "events.json").read_text(encoding="utf-8")),
        arrays=arrays,
    )


def primary_model_forces(frame: Frame) -> np.ndarray:
    """PHAL's primary MLIP = committee member 0 (stored in the fixture)."""

    assert frame.committee_forces is not None
    return frame.committee_forces[0].copy()


class NegatedForceLJ:
    """PHAL's inverted LJ convention expressed through the corrected estimator: corrected(-F) == PHAL(F)."""

    def __init__(self, estimator: LocalLJEstimator) -> None:
        self.estimator = estimator
        self.calls = 0

    def __call__(self, pairs: NeighborPairs, forces: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, list[LocalLJFit]]:
        self.calls += 1
        return self.estimator(pairs, -np.asarray(forces, dtype=float), mask)
