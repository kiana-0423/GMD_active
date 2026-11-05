"""Small synthetic frame factories for monitor tests (no MD engine involved)."""

from __future__ import annotations

from typing import Any

import numpy as np
from ase.build import molecule

from gmd_active_learning.contracts.frame import Frame

H4_POSITIONS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def h4_frame(step: int, force_scale: float, *, run_id: str = "run") -> Frame:
    """PHAL's test_online_monitor frame: four H atoms, force on atom 0 only."""

    forces = np.zeros((4, 3))
    forces[0, 0] = force_scale
    return Frame(run_id, step, np.arange(4), np.ones(4, dtype=int), H4_POSITIONS.copy(), np.eye(3) * 8.0, (False,) * 3, forces=forces)


class CountingForces:
    """Reference-force provider returning zeros (or a fixed array), counting calls."""

    def __init__(self, fail_first: bool = False) -> None:
        self.calls = 0
        self.fail_first = fail_first

    def __call__(self, frame: Frame) -> np.ndarray:
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("inference failed")
        return np.zeros((frame.n_atoms, 3))


class CountingCommittee:
    def __init__(self, members: int = 2) -> None:
        self.calls = 0
        self.members = members

    def __call__(self, frame: Frame) -> np.ndarray:
        self.calls += 1
        return np.zeros((self.members, frame.n_atoms, 3))


def ethane_box_frames(
    n_frames: int = 12,
    *,
    seed: int = 7,
    burst: dict[int, tuple[int, list[float]]] | None = None,
    committee_members: int = 3,
    run_id: str = "ethane",
    step_stride: int = 10,
) -> list[Frame]:
    """2x2x2 ethane box with a harmonic random walk, optional force bursts and committee forces."""

    rng = np.random.default_rng(seed)
    spacing, repeats = 4.6, 2
    numbers: list[int] = []
    positions: list[np.ndarray] = []
    for i in range(repeats):
        for j in range(repeats):
            for k in range(repeats):
                mol = molecule("C2H6")
                mol.translate(np.array([i + 0.5, j + 0.5, k + 0.5]) * spacing - mol.get_center_of_mass())
                numbers.extend(mol.numbers.tolist())
                positions.extend(mol.positions)
    reference = np.asarray(positions)
    cell = np.eye(3) * spacing * repeats
    drift = np.zeros_like(reference)
    frames = []
    for index in range(n_frames):
        drift += rng.normal(0.0, 0.01, size=drift.shape)
        forces = -5.0 * drift + rng.normal(0.0, 0.05, size=drift.shape)
        if burst and index in burst:
            atom, vector = burst[index]
            forces[atom] += np.asarray(vector)
        committee = None
        if committee_members:
            committee = forces[None] + rng.normal(0.0, 0.02, size=(committee_members, *drift.shape))
        frames.append(
            Frame(
                run_id,
                index * step_stride,
                np.arange(len(numbers)),
                np.asarray(numbers),
                reference + drift,
                cell,
                (True, True, True),
                forces=forces,
                committee_forces=committee,
            )
        )
    return frames


def staged_config(**overrides: Any) -> dict[str, Any]:
    """PHAL test_online_monitor configuration translated to the GMD_active schema."""

    metrics: dict[str, Any] = {
        name: {"weight": 0.0}
        for name in ("force", "delta_force", "rmin", "delta_q", "lj_residual", "committee", "displacement")
    }
    metrics["mlip_force_deviation"] = {"weight": 2.0}
    metrics["force"]["z_threshold"] = None
    metrics["lj_residual"]["lazy_threshold"] = 100.0
    metrics["rmin"]["threshold"] = None
    config: dict[str, Any] = {
        "neighbors": {"lj_cutoff": 4.0, "coordination_cutoff": 4.0},
        "metrics": metrics,
        "stages": {"screen_threshold": 1.0, "physics_threshold": 1.0, "label_threshold": 1.0, "scheduled_full_interval": 2},
        "running_stats": {"warmup_frames": 0, "min_std": 1.0},
        "actions": {"stop_min_distance": None},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config
