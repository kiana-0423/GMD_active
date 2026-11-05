"""Generate the tiny seeded synthetic H2O dataset fixture (ledger FIX-002, dataset part).

Toy harmonic bond/angle model, NOT DFT: no physical accuracy is claimed. The file is in the exact
layout GMD_active publishes (GMD_SGT keys force_weight / energy_weight / stress_weight /
split_group plus provenance) so model fixtures can be trained from it.

    python tests/fixtures/datasets/make_tiny_h2o.py   # rewrites tiny_h2o.extxyz deterministically
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

from gmd_active_learning.contracts import Frame, LabelResult, ValidationStatus
from gmd_active_learning.datasets import DatasetsConfig, LabeledSample, build_sample, write_samples_extxyz
from gmd_active_learning.extraction import ExtractionConfig, extract_cluster

SEED = 20261106
N_FRAMES = 8
N_EVENTS = 4
K_BOND, R0 = 25.0, 0.9572  # eV/A^2, A
K_ANGLE, THETA0 = 3.0, np.deg2rad(104.52)  # eV/rad^2
OUTPUT = Path(__file__).resolve().parent / "tiny_h2o.extxyz"


def toy_energy(positions: np.ndarray) -> float:
    o, h1, h2 = positions
    r1, r2 = h1 - o, h2 - o
    d1, d2 = np.linalg.norm(r1), np.linalg.norm(r2)
    theta = np.arccos(np.clip(r1 @ r2 / (d1 * d2), -1.0, 1.0))
    return float(0.5 * K_BOND * ((d1 - R0) ** 2 + (d2 - R0) ** 2) + 0.5 * K_ANGLE * (theta - THETA0) ** 2)


def toy_forces(positions: np.ndarray, step: float = 1.0e-6) -> np.ndarray:
    forces = np.zeros_like(positions)
    for atom in range(3):
        for axis in range(3):
            plus, minus = positions.copy(), positions.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            forces[atom, axis] = -(toy_energy(plus) - toy_energy(minus)) / (2.0 * step)
    return forces


def build() -> list[LabeledSample]:
    rng = np.random.default_rng(SEED)
    base = np.array([[0.0, 0.0, 0.0], [R0, 0.0, 0.0], [R0 * np.cos(THETA0), R0 * np.sin(THETA0), 0.0]]) + 5.0
    config = DatasetsConfig(energy={"nonperiodic_weight": 1.0})
    samples = []
    for k in range(N_FRAMES):
        positions = np.round(base + rng.normal(scale=0.05, size=base.shape), 6)
        frame = Frame(run_id="tiny-h2o", step=k, atom_ids=[0, 1, 2], numbers=[8, 1, 1], positions=positions, cell=np.eye(3) * 10.0, pbc=(False, False, False))
        region = extract_cluster(frame, [0, 1, 2], ExtractionConfig(extract_radius=2.0), event_id=f"evt-{k % N_EVENTS}")
        label = LabelResult(
            task_id=f"toy-{k}",
            region_id=region.region_id,
            forces=toy_forces(frame.positions),
            energy=toy_energy(frame.positions),
            converged=True,
            validation=ValidationStatus.VALID,
            engine="toy_harmonic",
        )
        samples.append(build_sample(region, label, config))
    return samples


def main() -> Path:
    return write_samples_extxyz(build(), OUTPUT)


if __name__ == "__main__":
    print(main())
