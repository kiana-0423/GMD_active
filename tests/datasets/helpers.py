"""Small labeled-sample builders for dataset tests."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule

from gmd_active_learning.contracts import ExtractedRegion, Frame, LabelResult, ValidationStatus
from gmd_active_learning.datasets import DatasetsConfig, LabeledSample, build_sample
from gmd_active_learning.extraction import ExtractionConfig, add_h_caps, extract_cluster, extract_slab


def ethanol_frame(step: int = 0, run_id: str = "run-a") -> Frame:
    mol = molecule("CH3CH2OH")
    mol.center(vacuum=6.0)
    return Frame.from_atoms(mol, run_id=run_id, step=step)


def capped_cluster(step: int = 0, run_id: str = "run-a", event_id: str | None = "evt-1") -> tuple[Frame, ExtractedRegion]:
    frame = ethanol_frame(step, run_id)
    region = extract_cluster(frame, [0], ExtractionConfig(extract_radius=1.6, core_radius=1.2, boundary_thickness=0.3), event_id=event_id)
    return frame, add_h_caps(region, frame)


def periodic_slab_region() -> tuple[Frame, ExtractedRegion]:
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((2, 2, 2))
    atoms.pbc = (True, True, False)
    atoms.center(vacuum=5.0, axis=2)
    frame = Frame.from_atoms(atoms, run_id="run-slab", step=3)
    region = extract_slab(frame, [0], ExtractionConfig(extract_radius=4.0, core_radius=2.0, slab={"lx": 7.22, "ly": 7.22, "z_margin": 3.0}), event_id="evt-slab")
    return frame, region


def _symmetric(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(scale=0.01, size=(3, 3))
    return 0.5 * (matrix + matrix.T)


def label_for(region: ExtractedRegion, task_id: str = "task-1", *, seed: int = 0, energy: float | None = -42.5, stress: bool = False) -> LabelResult:
    rng = np.random.default_rng(seed)
    return LabelResult(
        task_id=task_id,
        region_id=region.region_id,
        forces=rng.normal(scale=0.5, size=(region.n_atoms, 3)),
        energy=energy,
        converged=True,
        validation=ValidationStatus.VALID,
        stress=_symmetric(rng) if stress else None,
        engine="cp2k",
        engine_version="2026.2",
        output_sha256="ab" * 32,
    )


def sample_for(step: int = 0, run_id: str = "run-a", event_id: str | None = "evt-1", task: str = "task-1", config: DatasetsConfig | None = None) -> LabeledSample:
    _frame, region = capped_cluster(step, run_id, event_id)
    return build_sample(region, label_for(region, task, seed=step), config or DatasetsConfig())


def single_atom_frame() -> Frame:
    return Frame.from_atoms(Atoms("Ar", positions=[[0.0, 0.0, 0.0]]), run_id="r", step=0)
