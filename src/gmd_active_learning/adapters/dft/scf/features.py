# Ported from PHAL (hotspot_al.cp2k_ml_scf.features) at 96263b6; MIT License, see NOTICE.
"""Deterministic featurisation of SCF samples.

These features are deliberately simple and framework-free: they are computed
with numpy so the dataset can be inspected, exported and unit-tested without
PyTorch installed.  Stage 3 models consume them directly, and a model trained
on version ``FEATURE_VERSION`` must be re-trained if that string changes.

Two feature groups exist:

``composition`` / ``global`` features
    Fixed-length descriptors of the whole system.  Always available.

``per-atom`` features
    One row per atom, used by models that predict an atom-resolved density
    representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from gmd_active_learning.adapters.dft.scf.schema import SCFElectronicState, SCFSample, SCFStructure

FEATURE_VERSION = "phal-scf-features-v1"

#: Elements the composition vector resolves individually. Anything outside this
#: set is accumulated into a single "other" bin so an unexpected element widens
#: the error rather than silently colliding with a known one.
SUPPORTED_ELEMENTS: tuple[int, ...] = (1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 22, 26, 29, 30)

GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "n_atoms",
    "log_n_atoms",
    "cell_volume",
    "number_density",
    "mean_atomic_number",
    "max_atomic_number",
    "total_nuclear_charge",
    "net_charge",
    "multiplicity",
    "n_unpaired",
    "n_electrons",
    "electrons_per_atom",
    "spin_channels",
    "is_periodic",
)

PER_ATOM_FEATURE_NAMES: tuple[str, ...] = (
    "atomic_number",
    "atomic_number_scaled",
    "coordination_2p5",
    "coordination_3p5",
    "mean_neighbor_distance",
    "min_neighbor_distance",
    "local_z_sum",
    "electrons_per_atom",
    "spin_channels",
)

_COORDINATION_CUTOFFS = (2.5, 3.5)
_NEIGHBOR_CUTOFF = 5.0


@dataclass(frozen=True, slots=True)
class SCFFeatures:
    """Featurised view of one sample."""

    structure_hash: str
    composition: np.ndarray
    global_features: np.ndarray
    per_atom: np.ndarray
    feature_version: str = FEATURE_VERSION

    @property
    def n_atoms(self) -> int:
        return int(self.per_atom.shape[0])

    def flat(self) -> np.ndarray:
        """Concatenate the system-level descriptors into one vector."""

        return np.concatenate([self.composition, self.global_features])

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_version": self.feature_version,
            "structure_hash": self.structure_hash,
            "composition": self.composition.tolist(),
            "global_features": self.global_features.tolist(),
            "per_atom": self.per_atom.tolist(),
        }


def composition_feature_names() -> tuple[str, ...]:
    return tuple(f"count_Z{number}" for number in SUPPORTED_ELEMENTS) + ("count_other", "fraction_other")


def featurize_sample(sample: SCFSample) -> SCFFeatures:
    """Featurise a full sample, using only fields a predictor may see."""

    return featurize(sample.structure, sample.electronic_state, structure_hash=sample.structure_hash)


def featurize(
    structure: SCFStructure,
    electronic_state: SCFElectronicState,
    *,
    structure_hash: str | None = None,
) -> SCFFeatures:
    """Featurise a structure and its electronic state.

    The SCF trajectory is intentionally *not* an input: it is the label.
    """

    from gmd_active_learning.adapters.dft.scf.schema import structure_hash as compute_hash

    positions = np.asarray(structure.positions, dtype=float).reshape(-1, 3)
    numbers = np.asarray(structure.atomic_numbers, dtype=float)
    cell = np.asarray(structure.cell, dtype=float)

    return SCFFeatures(
        structure_hash=structure_hash or compute_hash(structure, electronic_state),
        composition=composition_vector(structure.atomic_numbers),
        global_features=_global_vector(structure, electronic_state, positions, numbers, cell),
        per_atom=_per_atom_matrix(positions, numbers, electronic_state),
    )


def composition_vector(atomic_numbers: Sequence[int]) -> np.ndarray:
    """Return per-element counts followed by the unsupported-element share."""

    counts = np.zeros(len(SUPPORTED_ELEMENTS) + 2, dtype=float)
    index_of = {number: index for index, number in enumerate(SUPPORTED_ELEMENTS)}
    other = 0
    for number in atomic_numbers:
        index = index_of.get(int(number))
        if index is None:
            other += 1
        else:
            counts[index] += 1.0
    counts[len(SUPPORTED_ELEMENTS)] = float(other)
    total = len(atomic_numbers)
    counts[len(SUPPORTED_ELEMENTS) + 1] = float(other) / total if total else 0.0
    return counts


def cell_volume(cell: np.ndarray) -> float:
    return float(abs(np.linalg.det(np.asarray(cell, dtype=float).reshape(3, 3))))


def _global_vector(
    structure: SCFStructure,
    electronic_state: SCFElectronicState,
    positions: np.ndarray,
    numbers: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    n_atoms = float(len(numbers))
    volume = cell_volume(cell)
    density = n_atoms / volume if volume > 1e-8 else 0.0
    return np.asarray(
        [
            n_atoms,
            float(np.log1p(n_atoms)),
            volume,
            density,
            float(numbers.mean()) if n_atoms else 0.0,
            float(numbers.max()) if n_atoms else 0.0,
            float(numbers.sum()),
            float(electronic_state.charge),
            float(electronic_state.multiplicity),
            float(electronic_state.multiplicity - 1),
            float(electronic_state.n_electrons),
            float(electronic_state.n_electrons) / n_atoms if n_atoms else 0.0,
            float(electronic_state.spin_channels),
            1.0 if any(structure.pbc) else 0.0,
        ],
        dtype=float,
    )


def _per_atom_matrix(
    positions: np.ndarray,
    numbers: np.ndarray,
    electronic_state: SCFElectronicState,
) -> np.ndarray:
    n_atoms = positions.shape[0]
    features = np.zeros((n_atoms, len(PER_ATOM_FEATURE_NAMES)), dtype=float)
    if n_atoms == 0:
        return features

    # Minimum-image conventions are deliberately skipped: extracted regions are
    # carved with vacuum padding, so raw distances are the physical ones.
    deltas = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    np.fill_diagonal(distances, np.inf)

    electrons_per_atom = float(electronic_state.n_electrons) / n_atoms
    within_cutoff = distances < _NEIGHBOR_CUTOFF
    for index in range(n_atoms):
        neighbors = distances[index][within_cutoff[index]]
        neighbor_numbers = numbers[within_cutoff[index]]
        features[index] = (
            float(numbers[index]),
            float(numbers[index]) / 100.0,
            float(np.count_nonzero(distances[index] < _COORDINATION_CUTOFFS[0])),
            float(np.count_nonzero(distances[index] < _COORDINATION_CUTOFFS[1])),
            float(neighbors.mean()) if neighbors.size else 0.0,
            float(distances[index].min()) if np.isfinite(distances[index]).any() else 0.0,
            float(neighbor_numbers.sum()),
            electrons_per_atom,
            float(electronic_state.spin_channels),
        )
    return features


def feature_metadata() -> dict[str, Any]:
    """Describe the feature layout so training runs can record it."""

    return {
        "feature_version": FEATURE_VERSION,
        "supported_elements": list(SUPPORTED_ELEMENTS),
        "composition_features": list(composition_feature_names()),
        "global_features": list(GLOBAL_FEATURE_NAMES),
        "per_atom_features": list(PER_ATOM_FEATURE_NAMES),
        "n_system_features": len(SUPPORTED_ELEMENTS) + 2 + len(GLOBAL_FEATURE_NAMES),
        "n_per_atom_features": len(PER_ATOM_FEATURE_NAMES),
    }


def stack_system_features(samples: Sequence[SCFSample]) -> np.ndarray:
    """Return an ``(n_samples, n_system_features)`` design matrix."""

    if not samples:
        return np.zeros((0, feature_metadata()["n_system_features"]), dtype=float)
    return np.vstack([featurize_sample(sample).flat() for sample in samples])


def trajectory_targets(samples: Sequence[SCFSample]) -> Mapping[str, np.ndarray]:
    """Return supervised targets derived from the recorded SCF trajectories."""

    iterations = np.asarray([sample.trajectory.n_iterations for sample in samples], dtype=float)
    converged = np.asarray([1.0 if sample.trajectory.converged else 0.0 for sample in samples], dtype=float)
    final_energy = np.asarray(
        [
            sample.trajectory.final_energy if sample.trajectory.final_energy is not None else np.nan
            for sample in samples
        ],
        dtype=float,
    )
    return {"n_iterations": iterations, "converged": converged, "final_energy": final_energy}


__all__ = [
    "FEATURE_VERSION",
    "GLOBAL_FEATURE_NAMES",
    "PER_ATOM_FEATURE_NAMES",
    "SUPPORTED_ELEMENTS",
    "SCFFeatures",
    "cell_volume",
    "composition_feature_names",
    "composition_vector",
    "feature_metadata",
    "featurize",
    "featurize_sample",
    "stack_system_features",
    "trajectory_targets",
]
