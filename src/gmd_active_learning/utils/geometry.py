from __future__ import annotations

from typing import Iterable

import numpy as np

from gmd_active_learning.utils.periodic import mic_displacements_from_reference, minimum_image_vectors


def minimum_image(displacement: np.ndarray, cell: np.ndarray | None, pbc: Iterable[bool]) -> np.ndarray:
    """Minimum-image displacement; exact for skewed (triclinic) cells as well as orthorhombic ones."""

    return minimum_image_vectors(displacement, cell=cell, pbc=np.asarray(list(pbc), dtype=bool))


def pairwise_distances(
    positions: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: Iterable[bool] = (False, False, False),
) -> tuple[np.ndarray, np.ndarray]:
    n_atoms = len(positions)
    distances: list[float] = []
    vectors: list[np.ndarray] = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            vec = minimum_image(np.asarray(positions[j]) - np.asarray(positions[i]), cell, pbc)
            distances.append(float(np.linalg.norm(vec)))
            vectors.append(vec)
    return np.asarray(distances, dtype=float), np.asarray(vectors, dtype=float)


def pair_distance_histogram(
    positions: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: Iterable[bool] = (False, False, False),
    cutoff: float = 6.0,
    bins: int = 16,
) -> np.ndarray:
    distances, _ = pairwise_distances(positions, cell=cell, pbc=pbc)
    if distances.size == 0:
        return np.zeros(bins, dtype=float)
    hist, _ = np.histogram(distances, bins=bins, range=(0.0, cutoff), density=False)
    hist = hist.astype(float)
    total = hist.sum()
    return hist / total if total > 0 else hist


def coordination_number_summary(
    positions: np.ndarray,
    cutoff: float,
    cell: np.ndarray | None = None,
    pbc: Iterable[bool] = (False, False, False),
) -> np.ndarray:
    n_atoms = len(positions)
    counts = np.zeros(n_atoms, dtype=float)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            vec = minimum_image(np.asarray(positions[j]) - np.asarray(positions[i]), cell, pbc)
            if np.linalg.norm(vec) <= cutoff:
                counts[i] += 1.0
                counts[j] += 1.0
    return np.asarray([counts.min(initial=0.0), counts.mean() if counts.size else 0.0, counts.max(initial=0.0)])


# --- Ported from PHAL (hotspot_al.utils.geometry) at 96263b6; MIT License, see NOTICE. ---
_DISTANCE_MATRIX_GROUP_CHUNK = 128


def row_norms(values: np.ndarray) -> np.ndarray:
    """Compute vector norms row-wise."""

    array = np.asarray(values, dtype=float)
    return np.linalg.norm(array, axis=-1)


def hotspot_center(
    positions: np.ndarray,
    hotspot_indices: list[int],
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
) -> np.ndarray:
    """Compute a contiguous centroid for hotspot atoms under PBC."""

    if not hotspot_indices:
        msg = "Cannot compute a hotspot center for an empty index list."
        raise ValueError(msg)
    anchor = positions[hotspot_indices[0]]
    displacements = mic_displacements_from_reference(anchor, positions[hotspot_indices], cell=cell, pbc=pbc)
    return anchor + np.mean(displacements, axis=0)


def distances_to_group(
    positions: np.ndarray,
    group_indices: list[int],
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
) -> np.ndarray:
    """Return the minimum distance from each atom to a group of atoms."""

    positions = np.asarray(positions, dtype=float)
    distances = np.full(len(positions), np.inf, dtype=float)
    if not group_indices:
        return distances
    group = positions[np.asarray(group_indices, dtype=int)]
    for start in range(0, len(group), _DISTANCE_MATRIX_GROUP_CHUNK):
        references = group[start : start + _DISTANCE_MATRIX_GROUP_CHUNK]
        displacements = minimum_image_vectors(positions[None, :, :] - references[:, None, :], cell=cell, pbc=pbc)
        distances = np.minimum(distances, np.min(np.linalg.norm(displacements, axis=2), axis=0))
    return distances


def padded_cluster_cell(positions: np.ndarray, padding: float = 4.0) -> np.ndarray:
    """Create a non-periodic orthorhombic cell around positions."""

    positions = np.asarray(positions, dtype=float)
    mins = positions.min(axis=0) - padding
    maxs = positions.max(axis=0) + padding
    lengths = np.maximum(maxs - mins, padding)
    return np.diag(lengths)
