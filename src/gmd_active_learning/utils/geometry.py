from __future__ import annotations

from typing import Iterable

import numpy as np


def minimum_image(displacement: np.ndarray, cell: np.ndarray | None, pbc: Iterable[bool]) -> np.ndarray:
    if cell is None or not any(pbc):
        return displacement
    cell_array = np.asarray(cell, dtype=float)
    inv_cell = np.linalg.inv(cell_array)
    frac = displacement @ inv_cell
    pbc_mask = np.asarray(list(pbc), dtype=bool)
    frac[..., pbc_mask] -= np.round(frac[..., pbc_mask])
    return frac @ cell_array


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
