# Ported from PHAL (hotspot_al.monitor.geometry_monitor) at 96263b6; MIT License, see NOTICE.
"""Per-atom displacement (exact minimum image) and nearest-neighbour distance metrics (Angstrom)."""

from __future__ import annotations

import numpy as np

from gmd_active_learning.core.exceptions import DataContractError
from gmd_active_learning.utils.geometry import row_norms
from gmd_active_learning.utils.neighbors import NeighborPairs
from gmd_active_learning.utils.periodic import minimum_image_vectors

_PAIRWISE_DISTANCE_CHUNK = 512

# Reported by the exact all-pairs search when an atom has nothing to measure against. It must be
# large, never zero: rmin is scored "lower is worse".
NO_NEIGHBOR_DISTANCE = 1.0e3


def displacement_norms(
    current_positions: np.ndarray,
    previous_positions: np.ndarray | None,
    *,
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
) -> np.ndarray:
    """|MIC(r_i(t) - r_i(t_prev))| per atom; zeros without a previous frame.

    Uses the exact minimum image (also for skewed, partially periodic cells), so a genuine jump
    longer than half a cell period is folded back (only relevant for tiny cells or long intervals).
    """

    current = np.asarray(current_positions, dtype=float)
    if previous_positions is None:
        return np.zeros(len(current), dtype=float)
    previous = np.asarray(previous_positions, dtype=float)
    if previous.shape != current.shape:
        raise DataContractError(f"previous positions {previous.shape} do not match current {current.shape}")
    return row_norms(minimum_image_vectors(current - previous, cell=cell, pbc=pbc))


def minimum_neighbor_distances(pairs: NeighborPairs) -> np.ndarray:
    """Nearest-neighbour distance per atom from a neighbour list.

    Atoms without a neighbour inside the list cutoff report the cutoff itself: a true lower bound
    on their nearest distance that never scores as a close contact (thresholds must be < cutoff).
    """

    minima = np.full(pairs.n_atoms, np.inf, dtype=float)
    if len(pairs.first):
        np.minimum.at(minima, pairs.first, pairs.distances)
    minima[np.isinf(minima)] = pairs.cutoff
    return minima


def minimum_neighbor_distances_exact(
    positions: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
) -> np.ndarray:
    """Reference O(N^2) nearest minimum-image distance (no cutoff); isolated atoms -> NO_NEIGHBOR_DISTANCE."""

    positions = np.asarray(positions, dtype=float)
    n_atoms = len(positions)
    if n_atoms <= 1:
        return np.full(n_atoms, NO_NEIGHBOR_DISTANCE, dtype=float)
    minima = np.full(n_atoms, np.inf, dtype=float)
    for start in range(0, n_atoms, _PAIRWISE_DISTANCE_CHUNK):
        stop = min(start + _PAIRWISE_DISTANCE_CHUNK, n_atoms)
        displacements = minimum_image_vectors(positions[None, :, :] - positions[start:stop, None, :], cell=cell, pbc=pbc)
        distances = np.linalg.norm(displacements, axis=2)
        distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
        minima[start:stop] = np.min(distances, axis=1)
    minima[np.isinf(minima)] = NO_NEIGHBOR_DISTANCE
    return minima


__all__ = [
    "NO_NEIGHBOR_DISTANCE",
    "displacement_norms",
    "minimum_neighbor_distances",
    "minimum_neighbor_distances_exact",
]
