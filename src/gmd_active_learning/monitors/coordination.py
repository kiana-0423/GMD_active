# Ported from PHAL (hotspot_al.monitor.coordination_monitor) at 96263b6; MIT License, see NOTICE.
"""Smooth coordination numbers and their per-atom change (dimensionless).

q_i = sum_j 1 / (1 + (r_ij / (scale * (R_cov,i + R_cov,j)))^power) over neighbours with
r_ij <= coordination_cutoff. PHAL validated ``coordination_scale`` / ``coordination_smoothing_power``
but never passed them (hard-coded 1.15 / 6); here both come from config
(``metrics.delta_q.scale`` / ``.power``) and are honoured (F-16).
"""

from __future__ import annotations

import numpy as np
from ase.data import covalent_radii

from gmd_active_learning.utils.neighbors import NeighborPairs
from gmd_active_learning.utils.periodic import mic_displacements_from_reference


def smooth_coordination_numbers(
    pairs: NeighborPairs,
    numbers: np.ndarray,
    *,
    cutoff: float,
    scale: float = 1.15,
    power: float = 6.0,
) -> np.ndarray:
    """q_i from a neighbour list, restricted to ``r_ij <= cutoff``."""

    numbers = np.asarray(numbers, dtype=int)
    q_values = np.zeros(pairs.n_atoms, dtype=float)
    first, second, _vectors, distances = pairs.within(cutoff)
    if len(first) == 0:
        return q_values
    radii = scale * (covalent_radii[numbers[first]] + covalent_radii[numbers[second]])
    weights = 1.0 / (1.0 + np.power(distances / radii, power))
    np.add.at(q_values, first, weights)
    return q_values


def smooth_coordination_numbers_reference(
    positions: np.ndarray,
    numbers: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
    *,
    cutoff: float | None = None,
    scale: float = 1.15,
    power: float = 6.0,
) -> np.ndarray:
    """O(N^2) minimum-image reference (one image per pair). ``cutoff=None`` sums every pair, as PHAL's reference did."""

    positions = np.asarray(positions, dtype=float)
    numbers = np.asarray(numbers, dtype=int)
    q_values = np.zeros(len(positions), dtype=float)
    for i, position in enumerate(positions):
        distances = np.linalg.norm(mic_displacements_from_reference(position, positions, cell=cell, pbc=pbc), axis=1)
        distances[i] = np.inf
        radii = scale * (covalent_radii[numbers[i]] + covalent_radii[numbers])
        weights = 1.0 / (1.0 + np.power(distances / radii, power))
        if cutoff is not None:
            weights[distances > cutoff] = 0.0
        weights[i] = 0.0
        q_values[i] = float(np.sum(weights))
    return q_values


def coordination_deltas(current_q: np.ndarray, previous_q: np.ndarray | None) -> np.ndarray:
    """|q_i(t) - q_i(t_prev)|; zeros without a previous frame."""

    current = np.asarray(current_q, dtype=float)
    if previous_q is None:
        return np.zeros(len(current), dtype=float)
    return np.abs(current - np.asarray(previous_q, dtype=float))


__all__ = ["coordination_deltas", "smooth_coordination_numbers", "smooth_coordination_numbers_reference"]
