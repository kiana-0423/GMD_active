"""Shared monitor neighbour list (MON-005): equals a brute-force image search within the cutoff."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from gmd_active_learning.core.exceptions import DataContractError
from gmd_active_learning.utils.neighbors import MonitorNeighbors, build_neighbor_pairs


def _brute_force(positions: np.ndarray, cell: np.ndarray | None, pbc: tuple[bool, bool, bool], cutoff: float) -> set[tuple[int, int, tuple[float, ...]]]:
    pairs = set()
    ranges = [range(-3, 4) if flag else range(1) for flag in pbc]
    shifts = [np.asarray(shift, dtype=float) @ cell for shift in itertools.product(*ranges)] if cell is not None else [np.zeros(3)]
    for i, j in itertools.product(range(len(positions)), repeat=2):
        for shift in shifts:
            vector = positions[j] + shift - positions[i]
            distance = float(np.linalg.norm(vector))
            if 1e-8 < distance < cutoff:
                pairs.add((i, j, tuple(np.round(vector, 9))))
    return pairs


@pytest.mark.parametrize(
    ("cell", "pbc"),
    [
        (None, (False, False, False)),
        (np.diag([7.0, 8.0, 9.0]), (True, True, True)),
        (np.array([[7.0, 0.0, 0.0], [2.5, 6.5, 0.0], [1.0, 1.5, 7.5]]), (True, True, True)),
        (np.array([[6.0, 0.0, 0.0], [5.0, 3.0, 0.0], [4.0, 2.5, 9.0]]), (True, True, False)),
    ],
    ids=["open", "orthorhombic", "triclinic", "skewed-partial"],
)
def test_pairs_equal_brute_force_image_search(cell: np.ndarray | None, pbc: tuple[bool, bool, bool]) -> None:
    rng = np.random.default_rng(4)
    positions = rng.uniform(0.0, 6.0, size=(14, 3))
    cutoff = 4.0
    pairs = build_neighbor_pairs(positions, cell, pbc, cutoff)
    found = {(int(i), int(j), tuple(np.round(v, 9))) for i, j, v in zip(pairs.first, pairs.second, pairs.vectors)}
    assert found == _brute_force(positions, cell, pbc, cutoff)
    np.testing.assert_allclose(pairs.distances, np.linalg.norm(pairs.vectors, axis=1))
    for index in range(len(positions)):
        indices, vectors, distances = pairs.neighbors_of(index, 3.0)
        assert np.all(distances <= 3.0)
        lattice_shift = vectors - (positions[indices] - positions[index])
        if cell is None:
            np.testing.assert_allclose(lattice_shift, 0.0, atol=1e-12)
        else:
            coefficients = lattice_shift @ np.linalg.inv(cell)
            np.testing.assert_allclose(coefficients, np.round(coefficients), atol=1e-9)


def test_vectors_point_from_i_to_j() -> None:
    pairs = build_neighbor_pairs(np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]), None, (False,) * 3, 3.0)
    indices, vectors, distances = pairs.neighbors_of(0)
    assert indices.tolist() == [1] and vectors.tolist() == [[1.5, 0.0, 0.0]] and distances.tolist() == [1.5]


def test_monitor_neighbors_rebuilds_for_new_atom_counts() -> None:
    neighbors = MonitorNeighbors(lj_cutoff=4.0, coordination_cutoff=5.0)
    assert neighbors.cutoff == 5.0
    with pytest.raises(DataContractError):
        _ = neighbors.pairs
    neighbors.rebuild(np.zeros((1, 3)) + [[0.0, 0.0, 0.0]], None, (False,) * 3)
    assert neighbors.n_atoms == 1 and len(neighbors.pairs.first) == 0
    neighbors.rebuild(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), None, (False,) * 3)
    assert neighbors.n_atoms == 3 and neighbors.builds == 2
    indices, _vectors, _distances = neighbors.get_displacements(0, 1.5)
    assert indices.tolist() == [1]
    with pytest.raises(DataContractError):
        build_neighbor_pairs(np.zeros((2, 3)), None, (True, True, True), 3.0)
