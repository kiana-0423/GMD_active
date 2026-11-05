from __future__ import annotations

import itertools

import numpy as np
import pytest

from gmd_active_learning.utils.bonds import infer_bonds
from gmd_active_learning.utils.geometry import distances_to_group, hotspot_center, minimum_image
from gmd_active_learning.utils.periodic import minimum_image_vectors

CELLS = {
    "cubic": np.eye(3) * 8.0,
    "mild_triclinic": np.array([[8.0, 0.0, 0.0], [2.0, 7.5, 0.0], [1.0, 1.5, 7.0]]),
    "strongly_skewed": np.array([[6.0, 0.0, 0.0], [5.0, 3.0, 0.0], [4.0, 2.5, 2.0]]),
}


def _brute_force(vector: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    ranges = [range(-12, 13) if flag else range(1) for flag in pbc]
    best = None
    for shift in itertools.product(*ranges):
        candidate = vector + np.asarray(shift, dtype=float) @ cell
        if best is None or np.dot(candidate, candidate) < np.dot(best, best):
            best = candidate
    assert best is not None
    return best


@pytest.mark.parametrize("name", sorted(CELLS))
@pytest.mark.parametrize("pbc", [(True, True, True), (True, True, False)])
def test_minimum_image_matches_explicit_lattice_search(name: str, pbc: tuple[bool, bool, bool]) -> None:
    cell = CELLS[name]
    rng = np.random.default_rng(0)
    vectors = rng.uniform(-15.0, 15.0, size=(200, 3))
    reduced = minimum_image_vectors(vectors, cell=cell, pbc=pbc)
    legacy = minimum_image(vectors, cell, pbc)
    inverse = np.linalg.inv(cell)
    for vector, got, got_legacy in zip(vectors, reduced, legacy):
        expected = _brute_force(vector, cell, np.asarray(pbc))
        shift = (got - vector) @ inverse  # must be an integer translation along periodic axes only
        assert np.allclose(shift, np.round(shift), atol=1e-9)
        assert np.allclose(shift[~np.asarray(pbc)], 0.0, atol=1e-9)
        assert np.linalg.norm(got) == pytest.approx(np.linalg.norm(expected), abs=1e-9)
        assert np.linalg.norm(got_legacy) == pytest.approx(np.linalg.norm(expected), abs=1e-9)


def test_hotspot_center_is_contiguous_across_the_boundary() -> None:
    positions = np.array([[0.2, 5.0, 5.0], [9.8, 5.0, 5.0], [5.0, 5.0, 5.0]])
    center = hotspot_center(positions, [0, 1], cell=np.eye(3) * 10.0, pbc=True)
    assert min(abs(center[0] - 0.0), abs(center[0] - 10.0)) == pytest.approx(0.0, abs=1e-12)
    distances = distances_to_group(positions, [0], cell=np.eye(3) * 10.0, pbc=True)
    assert distances[1] == pytest.approx(0.4)


def test_bonds_are_found_across_periodic_images_without_duplicates() -> None:
    positions = np.array([[0.3, 2.0, 2.0], [3.06, 2.0, 2.0]])  # 1.24 A apart through the x boundary of a 4 A cell
    bonds = infer_bonds(positions, np.array([6, 6]), cell=np.eye(3) * 4.0, pbc=True)
    assert bonds == [(0, 1)]
