"""Per-atom metric functions (MON-001..005, MON-008, MON-009)."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk, molecule

from gmd_active_learning.contracts.frame import Frame
from gmd_active_learning.core.exceptions import MonitorError
from gmd_active_learning.monitors.coordination import (
    coordination_deltas,
    smooth_coordination_numbers,
    smooth_coordination_numbers_reference,
)
from gmd_active_learning.monitors.ensemble_deviation import committee_force_deviation, compute_ensemble_deviation
from gmd_active_learning.monitors.force import delta_force_norms, force_norms
from gmd_active_learning.monitors.force_disagreement import checked_reference_forces, mlip_force_deviation
from gmd_active_learning.monitors.geometry import (
    NO_NEIGHBOR_DISTANCE,
    displacement_norms,
    minimum_neighbor_distances,
    minimum_neighbor_distances_exact,
)
from gmd_active_learning.utils.neighbors import build_neighbor_pairs
from gmd_active_learning.utils.periodic import mic_displacement

SKEWED = np.array([[10.0, 2.0, 0.0], [0.0, 8.0, 1.0], [0.0, 0.0, 12.0]])


# -- MON-001 -------------------------------------------------------------------------------------


def test_force_and_delta_force_norms() -> None:
    rng = np.random.default_rng(1)
    current, previous = rng.normal(size=(7, 3)), rng.normal(size=(7, 3))
    np.testing.assert_allclose(force_norms(current), np.sqrt(np.sum(current**2, axis=1)), rtol=1e-12)
    np.testing.assert_allclose(delta_force_norms(current, previous), np.linalg.norm(current - previous, axis=1), rtol=1e-12)
    first = delta_force_norms(current, None)
    assert first.shape == (7,) and np.all(first == 0.0)


# -- MON-002 -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cell", "pbc"),
    [(None, False), (np.diag([10.0, 10.0, 10.0]), True), (SKEWED, (True, True, False))],
    ids=["non-periodic", "orthorhombic", "skewed-partial"],
)
def test_displacement_norms_match_explicit_minimum_image(cell: np.ndarray | None, pbc: object) -> None:
    rng = np.random.default_rng(12)
    previous = rng.uniform(-2.0, 12.0, size=(300, 3))
    current = previous + rng.uniform(-6.0, 6.0, size=(300, 3))
    values = displacement_norms(current, previous, cell=cell, pbc=pbc)
    expected = np.array([np.linalg.norm(mic_displacement(p, c, cell=cell, pbc=pbc)) for p, c in zip(previous, current)])
    np.testing.assert_allclose(values, expected, atol=1e-12)
    assert np.all(displacement_norms(current, None) == 0.0)


# -- MON-003 -------------------------------------------------------------------------------------


def test_rmin_equals_brute_force_for_atoms_with_neighbours() -> None:
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((3, 3, 3))
    atoms.positions += np.random.default_rng(3).normal(0.0, 0.1, atoms.positions.shape)
    atoms.positions[10] = atoms.positions[11] + [0.6, 0.0, 0.0]
    pairs = build_neighbor_pairs(atoms.positions, atoms.cell.array, atoms.pbc, 5.0)
    np.testing.assert_allclose(
        minimum_neighbor_distances(pairs),
        minimum_neighbor_distances_exact(atoms.positions, atoms.cell.array, atoms.pbc),
        atol=1e-12,
    )
    assert minimum_neighbor_distances(pairs)[10] == pytest.approx(0.6, abs=1e-12)


def test_atom_without_neighbour_reports_cutoff_not_zero() -> None:
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((3, 3, 3))
    atoms.cell[2, 2] = 120.0
    atoms.positions[0] = [5.0, 5.0, 60.0]
    pairs = build_neighbor_pairs(atoms.positions, atoms.cell.array, atoms.pbc, 6.0)
    assert minimum_neighbor_distances(pairs)[0] == 6.0
    single = minimum_neighbor_distances_exact(np.zeros((1, 3)), np.eye(3) * 9.0, True)
    assert single.tolist() == [NO_NEIGHBOR_DISTANCE]


# -- MON-004 -------------------------------------------------------------------------------------


@pytest.mark.parametrize("repeat", [(2, 2, 2), (4, 4, 4)], ids=["32-atoms", "256-atoms"])
def test_smooth_coordination_matches_reference_within_cutoff(repeat: tuple[int, int, int]) -> None:
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True).repeat(repeat)
    atoms.positions += np.random.default_rng(4).normal(0.0, 0.05, atoms.positions.shape)
    cutoff = 3.5  # below half the smallest cell period, so one image per pair
    pairs = build_neighbor_pairs(atoms.positions, atoms.cell.array, atoms.pbc, cutoff)
    q_fast = smooth_coordination_numbers(pairs, atoms.numbers, cutoff=cutoff, scale=1.1, power=8.0)
    q_ref = smooth_coordination_numbers_reference(
        atoms.positions, atoms.numbers, atoms.cell.array, atoms.pbc, cutoff=cutoff, scale=1.1, power=8.0
    )
    np.testing.assert_allclose(q_fast, q_ref, rtol=1e-10)


def test_coordination_scale_and_power_are_honoured() -> None:
    mol = molecule("C2H6")
    pairs = build_neighbor_pairs(mol.positions, None, False, 5.0)
    default = smooth_coordination_numbers(pairs, mol.numbers, cutoff=5.0)
    assert not np.allclose(default, smooth_coordination_numbers(pairs, mol.numbers, cutoff=5.0, scale=1.3))
    assert not np.allclose(default, smooth_coordination_numbers(pairs, mol.numbers, cutoff=5.0, power=12.0))


def test_coordination_deltas_zero_on_first_frame() -> None:
    q = np.array([1.0, 2.5])
    assert np.all(coordination_deltas(q, None) == 0.0)
    np.testing.assert_allclose(coordination_deltas(q, np.array([0.5, 3.0])), [0.5, 0.5])


# -- MON-008 -------------------------------------------------------------------------------------


def test_committee_deviation_matches_gmd_ensemble_deviation() -> None:
    predictions = np.random.default_rng(5).normal(size=(4, 9, 3))
    expected = np.sqrt(np.mean(np.sum((predictions - predictions.mean(axis=0)) ** 2, axis=-1), axis=0))
    np.testing.assert_allclose(committee_force_deviation(predictions), expected, rtol=1e-12)
    np.testing.assert_allclose(
        compute_ensemble_deviation(predictions).per_atom_force_deviation,
        np.linalg.norm(np.std(predictions, axis=0), axis=1),
        rtol=1e-12,
    )
    with pytest.raises(ValueError):
        committee_force_deviation(predictions[0])


# -- MON-009 -------------------------------------------------------------------------------------


def test_mlip_force_deviation_is_delta_on_one_atom() -> None:
    forces = np.random.default_rng(6).normal(size=(5, 3))
    reference = forces.copy()
    reference[2] += [0.3, -0.4, 0.0]
    deviation = mlip_force_deviation(forces, reference)
    np.testing.assert_allclose(deviation, [0.0, 0.0, 0.5, 0.0, 0.0], atol=1e-12)


def test_reference_forces_are_validated() -> None:
    frame = Frame("r", 0, np.arange(2), np.array([1, 1]), np.zeros((2, 3)), None, (False, False, False), forces=np.zeros((2, 3)))
    with pytest.raises(MonitorError, match="shape"):
        checked_reference_forces(frame, np.zeros((3, 3)))
    with pytest.raises(MonitorError, match="non-finite"):
        checked_reference_forces(frame, np.full((2, 3), np.nan))
