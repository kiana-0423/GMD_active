"""Exact all-trans construction and rigid torsion/bend operations."""

from __future__ import annotations

import numpy as np
import pytest

from ._helpers import ROOT  # noqa: F401 - puts the repository root on sys.path
from gmd_active_learning.utils.bonds import infer_bonds
from tools.experiments.alkane_hotspots.geometry import (
    TETRAHEDRAL_DEG,
    AlkaneTopology,
    angle_deg,
    apply_bend,
    apply_torsion,
    build_all_trans,
    dihedral_deg,
    distance,
    gaussian_displacement,
    wrap_deg,
)


@pytest.mark.parametrize("n", [2, 3, 8, 16, 33])
def test_all_trans_has_exact_bonds_angles_and_topology(n: int) -> None:
    topology = AlkaneTopology(n)
    positions = build_all_trans(n)
    assert topology.n_atoms == 3 * n + 2 and topology.formula == f"C{n}H{2 * n + 2}"
    for i, j in topology.bonds:
        expected = 1.53 if topology.numbers[i] == topology.numbers[j] == 6 else 1.09
        assert distance(positions, i, j) == pytest.approx(expected, abs=1e-12)
    for triple in topology.angles:
        assert angle_deg(positions, *triple) == pytest.approx(TETRAHEDRAL_DEG, abs=1e-9)
    assert infer_bonds(positions, topology.numbers) == topology.bonds
    for c in range(n - 3):
        assert abs(dihedral_deg(positions, c, c + 1, c + 2, c + 3)) == pytest.approx(180.0, abs=1e-9)


def test_atom_order_carbons_first_then_hydrogens_by_carbon() -> None:
    topology = AlkaneTopology(5)
    assert list(topology.numbers[:5]) == [6] * 5 and set(topology.numbers[5:]) == {1}
    assert [len(h) for h in topology.hydrogens_of] == [3, 2, 2, 2, 3]
    assert topology.hydrogens_of[0] == [5, 6, 7] and topology.carbon_of[8] == 1


@pytest.mark.parametrize("delta", [120.0, -120.0, 60.0, 37.5, 180.0])
def test_torsion_changes_every_dihedral_about_the_bond_and_nothing_else(delta: float) -> None:
    n, k = 10, 4
    topology = AlkaneTopology(n)
    before = build_all_trans(n) + gaussian_displacement(topology.n_atoms, 0.02, 3)
    after, moved = apply_torsion(before, topology, (k, k + 1), delta)
    assert k + 1 in moved and k not in moved and all(c in moved for c in range(k + 1, n))
    for i, j in topology.bonds:
        assert distance(after, i, j) == pytest.approx(distance(before, i, j), abs=1e-12)
    for triple in topology.angles:
        assert angle_deg(after, *triple) == pytest.approx(angle_deg(before, *triple), abs=1e-9)
    left = [a for a in topology.adjacency[k] if a != k + 1]
    right = [a for a in topology.adjacency[k + 1] if a != k]
    for a in left:
        for d in right:
            change = wrap_deg(dihedral_deg(after, a, k, k + 1, d) - dihedral_deg(before, a, k, k + 1, d))
            assert change == pytest.approx(wrap_deg(delta), abs=1e-8)
    # Dihedrals about other bonds are untouched.
    for c in range(n - 3):
        if c + 1 != k:
            assert dihedral_deg(after, c, c + 1, c + 2, c + 3) == pytest.approx(dihedral_deg(before, c, c + 1, c + 2, c + 3), abs=1e-8)


@pytest.mark.parametrize("delta", [-15.0, 10.0, 25.0])
def test_bend_changes_only_the_angles_at_the_vertex_towards_the_moved_side(delta: float) -> None:
    n, k = 9, 4
    topology = AlkaneTopology(n)
    before = build_all_trans(n)
    after, _ = apply_bend(before, topology, k, delta)
    for i, j in topology.bonds:
        assert distance(after, i, j) == pytest.approx(distance(before, i, j), abs=1e-12)
    assert angle_deg(after, k - 1, k, k + 1) - angle_deg(before, k - 1, k, k + 1) == pytest.approx(delta, abs=1e-9)
    intended = {(k - 1, k, k + 1)} | {(h, k, k + 1) for h in topology.hydrogens_of[k]}
    changed = set()
    for triple in topology.angles:
        if abs(angle_deg(after, *triple) - angle_deg(before, *triple)) > 1e-9:
            i, c, j = triple
            changed.add(triple if triple in intended else (j, c, i))
    assert changed <= intended and (k - 1, k, k + 1) in changed


def test_bend_rejects_terminal_vertex_and_torsion_rejects_non_bond() -> None:
    topology = AlkaneTopology(5)
    positions = build_all_trans(5)
    with pytest.raises(ValueError):
        apply_bend(positions, topology, 0, 10.0)
    with pytest.raises(ValueError):
        apply_torsion(positions, topology, (0, 2), 10.0)


def test_gaussian_displacement_is_seeded() -> None:
    assert np.array_equal(gaussian_displacement(10, 0.03, 5), gaussian_displacement(10, 0.03, 5))
    assert not np.array_equal(gaussian_displacement(10, 0.03, 5), gaussian_displacement(10, 0.03, 6))
    assert np.std(gaussian_displacement(10000, 0.03, 1)) == pytest.approx(0.03, rel=0.05)
