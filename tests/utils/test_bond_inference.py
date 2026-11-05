"""Covalent bond inference (UTIL-003)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.build import molecule

from gmd_active_learning.utils.bonds import bonded_neighbors, infer_bonds

REF_SRC = Path(__file__).resolve().parents[2] / "ref" / "lammps_monitor" / "src"


def test_ethane_has_seven_bonds_and_symmetric_adjacency() -> None:
    mol = molecule("C2H6")
    bonds = infer_bonds(mol.positions, mol.numbers)
    assert len(bonds) == 7 and all(i < j for i, j in bonds)
    carbons = [index for index, number in enumerate(mol.numbers) if number == 6]
    assert tuple(carbons) in bonds
    adjacency = bonded_neighbors(mol.positions, mol.numbers)
    assert sorted(len(adjacency[c]) for c in carbons) == [4, 4]
    assert all(i in adjacency[j] for i, neighbours in adjacency.items() for j in neighbours)


def test_pair_bonded_through_two_images_is_reported_once() -> None:
    # 2.4 A cell along x: the two C atoms are 1.2 A apart directly and through the boundary.
    positions = np.array([[0.0, 2.0, 2.0], [1.2, 2.0, 2.0]])
    assert infer_bonds(positions, np.array([6, 6]), cell=np.diag([2.4, 6.0, 6.0]), pbc=True) == [(0, 1)]


def test_short_inputs() -> None:
    assert infer_bonds(np.zeros((1, 3)), np.array([1])) == []


@pytest.mark.skipif(not (REF_SRC / "hotspot_al").is_dir(), reason="PHAL reference tree not available")
def test_bond_sets_match_phal_on_periodic_and_molecular_fixtures() -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_phal_neighbor", REF_SRC / "hotspot_al" / "utils" / "neighbor.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(2)
    for mol_name in ("C2H6", "CH3CH2OH", "C6H6"):
        mol = molecule(mol_name)
        assert infer_bonds(mol.positions, mol.numbers) == module.infer_bonds(mol.positions, mol.numbers)
    positions = rng.uniform(0.0, 6.0, size=(30, 3))
    numbers = rng.choice([1, 6, 8], size=30)
    cell = np.array([[6.0, 0.0, 0.0], [1.0, 6.0, 0.0], [0.0, 0.5, 6.0]])
    assert infer_bonds(positions, numbers, cell=cell, pbc=True) == module.infer_bonds(positions, numbers, cell=cell, pbc=True)
    sys.modules.pop("_phal_neighbor", None)
