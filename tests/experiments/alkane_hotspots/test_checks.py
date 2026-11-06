"""Geometry checks reject dissociated, bonded, clashing and open-shell structures."""

from __future__ import annotations

import numpy as np

from ._helpers import ROOT  # noqa: F401
from tools.experiments.alkane_hotspots.checks import check_structure, segment_contact
from tools.experiments.alkane_hotspots.config import ChecksConfig
from tools.experiments.alkane_hotspots.geometry import TETRAHEDRAL_DEG, AlkaneTopology, apply_torsion, build_all_trans

CHECKS = ChecksConfig()


def _check(positions, topology, **kwargs):
    return check_structure(positions, topology, CHECKS, covalent_scale=1.2, ideal_angle_deg=TETRAHEDRAL_DEG, **kwargs)


def test_ideal_chain_passes_with_exactness_record() -> None:
    topology = AlkaneTopology(8)
    positions = build_all_trans(8)
    moved, _ = apply_torsion(positions, topology, (3, 4), 120.0)
    record = _check(moved, topology, reference_positions=positions, noise_free_positions=moved)
    assert record["passed"], record["failures"]
    assert record["topology"]["same_as_parent"] and record["rigid_exactness"]["max_bond_length_change"] < 1e-10


def test_dissociated_hydrogen_is_rejected() -> None:
    topology = AlkaneTopology(6)
    positions = build_all_trans(6)
    h = topology.hydrogens_of[2][0]
    positions[h] = positions[2] + 2.0 * (positions[h] - positions[2])
    record = _check(positions, topology)
    assert not record["passed"] and any(f.startswith("dissociated_bonds") for f in record["failures"])


def test_new_bond_and_clash_are_rejected() -> None:
    topology = AlkaneTopology(8)
    positions = build_all_trans(8)
    h = topology.hydrogens_of[0][0]
    positions[h] = positions[6] + np.array([0.0, 0.0, 1.1])  # H of C0 now next to C6
    record = _check(positions, topology)
    assert not record["passed"] and any(f.startswith("new_bonds") for f in record["failures"])
    clash = build_all_trans(8)
    a, b = topology.hydrogens_of[1][0], topology.hydrogens_of[5][0]
    clash[b] = clash[a] + np.array([1.2, 0.0, 0.0])
    record = _check(clash, topology)
    assert not record["passed"] and "nonbonded_HH" in record["failures"]


def test_untouched_angle_change_is_rejected_by_exactness_check() -> None:
    topology = AlkaneTopology(6)
    reference = build_all_trans(6)
    distorted = reference.copy()
    distorted[topology.hydrogens_of[3][0]] += np.array([0.0, 0.05, 0.0])
    record = _check(distorted, topology, reference_positions=reference, noise_free_positions=distorted)
    assert "rigid_angle_change" in record["failures"] and "rigid_bond_change" in record["failures"]


def test_open_shell_or_charged_requests_are_rejected() -> None:
    topology = AlkaneTopology(4)
    positions = build_all_trans(4)
    assert "not_neutral_closed_shell" in _check(positions, topology, multiplicity=3)["failures"]
    assert "not_neutral_closed_shell" in _check(positions, topology, charge=1)["failures"]


def test_contact_window_is_enforced() -> None:
    topology = AlkaneTopology(14)
    positions = build_all_trans(14)
    value, pair = segment_contact(positions, topology, 6)
    assert value > 5.0 and pair[0] < pair[1]
    record = _check(positions, topology, contact_separation=6)
    assert "contact_window" in record["failures"]
