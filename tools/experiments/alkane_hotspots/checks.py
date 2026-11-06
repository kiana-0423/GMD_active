"""Geometry acceptance checks for generated alkane parents (thresholds from ``checks`` in the YAML).

A structure is accepted only if every check passes; the full record (values and thresholds) is
stored with the parent, and rejected candidates are written to ``rejected.jsonl`` with the same
record, so nothing is silently dropped.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ase.data import covalent_radii

from gmd_active_learning.utils.bonds import infer_bonds
from tools.experiments.alkane_hotspots.config import ChecksConfig
from tools.experiments.alkane_hotspots.geometry import AlkaneTopology, angle_deg, distance


def pair_distances(positions: np.ndarray) -> np.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def nonbonded_minima(positions: np.ndarray, topology: AlkaneTopology, min_separation: int) -> dict[str, float]:
    """Minimum distance per species pair over atom pairs at least ``min_separation`` bonds apart."""

    distances = pair_distances(positions)
    far = topology.graph_distances >= int(min_separation)
    numbers = topology.numbers
    result: dict[str, float] = {}
    for name, (za, zb) in {"HH": (1, 1), "CH": (6, 1), "CC": (6, 6)}.items():
        mask = far & (((numbers[:, None] == za) & (numbers[None, :] == zb)) | ((numbers[:, None] == zb) & (numbers[None, :] == za)))
        result[name] = float(distances[mask].min()) if np.any(mask) else float("inf")
    return result


def segment_contact(positions: np.ndarray, topology: AlkaneTopology, min_separation: int) -> tuple[float, tuple[int, int]]:
    """Minimum distance between atoms whose carbons are at least ``min_separation`` apart along the chain."""

    carbon_index = np.asarray([i if i < topology.n_carbons else topology.carbon_of[i] for i in range(topology.n_atoms)])
    mask = np.abs(carbon_index[:, None] - carbon_index[None, :]) >= int(min_separation)
    if not np.any(mask):
        return float("inf"), (-1, -1)
    distances = np.where(mask, pair_distances(positions), np.inf)
    flat = int(np.argmin(distances))
    i, j = divmod(flat, topology.n_atoms)
    return float(distances[i, j]), (min(i, j), max(i, j))


def check_structure(
    positions: np.ndarray,
    topology: AlkaneTopology,
    checks: ChecksConfig,
    *,
    covalent_scale: float,
    ideal_angle_deg: float,
    intended_angles: list[tuple[int, int, int]] | None = None,
    reference_positions: np.ndarray | None = None,
    noise_free_positions: np.ndarray | None = None,
    contact_separation: int | None = None,
    charge: int = 0,
    multiplicity: int = 1,
) -> dict[str, Any]:
    """All checks for one candidate; ``record["passed"]`` is the conjunction.

    ``reference_positions`` (the unperturbed, noise-free base) and ``noise_free_positions`` (the
    perturbed structure before thermal noise) enable the exactness checks of the rigid operations.
    ``intended_angles`` are excluded from the untouched-angle checks.
    """

    positions = np.asarray(positions, dtype=float)
    intended = {tuple(item) for item in (intended_angles or [])}
    intended |= {(j, c, i) for i, c, j in intended}
    failures: list[str] = []
    record: dict[str, Any] = {}

    # Topology: no dissociation, no new bond.
    bonds = infer_bonds(positions, topology.numbers, scale=covalent_scale)
    expected = topology.bonds
    missing = sorted(set(expected) - set(bonds))
    extra = sorted(set(bonds) - set(expected))
    record["topology"] = {"same_as_parent": not missing and not extra, "missing_bonds": missing[:20], "new_bonds": extra[:20]}
    if missing:
        failures.append(f"dissociated_bonds:{len(missing)}")
    if extra:
        failures.append(f"new_bonds:{len(extra)}")

    # Bond lengths against covalent radii.
    deviations = [abs(distance(positions, i, j) - float(covalent_radii[topology.numbers[i]] + covalent_radii[topology.numbers[j]])) for i, j in expected]
    max_bond_dev = float(max(deviations)) if deviations else 0.0
    record["bond_length"] = {"max_deviation_from_covalent_sum": max_bond_dev, "tolerance": checks.bond_tolerance}
    if max_bond_dev > checks.bond_tolerance:
        failures.append("bond_length")

    # Untouched angles against the ideal angle (after noise).
    untouched = [a for a in topology.angles if a not in intended]
    angle_devs = [abs(angle_deg(positions, *a) - ideal_angle_deg) for a in untouched]
    max_angle_dev = float(max(angle_devs)) if angle_devs else 0.0
    record["angles"] = {"max_untouched_deviation_deg": max_angle_dev, "tolerance_deg": checks.max_angle_deviation_deg, "n_intended": len(intended) // 2}
    if max_angle_dev > checks.max_angle_deviation_deg:
        failures.append("angle_deviation")

    # Exactness of the rigid operation (before noise).
    if reference_positions is not None and noise_free_positions is not None:
        bond_change = max(abs(distance(noise_free_positions, i, j) - distance(reference_positions, i, j)) for i, j in expected)
        angle_change = max(
            (abs(angle_deg(noise_free_positions, *a) - angle_deg(reference_positions, *a)) for a in untouched), default=0.0
        )
        record["rigid_exactness"] = {
            "max_bond_length_change": float(bond_change),
            "max_untouched_angle_change_deg": float(angle_change),
            "tolerance": checks.exact_tolerance,
            "angle_tolerance_deg": checks.exact_angle_tolerance_deg,
        }
        if bond_change > checks.exact_tolerance:
            failures.append("rigid_bond_change")
        if angle_change > checks.exact_angle_tolerance_deg:
            failures.append("rigid_angle_change")

    # Nonbonded clashes.
    minima = nonbonded_minima(positions, topology, checks.nonbonded_min_bond_separation)
    limits = {"HH": checks.min_nonbonded_hh, "CH": checks.min_nonbonded_ch, "CC": checks.min_nonbonded_cc}
    record["nonbonded"] = {"minima": minima, "limits": limits, "min_bond_separation": checks.nonbonded_min_bond_separation}
    for key, limit in limits.items():
        if minima[key] < limit:
            failures.append(f"nonbonded_{key}")

    # Contact window (contact family only).
    if contact_separation is not None:
        value, pair = segment_contact(positions, topology, contact_separation)
        low, high = checks.contact_window
        record["contact"] = {"distance": value, "pair": list(pair), "window": [low, high], "min_segment_separation": contact_separation}
        if not low <= value <= high:
            failures.append("contact_window")

    # Neutral closed shell.
    n_electrons = int(topology.numbers.sum()) - int(charge)
    closed = charge == 0 and multiplicity == 1 and n_electrons % 2 == 0
    n_c = int(np.sum(topology.numbers == 6))
    n_h = int(np.sum(topology.numbers == 1))
    record["electronic"] = {"charge": charge, "multiplicity": multiplicity, "n_electrons": n_electrons, "saturated": n_h == 2 * n_c + 2}
    if not closed:
        failures.append("not_neutral_closed_shell")
    if n_h != 2 * n_c + 2:
        failures.append("not_saturated")

    record["failures"] = failures
    record["passed"] = not failures
    return record


__all__ = ["check_structure", "nonbonded_minima", "pair_distances", "segment_contact"]
