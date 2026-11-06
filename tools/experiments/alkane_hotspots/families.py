"""Deterministic parent generation: base conformations x perturbation families, with checks.

A *base conformation* is one chain length plus one seeded thermal displacement field; it is the split
group. Its parents are the unperturbed structure (family ``normal``) and rigidly perturbed ones
(``torsion``, ``bend``, ``contact``), each followed by the *same* thermal displacement. For every slot
the perturbation grid (magnitude x location) is tried in a seeded order until a candidate passes all
geometry checks; every rejected candidate is recorded with its check record.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from gmd_active_learning.contracts.ids import content_hash
from tools.experiments.alkane_hotspots.checks import check_structure, pair_distances, segment_contact
from tools.experiments.alkane_hotspots.config import FAMILIES, AlkaneConfig, ContactFamily
from tools.experiments.alkane_hotspots.geometry import (
    AlkaneTopology,
    angle_deg,
    apply_bend,
    apply_torsion,
    build_all_trans,
    dihedral_deg,
    gaussian_displacement,
    wrap_deg,
)
from tools.experiments.alkane_hotspots.splits import assign_splits, check_no_leakage

LOGGER = logging.getLogger(__name__)


@dataclass
class Candidate:
    family: str
    location: int | None
    magnitude: float
    units: str | None


@dataclass
class Perturbed:
    positions: np.ndarray  # noise-free
    perturbation: dict[str, Any]
    intended_angles: list[tuple[int, int, int]] = field(default_factory=list)
    contact_separation: int | None = None


@dataclass
class GenerationResult:
    parents: list[dict[str, Any]]
    positions: dict[str, np.ndarray]
    rejected: list[dict[str, Any]]


def derived_seed(*parts: Any) -> int:
    return int(content_hash(list(parts), length=8), 16)


def location_index(fraction: float, low: int, high: int, n_carbons: int) -> int:
    """Carbon index for a fractional chain position, clipped to ``[low, high]``."""

    return int(min(max(int(round(fraction * (n_carbons - 1))), low), high))


def torsion_site(topology: AlkaneTopology, k: int) -> list[int]:
    """Carbons k-1..k+2 (the dihedral about bond k-k+1) plus their hydrogens."""

    carbons = [c for c in range(k - 1, k + 3) if 0 <= c < topology.n_carbons]
    return topology.site_atoms(carbons)


def candidates_for(family: str, config: AlkaneConfig, n_carbons: int) -> list[Candidate]:
    fam = getattr(config.families, family)
    if family == "torsion":
        locations = sorted({location_index(f, 1, n_carbons - 3, n_carbons) for f in fam.locations})
        return [Candidate(family, k, float(d), "deg") for d, k in itertools.product(fam.deltas_deg, locations)]
    if family == "bend":
        locations = sorted({location_index(f, 1, n_carbons - 2, n_carbons) for f in fam.locations})
        return [Candidate(family, k, float(d), "deg") for d, k in itertools.product(fam.deltas_deg, locations)]
    if family == "contact":
        high = n_carbons - 2 - fam.fold_bonds
        if high < 1:
            return []
        locations = sorted({location_index(f, 1, high, n_carbons) for f in fam.locations})
        return [Candidate(family, k, float(t), "angstrom") for t, k in itertools.product(fam.target_distances, locations)]
    raise ValueError(f"no candidate grid for family {family}")


def fold_chain(reference: np.ndarray, topology: AlkaneTopology, start: int, target: float, contact: ContactFamily, config: AlkaneConfig) -> tuple[np.ndarray, list[float], float]:
    """Backbone torsions on bonds (start+i, start+i+1), i < fold_bonds, so that the segment contact
    distance equals ``target`` while every other nonbonded pair keeps its minimum (+ clash margin).

    Starts are the ``n_starts`` best trans/gauche rotamer combinations; each is refined with Powell.
    Returns (positions, torsion deltas, achieved contact distance). Deterministic.
    """

    checks = config.checks
    bonds = [(start + i, start + i + 1) for i in range(contact.fold_bonds)]
    numbers = topology.numbers
    far = topology.graph_distances >= checks.nonbonded_min_bond_separation
    carbon_index = np.asarray([i if i < topology.n_carbons else topology.carbon_of[i] for i in range(topology.n_atoms)])
    segment = np.abs(carbon_index[:, None] - carbon_index[None, :]) >= contact.min_segment_separation
    hh = (numbers[:, None] == 1) & (numbers[None, :] == 1)
    cc = (numbers[:, None] == 6) & (numbers[None, :] == 6)
    limits = np.where(hh, checks.min_nonbonded_hh, np.where(cc, checks.min_nonbonded_cc, checks.min_nonbonded_ch)) + contact.clash_margin
    inner = far & ~segment

    def build(deltas: np.ndarray) -> np.ndarray:
        positions = reference
        for bond, delta in zip(bonds, deltas):
            positions, _ = apply_torsion(positions, topology, bond, float(delta))
        return positions

    def objective(deltas: np.ndarray) -> float:
        distances = pair_distances(build(deltas))
        contact_distance = float(distances[segment].min())
        violation = np.clip(limits - distances, 0.0, None)[inner]
        return (contact_distance - target) ** 2 + 10.0 * float(np.sum(violation * violation))

    starts = sorted(itertools.product((0.0, 120.0, -120.0), repeat=contact.fold_bonds), key=lambda d: (objective(np.asarray(d)), d))
    best: tuple[float, np.ndarray] | None = None
    for start_deltas in starts[: contact.n_starts]:
        result = minimize(
            objective,
            np.asarray(start_deltas, dtype=float),
            method="Powell",
            options={"xtol": 1.0e-4, "ftol": 1.0e-12, "maxfev": contact.max_evaluations},
        )
        value = float(result.fun)
        if best is None or value < best[0] - 1.0e-15:
            best = (value, np.asarray(result.x, dtype=float))
    assert best is not None
    deltas = [wrap_deg(d) for d in best[1]]
    positions = build(np.asarray(deltas))
    achieved, _pair = segment_contact(positions, topology, contact.min_segment_separation)
    return positions, deltas, achieved


def perturb(reference: np.ndarray, topology: AlkaneTopology, candidate: Candidate, config: AlkaneConfig) -> Perturbed:
    if candidate.family == "normal":
        return Perturbed(reference.copy(), {"kind": "none", "atoms": [], "location": None, "magnitude": 0.0, "units": None})
    k = int(candidate.location)  # type: ignore[arg-type]
    if candidate.family == "torsion":
        positions, moved = apply_torsion(reference, topology, (k, k + 1), candidate.magnitude)
        before = dihedral_deg(reference, k - 1, k, k + 1, k + 2)
        after = dihedral_deg(positions, k - 1, k, k + 1, k + 2)
        return Perturbed(
            positions,
            {
                "kind": "torsion",
                "atoms": torsion_site(topology, k),
                "location": k,
                "magnitude": candidate.magnitude,
                "units": "deg",
                "bond": [k, k + 1],
                "dihedral_atoms": [k - 1, k, k + 1, k + 2],
                "dihedral_before_deg": before,
                "dihedral_after_deg": after,
                "dihedral_change_deg": wrap_deg(after - before),
                "n_moved_atoms": len(moved),
            },
        )
    if candidate.family == "bend":
        positions, moved = apply_bend(reference, topology, k, candidate.magnitude)
        intended = [(k - 1, k, k + 1)] + [(h, k, k + 1) for h in topology.hydrogens_of[k]]
        return Perturbed(
            positions,
            {
                "kind": "bend",
                "atoms": topology.site_atoms([k - 1, k, k + 1]),
                "location": k,
                "magnitude": candidate.magnitude,
                "units": "deg",
                "angle_atoms": [k - 1, k, k + 1],
                "angle_before_deg": angle_deg(reference, k - 1, k, k + 1),
                "angle_after_deg": angle_deg(positions, k - 1, k, k + 1),
                "changed_angles": [list(a) for a in intended],
                "n_moved_atoms": len(moved),
            },
            intended_angles=intended,
        )
    if candidate.family == "contact":
        contact = config.families.contact
        positions, deltas, achieved = fold_chain(reference, topology, k, candidate.magnitude, contact, config)
        _, pair = segment_contact(positions, topology, contact.min_segment_separation)
        carbons = sorted({p if p < topology.n_carbons else topology.carbon_of[p] for p in pair})
        return Perturbed(
            positions,
            {
                "kind": "contact",
                "atoms": topology.site_atoms(carbons),
                "location": k,
                "magnitude": candidate.magnitude,
                "units": "angstrom",
                "fold_bonds": [[k + i, k + i + 1] for i in range(contact.fold_bonds)],
                "fold_torsion_deltas_deg": deltas,
                "contact_pair": list(pair),
                "contact_carbons": carbons,
                "contact_distance_noise_free": achieved,
                "target_tolerance": contact.target_tolerance,
            },
            contact_separation=contact.min_segment_separation,
        )
    raise ValueError(f"unknown family {candidate.family}")


def _check(perturbed: Perturbed, reference: np.ndarray, final: np.ndarray, topology: AlkaneTopology, config: AlkaneConfig) -> dict[str, Any]:
    record = check_structure(
        final,
        topology,
        config.checks,
        covalent_scale=config.geometry.covalent_scale,
        ideal_angle_deg=config.geometry.angle_deg,
        intended_angles=perturbed.intended_angles,
        reference_positions=reference,
        noise_free_positions=perturbed.positions,
        contact_separation=perturbed.contact_separation,
        charge=config.cp2k.charge,
        multiplicity=config.cp2k.multiplicity,
    )
    p = perturbed.perturbation
    if p["kind"] == "torsion":
        error = abs(wrap_deg(p["dihedral_change_deg"] - p["magnitude"]))
        record["intended_change"] = {"requested": p["magnitude"], "achieved": p["dihedral_change_deg"], "error_deg": error}
        if error > config.checks.exact_angle_tolerance_deg:
            record["failures"].append("torsion_not_achieved")
    elif p["kind"] == "bend":
        achieved = p["angle_after_deg"] - p["angle_before_deg"]
        error = abs(achieved - p["magnitude"])
        record["intended_change"] = {"requested": p["magnitude"], "achieved": achieved, "error_deg": error}
        if error > config.checks.exact_angle_tolerance_deg:
            record["failures"].append("bend_not_achieved")
    elif p["kind"] == "contact":
        error = abs(p["contact_distance_noise_free"] - p["magnitude"])
        record["intended_change"] = {"requested": p["magnitude"], "achieved": p["contact_distance_noise_free"], "error": error}
        if error > p["target_tolerance"]:
            record["failures"].append("contact_target_not_reached")
    record["passed"] = not record["failures"]
    return record


def generate(config: AlkaneConfig) -> GenerationResult:
    """All parents of the configuration (deterministic: same config -> identical output)."""

    generation_hash = config.generation_hash()
    parents: list[dict[str, Any]] = []
    positions_by_id: dict[str, np.ndarray] = {}
    rejected: list[dict[str, Any]] = []
    geometry = config.geometry
    for n in config.systems.chain_lengths:
        topology = AlkaneTopology(n)
        reference = build_all_trans(n, cc_bond=geometry.cc_bond, ch_bond=geometry.ch_bond, angle_deg=geometry.angle_deg)
        for base in range(config.bases_per_length):
            base_id = f"C{n}-b{base:02d}"
            noise_seed = derived_seed("thermal", config.seeds.generation, n, base)
            noise = gaussian_displacement(topology.n_atoms, config.families.thermal.amplitude, noise_seed)
            for family in FAMILIES:
                count = config.generation.per_base.get(family, 0)
                if not count:
                    continue
                if family == "normal":
                    pool = [Candidate("normal", None, 0.0, None)]
                else:
                    pool = candidates_for(family, config, n)
                    order = np.random.default_rng(derived_seed("order", config.seeds.generation, n, base, family)).permutation(len(pool))
                    pool = [pool[i] for i in order]
                used = 0
                tried = 0
                for candidate in pool:
                    if used == count or tried >= config.generation.max_candidates_per_slot * count:
                        break
                    tried += 1
                    perturbed = perturb(reference, topology, candidate, config)
                    final = perturbed.positions + noise
                    record = _check(perturbed, reference, final, topology, config)
                    parent_id = f"{base_id}-{family}{used}"
                    entry = {
                        "parent_id": parent_id,
                        "base_id": base_id,
                        "chain_length": n,
                        "formula": topology.formula,
                        "family": family,
                        "seed": noise_seed,
                        "perturbation": perturbed.perturbation,
                        "geometry_checks": record,
                        "config_hash": generation_hash,
                    }
                    if not record["passed"]:
                        rejected.append({**entry, "parent_id": None, "candidate": candidate.__dict__})
                        LOGGER.info("rejected %s candidate %s: %s", base_id, candidate, record["failures"])
                        continue
                    entry.update({"n_atoms": topology.n_atoms, "split_group": base_id, "thermal_amplitude": config.families.thermal.amplitude})
                    parents.append(entry)
                    positions_by_id[parent_id] = final
                    used += 1
                if used < count:
                    raise RuntimeError(
                        f"{base_id}: only {used}/{count} {family} parents passed the checks after {tried} candidates "
                        "(see rejected.jsonl); widen the grid or relax nothing silently"
                    )
    groups = [(p["split_group"], f"C{p['chain_length']}") for p in parents]
    assignment = assign_splits(groups, config.splits, config.seeds.split)
    for parent in parents:
        parent["split"] = assignment[parent["split_group"]]
    check_no_leakage(parents)
    return GenerationResult(parents, positions_by_id, rejected)


def format_extxyz(parent: dict[str, Any], numbers: np.ndarray, positions: np.ndarray) -> str:
    """Deterministic extended XYZ (readable by ASE) with an ``atom_id`` column equal to the index."""

    symbols = {1: "H", 6: "C"}
    header = (
        f'Properties=species:S:1:pos:R:3:atom_id:I:1 pbc="F F F" parent_id={parent["parent_id"]} '
        f'family={parent["family"]} split={parent["split"]} split_group={parent["split_group"]} '
        f'config_hash={parent["config_hash"]}'
    )
    lines = [str(len(numbers)), header]
    for index, (z, xyz) in enumerate(zip(numbers.tolist(), positions)):
        lines.append(f"{symbols[int(z)]} {xyz[0]:.12f} {xyz[1]:.12f} {xyz[2]:.12f} {index}")
    return "\n".join(lines) + "\n"


def write_generation(result: GenerationResult, root: Path) -> dict[str, Path]:
    """Write ``parents.jsonl``, ``rejected.jsonl`` and ``structures/<parent_id>.extxyz`` under ``root``."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "structures").mkdir(exist_ok=True)
    lines = []
    for parent in result.parents:
        topology = AlkaneTopology(parent["chain_length"])
        relative = f"structures/{parent['parent_id']}.extxyz"
        text = format_extxyz(parent, topology.numbers, result.positions[parent["parent_id"]])
        _write_text(root / relative, text)
        parent["structure_path"] = relative
        parent["structure_sha256"] = content_hash(text)
        lines.append(json.dumps(parent, sort_keys=True))
    _write_text(root / "parents.jsonl", "\n".join(lines) + "\n")
    _write_text(root / "rejected.jsonl", "".join(json.dumps(item, sort_keys=True) + "\n" for item in result.rejected))
    summary = summarize_parents(result.parents, result.rejected)
    _write_text(root / "parents_summary.json", json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return {"parents": root / "parents.jsonl", "rejected": root / "rejected.jsonl", "structures": root / "structures",
            "summary": root / "parents_summary.json"}


def summarize_parents(parents: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> dict[str, Any]:
    """Small, deterministic summary of a generation (committable: no coordinates, no timestamps)."""

    rows = []
    for parent in parents:
        checks = parent["geometry_checks"]
        p = parent["perturbation"]
        rows.append(
            {
                "parent_id": parent["parent_id"],
                "split": parent["split"],
                "split_group": parent["split_group"],
                "family": parent["family"],
                "formula": parent["formula"],
                "location": p["location"],
                "magnitude": p["magnitude"],
                "units": p["units"],
                "n_site_atoms": len(p["atoms"]),
                "achieved": (checks.get("intended_change") or {}).get("achieved"),
                "contact_distance": (checks.get("contact") or {}).get("distance"),
                "min_nonbonded": {k: round(v, 4) for k, v in checks["nonbonded"]["minima"].items()},
                "max_untouched_angle_deviation_deg": round(checks["angles"]["max_untouched_deviation_deg"], 4),
                "structure_sha256": parent.get("structure_sha256"),
            }
        )
    reasons: dict[str, int] = {}
    for item in rejected:
        for failure in item["geometry_checks"]["failures"]:
            reasons[failure] = reasons.get(failure, 0) + 1
    return {
        "config_hash": parents[0]["config_hash"] if parents else None,
        "n_parents": len(parents),
        "n_rejected_candidates": len(rejected),
        "rejection_reasons": dict(sorted(reasons.items())),
        "parents": rows,
    }


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_parents(root: Path) -> list[dict[str, Any]]:
    path = root / "parents.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run the generate command first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_positions(root: Path, parent: dict[str, Any]) -> np.ndarray:
    """Positions from the parent's extxyz (rows in atom_id order)."""

    lines = (root / parent["structure_path"]).read_text(encoding="utf-8").splitlines()
    n = int(lines[0])
    rows = [line.split() for line in lines[2 : 2 + n]]
    order = [int(row[4]) for row in rows]
    if order != list(range(n)):
        raise ValueError(f"{parent['structure_path']}: atom_id column is not the row index")
    return np.asarray([[float(v) for v in row[1:4]] for row in rows])


__all__ = [
    "Candidate",
    "GenerationResult",
    "candidates_for",
    "fold_chain",
    "format_extxyz",
    "generate",
    "load_parents",
    "load_positions",
    "perturb",
    "summarize_parents",
    "torsion_site",
    "write_generation",
]
