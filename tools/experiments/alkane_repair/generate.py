"""Deterministic structures of the repair benchmark: independent base conformations, in-domain
(ordinary) samples and out-of-coverage challenge structures, assigned to partitions by base.

* A *base conformation* is an all-trans chain with seeded random deviations of every backbone dihedral
  and C-C-C angle (inside the coverage windows). It is the experimental unit and the leakage unit:
  every structure derived from a base (ordinary samples, challenges, MD frames and fragments from MD
  started at one of them) belongs to the base's partition.
* An *ordinary* structure is the base plus per-sample dihedral jitter plus thermal displacement; it
  must lie inside every coverage window (``coverage``), otherwise it is resampled.
* A *challenge* structure is the base (before sample jitter) with one rigid perturbation of the
  alkane demonstration (``families.perturb``: torsion, bend, contact fold) plus thermal displacement.
  The challenged descriptor must lie outside its coverage window by ``min_excess``; the other
  descriptors are recorded (a contact fold changes torsions too, by construction).

Geometry acceptance reuses ``checks.check_structure`` (topology, bond lengths, untouched angles, rigid
exactness, nonbonded clashes, contact window, closed shell). Every rejected candidate is recorded.
Partition assignment is a seeded permutation of bases. final_test structures are written to the
sealed root only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gmd_active_learning.contracts.ids import content_hash
from tools.experiments.alkane_hotspots.checks import check_structure, segment_contact
from tools.experiments.alkane_hotspots.config import BendFamily, ContactFamily, FamiliesConfig, TorsionFamily
from tools.experiments.alkane_hotspots.families import (
    Candidate,
    _check as family_check,
    candidates_for,
    derived_seed,
    format_extxyz,
    perturb,
)
from tools.experiments.alkane_hotspots.geometry import (
    AlkaneTopology,
    angle_deg,
    apply_bend,
    apply_torsion,
    build_all_trans,
    dihedral_deg,
    gaussian_displacement,
)

from .config import CHALLENGE_FAMILIES, PARTITION_SPLIT, PARTITIONS, LoadedBenchmark, RepairBenchmarkConfig

LOGGER = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    """A slot could not be filled with a structure that passes every check."""


# ---------------------------------------------------------------------------------------------- descriptors


def torsion_deviations(positions: np.ndarray, topology: AlkaneTopology) -> np.ndarray:
    """|180 - |phi|| for every backbone C-C-C-C dihedral (0 = trans)."""

    n = topology.n_carbons
    return np.asarray([180.0 - abs(dihedral_deg(positions, k, k + 1, k + 2, k + 3)) for k in range(n - 3)])


def angle_deviations(positions: np.ndarray, topology: AlkaneTopology, ideal_deg: float) -> np.ndarray:
    n = topology.n_carbons
    return np.asarray([abs(angle_deg(positions, v - 1, v, v + 1) - ideal_deg) for v in range(1, n - 1)])


def coverage_descriptors(positions: np.ndarray, topology: AlkaneTopology, config: RepairBenchmarkConfig, ideal_deg: float) -> dict[str, Any]:
    cov = config.coverage
    torsions = torsion_deviations(positions, topology)
    angles = angle_deviations(positions, topology, ideal_deg)
    contact, pair = segment_contact(positions, topology, cov.min_segment_separation)
    inside = {"torsion": bool(torsions.max() <= cov.max_torsion_deviation_deg),
              "bend": bool(angles.max() <= cov.max_angle_deviation_deg),
              "contact": bool(contact >= cov.min_segment_contact)}
    return {"max_torsion_deviation_deg": float(torsions.max()), "argmax_torsion_bond": int(np.argmax(torsions)) + 1,
            "max_angle_deviation_deg": float(angles.max()), "argmax_angle_vertex": int(np.argmax(angles)) + 1,
            "segment_contact": float(contact), "contact_pair": [int(pair[0]), int(pair[1])],
            "inside": inside, "in_domain": all(inside.values()),
            "windows": {"max_torsion_deviation_deg": cov.max_torsion_deviation_deg, "max_angle_deviation_deg": cov.max_angle_deviation_deg,
                        "min_segment_contact": cov.min_segment_contact, "min_segment_separation": cov.min_segment_separation}}


def challenged_descriptor(family: str, perturbation: dict[str, Any], positions: np.ndarray, topology: AlkaneTopology,
                          config: RepairBenchmarkConfig, ideal_deg: float) -> dict[str, Any]:
    """The family's own descriptor at the perturbed site and whether it clears the coverage window by the margin."""

    cov, ch = config.coverage, config.challenges
    if family == "torsion":
        a, b, c, d = perturbation["dihedral_atoms"]
        value = 180.0 - abs(dihedral_deg(positions, a, b, c, d))
        required = cov.max_torsion_deviation_deg + ch.torsion.min_excess_deg
        upper = ch.torsion.max_deviation_deg
        return {"name": "torsion_deviation_deg", "value": value, "required_min": required, "allowed_max": upper,
                "outside": bool(required <= value <= upper)}
    if family == "bend":
        a, b, c = perturbation["angle_atoms"]
        value = abs(angle_deg(positions, a, b, c) - ideal_deg)
        required = cov.max_angle_deviation_deg + ch.bend.min_excess_deg
        return {"name": "angle_deviation_deg", "value": value, "required_min": required, "outside": bool(value >= required)}
    if family == "contact":
        value, _ = segment_contact(positions, topology, cov.min_segment_separation)
        required = cov.min_segment_contact - ch.contact.min_excess
        return {"name": "segment_contact", "value": float(value), "required_max": required, "outside": bool(value <= required)}
    raise ValueError(f"unknown challenge family {family}")


# ---------------------------------------------------------------------------------------------- builders


def base_backbone(config: RepairBenchmarkConfig, alpha: Any, topology: AlkaneTopology, seed: int) -> np.ndarray:
    """All-trans chain with seeded Gaussian deviations of every backbone dihedral and C-C-C angle (noise-free)."""

    geo = alpha.geometry
    positions = build_all_trans(topology.n_carbons, cc_bond=geo.cc_bond, ch_bond=geo.ch_bond, angle_deg=geo.angle_deg)
    rng = np.random.default_rng(seed)
    sampler = config.ordinary
    for k in range(1, topology.n_carbons - 2):
        positions, _ = apply_torsion(positions, topology, (k, k + 1), float(rng.normal(0.0, sampler.base_torsion_sigma_deg)))
    for v in range(1, topology.n_carbons - 1):
        positions, _ = apply_bend(positions, topology, v, float(rng.normal(0.0, sampler.base_angle_sigma_deg)))
    return positions


def jitter(positions: np.ndarray, topology: AlkaneTopology, sigma_deg: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = positions
    for k in range(1, topology.n_carbons - 2):
        out, _ = apply_torsion(out, topology, (k, k + 1), float(rng.normal(0.0, sigma_deg)))
    return out


def generation_alpha(config: RepairBenchmarkConfig, alpha_config: Any) -> Any:
    """alpha's configuration with the benchmark's challenge grids as perturbation families (for ``perturb``)."""

    ch = config.challenges
    families = FamiliesConfig(
        torsion=TorsionFamily(enabled=ch.torsion.enabled, deltas_deg=ch.torsion.deltas_deg, locations=ch.torsion.locations),
        bend=BendFamily(enabled=ch.bend.enabled, deltas_deg=ch.bend.deltas_deg, locations=ch.bend.locations),
        contact=ContactFamily(enabled=ch.contact.enabled, target_distances=ch.contact.target_distances, target_tolerance=ch.contact.target_tolerance,
                              locations=ch.contact.locations, fold_bonds=ch.contact.fold_bonds,
                              min_segment_separation=config.coverage.min_segment_separation, n_starts=ch.contact.n_starts,
                              clash_margin=ch.contact.clash_margin, max_evaluations=ch.contact.max_evaluations))
    return alpha_config.model_copy(update={"families": families})


@dataclass
class Structures:
    records: list[dict[str, Any]]
    positions: dict[str, np.ndarray]
    rejected: list[dict[str, Any]]
    bases: list[dict[str, Any]]


def assign_partitions(config: RepairBenchmarkConfig) -> list[str]:
    """Partition of every base index (seeded permutation; sizes from ``partitions``)."""

    labels: list[str] = []
    for name in PARTITIONS:
        labels.extend([name] * getattr(config.partitions, name).bases)
    order = np.random.default_rng(int(config.seeds["assignment"])).permutation(len(labels))
    return [labels[i] for i in order]


def _record(parent_id: str, base: dict[str, Any], topology: AlkaneTopology, *, family: str, kind: str, seed: int, perturbation: dict[str, Any],
            checks: dict[str, Any], coverage: dict[str, Any], generation_hash: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    partition = base["partition"]
    return {"parent_id": parent_id, "base_id": base["base_id"], "chain_length": topology.n_carbons, "formula": topology.formula,
            "family": family, "kind": kind, "partition": partition, "split": PARTITION_SPLIT[partition], "split_group": base["base_id"],
            "ancestry": {"base_id": base["base_id"], "base_seed": base["seed"]}, "seed": seed, "perturbation": perturbation,
            "geometry_checks": checks, "coverage": coverage, "n_atoms": topology.n_atoms, "config_hash": generation_hash,
            "thermal_amplitude": None, **(extra or {})}


def generate(loaded: LoadedBenchmark) -> Structures:
    config = loaded.config
    alpha_config = loaded.alpha.config
    gen_alpha = generation_alpha(config, alpha_config)
    n = config.chain_length
    topology = AlkaneTopology(n)
    ideal = alpha_config.geometry.angle_deg
    generation_hash = content_hash({"benchmark": config.model_dump(mode="json", include={"seeds", "chain_length", "coverage", "ordinary", "challenges", "partitions"}),
                                    "geometry": alpha_config.geometry.model_dump(mode="json"), "checks": alpha_config.checks.model_dump(mode="json")}, length=16)
    gen_seed = int(config.seeds["generation"])
    amplitude = config.ordinary.thermal_amplitude
    records: list[dict[str, Any]] = []
    positions: dict[str, np.ndarray] = {}
    rejected: list[dict[str, Any]] = []
    bases: list[dict[str, Any]] = []
    for index, partition in enumerate(assign_partitions(config)):
        spec = getattr(config.partitions, partition)
        base_seed = derived_seed("repair-base", gen_seed, n, index)
        base = {"base_id": f"C{n}-r{index:03d}", "index": index, "partition": partition, "seed": base_seed}
        backbone = None
        for attempt in range(config.ordinary.max_attempts):
            candidate = base_backbone(config, alpha_config, topology, derived_seed("backbone", base_seed, attempt))
            cov = coverage_descriptors(candidate, topology, config, ideal)
            if cov["in_domain"]:
                backbone, base["backbone_attempt"] = candidate, attempt
                break
            rejected.append({"base_id": base["base_id"], "slot": "backbone", "attempt": attempt, "reason": "outside coverage", "coverage": cov})
        if backbone is None:
            raise GenerationError(f"{base['base_id']}: no in-domain backbone in {config.ordinary.max_attempts} attempts")
        bases.append(base)
        for j in range(spec.ordinary_per_base):
            made = False
            for attempt in range(config.ordinary.max_attempts):
                seed = derived_seed("ordinary", base_seed, j, attempt)
                noise_free = jitter(backbone, topology, config.ordinary.sample_torsion_sigma_deg, seed)
                final = noise_free + gaussian_displacement(topology.n_atoms, amplitude, derived_seed("thermal", seed))
                checks = check_structure(final, topology, alpha_config.checks, covalent_scale=alpha_config.geometry.covalent_scale,
                                         ideal_angle_deg=ideal, charge=alpha_config.cp2k.charge, multiplicity=alpha_config.cp2k.multiplicity)
                cov = coverage_descriptors(final, topology, config, ideal)
                if not checks["passed"] or not cov["in_domain"]:
                    rejected.append({"base_id": base["base_id"], "slot": f"ord{j}", "attempt": attempt, "failures": checks["failures"],
                                     "in_domain": cov["in_domain"], "coverage": cov})
                    continue
                parent_id = f"{base['base_id']}-ord{j}"
                record = _record(parent_id, base, topology, family="normal", kind="ordinary", seed=seed,
                                 perturbation={"kind": "none", "atoms": [], "location": None, "magnitude": 0.0, "units": None},
                                 checks=checks, coverage=cov, generation_hash=generation_hash, extra={"thermal_amplitude": amplitude})
                records.append(record)
                positions[parent_id] = final
                made = True
                break
            if not made:
                raise GenerationError(f"{base['base_id']}: ordinary slot {j} unfilled after {config.ordinary.max_attempts} attempts (see rejected.jsonl)")
        for family in config.challenges.enabled() if spec.challenges_per_family else []:
            pool = candidates_for(family, gen_alpha, n)
            order = np.random.default_rng(derived_seed("order", base_seed, family)).permutation(len(pool))
            pool = [pool[i] for i in order]
            used = 0
            for attempt, candidate in enumerate(pool):
                if used == spec.challenges_per_family:
                    break
                perturbed = perturb(backbone, topology, candidate, gen_alpha)
                seed = derived_seed("challenge", base_seed, family, attempt)
                final = perturbed.positions + gaussian_displacement(topology.n_atoms, amplitude, derived_seed("thermal", seed))
                checks = family_check(perturbed, backbone, final, topology, gen_alpha)
                cov = coverage_descriptors(final, topology, config, ideal)
                target = challenged_descriptor(family, perturbed.perturbation, final, topology, config, ideal)
                if not checks["passed"] or not target["outside"]:
                    rejected.append({"base_id": base["base_id"], "slot": f"{family}{used}", "attempt": attempt, "candidate": candidate.__dict__,
                                     "failures": checks["failures"], "challenged": target})
                    continue
                parent_id = f"{base['base_id']}-{family}{used}"
                others = sorted(name for name, inside in cov["inside"].items() if name != family and not inside)
                record = _record(parent_id, base, topology, family=family, kind="challenge", seed=seed, perturbation=perturbed.perturbation,
                                 checks=checks, coverage={**cov, "challenged": target, "also_outside": others}, generation_hash=generation_hash,
                                 extra={"thermal_amplitude": amplitude})
                records.append(record)
                positions[parent_id] = final
                used += 1
            if used < spec.challenges_per_family:
                raise GenerationError(f"{base['base_id']}: only {used}/{spec.challenges_per_family} {family} challenges passed (see rejected.jsonl)")
    check_partition_isolation(records)
    return Structures(records, positions, rejected, bases)


def check_partition_isolation(records: list[dict[str, Any]]) -> None:
    """Every base (and so every derived structure) belongs to exactly one partition; splits follow partitions."""

    seen: dict[str, str] = {}
    for record in records:
        partition = record["partition"]
        if partition not in PARTITIONS:
            raise ValueError(f"unknown partition {partition!r} for {record['parent_id']}")
        if record["split"] != PARTITION_SPLIT[partition]:
            raise ValueError(f"{record['parent_id']}: split {record['split']} does not match partition {partition}")
        if seen.setdefault(record["base_id"], partition) != partition:
            raise ValueError(f"base {record['base_id']} straddles partitions {seen[record['base_id']]} and {partition}")
        if record["kind"] == "challenge" and partition in ("id_train", "id_val", "calibration"):
            raise ValueError(f"{record['parent_id']}: challenge structure in in-domain partition {partition}")
        if record["kind"] == "ordinary" and not record["coverage"]["in_domain"]:
            raise ValueError(f"{record['parent_id']}: ordinary structure outside the coverage windows")


# ---------------------------------------------------------------------------------------------- writing


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_structures(loaded: LoadedBenchmark, result: Structures) -> dict[str, Any]:
    """Public parents (every partition but final_test) under the root, final_test under the sealed root.

    Refuses to overwrite an existing generation with different content (a regenerated set is a new
    benchmark, never a silent replacement)."""

    targets = {"public": (loaded.root, [r for r in result.records if r["partition"] != "final_test"]),
               "sealed": (loaded.sealed_root, [r for r in result.records if r["partition"] == "final_test"])}
    written: dict[str, Any] = {}
    for name, (root, records) in targets.items():
        lines = []
        for record in records:
            topology = AlkaneTopology(record["chain_length"])
            relative = f"structures/{record['parent_id']}.extxyz"
            text = format_extxyz(record, topology.numbers, result.positions[record["parent_id"]])
            record["structure_path"] = relative
            record["structure_sha256"] = content_hash(text)
            lines.append((relative, text, json.dumps(record, sort_keys=True)))
        payload = "\n".join(line for _, _, line in lines) + ("\n" if lines else "")
        parents_path = root / "parents.jsonl"
        if parents_path.is_file() and parents_path.read_text(encoding="utf-8") != payload:
            raise GenerationError(f"{parents_path} exists with different content; a new generation needs a new benchmark name/root")
        for relative, text, _ in lines:
            _write_text(root / relative, text)
        _write_text(parents_path, payload)
        written[name] = {"root": str(root), "n_structures": len(records), "parents_sha256": content_hash(payload)}
    _write_text(loaded.root / "rejected.jsonl", "".join(json.dumps(item, sort_keys=True, default=str) + "\n" for item in result.rejected))
    summary = summarize(result)
    summary["files"] = written
    _write_text(loaded.root / "structures_summary.json", json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
    return summary


def summarize(result: Structures) -> dict[str, Any]:
    """Committable summary: counts per partition/kind/family, coverage descriptors, no coordinates."""

    counts: dict[str, dict[str, int]] = {}
    for record in result.records:
        key = record["family"] if record["kind"] == "challenge" else "ordinary"
        counts.setdefault(record["partition"], {}).setdefault(key, 0)
        counts[record["partition"]][key] += 1
    rows = [{"parent_id": r["parent_id"], "partition": r["partition"], "base_id": r["base_id"], "kind": r["kind"], "family": r["family"],
             "max_torsion_deviation_deg": round(r["coverage"]["max_torsion_deviation_deg"], 2),
             "max_angle_deviation_deg": round(r["coverage"]["max_angle_deviation_deg"], 2),
             "segment_contact": round(r["coverage"]["segment_contact"], 3),
             "challenged": None if r["kind"] != "challenge" else {k: (round(v, 3) if isinstance(v, float) else v)
                                                                   for k, v in r["coverage"]["challenged"].items()},
             "also_outside": r["coverage"].get("also_outside"), "location": r["perturbation"].get("location"),
             "magnitude": r["perturbation"].get("magnitude")} for r in result.records]
    reasons: dict[str, int] = {}
    for item in result.rejected:
        for failure in item.get("failures") or [item.get("reason", "outside coverage")]:
            reasons[failure] = reasons.get(failure, 0) + 1
    return {"n_structures": len(result.records), "n_bases": len(result.bases),
            "bases": [{k: b[k] for k in ("base_id", "partition")} for b in result.bases],
            "counts": counts, "n_rejected_candidates": len(result.rejected), "rejection_reasons": dict(sorted(reasons.items())),
            "config_hash": result.records[0]["config_hash"] if result.records else None, "structures": rows}


def load_records(root: Path) -> list[dict[str, Any]]:
    path = Path(root) / "parents.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run `generate` first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


__all__ = ["CHALLENGE_FAMILIES", "GenerationError", "Structures", "assign_partitions", "challenged_descriptor", "check_partition_isolation",
           "coverage_descriptors", "generate", "load_records", "summarize", "write_structures"]
