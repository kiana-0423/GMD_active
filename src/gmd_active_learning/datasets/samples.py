"""Labeled training samples: an extracted region plus its validated label and provenance.

Energy semantics (finding F-12 context): the energy stored on a sample is the DFT energy of the
labeled *region* (extracted, possibly capped) and is never attached to the parent frame. Whether it
is a training target is an explicit per-sample ``energy_weight`` decided by
:class:`~gmd_active_learning.datasets.config.EnergyPolicy`. Stress is kept only when it was labeled
and the region is periodic; otherwise ``stress`` is ``None`` and ``stress_weight`` is 0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gmd_active_learning.contracts import (
    ExtractedRegion,
    LabelResult,
    RegionRole,
    ValidationStatus,
    content_hash,
    stable_id,
)
from gmd_active_learning.core.exceptions import DataContractError
from gmd_active_learning.datasets.config import DatasetsConfig
from gmd_active_learning.datasets.masks import roles_from_codes

#: Bumped whenever the canonical array layout changes.
SAMPLE_FORMAT_VERSION = 1


@dataclass(slots=True)
class LabeledSample:
    sample_id: str
    region: ExtractedRegion
    forces: np.ndarray
    energy: float | None
    stress: np.ndarray | None
    energy_weight: float
    stress_weight: float
    split_group: str
    task_id: str
    label_engine: str
    engine_version: str | None = None
    converged: bool = True
    validation: str = ValidationStatus.VALID.value
    output_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.forces = np.asarray(self.forces, dtype=float)
        if self.forces.shape != (self.region.n_atoms, 3):
            raise DataContractError(
                f"sample {self.sample_id}: forces shape {self.forces.shape} != ({self.region.n_atoms}, 3)"
            )
        if not np.all(np.isfinite(self.forces)):
            raise DataContractError(f"sample {self.sample_id}: forces must be finite")
        if self.stress is not None:
            self.stress = np.asarray(self.stress, dtype=float).reshape(3, 3)
            if not any(self.region.pbc):
                raise DataContractError(f"sample {self.sample_id}: stress on a non-periodic region")
            if not np.allclose(self.stress, self.stress.T, rtol=0.0, atol=1e-10 * max(1.0, float(np.abs(self.stress).max()))):
                raise DataContractError(f"sample {self.sample_id}: stress must be symmetric (extxyz stores Voigt components)")
        if self.energy is not None:
            self.energy = float(self.energy)
            if not np.isfinite(self.energy):
                raise DataContractError(f"sample {self.sample_id}: energy must be finite")
        self.energy_weight = float(self.energy_weight)
        self.stress_weight = float(self.stress_weight)
        if self.energy_weight < 0 or self.stress_weight < 0:
            raise DataContractError(f"sample {self.sample_id}: weights must be >= 0")
        if self.energy_weight > 0 and self.energy is None:
            raise DataContractError(f"sample {self.sample_id}: energy_weight > 0 needs an energy label")
        if self.stress_weight > 0 and self.stress is None:
            raise DataContractError(f"sample {self.sample_id}: stress_weight > 0 needs a stress label")

    @property
    def model_version(self) -> str | None:
        return self.region.model_version

    def provenance(self) -> dict[str, Any]:
        """Scalar provenance written next to the arrays (extxyz info / JSON index)."""

        region = self.region
        return {
            "sample_id": self.sample_id,
            "region_id": region.region_id,
            "extraction_mode": region.mode,
            "run_id": region.run_id,
            "parent_step": int(region.parent_step),
            "event_id": region.event_id,
            "hotspot_atom_ids": [int(value) for value in region.hotspot_atom_ids],
            "charge": int(region.charge),
            "multiplicity": region.multiplicity,
            "model_version": region.model_version,
            "task_id": self.task_id,
            "label_engine": self.label_engine,
            "engine_version": self.engine_version,
            "converged": bool(self.converged),
            "validation": self.validation,
            "output_sha256": self.output_sha256,
            "split_group": self.split_group,
            "energy_scope": "region",
            "energy_is_parent_energy": False,
        }


def sample_id_for(region_id: str, task_id: str) -> str:
    """Deterministic sample identity: one sample per (region, labeling task)."""

    return stable_id("sample", region_id, task_id)


def _energy_weight(region: ExtractedRegion, config: DatasetsConfig) -> tuple[float, str]:
    capped = any(role is RegionRole.H_CAP for role in region.roles)
    if capped and not config.energy.allow_capped:
        return 0.0, "capped_region"
    if any(region.pbc):
        return float(config.energy.periodic_weight), "periodic_region"
    return float(config.energy.nonperiodic_weight), "nonperiodic_region"


def split_group_for(region: ExtractedRegion, config: DatasetsConfig) -> str:
    group_by = config.split.group_by
    if group_by == "region":
        return f"region:{region.region_id}"
    if group_by == "event" and region.event_id:
        return f"event:{region.event_id}"
    return f"run:{region.run_id}"


def build_sample(
    region: ExtractedRegion,
    label: LabelResult,
    config: DatasetsConfig | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> LabeledSample:
    """Combine a region with its label under the configured energy/stress policy."""

    config = config or DatasetsConfig()
    if label.region_id != region.region_id:
        raise DataContractError(f"label {label.task_id} belongs to region {label.region_id}, not {region.region_id}")
    if config.require_valid_labels and label.validation is not ValidationStatus.VALID:
        raise DataContractError(f"label {label.task_id} is {label.validation.value}; only VALID labels enter datasets")
    forces = np.asarray(label.forces, dtype=float)
    if forces.shape != (region.n_atoms, 3):
        raise DataContractError(f"label {label.task_id}: forces {forces.shape} for a region of {region.n_atoms} atoms")
    energy_weight, energy_reason = _energy_weight(region, config)
    energy = label.energy if config.energy.write_energy or energy_weight > 0 else None
    if energy_weight > 0 and energy is None:
        raise DataContractError(f"label {label.task_id}: energy policy gives weight {energy_weight} but no energy")
    stress, stress_weight, stress_reason = None, 0.0, "not_labeled"
    if label.stress is not None:
        if not config.stress.enabled:
            stress_reason = "disabled_by_config"
        elif not any(region.pbc):
            stress_reason = "nonperiodic_region"
        else:
            stress, stress_weight, stress_reason = np.asarray(label.stress, dtype=float), float(config.stress.weight), "periodic_region"
    sample_id = sample_id_for(region.region_id, label.task_id)
    return LabeledSample(
        sample_id=sample_id,
        region=region,
        forces=forces,
        energy=energy,
        stress=stress,
        energy_weight=energy_weight,
        stress_weight=stress_weight,
        split_group=split_group_for(region, config),
        task_id=label.task_id,
        label_engine=label.engine,
        engine_version=label.engine_version,
        converged=bool(label.converged),
        validation=label.validation.value,
        output_sha256=label.output_sha256,
        metadata={
            "energy_policy": energy_reason,
            "stress_policy": stress_reason,
            "scf_steps": label.scf_steps,
            "label_content_hash": content_hash({"forces": forces, "energy": energy, "stress": stress}),
            "label_metadata": dict(label.metadata),
            **(metadata or {}),
        },
    )


# --------------------------------------------------------------------------- canonical encoding
_PER_ATOM = ("numbers", "positions", "forces", "force_weight", "region_code", "frozen", "parent_atom_id", "image_shift")


def samples_to_arrays(samples: list[LabeledSample]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Lossless, pickle-free encoding: concatenated per-atom arrays + per-sample arrays + JSON records."""

    if not samples:
        raise DataContractError("cannot encode an empty sample list")
    n_atoms = np.asarray([sample.region.n_atoms for sample in samples], dtype=np.int64)
    arrays: dict[str, np.ndarray] = {
        "format_version": np.asarray(SAMPLE_FORMAT_VERSION, dtype=np.int64),
        "n_atoms": n_atoms,
        "numbers": np.concatenate([s.region.numbers for s in samples]).astype(np.int64),
        "positions": np.vstack([s.region.positions for s in samples]),
        "forces": np.vstack([s.forces for s in samples]),
        "force_weight": np.concatenate([s.region.force_weights for s in samples]),
        "region_code": np.concatenate([s.region.region_codes for s in samples]).astype(np.int64),
        "frozen": np.concatenate([s.region.frozen for s in samples]).astype(bool),
        "parent_atom_id": np.concatenate([s.region.parent_atom_ids for s in samples]).astype(np.int64),
        "image_shift": np.vstack([s.region.image_shifts for s in samples]).astype(np.int64),
        "cell": np.stack([np.full((3, 3), np.nan) if s.region.cell is None else s.region.cell for s in samples]),
        "pbc": np.asarray([s.region.pbc for s in samples], dtype=bool),
        "energy": np.asarray([np.nan if s.energy is None else s.energy for s in samples], dtype=float),
        "stress": np.stack([np.full((3, 3), np.nan) if s.stress is None else s.stress for s in samples]),
        "energy_weight": np.asarray([s.energy_weight for s in samples], dtype=float),
        "stress_weight": np.asarray([s.stress_weight for s in samples], dtype=float),
    }
    records = [
        {
            **sample.provenance(),
            "has_cell": sample.region.cell is not None,
            "has_energy": sample.energy is not None,
            "region_metadata": sample.region.metadata,
            "sample_metadata": sample.metadata,
        }
        for sample in samples
    ]
    json.dumps(records)  # fail early on non-JSON metadata
    return arrays, records


def samples_from_arrays(arrays: dict[str, np.ndarray], records: list[dict[str, Any]]) -> list[LabeledSample]:
    version = int(np.asarray(arrays["format_version"]))
    if version != SAMPLE_FORMAT_VERSION:
        raise DataContractError(f"unsupported sample format version {version}")
    n_atoms = np.asarray(arrays["n_atoms"], dtype=np.int64)
    if len(n_atoms) != len(records):
        raise DataContractError("sample arrays and records disagree on the number of samples")
    offsets = np.concatenate([[0], np.cumsum(n_atoms)])
    samples = []
    for k, record in enumerate(records):
        a, b = int(offsets[k]), int(offsets[k + 1])
        region = ExtractedRegion(
            region_id=record["region_id"],
            mode=record["extraction_mode"],
            run_id=record["run_id"],
            parent_step=int(record["parent_step"]),
            event_id=record.get("event_id"),
            numbers=arrays["numbers"][a:b],
            positions=arrays["positions"][a:b],
            cell=arrays["cell"][k] if record["has_cell"] else None,
            pbc=tuple(bool(flag) for flag in arrays["pbc"][k]),  # type: ignore[arg-type]
            parent_atom_ids=arrays["parent_atom_id"][a:b],
            image_shifts=arrays["image_shift"][a:b],
            roles=roles_from_codes(arrays["region_code"][a:b].tolist()),
            force_weights=arrays["force_weight"][a:b],
            frozen=arrays["frozen"][a:b],
            hotspot_atom_ids=list(record.get("hotspot_atom_ids", [])),
            charge=int(record.get("charge", 0)),
            multiplicity=record.get("multiplicity"),
            model_version=record.get("model_version"),
            metadata=dict(record.get("region_metadata", {})),
        )
        stress = arrays["stress"][k]
        samples.append(
            LabeledSample(
                sample_id=record["sample_id"],
                region=region,
                forces=arrays["forces"][a:b],
                energy=float(arrays["energy"][k]) if record["has_energy"] else None,
                stress=None if np.all(np.isnan(stress)) else stress,
                energy_weight=float(arrays["energy_weight"][k]),
                stress_weight=float(arrays["stress_weight"][k]),
                split_group=record["split_group"],
                task_id=record["task_id"],
                label_engine=record["label_engine"],
                engine_version=record.get("engine_version"),
                converged=bool(record.get("converged", True)),
                validation=record.get("validation", ValidationStatus.VALID.value),
                output_sha256=record.get("output_sha256"),
                metadata=dict(record.get("sample_metadata", {})),
            )
        )
    return samples


__all__ = [
    "SAMPLE_FORMAT_VERSION",
    "LabeledSample",
    "build_sample",
    "sample_id_for",
    "samples_from_arrays",
    "samples_to_arrays",
    "split_group_for",
]
