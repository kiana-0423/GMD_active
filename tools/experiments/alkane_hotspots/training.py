"""Training data and GMD_SGT training for the alkane demo (all arms share this code path).

Supervision policy (YAML ``training``):

* fragment labels: per-atom ``force_weight`` from the atom's region role (``force_weights.core``,
  ``inner_buffer``, ``outer_buffer``, ``boundary``, ``h_cap``); ``energy_weight`` is **0 for every
  capped fragment** -- its total energy belongs to an artificial H-terminated molecule, never to the
  parent -- and the reason is written into the sample metadata and the dataset manifest. Uncapped
  fragments are finite clusters cut out of a molecule; their weight is ``energy.uncapped_fragment_weight``
  (default 0, reason recorded likewise).
* full-system labels: every atom gets ``full_system_force_weight``; the energy is the molecule's own
  DFT energy and is supervised with ``energy.full_system_weight``.

Training goes through :class:`gmd_active_learning.adapters.sgt.backend.GMDSGTBackend` (GMD_SGT's own
reader honours ``force_weight`` / ``energy_weight``; masked atoms contribute neither to the loss nor
to any gradient, see ``tests/experiments/alkane_hotspots/test_training.py``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gmd_active_learning.contracts import DatasetVersion, ExtractedRegion, ModelArtifact, RegionRole, stable_id
from gmd_active_learning.core.exceptions import ConfigurationError, DataContractError
from gmd_active_learning.datasets.config import RoleWeights
from gmd_active_learning.datasets.extxyz_io import write_samples_extxyz
from gmd_active_learning.datasets.masks import reweight
from gmd_active_learning.datasets.samples import LabeledSample, sample_id_for

from .ground_truth import label_forces, parent_frame, resolve_artifact_path

LOGGER = logging.getLogger(__name__)

CAPPED_ENERGY_REASON = "capped fragment: the DFT energy belongs to an artificial H-terminated molecule, not to the parent; energy loss disabled"
UNCAPPED_ENERGY_REASON = "uncapped fragment: finite cluster cut from a molecule; energy weight from training.energy.uncapped_fragment_weight"
FULL_ENERGY_REASON = "full system: the molecule's own DFT energy; weight from training.energy.full_system_weight"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnergySupervision(_Strict):
    full_system_weight: float = Field(1.0, ge=0.0)
    uncapped_fragment_weight: float = Field(0.0, ge=0.0)


class SGTRuntime(_Strict):
    source_path: str | None = Field("../GMD_SGT", description="GMD_SGT checkout, relative to the GMD_active root")
    device: str = "cpu"
    committee_size: int = Field(2, ge=0, description="extra committee members (seeds seed+1..)")


class TrainingConfig(_Strict):
    sgt: SGTRuntime = Field(default_factory=SGTRuntime)
    model: dict[str, Any] = Field(default_factory=dict, description="GMD_SGT model section (also used by receptive-field)")
    train: dict[str, Any] = Field(default_factory=dict, description="GMD_SGT train section (identical for every arm)")
    force_weights: RoleWeights = Field(default_factory=RoleWeights)
    full_system_force_weight: float = Field(1.0, gt=0.0)
    energy: EnergySupervision = Field(default_factory=EnergySupervision)
    initial_splits: list[str] = Field(default_factory=lambda: ["initial_train"])
    validation_splits: list[str] = Field(default_factory=lambda: ["validation"])
    mode: Literal["scratch", "finetune"] = "scratch"

    @model_validator(mode="after")
    def _check(self) -> TrainingConfig:
        for splits in (self.initial_splits, self.validation_splits):
            if "final_eval" in splits:
                raise ValueError("final_eval must never be used for training or model selection")
        if self.train.get("seed") is not None:
            raise ValueError("training.train.seed is set per repetition by the arms runner; remove it from the YAML")
        return self

    def sgt_train_section(self) -> dict[str, Any]:
        return {"model": dict(self.model), "train": dict(self.train)}


# ---------------------------------------------------------------------------------------------- samples


def _is_capped(region: ExtractedRegion) -> bool:
    return any(role is RegionRole.H_CAP for role in region.roles)


def fragment_sample(label: Mapping[str, Any], parent: Mapping[str, Any], config: TrainingConfig, *bases: str | Path | None) -> LabeledSample:
    """A fragment label as a training sample: role-based force weights, energy weight 0 when capped."""

    if label.get("kind") != "fragment":
        raise DataContractError(f"label {label.get('label_id')} is not a fragment label")
    region = ExtractedRegion.from_dict(json.loads(resolve_artifact_path(label["region_path"], *bases).read_text()))
    region = reweight(region, config.force_weights)
    forces = label_forces(label, *bases)
    capped = _is_capped(region)
    weight = 0.0 if capped else config.energy.uncapped_fragment_weight
    energy = label.get("energy_ev")
    if weight > 0 and energy is None:
        raise DataContractError(f"label {label['label_id']}: energy weight {weight} but no energy")
    metadata = {"label_id": label["label_id"], "label_kind": "fragment", "parent_id": label.get("parent_id"),
                "energy_weight_reason": CAPPED_ENERGY_REASON if capped else UNCAPPED_ENERGY_REASON,
                "force_weights": config.force_weights.as_dict(), "cpu_core_s": (label.get("cost") or {}).get("cpu_core_s")}
    return LabeledSample(sample_id=sample_id_for(region.region_id, str(label["label_id"])), region=region, forces=forces,
                         energy=None if energy is None else float(energy), stress=None, energy_weight=weight, stress_weight=0.0,
                         split_group=str(parent.get("split_group", parent["parent_id"])), task_id=str(label.get("task_id") or label["label_id"]),
                         label_engine="cp2k", engine_version=(label.get("cp2k") or {}).get("version"), metadata=metadata)


def full_region(parent: Mapping[str, Any], config: TrainingConfig, *bases: str | Path | None) -> ExtractedRegion:
    frame = parent_frame(parent, *bases)
    n = frame.n_atoms
    return ExtractedRegion(region_id=stable_id("full", str(parent["parent_id"])), mode="full_system", run_id=str(parent["parent_id"]),
                           parent_step=0, event_id=None, numbers=frame.numbers, positions=frame.positions, cell=None,
                           pbc=(False, False, False), parent_atom_ids=frame.atom_ids, image_shifts=np.zeros((n, 3), dtype=np.int64),
                           roles=[RegionRole.CORE] * n, force_weights=np.full(n, config.full_system_force_weight), frozen=np.zeros(n, dtype=bool),
                           hotspot_atom_ids=[], metadata={"full_system": True})


def full_sample(label: Mapping[str, Any], parent: Mapping[str, Any], config: TrainingConfig, *bases: str | Path | None) -> LabeledSample:
    if label.get("kind") != "full":
        raise DataContractError(f"label {label.get('label_id')} is not a full-system label")
    if str(label.get("parent_id")) != str(parent["parent_id"]):
        raise DataContractError(f"label {label['label_id']} belongs to {label.get('parent_id')}, not {parent['parent_id']}")
    region = full_region(parent, config, *bases)
    forces = label_forces(label, *bases)
    energy = label.get("energy_ev")
    weight = config.energy.full_system_weight if energy is not None else 0.0
    metadata = {"label_id": label["label_id"], "label_kind": "full", "parent_id": parent["parent_id"],
                "energy_weight_reason": FULL_ENERGY_REASON if energy is not None else "no energy in the label record",
                "cpu_core_s": (label.get("cost") or {}).get("cpu_core_s")}
    return LabeledSample(sample_id=sample_id_for(region.region_id, str(label["label_id"])), region=region, forces=forces,
                         energy=None if energy is None else float(energy), stress=None, energy_weight=weight, stress_weight=0.0,
                         split_group=str(parent.get("split_group", parent["parent_id"])), task_id=str(label.get("task_id") or label["label_id"]),
                         label_engine="cp2k", engine_version=(label.get("cp2k") or {}).get("version"), metadata=metadata)


def sample_for_label(label: Mapping[str, Any], parents: Mapping[str, Mapping[str, Any]], config: TrainingConfig, *bases: str | Path | None) -> LabeledSample:
    parent = parents.get(str(label.get("parent_id")))
    if parent is None:
        raise DataContractError(f"label {label.get('label_id')} references unknown parent {label.get('parent_id')}")
    if label.get("kind") == "full":
        return full_sample(label, parent, config, *bases)
    if label.get("kind") == "fragment":
        return fragment_sample(label, parent, config, *bases)
    raise DataContractError(f"label {label.get('label_id')} has unknown kind {label.get('kind')!r}")


# ---------------------------------------------------------------------------------------------- datasets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset(train: Sequence[LabeledSample], val: Sequence[LabeledSample], out_dir: str | Path, *, version: str) -> DatasetVersion:
    """``train.extxyz`` + ``val.extxyz`` + ``manifest.json`` (energy-weight reasons, label ids, hashes)."""

    if not train:
        raise DataContractError("empty training set")
    if not val:
        raise DataContractError("empty validation set (training needs a validation file; final_eval is never used)")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = write_samples_extxyz(train, out / "train.extxyz")
    val_path = write_samples_extxyz(val, out / "val.extxyz")
    def describe(sample: LabeledSample) -> dict[str, Any]:
        return {"sample_id": sample.sample_id, "label_id": sample.metadata.get("label_id"), "kind": sample.metadata.get("label_kind"),
                "n_atoms": sample.region.n_atoms, "energy_weight": sample.energy_weight,
                "energy_weight_reason": sample.metadata.get("energy_weight_reason"),
                "force_weight_sum": float(np.sum(sample.region.force_weights)), "split_group": sample.split_group}
    manifest = {"version": version, "train": [describe(s) for s in train], "val": [describe(s) for s in val],
                "files": {"train.extxyz": _sha256(train_path), "val.extxyz": _sha256(val_path)}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    digest = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
    return DatasetVersion(version, str(out), digest, len(train) + len(val), len(train), len(val), None,
                          tuple(s.sample_id for s in [*train, *val]), "", {"train_file": str(train_path), "val_file": str(val_path),
                                                                           "manifest": str(out / "manifest.json")})


# ---------------------------------------------------------------------------------------------- training


@dataclass(slots=True)
class TrainedModel:
    checkpoints: list[str]  # primary first, then committee members
    species: list[int]
    cutoff: float | None
    summary: dict[str, Any]

    @property
    def primary(self) -> str:
        return self.checkpoints[0]

    def to_dict(self) -> dict[str, Any]:
        return {"checkpoints": self.checkpoints, "species": self.species, "cutoff": self.cutoff,
                "best_val": self.summary.get("best_val"), "mode": self.summary.get("mode")}


def make_backend(config: TrainingConfig, *, root: str | Path | None = None) -> Any:
    from gmd_active_learning.adapters.sgt.backend import GMDSGTBackend, SGTConfig

    source = config.sgt.source_path
    if source is not None and root is not None and not Path(source).is_absolute():
        source = str((Path(root) / source).resolve())
    return GMDSGTBackend(SGTConfig(source_path=source, device=config.sgt.device, train=config.sgt_train_section(),
                                   committee_size=config.sgt.committee_size, finetune_from_active=config.mode == "finetune"))


def train_model(backend: Any, dataset: DatasetVersion, work_dir: str | Path, *, seed: int, parent: TrainedModel | None = None,
                config: TrainingConfig | None = None) -> TrainedModel:
    """Primary + committee through GMDSGTBackend (idempotent per work_dir: finished runs are reused)."""

    parent_artifact = None
    if parent is not None and config is not None and config.mode == "finetune":
        parent_artifact = ModelArtifact(model_id=stable_id("model", parent.primary), checkpoint_path=parent.primary, checkpoint_sha256="",
                                        export_path=None, export_sha256=None, dataset_version=None, parent_model_id=None,
                                        species=tuple(parent.species), cutoff=parent.cutoff, units={}, metrics={}, train_mode="fresh",
                                        backend="gmd_sgt", created_at="")
    output = backend.train(dataset, parent=parent_artifact, work_dir=Path(work_dir), seed=int(seed))
    checkpoints = [str(output.checkpoint), *output.summary.get("committee_checkpoints", [])]
    return TrainedModel(checkpoints, list(output.species), output.cutoff, dict(output.summary))


def predictor_for(backend: Any, model: TrainedModel):
    """``predictor(numbers, positions) -> (primary forces, committee forces or None)`` for ground truth / arms."""

    def predict(numbers: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        primary = backend.predict(model.primary, positions, numbers).forces
        committee = None
        if len(model.checkpoints) >= 2:
            committee = backend.predict_committee(model.checkpoints, positions, numbers)
        return primary, committee

    return predict


def check_training_config(config: TrainingConfig) -> None:
    if not config.model:
        raise ConfigurationError("training.model (GMD_SGT model section) is required")


__all__ = [
    "CAPPED_ENERGY_REASON",
    "TrainedModel",
    "TrainingConfig",
    "fragment_sample",
    "full_sample",
    "make_backend",
    "predictor_for",
    "sample_for_label",
    "train_model",
    "write_dataset",
]
