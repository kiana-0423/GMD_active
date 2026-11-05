"""Configuration models for training masks and the versioned dataset store.

Every threshold and switch used by :mod:`gmd_active_learning.datasets` comes from these models;
the defaults are written to ``configs/sections/datasets.yaml``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gmd_active_learning.contracts import RegionRole


class RoleWeights(BaseModel):
    """Per-atom force-loss weight for each region role (GMD_SGT ``force_weight``).

    Defaults follow PHAL's mask generator (``training/mask_generator.py``): core 1.0, inner buffer
    0.3, everything else 0. PHAL's block metadata carried a different, informational-only inner
    buffer default of 0.2 (``extraction/block.py:543``, finding F-18); the value that actually
    reached the training files was 0.3, so 0.3 is the single default for every mode.
    ``label_core`` / ``frozen_boundary`` (block mode) share the ``core`` / ``boundary`` weights.
    """

    model_config = ConfigDict(extra="forbid")

    core: float = Field(1.0, ge=0.0)
    inner_buffer: float = Field(0.3, ge=0.0)
    outer_buffer: float = Field(0.0, ge=0.0)
    boundary: float = Field(0.0, ge=0.0)
    h_cap: float = Field(0.0, ge=0.0)
    unassigned: float = Field(0.0, ge=0.0)

    def weight(self, role: RegionRole) -> float:
        return float(getattr(self, RegionRole(role).value))

    def as_dict(self) -> dict[str, float]:
        return {role.value: self.weight(role) for role in RegionRole}


class EnergyPolicy(BaseModel):
    """Whether a region's DFT energy becomes a training target (GMD_SGT ``energy_weight``).

    A region's energy describes the extracted (possibly capped) system only; it is never attached
    to the parent frame. Non-periodic clusters are artificial finite systems, so their energy is
    excluded by default (weight 0, the energy is still written as provenance). Capped regions
    contain artificial H atoms and are excluded unless ``allow_capped`` is true.
    """

    model_config = ConfigDict(extra="forbid")

    nonperiodic_weight: float = Field(0.0, ge=0.0)
    periodic_weight: float = Field(0.0, ge=0.0)
    allow_capped: bool = False
    write_energy: bool = True  # write the region energy as provenance even when its weight is 0


class StressPolicy(BaseModel):
    """Stress is written only when labeled and the region is periodic; otherwise weight 0."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    weight: float = Field(1.0, ge=0.0)


class SplitConfig(BaseModel):
    """Deterministic, group-aware train/validation split.

    ``group_by: event`` keeps every sample cut from one MD event together (samples without an
    event fall back to their run, recorded in the sample's ``split_group``); ``run`` groups by
    trajectory; ``region`` groups by extracted region.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    val_fraction: float = Field(0.1, ge=0.0, lt=1.0)
    seed: int = 20261106
    group_by: Literal["event", "run", "region"] = "event"


class ExportConfig(BaseModel):
    """Which files a published version contains (each switchable)."""

    model_config = ConfigDict(extra="forbid")

    extxyz: bool = True
    sgt_npz: bool = True
    split_files: bool = True  # train.extxyz / val.extxyz next to all.extxyz


class DatasetsConfig(BaseModel):
    """``datasets`` configuration section.

    Per-atom force weights are decided at extraction time (``extraction.force_weights``, a
    :class:`RoleWeights`) and travel with the region; this section does not override them.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    store_root: str = "datasets_store"
    energy: EnergyPolicy = Field(default_factory=EnergyPolicy)
    stress: StressPolicy = Field(default_factory=StressPolicy)
    split: SplitConfig = Field(default_factory=SplitConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    require_valid_labels: bool = True
    float_format: str = "shortest"  # exact shortest repr; or a %-format such as "%.17g"

    @field_validator("float_format")
    @classmethod
    def _check_float_format(cls, value: str) -> str:
        if value == "shortest":
            return value
        try:
            float(value % 0.1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"float_format {value!r} is not a valid %-format for floats") from exc
        return value

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> DatasetsConfig:
        return cls.model_validate(data or {})


__all__ = [
    "DatasetsConfig",
    "EnergyPolicy",
    "ExportConfig",
    "RoleWeights",
    "SplitConfig",
    "StressPolicy",
]
