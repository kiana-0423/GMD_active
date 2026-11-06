"""Configuration models for extraction, H capping and embedding preparation.

Defaults reproduce PHAL's ``configs/default.yaml`` (96263b6) unless a finding says otherwise; every
deviation is listed in ``docs/extraction_and_datasets.md``. The default YAML for each section lives
in ``configs/sections/{extraction,capping,embedding}.yaml``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gmd_active_learning.datasets.config import RoleWeights

ExtractionMode = Literal["cluster", "slab", "graph", "block"]


class PeriodicGuardConfig(BaseModel):
    """Detect regions that cannot be represented once in a non-periodic box (finding F-09).

    A region is *torn* when two of its atoms that are within ``pair_cutoff`` of each other in the
    periodic parent frame are placed at different relative images in the region (typically because
    the region is wider than half a cell, so the minimum image from its centre wraps). ``error``
    refuses such regions (:class:`~gmd_active_learning.core.exceptions.ExtractionError`); ``flag``
    keeps them and records ``metadata['periodic_guard']`` (H capping then caps the torn bonds
    explicitly); ``allow`` records the report without warning.
    """

    model_config = ConfigDict(extra="forbid")

    policy: Literal["error", "flag", "allow"] = "error"
    pair_cutoff: float = Field(3.0, gt=0.0)


class SlabConfig(BaseModel):
    """Slab patch window (PHAL ``extraction.slab``; window = (lx, ly), z thickness = 2*z_margin).

    ``multi_period`` decides what happens when a lateral window covers two or more periods of a
    periodic source axis (PHAL produced a multi-period cell holding a single image, finding F-08):
    ``replicate`` fills the supercell with the periodic images, ``single_period`` keeps one period
    (recorded as ``window_reduced``), ``error`` refuses. ``skewed_policy`` applies when only one
    lateral axis of a skewed in-plane cell can stay periodic, which a Cartesian window cannot
    represent: ``error`` refuses, ``nonperiodic`` drops lateral periodicity (recorded).
    """

    model_config = ConfigDict(extra="forbid")

    lx: float = Field(16.0, gt=0.0)
    ly: float = Field(16.0, gt=0.0)
    z_margin: float = Field(8.0, gt=0.0)
    z_vacuum: float = Field(6.0, ge=0.0)  # PHAL hard-coded 6 A (slab_extractor.py:101)
    multi_period: Literal["replicate", "single_period", "error"] = "replicate"
    skewed_policy: Literal["error", "nonperiodic"] = "error"
    period_tolerance: float = Field(1.0e-8, ge=0.0)


class GraphConfig(BaseModel):
    """Bonded-subgraph extraction (PHAL ``extraction.graph``).

    PHAL read the bond scale from ``h_capping.covalent_scale``; it is its own key here (same
    default) so graph extraction does not depend on the capping section.
    """

    model_config = ConfigDict(extra="forbid")

    hops: int = Field(2, ge=0)
    radial_shell: bool = True
    covalent_scale: float = Field(1.2, gt=0.0)


class BlockBufferConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inner: float = Field(3.0, ge=0.0)
    outer: float = Field(5.0, ge=0.0)


class BlockFrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    thickness: float = Field(2.0, ge=0.0)


class BlockConfig(BaseModel):
    """Spatial-grid block extraction (PHAL ``extraction.block``)."""

    model_config = ConfigDict(extra="forbid")

    scheme: Literal["spatial_grid"] = "spatial_grid"
    size: tuple[float, float, float] | None = None
    halo: float = Field(5.0, ge=0.0)
    buffer: BlockBufferConfig = Field(default_factory=BlockBufferConfig)
    frozen: BlockFrozenConfig = Field(default_factory=BlockFrozenConfig)
    merge_adjacent: bool = True
    max_merged_blocks: int = Field(4, ge=0)  # 0 = unlimited
    max_atoms: int | None = Field(None, ge=1)
    min_atoms: int | None = Field(None, ge=0)
    min_atoms_policy: Literal["warn", "error", "allow"] = "warn"
    cooldown_steps: int = Field(1000, ge=0)
    cooldown_state_path: str | None = None
    vacuum_offset: float = Field(2.0, ge=0.0)  # PHAL hard-coded (block.py:511-515)
    cell_padding: float = Field(4.0, ge=0.0)

    @field_validator("size")
    @classmethod
    def _positive_size(cls, value: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
        if value is not None and any(float(item) <= 0.0 for item in value):
            raise ValueError("extraction.block.size must contain three positive numbers")
        return value


class ExtractionConfig(BaseModel):
    """``extraction`` configuration section.

    ``extract_radius`` defaults to ``mlip_cutoff + buffer_radius`` as in PHAL. ``max_atoms`` /
    ``min_atoms`` apply to the radial modes (cluster, graph); PHAL trimmed and padded silently,
    here ``max_atoms_policy`` / ``min_atoms_policy`` make that explicit and every trimmed or padded
    atom is recorded in the region metadata.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    mode: ExtractionMode = "cluster"
    extract_radius: float | None = Field(None, ge=0.0)
    mlip_cutoff: float = Field(6.0, ge=0.0)
    buffer_radius: float = Field(4.0, ge=0.0)
    core_radius: float = Field(4.0, ge=0.0)
    boundary_thickness: float = Field(2.0, ge=0.0)
    vacuum_padding: float = Field(4.0, ge=0.0)
    max_atoms: int | None = Field(None, ge=1)
    min_atoms: int | None = Field(None, ge=0)
    max_atoms_policy: Literal["trim", "error"] = "trim"
    min_atoms_policy: Literal["pad", "error", "allow"] = "pad"
    #: ``as_selected`` (default, PHAL behaviour) keeps the radial/graph selection as is.
    #: ``follow_partner``: a terminal atom (exactly one covalent bond, e.g. H) is in the region if and
    #: only if its bonded partner is, so molecular fragments never contain a lone H whose carbon was
    #: cut away and never lose the H of an included carbon. Bonds use ``graph.covalent_scale``.
    terminal_atoms: Literal["as_selected", "follow_partner"] = "as_selected"
    freeze_boundary: bool = True
    periodic_guard: PeriodicGuardConfig = Field(default_factory=PeriodicGuardConfig)
    force_weights: RoleWeights = Field(default_factory=RoleWeights)
    slab: SlabConfig = Field(default_factory=SlabConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    block: BlockConfig = Field(default_factory=BlockConfig)

    @property
    def effective_extract_radius(self) -> float:
        if self.extract_radius is not None:
            return float(self.extract_radius)
        return float(self.mlip_cutoff + self.buffer_radius)

    def identity_payload(self, exclude: set[str]) -> dict[str, Any]:
        """Configuration payload hashed into region IDs; a default ``terminal_atoms`` is left out so
        region IDs of existing configurations do not change."""

        payload = self.model_dump(mode="json", exclude=exclude)
        if payload.get("terminal_atoms") == "as_selected":
            payload.pop("terminal_atoms")
        return payload

    @model_validator(mode="after")
    def _block_needs_size(self) -> ExtractionConfig:
        if self.mode == "block" and self.block.size is None:
            raise ValueError("extraction.mode='block' requires extraction.block.size")
        return self

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> ExtractionConfig:
        return cls.model_validate(data or {})


DEFAULT_BOND_LENGTHS: dict[str, float] = {
    "C": 1.09,
    "Si": 1.48,
    "O": 0.96,
    "N": 1.01,
    "S": 1.34,
    "P": 1.42,
    "B": 1.19,
}


class CappingConfig(BaseModel):
    """``capping`` section (PHAL ``h_capping``).

    ``allow_core_capping`` keeps PHAL's semantics, which differ from its name: it extends capping
    from boundary/outer-buffer atoms to *inner-buffer* atoms; PHAL never capped core atoms
    (``h_capping.py:115-129``). ``cap_core_atoms`` is the explicit switch that really caps core
    atoms. ``uncapped_policy`` decides what a dangling bond that could not be capped means for
    :func:`~gmd_active_learning.extraction.h_capping.check_labelable`.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    allow_core_capping: bool = False
    cap_core_atoms: bool = False
    covalent_scale: float = Field(1.2, gt=0.0)
    bond_lengths: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_BOND_LENGTHS))
    disabled_for_metals: bool = True
    disabled_for_oxides_by_default: bool = True
    optimize_h_only: bool = False  # PHAL wrote the input but never ran it (F-17); labeling decides
    uncapped_policy: Literal["error", "warn", "allow"] = "error"
    partial_policy: Literal["error", "warn", "allow"] = "warn"

    @field_validator("bond_lengths")
    @classmethod
    def _positive_lengths(cls, value: dict[str, float]) -> dict[str, float]:
        if any(float(length) <= 0.0 for length in value.values()):
            raise ValueError("capping.bond_lengths must be positive")
        return {str(key): float(length) for key, length in value.items()}

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> CappingConfig:
        return cls.model_validate(data or {})


class EmbeddingConfig(BaseModel):
    """``embedding`` section: point-charge data only. QM/MM execution is unsupported."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: str = "point_charge"
    charges: dict[str, float] = Field(default_factory=dict)
    default_charge: float | None = None
    cutoff: float = Field(12.0, gt=0.0)
    rigid_tolerance: float = Field(1.0e-6, gt=0.0)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> EmbeddingConfig:
        return cls.model_validate(data or {})


__all__ = [
    "DEFAULT_BOND_LENGTHS",
    "BlockConfig",
    "CappingConfig",
    "EmbeddingConfig",
    "ExtractionConfig",
    "ExtractionMode",
    "GraphConfig",
    "PeriodicGuardConfig",
    "SlabConfig",
]
