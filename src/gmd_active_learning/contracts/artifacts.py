"""Immutable dataset versions and model artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetVersion:
    """A published, immutable dataset. ``sha256`` covers every file in ``path``."""

    version: str
    path: str
    sha256: str
    n_structures: int
    n_train: int
    n_val: int
    parent_version: str | None
    sample_ids: tuple[str, ...]
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sample_ids"] = list(self.sample_ids)
        return data


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """A registered model: the files are copied into the registry and never modified."""

    model_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    export_path: str | None
    export_sha256: str | None
    dataset_version: str | None
    parent_model_id: str | None
    species: tuple[int, ...]
    cutoff: float | None
    units: dict[str, str]
    metrics: dict[str, float]
    train_mode: str
    backend: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["species"] = list(self.species)
        return data
