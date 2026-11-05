"""Labeling contracts. Execution status and scientific validation are tracked separately:
a process that exits 0 is not, by itself, a valid label."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from gmd_active_learning.core.exceptions import DataContractError


class ExecutionStatus(str, Enum):
    PREPARED = "prepared"
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"  # process finished; says nothing about scientific validity
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"  # scheduler could not tell; reconcile before any retry

    @property
    def terminal(self) -> bool:
        return self in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}


class ValidationStatus(str, Enum):
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"


@dataclass(slots=True)
class SchedulerJob:
    job_id: str
    backend: str
    state: ExecutionStatus
    work_dir: str
    submitted_at: str | None = None
    exit_code: int | None = None
    raw_state: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LabelTask:
    """One labeling calculation for one region.

    ``task_id`` is derived from the region ID and the engine settings hash, so preparing the same
    region twice yields the same task (idempotent submission).
    """

    task_id: str
    region_id: str
    engine: str
    settings_hash: str
    work_dir: str
    attempt: int = 0
    execution: ExecutionStatus = ExecutionStatus.PREPARED
    validation: ValidationStatus = ValidationStatus.PENDING
    job: SchedulerJob | None = None
    failure_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LabelResult:
    """Parsed and validated output of a labeling calculation, in eV / eV/A / eV/A^3.

    ``energy`` refers to the labeled region (extracted, possibly capped), never to the parent frame.
    """

    task_id: str
    region_id: str
    forces: np.ndarray
    energy: float | None
    converged: bool
    validation: ValidationStatus
    validation_reasons: list[str] = field(default_factory=list)
    stress: np.ndarray | None = None
    scf_steps: int | None = None
    engine: str = "cp2k"
    engine_version: str | None = None
    output_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.forces = np.asarray(self.forces, dtype=float)
        if self.forces.ndim != 2 or self.forces.shape[1] != 3:
            raise DataContractError(f"label {self.task_id}: forces shape {self.forces.shape}")
        if self.stress is not None:
            self.stress = np.asarray(self.stress, dtype=float).reshape(3, 3)
        if self.validation is ValidationStatus.VALID:
            if not self.converged:
                raise DataContractError(f"label {self.task_id}: an unconverged result cannot be VALID")
            if not np.all(np.isfinite(self.forces)) or (self.energy is not None and not np.isfinite(self.energy)):
                raise DataContractError(f"label {self.task_id}: a VALID result must be finite")
