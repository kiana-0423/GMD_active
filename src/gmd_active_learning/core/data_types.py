from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class LJFitResult:
    pair_params_ab: dict[str, dict[str, float]]
    pair_params_epsilon_sigma: dict[str, dict[str, float | None]]
    relative_force_residual: float
    per_atom_residual: np.ndarray
    is_physical: bool
    warnings: list[str]
    design_matrix_rank: int
    condition_number: float

    def to_json_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["per_atom_residual"] = self.per_atom_residual.tolist()
        return result


@dataclass(slots=True)
class EnsembleDeviationResult:
    per_atom_force_deviation: np.ndarray
    max_force_deviation: float
    mean_force_deviation: float
    min_force_deviation: float


@dataclass(slots=True)
class PhysicalCheckResult:
    max_force: float
    min_distance: float
    has_close_contact: bool
    energy_drift: float | None = None
    temperature_anomaly: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass(slots=True)
class ReliabilityResult:
    status: str
    risk_score: float
    metrics: dict[str, float | bool | None]
    reasons: list[str]
    lj_fit_result: LJFitResult | None
    should_save_frame: bool
    should_stop_md: bool

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.lj_fit_result is not None:
            payload["lj_fit_result"] = self.lj_fit_result.to_json_dict()
        return payload


@dataclass(slots=True)
class MDFrame:
    step: int
    positions: np.ndarray
    symbols: list[str]
    forces: np.ndarray
    cell: np.ndarray | None = None
    pbc: tuple[bool, bool, bool] = (False, False, False)
    energy: float | None = None
    ensemble_forces: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MDRunResult:
    frames: list[MDFrame]
    stopped_early: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateRecord:
    candidate_id: str
    path: str
    risk_score: float
    reasons: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
