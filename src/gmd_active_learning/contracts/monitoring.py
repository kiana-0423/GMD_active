"""Monitor result contract: per-atom metrics with explicit evaluation status."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from gmd_active_learning.core.exceptions import DataContractError


class Stage(str, Enum):
    """Evaluation depth reached for a frame (PHAL's staged light -> physics -> full)."""

    SKIPPED = "skipped"  # frame not scored (e.g. outside light_interval)
    LIGHT = "light"
    PHYSICS = "physics"
    FULL = "full"


class MetricStatus(str, Enum):
    MEASURED = "measured"  # values are real measurements for every evaluated atom
    UNAVAILABLE = "unavailable"  # the metric could not be computed (e.g. no committee configured)
    NOT_EVALUATED = "not_evaluated"  # skipped by staging / lazy evaluation policy


class FrameAction(str, Enum):
    CONTINUE = "continue"
    SAVE = "save"  # capture an event / candidate, keep running
    CHECKPOINT = "checkpoint"  # request a checkpoint, keep running
    STOP = "stop"  # request the MD engine to stop at its next control boundary


@dataclass(slots=True)
class MetricValues:
    """Per-atom values of one metric.

    ``evaluated[i]`` is False for atoms the metric skipped (lazy evaluation); their ``values`` are NaN.
    An ``UNAVAILABLE`` metric has all-NaN values and all-False ``evaluated``; it never reads as zero.
    """

    name: str
    values: np.ndarray
    evaluated: np.ndarray
    status: MetricStatus = MetricStatus.MEASURED
    reason: str | None = None

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        self.evaluated = np.asarray(self.evaluated, dtype=bool)
        if self.values.shape != self.evaluated.shape or self.values.ndim != 1:
            raise DataContractError(f"metric {self.name}: values/evaluated shapes {self.values.shape}/{self.evaluated.shape}")
        if self.status is not MetricStatus.MEASURED and np.any(self.evaluated):
            raise DataContractError(f"metric {self.name}: status {self.status.value} cannot have evaluated atoms")
        if np.any(~np.isfinite(self.values[self.evaluated])):
            raise DataContractError(f"metric {self.name}: evaluated values must be finite")

    @classmethod
    def measured(cls, name: str, values: np.ndarray, evaluated: np.ndarray | None = None) -> MetricValues:
        values = np.asarray(values, dtype=float)
        mask = np.ones(values.shape, dtype=bool) if evaluated is None else np.asarray(evaluated, dtype=bool)
        return cls(name, np.where(mask, values, np.nan), mask, MetricStatus.MEASURED)

    @classmethod
    def unavailable(cls, name: str, n_atoms: int, reason: str) -> MetricValues:
        return cls(name, np.full(n_atoms, np.nan), np.zeros(n_atoms, dtype=bool), MetricStatus.UNAVAILABLE, reason)

    @classmethod
    def not_evaluated(cls, name: str, n_atoms: int, reason: str) -> MetricValues:
        return cls(name, np.full(n_atoms, np.nan), np.zeros(n_atoms, dtype=bool), MetricStatus.NOT_EVALUATED, reason)

    def max(self) -> float | None:
        return float(np.max(self.values[self.evaluated])) if np.any(self.evaluated) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "reason": self.reason,
            "values": [None if not flag else float(value) for value, flag in zip(self.values, self.evaluated)],
        }


@dataclass(slots=True)
class MonitorResult:
    """Outcome of monitoring one frame.

    ``hotspot_atom_ids`` are stable atom IDs (not frame indices). ``atom_scores`` combine the
    evaluated metrics; atoms with no evaluated metric score 0 and are never hotspots.
    """

    run_id: str
    step: int
    stage: Stage
    atom_scores: np.ndarray
    metrics: dict[str, MetricValues]
    max_score: float
    triggered: bool
    hotspot_atom_ids: list[int]
    trigger_reasons: list[str]
    action: FrameAction = FrameAction.CONTINUE
    metadata: dict[str, Any] = field(default_factory=dict)

    def evaluated_metrics(self) -> list[str]:
        return sorted(name for name, metric in self.metrics.items() if metric.status is MetricStatus.MEASURED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step": self.step,
            "stage": self.stage.value,
            "max_score": self.max_score,
            "triggered": self.triggered,
            "hotspot_atom_ids": list(map(int, self.hotspot_atom_ids)),
            "trigger_reasons": list(self.trigger_reasons),
            "action": self.action.value,
            "atom_scores": np.asarray(self.atom_scores, dtype=float).tolist(),
            "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()},
            "metadata": self.metadata,
        }
