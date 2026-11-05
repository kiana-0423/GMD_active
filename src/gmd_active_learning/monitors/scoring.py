# Ported from PHAL (hotspot_al.monitor.ood_score) at 96263b6; MIT License, see NOTICE.
"""Atom-wise OOD scoring: per-metric score transforms, staged aggregation, trigger and reasons.

Inputs are :class:`~gmd_active_learning.contracts.MetricValues` (with explicit evaluated masks), so a
metric that was not evaluated or is unavailable contributes exactly nothing and never enters the
running statistics (PHAL zero-filled absent committee/MLIP metrics, F-06).

Per-metric scores (all >= 0, zero for unevaluated atoms):

=====================  =====================================================================
force, delta_force,    gated z: z if z >= z_threshold else 0
displacement
rmin                   max(0, (threshold - rmin) / threshold); threshold null -> 0
delta_q                max(z, max(0, (|dq| - threshold) / threshold))
lj_residual            max(residual, z)   (raw dimensionless residual competes with z, as PHAL)
committee              max(sigma, z)      (raw eV/A competes with z, as PHAL)
mlip_force_deviation   z
=====================  =====================================================================

z = |x - mean| / max(std, min_std) once a metric has ``warmup_frames`` stats-updating frames.
During warmup ``warmup_policy=suppress`` (default) drops the z-based parts; ``raw`` reproduces PHAL,
which compared raw values (eV/A, A) against z thresholds (F-16). With running stats disabled the
z-score is replaced by |x| (PHAL).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from gmd_active_learning.contracts.monitoring import MetricStatus, MetricValues, Stage
from gmd_active_learning.monitors.config import MonitorConfig
from gmd_active_learning.monitors.running_stats import RunningMetricStats

logger = logging.getLogger(__name__)

METRIC_NAMES: tuple[str, ...] = (
    "force",
    "delta_force",
    "rmin",
    "delta_q",
    "lj_residual",
    "committee",
    "displacement",
    "mlip_force_deviation",
)
REASON_MAP: dict[str, str] = {
    "force": "force_large",
    "delta_force": "force_burst",
    "rmin": "close_contact",
    "delta_q": "coordination_change",
    "lj_residual": "lj_residual",
    "committee": "committee_deviation",
    "displacement": "trajectory_jump",
    "mlip_force_deviation": "model_drift",
}
LIGHT_METRICS: tuple[str, ...] = ("force", "delta_force", "rmin", "delta_q", "displacement")
PHYSICS_METRICS: tuple[str, ...] = (*LIGHT_METRICS, "lj_residual", "mlip_force_deviation")
FULL_METRICS: tuple[str, ...] = (*LIGHT_METRICS, "lj_residual", "committee", "mlip_force_deviation")


def stage_metrics(stage: Stage) -> tuple[str, ...]:
    if stage is Stage.LIGHT:
        return LIGHT_METRICS
    if stage is Stage.PHYSICS:
        return PHYSICS_METRICS
    if stage is Stage.FULL:
        return FULL_METRICS
    return ()


@dataclass(slots=True)
class StageScore:
    """Aggregated outcome of one stage. ``metric_scores`` holds the stage's metrics in stage order."""

    stage: Stage
    atom_scores: np.ndarray
    metric_scores: dict[str, np.ndarray]
    max_score: float
    triggered: bool
    hotspot_indices: list[int]
    trigger_reasons: list[str]
    threshold: float


class OODScorer:
    """Turns per-atom metric values into per-atom scores and a staged trigger decision."""

    def __init__(self, config: MonitorConfig) -> None:
        self.config = config
        self.stats: dict[str, RunningMetricStats] = {name: RunningMetricStats() for name in METRIC_NAMES}

    # -- normalisation ---------------------------------------------------------------------------

    def _z_score(self, name: str, values: np.ndarray) -> np.ndarray | None:
        """z-scores, raw |x| (stats disabled / raw warmup), or None (suppressed warmup)."""

        settings = self.config.running_stats
        if not settings.enabled:
            return np.abs(values)
        stats = self.stats[name]
        if stats.frame_count < settings.warmup_frames:
            return np.abs(values) if settings.warmup_policy == "raw" else None
        std = max(stats.std, settings.min_std)
        return np.abs((values - stats.mean) / std)

    @staticmethod
    def _threshold_score(values: np.ndarray, threshold: float | None, *, lower_is_worse: bool = False) -> np.ndarray:
        if threshold is None:
            return np.zeros_like(values, dtype=float)
        scale = max(abs(threshold), 1.0e-8)
        if lower_is_worse:
            return np.clip((threshold - values) / scale, 0.0, None)
        return np.clip((values - threshold) / scale, 0.0, None)

    def metric_score(self, name: str, metric: MetricValues | None) -> np.ndarray:
        """Non-negative per-atom score of one metric; zero wherever the metric was not evaluated."""

        if metric is None:
            raise KeyError(name)
        n_atoms = len(metric.values)
        if metric.status is not MetricStatus.MEASURED or not np.any(metric.evaluated):
            return np.zeros(n_atoms, dtype=float)
        values = np.where(metric.evaluated, metric.values, 0.0)
        metrics_cfg = self.config.metrics
        if name in {"force", "delta_force", "displacement"}:
            z = self._z_score(name, values)
            if z is None:
                score = np.zeros(n_atoms, dtype=float)
            else:
                threshold = getattr(metrics_cfg, name).z_threshold
                score = z if threshold is None else np.where(z >= float(threshold), z, 0.0)
        elif name == "rmin":
            score = self._threshold_score(values, metrics_cfg.rmin.threshold, lower_is_worse=True)
        elif name == "delta_q":
            z = self._z_score(name, values)
            threshold_part = self._threshold_score(values, metrics_cfg.delta_q.threshold)
            score = threshold_part if z is None else np.maximum(z, threshold_part)
        elif name in {"lj_residual", "committee"}:
            z = self._z_score(name, values)
            score = values.copy() if z is None else np.maximum(values, z)
        elif name == "mlip_force_deviation":
            z = self._z_score(name, values)
            score = np.zeros(n_atoms, dtype=float) if z is None else z
        else:
            raise KeyError(f"unknown metric {name!r}")
        return np.where(metric.evaluated, score, 0.0)

    # -- aggregation -----------------------------------------------------------------------------

    def stage_threshold(self, stage: Stage) -> float:
        stages = self.config.stages
        if stage is Stage.LIGHT:
            return float(stages.screen_threshold)
        if stage is Stage.PHYSICS:
            return float(stages.physics_threshold)
        return float(stages.label_threshold)

    def weighted_sum(self, metric_scores: dict[str, np.ndarray], names: tuple[str, ...], n_atoms: int) -> np.ndarray:
        total = np.zeros(n_atoms, dtype=float)
        for name in names:
            total += self.config.metrics.weight(name) * metric_scores[name]
        return total

    def score_stage(
        self,
        metrics: dict[str, MetricValues],
        stage: Stage,
        *,
        n_atoms: int,
        cache: dict[str, np.ndarray] | None = None,
    ) -> StageScore:
        """Aggregate the stage's metrics. ``cache`` (name -> score) avoids recomputing scores across stages."""

        names = stage_metrics(stage)
        scores_cache = {} if cache is None else cache
        metric_scores: dict[str, np.ndarray] = {}
        for name in names:
            if name not in scores_cache:
                metric = metrics.get(name)
                scores_cache[name] = (
                    np.zeros(n_atoms, dtype=float) if metric is None else self.metric_score(name, metric)
                )
            metric_scores[name] = scores_cache[name]
        atom_scores = self.weighted_sum(metric_scores, names, n_atoms)
        threshold = self.stage_threshold(stage)
        stages_cfg = self.config.stages
        max_score = float(np.max(atom_scores)) if n_atoms else 0.0
        triggered = bool(np.sum(atom_scores >= threshold) >= stages_cfg.min_trigger_atoms)
        hotspot_indices = np.flatnonzero(atom_scores >= float(stages_cfg.label_threshold)).tolist()
        if triggered and not hotspot_indices and n_atoms:
            hotspot_indices = np.flatnonzero(atom_scores == np.max(atom_scores)).tolist()
        return StageScore(
            stage=stage,
            atom_scores=atom_scores,
            metric_scores=metric_scores,
            max_score=max_score,
            triggered=triggered,
            hotspot_indices=[int(index) for index in hotspot_indices],
            trigger_reasons=infer_reasons(metric_scores, hotspot_indices),
            threshold=threshold,
        )

    # -- statistics ------------------------------------------------------------------------------

    def update_stats(self, metrics: dict[str, MetricValues]) -> None:
        """Add the evaluated atoms of every measured metric (called once per frame, after scoring)."""

        for name in METRIC_NAMES:
            metric = metrics.get(name)
            if metric is None or metric.status is not MetricStatus.MEASURED:
                continue
            observed = metric.values[metric.evaluated]
            if len(observed):
                self.stats[name].update(observed)

    def to_state(self) -> dict[str, Any]:
        return {name: stats.to_state() for name, stats in self.stats.items()}

    def load_state(self, state: dict[str, Any]) -> None:
        for name in METRIC_NAMES:
            if name in state:
                self.stats[name] = RunningMetricStats.from_state(state[name])


def infer_reasons(metric_scores: dict[str, np.ndarray], hotspot_indices: list[int]) -> list[str]:
    """Reason labels ranked by the mean UNWEIGHTED metric score over hotspot atoms (all atoms if none)."""

    if not metric_scores:
        return []
    sample = next(iter(metric_scores.values()))
    candidate = list(hotspot_indices) if hotspot_indices else list(range(len(sample)))
    ranked: list[tuple[str, float]] = []
    for name, values in metric_scores.items():
        score = float(np.mean(values[candidate])) if candidate else 0.0
        if score > 0.0:
            ranked.append((name, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    reasons: list[str] = []
    for name, _score in ranked:
        label = REASON_MAP.get(name, name)
        if label not in reasons:
            reasons.append(label)
    return reasons


__all__ = [
    "FULL_METRICS",
    "LIGHT_METRICS",
    "METRIC_NAMES",
    "PHYSICS_METRICS",
    "REASON_MAP",
    "OODScorer",
    "StageScore",
    "infer_reasons",
    "stage_metrics",
]
