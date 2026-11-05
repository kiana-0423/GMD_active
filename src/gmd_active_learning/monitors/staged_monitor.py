# Ported from PHAL (hotspot_al.monitor.online_monitor.OnlineMonitor.process_frame and interval rules)
# at 96263b6; MIT License, see NOTICE.
"""Staged per-frame OOD monitor: light -> physics -> full, producing a ``MonitorResult``.

Per frame (``frame_index`` counts processed frames, persisted across segments):

1. Light metrics are always computed (force, delta_force, displacement, rmin, delta_q) so the
   temporal references and the hard STOP rules stay current.
2. ``light_interval`` not due -> stage SKIPPED (no score, no stats update).
3. ``scheduled_full_interval`` k > 1 and ``frame_index % k == 0`` -> everything is evaluated (FULL).
4. Otherwise score LIGHT; escalate to PHYSICS when ``physics_interval`` is due (``triggered`` = the
   light stage triggered), computing the lazy local-LJ residual and the MLIP force deviation; then to
   FULL when ``full_interval`` is due, adding the committee deviation.

Each metric is computed at most once per frame (PHAL fitted LJ twice per escalated frame, F-16), and
its :class:`~gmd_active_learning.contracts.MetricStatus` records whether it was measured, skipped
by staging/laziness/config (NOT_EVALUATED) or impossible (UNAVAILABLE, e.g. no committee, F-06).
Running statistics are updated once, after the final stage, with evaluated atoms only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gmd_active_learning.contracts.frame import Frame
from gmd_active_learning.contracts.monitoring import FrameAction, MetricStatus, MetricValues, MonitorResult, Stage
from gmd_active_learning.core.exceptions import MonitorError
from gmd_active_learning.monitors.config import ActionsConfig, IntervalSpec, MonitorConfig, load_monitor_config
from gmd_active_learning.monitors.coordination import coordination_deltas, smooth_coordination_numbers
from gmd_active_learning.monitors.ensemble_deviation import committee_force_deviation
from gmd_active_learning.monitors.force import delta_force_norms, force_norms
from gmd_active_learning.monitors.force_disagreement import (
    CommitteeForceProvider,
    ReferenceForceProvider,
    checked_reference_forces,
    mlip_force_deviation,
)
from gmd_active_learning.monitors.geometry import displacement_norms, minimum_neighbor_distances
from gmd_active_learning.monitors.lj_local import LocalLJEstimator, LocalLJFit
from gmd_active_learning.monitors.scoring import LIGHT_METRICS, METRIC_NAMES, OODScorer, StageScore
from gmd_active_learning.utils.neighbors import MonitorNeighbors, NeighborPairs

logger = logging.getLogger(__name__)

LJEstimatorFn = Callable[[NeighborPairs, np.ndarray, "np.ndarray | None"], tuple[np.ndarray, list[LocalLJFit]]]
TEMPORAL_METRICS = ("delta_force", "displacement", "delta_q")


def interval_due(spec: IntervalSpec, frame_index: int, *, triggered: bool) -> bool:
    """PHAL ``_interval_due``: ``triggered`` follows the previous stage; integers k fire on frame_index % k == 0."""

    if isinstance(spec, str):
        if spec == "triggered":
            return triggered
        if spec in {"never", "disabled", "off"}:
            return False
        if spec in {"always", "all"}:
            return True
        raise MonitorError(f"unknown interval keyword {spec!r}")
    return spec > 0 and frame_index % spec == 0


def decide_action(
    actions: ActionsConfig,
    *,
    triggered: bool,
    max_score: float,
    metrics: dict[str, MetricValues],
) -> tuple[FrameAction, list[str]]:
    """Frame action with precedence stop > checkpoint > save > continue, and the rules that fired."""

    reasons: list[str] = []
    action = FrameAction.CONTINUE
    if triggered and actions.save_on_trigger:
        action = FrameAction.SAVE
        reasons.append("triggered")
    if actions.checkpoint_score is not None and max_score >= actions.checkpoint_score:
        action = FrameAction.CHECKPOINT
        reasons.append("checkpoint_score")
    stop_reasons: list[str] = []
    if actions.stop_score is not None and max_score >= actions.stop_score:
        stop_reasons.append("stop_score")
    rmin = metrics.get("rmin")
    if actions.stop_min_distance is not None and rmin is not None and rmin.status is MetricStatus.MEASURED:
        if np.any(rmin.values[rmin.evaluated] < actions.stop_min_distance):
            stop_reasons.append("stop_min_distance")
    force = metrics.get("force")
    if actions.stop_max_force is not None and force is not None and force.status is MetricStatus.MEASURED:
        if np.any(force.values[force.evaluated] > actions.stop_max_force):
            stop_reasons.append("stop_max_force")
    if stop_reasons:
        action = FrameAction.STOP
        reasons.extend(stop_reasons)
    return action, reasons


@dataclass(slots=True)
class _TemporalState:
    step: int
    atom_ids: np.ndarray
    positions: np.ndarray
    forces: np.ndarray
    q_values: np.ndarray | None

    def aligned_to(self, atom_ids: np.ndarray) -> _TemporalState | None:
        """The previous frame in the current frame's atom order, or None if the atom set changed."""

        if np.array_equal(self.atom_ids, atom_ids):
            return self
        if len(self.atom_ids) != len(atom_ids) or set(self.atom_ids.tolist()) != set(atom_ids.tolist()):
            return None
        lookup = {int(atom_id): index for index, atom_id in enumerate(self.atom_ids)}
        order = np.asarray([lookup[int(atom_id)] for atom_id in atom_ids], dtype=np.int64)
        return _TemporalState(
            self.step,
            self.atom_ids[order],
            self.positions[order],
            self.forces[order],
            None if self.q_values is None else self.q_values[order],
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "atom_ids": self.atom_ids.tolist(),
            "positions": self.positions.tolist(),
            "forces": self.forces.tolist(),
            "q_values": None if self.q_values is None else self.q_values.tolist(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> _TemporalState:
        return cls(
            int(state["step"]),
            np.asarray(state["atom_ids"], dtype=np.int64),
            np.asarray(state["positions"], dtype=float).reshape(-1, 3),
            np.asarray(state["forces"], dtype=float).reshape(-1, 3),
            None if state.get("q_values") is None else np.asarray(state["q_values"], dtype=float),
        )


@dataclass(slots=True)
class StagedEvaluation:
    """A ``MonitorResult`` plus the per-metric scores and LJ fits behind it (diagnostics, parity)."""

    result: MonitorResult
    metric_scores: dict[str, np.ndarray] = field(default_factory=dict)
    hotspot_indices: list[int] = field(default_factory=list)
    lj_fits: list[LocalLJFit] | None = None
    stage_scores: dict[str, StageScore] = field(default_factory=dict)


class StagedOODMonitor:
    """PHAL's atom-wise staged OOD monitor on :class:`~gmd_active_learning.contracts.Frame` objects."""

    engine = "staged"

    def __init__(
        self,
        config: MonitorConfig | dict[str, Any] | None = None,
        *,
        reference_forces: ReferenceForceProvider | None = None,
        committee_provider: CommitteeForceProvider | None = None,
        lj_estimator: LJEstimatorFn | None = None,
    ) -> None:
        self.config = load_monitor_config(config)
        self.reference_forces = reference_forces
        self.committee_provider = committee_provider
        lj_cfg = self.config.metrics.lj_residual
        self.lj_estimator: LJEstimatorFn = lj_estimator or LocalLJEstimator(
            estimator=lj_cfg.estimator,
            cutoff=self.config.neighbors.lj_cutoff,
            delta=lj_cfg.relative_delta,
            condition_limit=lj_cfg.condition_limit,
            sigma_min=lj_cfg.sigma_min,
            sigma_max=lj_cfg.sigma_max,
            nonlinear_max_nfev=lj_cfg.nonlinear_max_nfev,
        )
        self.scorer = OODScorer(self.config)
        self.neighbors = MonitorNeighbors(
            lj_cutoff=self.config.neighbors.lj_cutoff,
            coordination_cutoff=self.config.neighbors.coordination_cutoff,
        )
        self.frame_index = 0
        self._previous: _TemporalState | None = None
        self.counters = {"reference_force_calls": 0, "committee_provider_calls": 0, "lj_fits": 0}

    # -- public API ------------------------------------------------------------------------------

    def evaluate(self, frame: Frame) -> MonitorResult:
        return self.evaluate_detailed(frame).result

    def evaluate_detailed(self, frame: Frame) -> StagedEvaluation:
        if not self.config.enabled:
            return self._disabled_result(frame)
        if frame.forces is None:
            raise MonitorError(f"frame {frame.run_id}:{frame.step} has no forces; online monitoring needs force-bearing frames")
        frame_index = self.frame_index
        metrics_cfg = self.config.metrics
        stages_cfg = self.config.stages
        n_atoms = frame.n_atoms
        previous = None if self._previous is None else self._previous.aligned_to(frame.atom_ids)
        if self._previous is not None and previous is None:
            logger.warning("atom set changed at step %d; temporal metrics restart", frame.step)

        pairs = self._neighbor_pairs(frame)
        metrics, q_values = self._light_metrics(frame, pairs, previous)

        evaluation = StagedEvaluation(result=None)  # type: ignore[arg-type]
        scheduled_full = stages_cfg.scheduled_full_interval > 1 and frame_index % stages_cfg.scheduled_full_interval == 0
        cache: dict[str, np.ndarray] = {}
        metadata: dict[str, Any] = {"frame_index": frame_index}
        lj_fits: list[LocalLJFit] | None = None

        if not interval_due(stages_cfg.light_interval, frame_index, triggered=True):
            stage = Stage.SKIPPED
            final: StageScore | None = None
            metadata["skipped_by_light_interval"] = True
        elif scheduled_full:
            light = self.scorer.score_stage(metrics, Stage.LIGHT, n_atoms=n_atoms, cache=cache)
            lj_fits = self._add_physics_metrics(frame, pairs, metrics, light)
            self._add_committee_metric(frame, metrics)
            final = self.scorer.score_stage(metrics, Stage.FULL, n_atoms=n_atoms, cache=cache)
            stage = Stage.FULL
            metadata.update(scheduled_full=True, screen_max_score=light.max_score)
            evaluation.stage_scores["light"] = light
        else:
            light = self.scorer.score_stage(metrics, Stage.LIGHT, n_atoms=n_atoms, cache=cache)
            evaluation.stage_scores["light"] = light
            final, stage = light, Stage.LIGHT
            if interval_due(stages_cfg.physics_interval, frame_index, triggered=light.triggered):
                lj_fits = self._add_physics_metrics(frame, pairs, metrics, light)
                physics = self.scorer.score_stage(metrics, Stage.PHYSICS, n_atoms=n_atoms, cache=cache)
                evaluation.stage_scores["physics"] = physics
                metadata["screen_max_score"] = light.max_score
                final, stage = physics, Stage.PHYSICS
                if interval_due(stages_cfg.full_interval, frame_index, triggered=physics.triggered):
                    self._add_committee_metric(frame, metrics)
                    final = self.scorer.score_stage(metrics, Stage.FULL, n_atoms=n_atoms, cache=cache)
                    stage = Stage.FULL
                    metadata["physics_max_score"] = physics.max_score

        self._mark_unreached(metrics, n_atoms, stage)
        if final is not None:
            self.scorer.update_stats(metrics)
            atom_scores, triggered, max_score = final.atom_scores, final.triggered, final.max_score
            hotspot_indices, reasons = final.hotspot_indices, final.trigger_reasons
            evaluation.metric_scores = dict(final.metric_scores)
            metadata["stage_threshold"] = final.threshold
        else:
            atom_scores, triggered, max_score = np.zeros(n_atoms, dtype=float), False, 0.0
            hotspot_indices, reasons = [], []

        if lj_fits is not None:
            evaluated = [fit for fit in lj_fits if fit.evaluated]
            metadata["lj_fit_count"] = len(evaluated)
            metadata["lj_invalid_count"] = sum(1 for fit in evaluated if not fit.valid)
        metadata.update(
            screen_threshold=stages_cfg.screen_threshold,
            physics_threshold=stages_cfg.physics_threshold,
            label_threshold=stages_cfg.label_threshold,
        )
        action, action_reasons = decide_action(self.config.actions, triggered=triggered, max_score=max_score, metrics=metrics)
        metadata["action_reasons"] = action_reasons
        result = MonitorResult(
            run_id=frame.run_id,
            step=frame.step,
            stage=stage,
            atom_scores=np.asarray(atom_scores, dtype=float),
            metrics={name: metrics[name] for name in METRIC_NAMES},
            max_score=float(max_score),
            triggered=bool(triggered),
            hotspot_atom_ids=[int(frame.atom_ids[index]) for index in hotspot_indices],
            trigger_reasons=list(reasons),
            action=action,
            metadata=metadata,
        )
        evaluation.result = result
        evaluation.hotspot_indices = list(hotspot_indices)
        evaluation.lj_fits = lj_fits
        self._previous = _TemporalState(
            frame.step,
            frame.atom_ids.copy(),
            frame.positions.copy(),
            np.asarray(frame.forces, dtype=float).copy(),
            None if q_values is None else q_values.copy(),
        )
        self.frame_index += 1
        if triggered:
            logger.info(
                "OOD trigger step=%d stage=%s max_score=%.3f hotspots=%s reasons=%s",
                frame.step, stage.value, max_score, result.hotspot_atom_ids, reasons,
            )
        return evaluation

    def reset_temporal_state(self) -> None:
        """Forget the previous frame (e.g. after a discontinuous restart); statistics are kept."""

        self._previous = None

    def to_state(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "frame_index": self.frame_index,
            "stats": self.scorer.to_state(),
            "previous": None if self._previous is None else self._previous.to_state(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if state.get("engine", self.engine) != self.engine:
            raise MonitorError(f"cannot load {state.get('engine')!r} monitor state into the {self.engine!r} monitor")
        self.frame_index = int(state.get("frame_index", 0))
        self.scorer.load_state(state.get("stats", {}))
        previous = state.get("previous")
        self._previous = None if previous is None else _TemporalState.from_state(previous)

    # -- metric computation ----------------------------------------------------------------------

    def _neighbor_pairs(self, frame: Frame) -> NeighborPairs | None:
        metrics_cfg = self.config.metrics
        if not (metrics_cfg.rmin.enabled or metrics_cfg.delta_q.enabled or metrics_cfg.lj_residual.enabled):
            return None
        return self.neighbors.rebuild(frame.positions, frame.cell, frame.pbc)

    def _temporal(self, name: str, values: np.ndarray | None, n_atoms: int) -> MetricValues:
        if values is not None:
            return MetricValues.measured(name, values)
        if self.config.running_stats.first_frame_deltas == "zero":
            return MetricValues.measured(name, np.zeros(n_atoms, dtype=float))
        return MetricValues.not_evaluated(name, n_atoms, "no previous frame")

    def _light_metrics(
        self,
        frame: Frame,
        pairs: NeighborPairs | None,
        previous: _TemporalState | None,
    ) -> tuple[dict[str, MetricValues], np.ndarray | None]:
        cfg = self.config.metrics
        n_atoms = frame.n_atoms
        forces = np.asarray(frame.forces, dtype=float)
        metrics: dict[str, MetricValues] = {}
        metrics["force"] = (
            MetricValues.measured("force", force_norms(forces)) if cfg.force.enabled else MetricValues.not_evaluated("force", n_atoms, "disabled")
        )
        if cfg.delta_force.enabled:
            values = None if previous is None else delta_force_norms(forces, previous.forces)
            metrics["delta_force"] = self._temporal("delta_force", values, n_atoms)
        else:
            metrics["delta_force"] = MetricValues.not_evaluated("delta_force", n_atoms, "disabled")
        if cfg.displacement.enabled:
            values = (
                None if previous is None else displacement_norms(frame.positions, previous.positions, cell=frame.cell, pbc=frame.pbc)
            )
            metrics["displacement"] = self._temporal("displacement", values, n_atoms)
        else:
            metrics["displacement"] = MetricValues.not_evaluated("displacement", n_atoms, "disabled")
        if cfg.rmin.enabled and pairs is not None:
            metrics["rmin"] = MetricValues.measured("rmin", minimum_neighbor_distances(pairs))
        else:
            metrics["rmin"] = MetricValues.not_evaluated("rmin", n_atoms, "disabled")
        q_values = None
        if cfg.delta_q.enabled and pairs is not None:
            q_values = smooth_coordination_numbers(
                pairs,
                frame.numbers,
                cutoff=self.config.neighbors.coordination_cutoff,
                scale=cfg.delta_q.scale,
                power=cfg.delta_q.power,
            )
            previous_q = None if previous is None else previous.q_values
            values = None if previous_q is None else coordination_deltas(q_values, previous_q)
            metrics["delta_q"] = self._temporal("delta_q", values, n_atoms)
        else:
            metrics["delta_q"] = MetricValues.not_evaluated("delta_q", n_atoms, "disabled")
        return metrics, q_values

    def _add_physics_metrics(
        self,
        frame: Frame,
        pairs: NeighborPairs | None,
        metrics: dict[str, MetricValues],
        light: StageScore,
    ) -> list[LocalLJFit] | None:
        cfg = self.config.metrics
        n_atoms = frame.n_atoms
        forces = np.asarray(frame.forces, dtype=float)
        lj_fits: list[LocalLJFit] | None = None
        if not cfg.lj_residual.enabled or pairs is None:
            metrics["lj_residual"] = MetricValues.not_evaluated("lj_residual", n_atoms, "disabled")
        else:
            mask = self._lazy_lj_mask(light.atom_scores)
            if not np.any(mask):
                metrics["lj_residual"] = MetricValues.not_evaluated("lj_residual", n_atoms, "lazy: no suspicious atom")
                lj_fits = []
            else:
                residuals, lj_fits = self.lj_estimator(pairs, forces, mask)
                evaluated = np.fromiter((fit.evaluated for fit in lj_fits), dtype=bool, count=len(lj_fits))
                self.counters["lj_fits"] += int(np.sum(evaluated))
                metrics["lj_residual"] = MetricValues.measured("lj_residual", residuals, evaluated)
        if not cfg.mlip_force_deviation.enabled:
            metrics["mlip_force_deviation"] = MetricValues.not_evaluated("mlip_force_deviation", n_atoms, "disabled")
        elif self.reference_forces is None:
            metrics["mlip_force_deviation"] = MetricValues.unavailable(
                "mlip_force_deviation", n_atoms, "no reference force provider configured"
            )
        else:
            self.counters["reference_force_calls"] += 1
            reference = checked_reference_forces(frame, self.reference_forces(frame))
            metrics["mlip_force_deviation"] = MetricValues.measured("mlip_force_deviation", mlip_force_deviation(forces, reference))
        return lj_fits

    def _lazy_lj_mask(self, light_scores: np.ndarray) -> np.ndarray:
        n_atoms = len(light_scores)
        if not self.config.metrics.lj_residual.lazy:
            return np.ones(n_atoms, dtype=bool)
        mask = light_scores >= self.config.metrics.lj_residual.lazy_threshold
        if not np.any(mask) and n_atoms:
            mask[int(np.argmax(light_scores))] = bool(np.max(light_scores) > 0.0)
        return mask

    def _add_committee_metric(self, frame: Frame, metrics: dict[str, MetricValues]) -> None:
        cfg = self.config.metrics.committee
        n_atoms = frame.n_atoms
        if not cfg.enabled:
            metrics["committee"] = MetricValues.not_evaluated("committee", n_atoms, "disabled")
            return
        committee = frame.committee_forces
        if committee is None and self.committee_provider is not None:
            self.counters["committee_provider_calls"] += 1
            committee = np.asarray(self.committee_provider(frame), dtype=float)
            if committee.ndim != 3 or committee.shape[1:] != (n_atoms, 3) or not np.all(np.isfinite(committee)):
                raise MonitorError(f"committee provider returned invalid forces of shape {committee.shape} for step {frame.step}")
        if committee is None:
            metrics["committee"] = MetricValues.unavailable("committee", n_atoms, "no committee forces")
        elif len(committee) < cfg.min_members:
            metrics["committee"] = MetricValues.unavailable(
                "committee", n_atoms, f"committee has {len(committee)} member(s), need >= {cfg.min_members}"
            )
        else:
            metrics["committee"] = MetricValues.measured("committee", committee_force_deviation(committee))

    @staticmethod
    def _mark_unreached(metrics: dict[str, MetricValues], n_atoms: int, stage: Stage) -> None:
        for name in METRIC_NAMES:
            if name not in metrics:
                reason = "light_interval" if stage is Stage.SKIPPED else f"not escalated beyond {stage.value}"
                metrics[name] = MetricValues.not_evaluated(name, n_atoms, reason)

    def _disabled_result(self, frame: Frame) -> StagedEvaluation:
        n_atoms = frame.n_atoms
        result = MonitorResult(
            run_id=frame.run_id,
            step=frame.step,
            stage=Stage.SKIPPED,
            atom_scores=np.zeros(n_atoms, dtype=float),
            metrics={name: MetricValues.not_evaluated(name, n_atoms, "monitor disabled") for name in METRIC_NAMES},
            max_score=0.0,
            triggered=False,
            hotspot_atom_ids=[],
            trigger_reasons=[],
            action=FrameAction.CONTINUE,
            metadata={"monitor_disabled": True},
        )
        return StagedEvaluation(result=result)


__all__ = ["LIGHT_METRICS", "StagedEvaluation", "StagedOODMonitor", "decide_action", "interval_due"]
