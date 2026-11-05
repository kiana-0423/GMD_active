"""Frame-level ``global_lj`` engine: GMD's ReliabilityMonitor exposed through the MonitorResult contract.

It fits per-species-pair LJ parameters over a rolling window of frames (``mode=global_pair``,
sign-corrected), runs the frame-global physical checks and the committee deviation, and maps
SAFE / CANDIDATE / STOP onto ``triggered`` and ``FrameAction``. It cannot localise anomalies, so
the hotspot atoms are the atoms with the largest per-atom LJ fit residual of the current frame.

Metrics reported per atom: ``force`` (|F|), ``lj_global_residual`` (|F_fit - F| in eV/A, current
frame) and ``committee`` (UNAVAILABLE without committee forces). ``max_score`` is the frame-level
risk score of the ReliabilityMonitor; ``atom_scores`` are the per-atom LJ residuals.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from gmd_active_learning.contracts.frame import Frame
from gmd_active_learning.contracts.monitoring import FrameAction, MetricValues, MonitorResult, Stage
from gmd_active_learning.core.constants import CANDIDATE, STOP
from gmd_active_learning.core.exceptions import MonitorError
from gmd_active_learning.monitors.config import MonitorConfig, load_monitor_config
from gmd_active_learning.monitors.ensemble_deviation import committee_force_deviation
from gmd_active_learning.monitors.force import force_norms
from gmd_active_learning.monitors.reliability_monitor import ReliabilityMonitor

logger = logging.getLogger(__name__)


class GlobalLJMonitor:
    engine = "global_lj"

    def __init__(self, config: MonitorConfig | dict[str, Any] | None = None) -> None:
        self.config = load_monitor_config(config)
        self.monitor = ReliabilityMonitor(self.config.global_lj.as_reliability_config())
        self.frame_index = 0
        self._previous_lj_params: dict[str, dict[str, float]] | None = None

    def evaluate(self, frame: Frame) -> MonitorResult:
        if frame.forces is None:
            raise MonitorError(f"frame {frame.run_id}:{frame.step} has no forces")
        n_atoms = frame.n_atoms
        committee = frame.committee_forces
        use_committee = committee is not None and len(committee) >= self.config.metrics.committee.min_members
        reliability = self.monitor.evaluate(
            step=frame.step,
            positions=frame.positions,
            symbols=frame.symbols,
            ml_forces=frame.forces,
            cell=frame.cell,
            pbc=frame.pbc,
            energy=frame.energy,
            ensemble_forces=committee if use_committee else None,
            previous_lj_params=self._previous_lj_params,
        )
        lj_fit = reliability.lj_fit_result
        per_atom = np.zeros(n_atoms, dtype=float)
        if lj_fit is not None and len(lj_fit.per_atom_residual) >= n_atoms:
            per_atom = np.asarray(lj_fit.per_atom_residual[-n_atoms:], dtype=float)
            self._previous_lj_params = {
                pair: {key: float(value) for key, value in params.items() if value is not None}
                for pair, params in lj_fit.pair_params_epsilon_sigma.items()
            }
        metrics = {
            "force": MetricValues.measured("force", force_norms(frame.forces)),
            "lj_global_residual": MetricValues.measured("lj_global_residual", per_atom),
            "committee": (
                MetricValues.measured("committee", committee_force_deviation(committee))
                if use_committee
                else MetricValues.unavailable("committee", n_atoms, "no committee forces (or too few members)")
            ),
        }
        triggered = reliability.status in {CANDIDATE, STOP}
        hotspot_ids: list[int] = []
        if triggered and n_atoms:
            hotspot_ids = [int(frame.atom_ids[index]) for index in np.flatnonzero(per_atom == np.max(per_atom))]
        action = FrameAction.CONTINUE
        if reliability.should_save_frame and self.config.actions.save_on_trigger:
            action = FrameAction.SAVE
        if reliability.should_stop_md:
            action = FrameAction.STOP
        result = MonitorResult(
            run_id=frame.run_id,
            step=frame.step,
            stage=Stage.FULL if use_committee else Stage.PHYSICS,
            atom_scores=per_atom,
            metrics=metrics,
            max_score=float(reliability.risk_score),
            triggered=triggered,
            hotspot_atom_ids=hotspot_ids,
            trigger_reasons=list(reliability.reasons),
            action=action,
            metadata={
                "frame_index": self.frame_index,
                "engine": self.engine,
                "status": reliability.status,
                "frame_metrics": {key: value for key, value in reliability.metrics.items()},
            },
        )
        self.frame_index += 1
        return result

    def to_state(self) -> dict[str, Any]:
        window = [
            {
                "positions": np.asarray(item["positions"]).tolist(),
                "symbols": list(item["symbols"]),  # type: ignore[arg-type]
                "ml_forces": np.asarray(item["ml_forces"]).tolist(),
                "cell": None if item["cell"] is None else np.asarray(item["cell"]).tolist(),
                "pbc": list(item["pbc"]),  # type: ignore[arg-type]
            }
            for item in self.monitor.window
        ]
        return {
            "engine": self.engine,
            "frame_index": self.frame_index,
            "window": window,
            "previous_lj_params": self._previous_lj_params,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if state.get("engine") != self.engine:
            raise MonitorError(f"cannot load {state.get('engine')!r} monitor state into the {self.engine!r} monitor")
        self.frame_index = int(state.get("frame_index", 0))
        self.monitor.window.clear()
        for item in state.get("window", []):
            self.monitor.window.append(
                {
                    "positions": np.asarray(item["positions"], dtype=float),
                    "symbols": list(item["symbols"]),
                    "ml_forces": np.asarray(item["ml_forces"], dtype=float),
                    "cell": None if item["cell"] is None else np.asarray(item["cell"], dtype=float),
                    "pbc": tuple(item["pbc"]),
                }
            )
        self._previous_lj_params = state.get("previous_lj_params")


__all__ = ["GlobalLJMonitor"]
