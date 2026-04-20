from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from gmd_active_learning.core.constants import CANDIDATE, SAFE, STOP
from gmd_active_learning.core.data_types import ReliabilityResult
from gmd_active_learning.monitors.base_monitor import BaseMonitor
from gmd_active_learning.monitors.ensemble_deviation import compute_ensemble_deviation
from gmd_active_learning.monitors.lj_projection import LJProjectionFitter, compute_parameter_jump_ratio
from gmd_active_learning.monitors.physical_checks import run_physical_checks
from gmd_active_learning.monitors.risk_score import compute_risk_score


class ReliabilityMonitor(BaseMonitor):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.window: deque[dict[str, object]] = deque(maxlen=int(config.get("window_size", 20)))
        self.fitter = LJProjectionFitter(
            cutoff=float(config.get("cutoff", 6.0)),
            ridge_lambda=float(config.get("ridge_lambda", 1.0e-6)),
            parameter_bounds=config.get("parameter_bounds"),
        )

    def evaluate(
        self,
        *,
        step: int,
        positions: np.ndarray,
        symbols: list[str],
        ml_forces: np.ndarray,
        cell: np.ndarray | None = None,
        pbc: tuple[bool, bool, bool] = (False, False, False),
        energy: float | None = None,
        ensemble_forces: np.ndarray | None = None,
        previous_lj_params: dict[str, dict[str, float]] | None = None,
        unsafe_probability: float | None = None,
    ) -> ReliabilityResult:
        thresholds = self.config.get("thresholds", {})
        frame = {
            "positions": np.asarray(positions, dtype=float),
            "symbols": list(symbols),
            "ml_forces": np.asarray(ml_forces, dtype=float),
            "cell": None if cell is None else np.asarray(cell, dtype=float),
            "pbc": pbc,
        }
        self.window.append(frame)
        lj_fit = self.fitter.fit_window(list(self.window))
        param_jump_ratio = compute_parameter_jump_ratio(lj_fit, previous_lj_params)
        physical = run_physical_checks(positions, ml_forces, cell=cell, pbc=pbc, energy=energy)
        ensemble = compute_ensemble_deviation(ensemble_forces) if ensemble_forces is not None else None
        metrics: dict[str, float | bool | None] = {
            "step": float(step),
            "lj_residual": lj_fit.relative_force_residual,
            "param_anomaly": (not lj_fit.is_physical) or bool(lj_fit.warnings),
            "param_jump_ratio": param_jump_ratio,
            "ensemble_max_deviation": ensemble.max_force_deviation if ensemble else 0.0,
            "ensemble_mean_deviation": ensemble.mean_force_deviation if ensemble else 0.0,
            "max_force": physical.max_force,
            "min_distance": physical.min_distance,
            "unsafe_probability": unsafe_probability if unsafe_probability is not None else 0.0,
        }
        risk_score = compute_risk_score(metrics, self.config)
        reasons: list[str] = []
        status = SAFE
        should_save_frame = False
        should_stop_md = False
        if physical.min_distance < float(thresholds.get("min_distance_stop", 0.6)):
            reasons.append("min_distance_stop")
            status = STOP
        if physical.max_force > float(thresholds.get("max_force_stop", 20.0)):
            reasons.append("max_force_stop")
            status = STOP
        if lj_fit.relative_force_residual > float(thresholds.get("lj_residual_stop", 0.6)):
            reasons.append("lj_residual_stop")
            status = STOP
        if ensemble and ensemble.max_force_deviation > float(thresholds.get("ensemble_deviation_stop", 0.3)):
            reasons.append("ensemble_deviation_stop")
            status = STOP
        if status != STOP:
            if lj_fit.relative_force_residual > float(thresholds.get("lj_residual_candidate", 0.3)):
                reasons.append("lj_residual_candidate")
                status = CANDIDATE
            if ensemble and ensemble.max_force_deviation > float(thresholds.get("ensemble_deviation_candidate", 0.15)):
                reasons.append("ensemble_deviation_candidate")
                status = CANDIDATE
            if param_jump_ratio > float(thresholds.get("parameter_jump_ratio", 5.0)):
                reasons.append("parameter_jump_candidate")
                status = CANDIDATE
            if (not lj_fit.is_physical) or bool(lj_fit.warnings):
                reasons.append("lj_parameter_anomaly")
                status = CANDIDATE
        if status in {CANDIDATE, STOP}:
            should_save_frame = True
        if status == STOP:
            should_stop_md = True
        return ReliabilityResult(
            status=status,
            risk_score=risk_score,
            metrics=metrics,
            reasons=sorted(set(reasons)),
            lj_fit_result=lj_fit,
            should_save_frame=should_save_frame,
            should_stop_md=should_stop_md,
        )
