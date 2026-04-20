from __future__ import annotations

import numpy as np

from gmd_active_learning.core.constants import SAFE, STOP
from gmd_active_learning.monitors.reliability_monitor import ReliabilityMonitor


def monitor_config() -> dict[str, object]:
    return {
        "window_size": 4,
        "cutoff": 6.0,
        "ridge_lambda": 1.0e-8,
        "thresholds": {
            "lj_residual_candidate": 0.3,
            "lj_residual_stop": 100.0,
            "max_force_stop": 20.0,
            "min_distance_stop": 0.6,
            "ensemble_deviation_candidate": 0.15,
            "ensemble_deviation_stop": 0.3,
            "parameter_jump_ratio": 5.0,
        },
        "parameter_bounds": {
            "epsilon_min": 1.0e-6,
            "epsilon_max": 10.0,
            "sigma_min": 0.5,
            "sigma_max": 8.0,
        },
        "weights": {
            "lj_residual": 1.0,
            "param_anomaly": 1.0,
            "param_jump": 0.5,
            "ensemble_deviation": 1.0,
            "max_force": 0.5,
            "min_distance": 0.5,
            "reliability_model": 0.5,
        },
    }


def test_reliability_monitor_safe() -> None:
    monitor = ReliabilityMonitor(monitor_config())
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.8, 0.0, 0.0]], dtype=float)
    A = 1.5
    B = 0.7
    forces = np.zeros_like(positions)
    for i in range(3):
        for j in range(i + 1, 3):
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)
            pair_force = (12.0 * A / r**14 - 6.0 * B / r**8) * r_vec
            forces[i] += pair_force
            forces[j] -= pair_force
    result = monitor.evaluate(step=0, positions=positions, symbols=["Ar", "Ar", "Ar"], ml_forces=forces)
    assert result.status == SAFE
    assert result.should_stop_md is False


def test_reliability_monitor_stop() -> None:
    monitor = ReliabilityMonitor(monitor_config())
    positions = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=float)
    forces = np.array([[50.0, 0.0, 0.0], [-60.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float)
    result = monitor.evaluate(step=0, positions=positions, symbols=["Ar", "Ar", "Ar"], ml_forces=forces)
    assert result.status == STOP
    assert result.should_stop_md is True
