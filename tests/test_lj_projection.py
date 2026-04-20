from __future__ import annotations

import numpy as np

from gmd_active_learning.monitors.lj_projection import LJProjectionFitter


def test_lj_projection_two_atoms() -> None:
    symbols = ["Ar", "Ar", "Ar"]
    A = 2.0
    B = 1.0
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.7, 0.0, 0.0]], dtype=float)
    ml_forces = np.zeros_like(positions)
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)
            pair_force = (12.0 * A / r**14 - 6.0 * B / r**8) * r_vec
            ml_forces[i] += pair_force
            ml_forces[j] -= pair_force
    fitter = LJProjectionFitter(cutoff=6.0, ridge_lambda=1.0e-10)
    result = fitter.fit_single_frame(positions, symbols, ml_forces)
    params = result.pair_params_ab["Ar-Ar"]
    assert np.isclose(params["A"], A, rtol=1.0e-3, atol=1.0e-6)
    assert np.isclose(params["B"], B, rtol=1.0e-3, atol=1.0e-6)
    assert result.relative_force_residual < 1.0e-6


def test_lj_projection_nonphysical() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=float)
    symbols = ["Ar", "Ar"]
    ml_forces = np.array([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]], dtype=float)
    fitter = LJProjectionFitter(cutoff=6.0, ridge_lambda=1.0e-10)
    result = fitter.fit_single_frame(positions, symbols, ml_forces)
    assert result.is_physical is False
    assert any("nonphysical" in warning for warning in result.warnings)
