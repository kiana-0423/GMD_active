from __future__ import annotations

import numpy as np

from gmd_active_learning.core.data_types import EnsembleDeviationResult


def compute_ensemble_deviation(ensemble_forces: np.ndarray) -> EnsembleDeviationResult:
    if ensemble_forces.ndim != 3:
        raise ValueError("ensemble_forces must have shape (M, N, 3)")
    std = np.std(ensemble_forces, axis=0)
    per_atom = np.linalg.norm(std, axis=1)
    return EnsembleDeviationResult(
        per_atom_force_deviation=per_atom,
        max_force_deviation=float(np.max(per_atom, initial=0.0)),
        mean_force_deviation=float(np.mean(per_atom)) if per_atom.size else 0.0,
        min_force_deviation=float(np.min(per_atom, initial=0.0)),
    )
