"""Committee (ensemble) force deviation, eV/A.

sigma_i = sqrt(mean_m |F_i^m - mean_m F_i^m|^2) (population std, ddof=0); identical to the norm of
the per-component standard deviation used by GMD before the merge and to PHAL's
``committee_force_deviation`` (hotspot_al.monitor.committee_deviation at 96263b6, MIT License).
"""

from __future__ import annotations

import numpy as np

from gmd_active_learning.core.data_types import EnsembleDeviationResult


def committee_force_deviation(force_predictions: np.ndarray) -> np.ndarray:
    """Per-atom committee force deviation for predictions of shape (M, N, 3)."""

    predictions = np.asarray(force_predictions, dtype=float)
    if predictions.ndim != 3 or predictions.shape[-1] != 3:
        raise ValueError(f"committee predictions must have shape (M, N, 3), got {predictions.shape}")
    mean_forces = np.mean(predictions, axis=0)
    squared = np.sum((predictions - mean_forces) ** 2, axis=-1)
    return np.sqrt(np.mean(squared, axis=0))


def compute_ensemble_deviation(ensemble_forces: np.ndarray) -> EnsembleDeviationResult:
    if np.asarray(ensemble_forces).ndim != 3:
        raise ValueError("ensemble_forces must have shape (M, N, 3)")
    per_atom = committee_force_deviation(ensemble_forces)
    return EnsembleDeviationResult(
        per_atom_force_deviation=per_atom,
        max_force_deviation=float(np.max(per_atom, initial=0.0)),
        mean_force_deviation=float(np.mean(per_atom)) if per_atom.size else 0.0,
        min_force_deviation=float(np.min(per_atom, initial=0.0)),
    )


__all__ = ["committee_force_deviation", "compute_ensemble_deviation"]
