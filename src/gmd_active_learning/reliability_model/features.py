from __future__ import annotations

import numpy as np

from gmd_active_learning.core.constants import DEFAULT_HISTOGRAM_BINS
from gmd_active_learning.core.data_types import ReliabilityResult
from gmd_active_learning.utils.geometry import coordination_number_summary, pair_distance_histogram


def build_reliability_features(
    reliability_result: ReliabilityResult,
    positions: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: tuple[bool, bool, bool] = (False, False, False),
    cutoff: float = 6.0,
    latent_descriptor: np.ndarray | None = None,
) -> np.ndarray:
    metrics = reliability_result.metrics
    scalar_features = np.asarray(
        [
            float(metrics.get("lj_residual", 0.0) or 0.0),
            1.0 if bool(metrics.get("param_anomaly", False)) else 0.0,
            float(metrics.get("param_jump_ratio", 0.0) or 0.0),
            float(metrics.get("ensemble_max_deviation", 0.0) or 0.0),
            float(metrics.get("ensemble_mean_deviation", 0.0) or 0.0),
            float(metrics.get("max_force", 0.0) or 0.0),
            float(metrics.get("min_distance", 0.0) or 0.0),
        ],
        dtype=float,
    )
    coordination = coordination_number_summary(positions, cutoff=cutoff, cell=cell, pbc=pbc)
    histogram = pair_distance_histogram(positions, cell=cell, pbc=pbc, cutoff=cutoff, bins=DEFAULT_HISTOGRAM_BINS)
    latent = np.asarray(latent_descriptor, dtype=float).ravel() if latent_descriptor is not None else np.array([], dtype=float)
    return np.concatenate([scalar_features, coordination, histogram, latent])
