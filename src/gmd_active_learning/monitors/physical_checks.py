from __future__ import annotations

import numpy as np

from gmd_active_learning.core.data_types import PhysicalCheckResult
from gmd_active_learning.utils.geometry import pairwise_distances


def run_physical_checks(
    positions: np.ndarray,
    forces: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: tuple[bool, bool, bool] = (False, False, False),
    close_contact_distance: float = 0.8,
    energy: float | None = None,
    reference_energy: float | None = None,
    temperature: float | None = None,
    temperature_bounds: tuple[float, float] | None = None,
) -> PhysicalCheckResult:
    force_norms = np.linalg.norm(forces, axis=1)
    distances, _ = pairwise_distances(positions, cell=cell, pbc=pbc)
    min_distance = float(np.min(distances)) if distances.size else float("inf")
    warnings: list[str] = []
    has_close_contact = min_distance < close_contact_distance
    if has_close_contact:
        warnings.append("close_contact")
    energy_drift = None
    if energy is not None and reference_energy is not None:
        energy_drift = float(abs(energy - reference_energy))
    temperature_anomaly = None
    if temperature is not None and temperature_bounds is not None:
        low, high = temperature_bounds
        if temperature < low:
            temperature_anomaly = float(low - temperature)
        elif temperature > high:
            temperature_anomaly = float(temperature - high)
        else:
            temperature_anomaly = 0.0
    return PhysicalCheckResult(
        max_force=float(np.max(force_norms, initial=0.0)),
        min_distance=min_distance,
        has_close_contact=has_close_contact,
        energy_drift=energy_drift,
        temperature_anomaly=temperature_anomaly,
        warnings=warnings,
    )
