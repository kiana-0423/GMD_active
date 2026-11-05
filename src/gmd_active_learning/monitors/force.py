# Ported from PHAL (hotspot_al.monitor.force_monitor) at 96263b6; MIT License, see NOTICE.
"""Per-atom force magnitude and force-burst metrics (eV/A)."""

from __future__ import annotations

import numpy as np

from gmd_active_learning.core.exceptions import DataContractError
from gmd_active_learning.utils.geometry import row_norms


def force_norms(forces: np.ndarray) -> np.ndarray:
    """|F_i| per atom, shape (N,)."""

    array = np.asarray(forces, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise DataContractError(f"forces must have shape (N, 3), got {array.shape}")
    return row_norms(array)


def delta_force_norms(current_forces: np.ndarray, previous_forces: np.ndarray | None) -> np.ndarray:
    """|F_i(t) - F_i(t_prev)| per atom against the previous processed frame; zeros without one."""

    current = np.asarray(current_forces, dtype=float)
    if previous_forces is None:
        return np.zeros(len(current), dtype=float)
    previous = np.asarray(previous_forces, dtype=float)
    if previous.shape != current.shape:
        raise DataContractError(f"previous forces {previous.shape} do not match current {current.shape}")
    return row_norms(current - previous)


__all__ = ["delta_force_norms", "force_norms"]
