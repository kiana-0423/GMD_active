"""MD-force vs reference-model force disagreement (PHAL ``mlip_force_deviation``, reason ``model_drift``).

The monitor never constructs an inference backend. The caller (the workflow / GMD adapter) passes
a :class:`ReferenceForceProvider`, e.g. a GMD_SGT re-evaluation of the frame with the reference
model; it is only called on physics/full frames. With no provider the metric is UNAVAILABLE.
If the MD engine runs the same deployed model, the deviation measures deployment/precision drift.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from gmd_active_learning.contracts.frame import Frame
from gmd_active_learning.core.exceptions import MonitorError
from gmd_active_learning.utils.geometry import row_norms


@runtime_checkable
class ReferenceForceProvider(Protocol):
    """Callable returning reference-model forces (N, 3) in eV/A for a frame, in frame atom order."""

    def __call__(self, frame: Frame) -> np.ndarray: ...


@runtime_checkable
class CommitteeForceProvider(Protocol):
    """Callable returning committee forces (M, N, 3) for a frame that carries none."""

    def __call__(self, frame: Frame) -> np.ndarray: ...


def checked_reference_forces(frame: Frame, forces: object) -> np.ndarray:
    array = np.asarray(forces, dtype=float)
    if array.shape != (frame.n_atoms, 3):
        raise MonitorError(f"reference forces for step {frame.step} have shape {array.shape}, expected ({frame.n_atoms}, 3)")
    if not np.all(np.isfinite(array)):
        raise MonitorError(f"reference forces for step {frame.step} contain non-finite values")
    return array


def mlip_force_deviation(md_forces: np.ndarray, reference_forces: np.ndarray) -> np.ndarray:
    """|F_i(MD) - F_i(reference)| per atom."""

    return row_norms(np.asarray(md_forces, dtype=float) - np.asarray(reference_forces, dtype=float))


__all__ = [
    "CommitteeForceProvider",
    "ReferenceForceProvider",
    "checked_reference_forces",
    "mlip_force_deviation",
]
