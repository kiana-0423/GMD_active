from __future__ import annotations

import numpy as np

from gmd_active_learning.monitors.physical_checks import run_physical_checks


def test_physical_checks_close_contact() -> None:
    result = run_physical_checks(
        positions=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=float),
        forces=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float),
        close_contact_distance=0.8,
    )
    assert result.has_close_contact is True
    assert result.min_distance < 0.8
