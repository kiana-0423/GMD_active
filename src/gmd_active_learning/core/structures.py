from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class AtomisticStructure:
    symbols: list[str]
    positions: np.ndarray
    cell: np.ndarray | None = None
    pbc: tuple[bool, bool, bool] = (False, False, False)
    metadata: dict[str, Any] = field(default_factory=dict)
