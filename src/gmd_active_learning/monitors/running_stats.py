# Ported from PHAL (hotspot_al.monitor.ood_score.RunningMetricStats) at 96263b6; MIT License, see NOTICE.
"""Running per-metric statistics (Chan parallel update) used for z-score normalisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class RunningMetricStats:
    """Pooled mean/variance over every atom sample of every stats-updating frame (no decay)."""

    count: int = 0
    frame_count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        batch = np.asarray(values, dtype=float).ravel()
        n_new = len(batch)
        if n_new == 0:
            return
        self.frame_count += 1
        batch_mean = float(batch.mean())
        batch_m2 = float(np.sum((batch - batch_mean) ** 2))
        if self.count == 0:
            self.mean, self.m2, self.count = batch_mean, batch_m2, n_new
            return
        total = self.count + n_new
        delta = batch_mean - self.mean
        self.mean += delta * n_new / total
        self.m2 += batch_m2 + delta**2 * self.count * n_new / total
        self.count = total

    @property
    def std(self) -> float:
        """Sample standard deviation (ddof=1)."""

        if self.count < 2:
            return 0.0
        return float(np.sqrt(self.m2 / (self.count - 1)))

    def to_state(self) -> dict[str, Any]:
        return {"count": self.count, "frame_count": self.frame_count, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> RunningMetricStats:
        return cls(int(state["count"]), int(state["frame_count"]), float(state["mean"]), float(state["m2"]))


__all__ = ["RunningMetricStats"]
