from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gmd_active_learning.adapters.base import BaseMDAdapter, MonitorCallback
from gmd_active_learning.core.data_types import MDFrame, MDRunResult


class ASEAdapter(BaseMDAdapter):
    def run_md(self, model_path: str | Path, md_config: dict[str, Any], monitor_callback: MonitorCallback | None = None) -> MDRunResult:
        n_steps = int(md_config.get("n_steps", 10))
        frames: list[MDFrame] = []
        for step in range(n_steps):
            positions = np.array([[0.0, 0.0, 0.0], [1.2 + 0.01 * step, 0.0, 0.0]], dtype=float)
            forces = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=float)
            frame = MDFrame(step=step, positions=positions, symbols=["Ar", "Ar"], forces=forces)
            frames.append(frame)
            if monitor_callback is not None:
                monitor_callback(frame)
        return MDRunResult(frames=frames, stopped_early=False, metadata={"adapter": "ase"})
