from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from gmd_active_learning.adapters.base import BaseMDAdapter, MonitorCallback
from gmd_active_learning.core.data_types import MDFrame, MDRunResult


class GMDAdapter(BaseMDAdapter):
    def __init__(self, command_template: str | None = None, dry_run: bool = False) -> None:
        self.command_template = command_template
        self.dry_run = dry_run

    def run_md(self, model_path: str | Path, md_config: dict[str, Any], monitor_callback: MonitorCallback | None = None) -> MDRunResult:
        if self.dry_run or not self.command_template:
            frame = MDFrame(
                step=0,
                positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=float),
                symbols=["Ar", "Ar"],
                forces=np.array([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]], dtype=float),
            )
            if monitor_callback is not None:
                monitor_callback(frame)
            return MDRunResult(frames=[frame], stopped_early=False, metadata={"mode": "dry_run"})
        command = self.command_template.format(model=model_path, config=md_config.get("config", ""))
        subprocess.run(shlex.split(command), check=True)
        return MDRunResult(frames=[], stopped_early=False, metadata={"mode": "subprocess"})
