from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from gmd_active_learning.adapters.base import BaseMLIPTrainerAdapter


class GMDSE3GNNAdapter(BaseMLIPTrainerAdapter):
    def __init__(self, call_mode: str = "subprocess", train_command: str | None = None, export_command: str | None = None, dry_run: bool = False) -> None:
        self.call_mode = call_mode
        self.train_command = train_command
        self.export_command = export_command
        self.dry_run = dry_run

    def train(self, dataset_path: str | Path, train_config: dict[str, Any]) -> str:
        output = Path(train_config.get("output_dir", "trained_model"))
        output.mkdir(parents=True, exist_ok=True)
        model_path = output / "model.pt"
        if self.dry_run or not self.train_command:
            model_path.write_text("dry-run-model", encoding="utf-8")
            return str(model_path)
        command = self.train_command.format(config=train_config.get("config", ""), data=dataset_path, output=output)
        subprocess.run(shlex.split(command), check=True)
        return str(model_path)

    def predict(self, structure: dict[str, Any]) -> tuple[float | None, object]:
        positions = np.asarray(structure["positions"], dtype=float)
        forces = np.zeros_like(positions)
        return 0.0, forces

    def export_model(self, model_path: str | Path, output_path: str | Path, export_config: dict[str, Any]) -> str:
        output = Path(output_path)
        output.mkdir(parents=True, exist_ok=True)
        exported = output / "exported_model.pt"
        if self.dry_run or not self.export_command:
            exported.write_text(f"exported from {model_path}", encoding="utf-8")
            return str(exported)
        command = self.export_command.format(model=model_path, output=output)
        subprocess.run(shlex.split(command), check=True)
        return str(exported)
