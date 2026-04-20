from __future__ import annotations

from pathlib import Path
from typing import Any

from gmd_active_learning.adapters.gmd_se3gnn_adapter import GMDSE3GNNAdapter


class RetrainRunner:
    def __init__(self, adapter: GMDSE3GNNAdapter) -> None:
        self.adapter = adapter

    def run(self, dataset_path: str | Path, train_config: dict[str, Any]) -> str:
        return self.adapter.train(dataset_path, train_config)
