from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gmd_active_learning.utils.io import ensure_dir, write_json


class ModelRegistry:
    def __init__(self, registry_dir: str | Path) -> None:
        self.registry_dir = ensure_dir(registry_dir)

    def register(
        self,
        *,
        model_path: str | Path,
        training_config: dict[str, Any],
        dataset_version: str,
        metrics: dict[str, Any],
        parent_model: str | None = None,
    ) -> Path:
        existing = sorted(
            path for path in self.registry_dir.iterdir() if path.is_dir() and path.name.startswith("model_v")
        )
        version = f"model_v{len(existing):03d}"
        model_dir = ensure_dir(self.registry_dir / version)
        metadata = {
            "model_path": str(model_path),
            "training_config": training_config,
            "dataset_version": dataset_version,
            "metrics": metrics,
            "created_time": datetime.now(timezone.utc).isoformat(),
            "parent_model": parent_model,
        }
        write_json(model_dir / "metadata.json", metadata)
        return model_dir
