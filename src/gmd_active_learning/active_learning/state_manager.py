from __future__ import annotations

from pathlib import Path
from typing import Any

from gmd_active_learning.utils.io import read_json, write_json


class StateManager:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"iteration": 0, "selected_candidates": []}
        return read_json(self.path)

    def save(self, state: dict[str, Any]) -> None:
        write_json(self.path, state)
