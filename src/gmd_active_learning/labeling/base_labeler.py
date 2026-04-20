from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseLabeler(ABC):
    @abstractmethod
    def generate(self, candidate_paths: list[Path], config: dict[str, Any]) -> list[Path]:
        raise NotImplementedError
