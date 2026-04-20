from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from gmd_active_learning.core.data_types import MDFrame, MDRunResult


MonitorCallback = Callable[[MDFrame], object]


class BaseMDAdapter(ABC):
    @abstractmethod
    def run_md(self, model_path: str | Path, md_config: dict[str, Any], monitor_callback: MonitorCallback | None = None) -> MDRunResult:
        raise NotImplementedError


class BaseMLIPTrainerAdapter(ABC):
    @abstractmethod
    def train(self, dataset_path: str | Path, train_config: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def predict(self, structure: dict[str, Any]) -> tuple[float | None, object]:
        raise NotImplementedError

    @abstractmethod
    def export_model(self, model_path: str | Path, output_path: str | Path, export_config: dict[str, Any]) -> str:
        raise NotImplementedError
