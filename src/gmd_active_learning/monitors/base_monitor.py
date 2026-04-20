from __future__ import annotations

from abc import ABC, abstractmethod

from gmd_active_learning.core.data_types import ReliabilityResult


class BaseMonitor(ABC):
    @abstractmethod
    def evaluate(self, **kwargs: object) -> ReliabilityResult:
        raise NotImplementedError
