# Ported from PHAL (hotspot_al.cp2k_ml_scf.training.models: require_torch, torch_available,
# model_digest) at 96263b6; MIT License, see NOTICE.
"""Optional-torch gate for the SCF subsystem.

PyTorch is an optional dependency of the SCF tooling.  Every torch-only code
path calls :func:`require_torch` at *call* time, never at import time, so
importing :mod:`gmd_active_learning.adapters.dft.scf` (or anything else in
``gmd_active_learning``) never imports torch.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any

from gmd_active_learning.core.exceptions import CapabilityUnavailableError

LOGGER = logging.getLogger(__name__)


class TorchUnavailableError(CapabilityUnavailableError, ImportError):
    """Raised when a torch-only code path is reached without torch installed."""


def require_torch() -> Any:
    """Import torch lazily, converting the failure into an actionable message."""

    try:
        return importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise TorchUnavailableError(
            "PyTorch is required for the experimental ML-SCF density models. "
            "Install torch into this environment to use them; nothing in the default "
            "labeling path needs it."
        ) from exc


def torch_available() -> bool:
    """Whether torch could be imported, without importing it."""

    return importlib.util.find_spec("torch") is not None


def model_digest(path: str | Path) -> str:
    """Digest a model/checkpoint file so predictions can name their exact weights."""

    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


__all__ = ["TorchUnavailableError", "model_digest", "require_torch", "torch_available"]
