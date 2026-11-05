# Ported from PHAL (hotspot_al.cp2k_ml_scf.density.model_registry) at 96263b6; MIT License, see NOTICE.
"""Registry and configuration resolution for SCF density predictors.

Predictors are addressed by a short name (``dummy``, ``torch``) so that the
accelerated mode can be switched on with one config key.  A dotted
``module:Class`` path is also accepted, which lets a project keep a private
model outside this repository without patching the registry.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gmd_active_learning.adapters.dft.scf.density.predictor import (
    DummyDensityPredictor,
    SCFDensityPredictor,
)
from gmd_active_learning.adapters.dft.scf.density.prediction import (
    SCFDensityPrediction,
    utc_timestamp,
)
from gmd_active_learning.utils.atomic_io import write_json_atomic

DEFAULT_PREDICTOR_TYPE = "dummy"
DEFAULT_PREDICTION_DIR = Path("runtime/cp2k_ml/predictions")
PROVENANCE_LOG_NAME = "predictions.jsonl"

_REGISTRY: dict[str, type[SCFDensityPredictor]] = {}

#: Built-in predictors whose modules need torch (or the source-checkout tooling).
#: They are resolved by import path on first use, so a missing torch only affects
#: the predictor that needs it. PHAL returned early from its registration when
#: ``torch`` failed to import, which also left ``density_gnn`` unregistered.
_LAZY_BUILTINS: dict[str, str] = {
    "density_gnn": "gmd_active_learning.adapters.dft.scf.density.gnn_predictor:AODensityMatrixPredictor",
    # Experimental atomic-population MLP (ledger SCF-025); needs scf.ml_scf.experimental.
    "torch": "tools.scf.experimental.atomic_population_mlp.model:TorchDensityPredictor",
}


class PredictorNotFoundError(KeyError):
    """Raised when a configured predictor type cannot be resolved."""


@dataclass(frozen=True, slots=True)
class PredictorSpec:
    """The resolved ``cp2k.ml_scf.predictor`` configuration block."""

    type: str
    model_path: Path | None = None
    options: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.options is None:
            object.__setattr__(self, "options", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "model_path": None if self.model_path is None else str(self.model_path),
            "options": dict(self.options),
        }


def register_predictor(name: str, predictor: type[SCFDensityPredictor], *, replace: bool = False) -> None:
    """Register a predictor class under ``name``."""

    key = _normalize(name)
    if key in _REGISTRY and not replace:
        raise ValueError(f"predictor {key!r} is already registered; pass replace=True to override")
    if not (isinstance(predictor, type) and issubclass(predictor, SCFDensityPredictor)):
        raise TypeError(f"{predictor!r} is not an SCFDensityPredictor subclass")
    _REGISTRY[key] = predictor


def unregister_predictor(name: str) -> None:
    _REGISTRY.pop(_normalize(name), None)


def available_predictors() -> tuple[str, ...]:
    """Registered names, including lazily resolved built-ins (whether or not torch is present)."""

    _register_builtins()
    return tuple(sorted(set(_REGISTRY) | set(_LAZY_BUILTINS)))


def resolve_predictor_class(name: str) -> type[SCFDensityPredictor]:
    """Resolve a registered name or a ``module:Class`` path."""

    _register_builtins()
    key = _normalize(name)
    if key in _REGISTRY:
        return _REGISTRY[key]
    if key in _LAZY_BUILTINS:
        resolved = _import_predictor(_LAZY_BUILTINS[key], label=key)
        _REGISTRY[key] = resolved
        return resolved
    if ":" in name:
        return _import_predictor(name, label=name)
    raise PredictorNotFoundError(
        f"unknown predictor {name!r}; registered predictors: {', '.join(available_predictors()) or 'none'}"
    )


def _import_predictor(path: str, *, label: str) -> type[SCFDensityPredictor]:
    module_name, _, attribute = path.partition(":")
    try:
        module = importlib.import_module(module_name)
        candidate = getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise PredictorNotFoundError(f"could not import predictor {label!r} ({path}): {exc}") from exc
    if not (isinstance(candidate, type) and issubclass(candidate, SCFDensityPredictor)):
        raise PredictorNotFoundError(f"{path!r} is not an SCFDensityPredictor subclass")
    return candidate


def predictor_spec(config: Any) -> PredictorSpec:
    """Read ``scf.ml_scf.predictor`` (PHAL: ``cp2k.ml_scf.predictor``) out of a configuration."""

    predictor = _ml_section(config).predictor
    return PredictorSpec(
        type=str(predictor.type or DEFAULT_PREDICTOR_TYPE),
        model_path=None if predictor.model_path in (None, "") else Path(str(predictor.model_path)),
        options=dict(predictor.options),
    )


def ml_scf_config(config: Any) -> Mapping[str, Any]:
    """Return the ``scf.ml_scf`` section as a plain mapping."""

    return _ml_section(config).model_dump()


def ml_scf_enabled(config: Any) -> bool:
    """Whether the ML initial-guess wrapper is switched on (``scf.ml_scf.enabled``).

    For a PHAL-shaped configuration this is PHAL's rule: ``backend.dft.engine``
    must be ``cp2k_ml_scf`` and ``cp2k.ml_scf.enabled`` (default true) not off.
    """

    return bool(_ml_section(config).enabled)


def _ml_section(config: Any) -> Any:
    from gmd_active_learning.adapters.dft.scf.config import scf_config

    return scf_config(config).ml_scf


def create_predictor(
    config: Mapping[str, Any] | None = None,
    *,
    spec: PredictorSpec | None = None,
    load: bool = True,
) -> SCFDensityPredictor:
    """Construct the configured predictor."""

    resolved = spec or predictor_spec(config or {})
    predictor_class = resolve_predictor_class(resolved.type)
    predictor = predictor_class(
        config={
            "type": resolved.type,
            "model_path": None if resolved.model_path is None else str(resolved.model_path),
            "options": dict(resolved.options),
        }
    )
    if load:
        predictor.load_model()
    return predictor


def record_prediction(
    prediction: SCFDensityPrediction,
    *,
    output_dir: str | Path = DEFAULT_PREDICTION_DIR,
    task_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist a prediction plus an append-only provenance line.

    Two artifacts are written: a per-prediction JSON (with its ``.npz`` values)
    and one line in ``predictions.jsonl``.  The log is what makes an SCF
    regression attributable months later, so it is written even when the
    prediction itself is later rejected by CP2K.
    """

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    identifier = task_id or prediction.structure_hash.rsplit(":", 1)[-1][:16]
    prediction_path = directory / f"{identifier}.prediction.json"

    from gmd_active_learning.adapters.dft.scf.density.prediction import write_prediction

    write_prediction(prediction, prediction_path)

    log_path = directory / PROVENANCE_LOG_NAME
    entry = {
        **prediction.provenance.to_dict(),
        "task_id": task_id,
        "model_name": prediction.model_name,
        "model_version": prediction.model_version,
        "density_format": prediction.density_format,
        "electron_count": prediction.electron_count,
        "spin_channels": prediction.spin_channels,
        "prediction_path": str(prediction_path),
        **dict(extra or {}),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        import json

        handle.write(json.dumps(entry, sort_keys=True) + "\n")

    return {"prediction": prediction_path, "log": log_path}


def write_predictor_manifest(
    predictor: SCFDensityPredictor,
    path: str | Path,
    *,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Record which predictor build a run used."""

    return write_json_atomic(
        path,
        {
            "predictor_type": predictor.predictor_type,
            "model_name": predictor.model_name,
            "model_version": predictor.model_version,
            "model_digest": predictor.model_digest,
            "density_format": predictor.density_format,
            "timestamp": utc_timestamp(),
            **dict(extra or {}),
        },
    )


def _register_builtins() -> None:
    if "dummy" not in _REGISTRY:
        _REGISTRY["dummy"] = DummyDensityPredictor


def _normalize(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


__all__ = [
    "DEFAULT_PREDICTION_DIR",
    "DEFAULT_PREDICTOR_TYPE",
    "PROVENANCE_LOG_NAME",
    "PredictorNotFoundError",
    "PredictorSpec",
    "available_predictors",
    "create_predictor",
    "ml_scf_config",
    "ml_scf_enabled",
    "predictor_spec",
    "record_prediction",
    "register_predictor",
    "resolve_predictor_class",
    "unregister_predictor",
    "write_predictor_manifest",
]
