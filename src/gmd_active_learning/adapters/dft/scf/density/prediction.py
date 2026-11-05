# Ported from PHAL (hotspot_al.cp2k_ml_scf.density.prediction) at 96263b6; MIT License, see NOTICE.
"""The prediction record handed from an ML model to CP2K.

A prediction is a *proposal*.  CP2K decides whether to use it, and it never
changes what CP2K converges to: the initial density only affects how many SCF
iterations are needed to satisfy ``EPS_SCF``.  Because a bad proposal can still
waste a whole calculation, every prediction carries enough provenance to
attribute a regression to one model build and one input structure.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PREDICTION_VERSION = "phal-scf-density-prediction-v1"

#: Per-atom valence populations, shape ``(spin_channels, n_atoms)``.
DENSITY_FORMAT_ATOMIC_POPULATION = "phal-atomic-population-v1"
#: A dense density matrix per spin channel, shape ``(spin_channels, nao, nao)``.
DENSITY_FORMAT_DENSITY_MATRIX = "phal-density-matrix-v1"

SUPPORTED_DENSITY_FORMATS = (DENSITY_FORMAT_ATOMIC_POPULATION, DENSITY_FORMAT_DENSITY_MATRIX)

#: Electron count tolerance CP2K's guess can absorb without harming the SCF.
DEFAULT_ELECTRON_TOLERANCE = 1.0e-3


class PredictionError(RuntimeError):
    """Raised when a prediction cannot be built, written, or read back."""


@dataclass(frozen=True, slots=True)
class PredictionProvenance:
    """Who produced a prediction, from what, and when."""

    model_digest: str
    structure_hash: str
    timestamp: str
    predictor_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_digest": self.model_digest,
            "structure_hash": self.structure_hash,
            "timestamp": self.timestamp,
            "predictor_type": self.predictor_type,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PredictionProvenance":
        try:
            return cls(
                model_digest=str(payload["model_digest"]),
                structure_hash=str(payload["structure_hash"]),
                timestamp=str(payload["timestamp"]),
                predictor_type=str(payload["predictor_type"]),
            )
        except KeyError as exc:
            raise PredictionError(f"provenance is missing {exc.args[0]!r}") from exc


@dataclass(frozen=True, slots=True)
class SCFDensityPrediction:
    """One predicted initial electronic state."""

    model_name: str
    model_version: str
    model_digest: str
    structure_hash: str
    electron_count: float
    spin_channels: int
    density_format: str
    output_path: Path | None = None
    values: np.ndarray | None = field(default=None, repr=False)
    #: AO overlap S, (n_ao, n_ao). Required to export a density *matrix*: the
    #: electron count is Tr(P S), and CP2K refuses a matrix it cannot check.
    #: Unused for atomic populations, where the count is the plain sum.
    #: See docs/density_representation.md.
    overlap: np.ndarray | None = field(default=None, repr=False)
    predictor_type: str = "unknown"
    timestamp: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.spin_channels not in (1, 2):
            raise PredictionError(f"spin_channels must be 1 or 2, got {self.spin_channels}")
        if self.density_format not in SUPPORTED_DENSITY_FORMATS:
            raise PredictionError(
                f"unsupported density_format {self.density_format!r}; "
                f"expected one of {', '.join(SUPPORTED_DENSITY_FORMATS)}"
            )
        object.__setattr__(self, "timestamp", self.timestamp or utc_timestamp())
        if self.output_path is not None:
            object.__setattr__(self, "output_path", Path(self.output_path))

    @property
    def provenance(self) -> PredictionProvenance:
        return PredictionProvenance(
            model_digest=self.model_digest,
            structure_hash=self.structure_hash,
            timestamp=str(self.timestamp),
            predictor_type=self.predictor_type,
        )

    def to_dict(self, *, include_values: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prediction_version": PREDICTION_VERSION,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_digest": self.model_digest,
            "structure_hash": self.structure_hash,
            "electron_count": float(self.electron_count),
            "spin_channels": int(self.spin_channels),
            "density_format": self.density_format,
            "output_path": None if self.output_path is None else str(self.output_path),
            "provenance": self.provenance.to_dict(),
            "metadata": dict(self.metadata),
        }
        if self.values is not None:
            payload["shape"] = list(self.values.shape)
            payload["values_digest"] = array_digest(self.values)
            if include_values:
                payload["values"] = self.values.tolist()
        if self.overlap is not None:
            payload["overlap_digest"] = array_digest(self.overlap)
            if include_values:
                payload["overlap"] = np.asarray(self.overlap).tolist()
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        values: np.ndarray | None = None,
        overlap: np.ndarray | None = None,
    ) -> "SCFDensityPrediction":
        provenance = payload.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise PredictionError("provenance must be a mapping")
        inline = payload.get("values")
        inline_overlap = payload.get("overlap")
        resolved_overlap = overlap
        if resolved_overlap is None and inline_overlap is not None:
            resolved_overlap = np.asarray(inline_overlap, dtype=float)
        resolved = values
        if resolved is None and inline is not None:
            resolved = np.asarray(inline, dtype=float)
        try:
            return cls(
                model_name=str(payload["model_name"]),
                model_version=str(payload["model_version"]),
                model_digest=str(payload["model_digest"]),
                structure_hash=str(payload["structure_hash"]),
                electron_count=float(payload["electron_count"]),
                spin_channels=int(payload["spin_channels"]),
                density_format=str(payload["density_format"]),
                output_path=None if payload.get("output_path") is None else Path(str(payload["output_path"])),
                values=resolved,
                overlap=resolved_overlap,
                predictor_type=str(provenance.get("predictor_type", "unknown")),
                timestamp=None if provenance.get("timestamp") is None else str(provenance["timestamp"]),
                metadata=dict(payload.get("metadata", {})),
            )
        except KeyError as exc:
            raise PredictionError(f"prediction is missing {exc.args[0]!r}") from exc

    def with_output_path(self, path: str | Path) -> "SCFDensityPrediction":
        # Deliberate fix (ledger SCF-020): PHAL dropped ``overlap`` here, so a
        # recorded density-matrix prediction could never be re-exported.
        return replace(self, output_path=Path(path), metadata=dict(self.metadata))

    def with_metadata(self, **changes: Any) -> "SCFDensityPrediction":
        return replace(self, metadata={**dict(self.metadata), **changes})

    def with_overlap(self, overlap: np.ndarray | None) -> "SCFDensityPrediction":
        return replace(self, overlap=None if overlap is None else np.asarray(overlap, dtype=float))


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for provenance records."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def array_digest(values: np.ndarray) -> str:
    """Digest an array's exact bytes so a prediction file can be verified."""

    contiguous = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.shape).encode("utf-8"))
    hasher.update(contiguous.tobytes())
    return f"sha256:{hasher.hexdigest()}"


def write_prediction(prediction: SCFDensityPrediction, path: str | Path) -> Path:
    """Write a prediction as ``<path>`` (JSON) plus ``<path>.npz`` (values)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    located = prediction.with_output_path(target)

    if located.values is not None:
        values_path = target.with_suffix(target.suffix + ".npz")
        arrays: dict[str, np.ndarray] = {
            "values": np.asarray(located.values, dtype=np.float64),
            "density_format": np.asarray(located.density_format),
        }
        if located.overlap is not None:
            arrays["overlap"] = np.asarray(located.overlap, dtype=np.float64)
        np.savez_compressed(values_path, **arrays)
        payload = located.to_dict()
        payload["values_path"] = str(values_path)
    else:
        payload = located.to_dict()

    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def read_prediction(path: str | Path) -> SCFDensityPrediction:
    """Read back a prediction written by :func:`write_prediction`."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PredictionError(f"{source}: invalid JSON ({exc})") from exc
    if not isinstance(payload, Mapping):
        raise PredictionError(f"{source}: prediction must be a JSON object")

    values = None
    overlap = None
    values_path = payload.get("values_path")
    if values_path and Path(values_path).is_file():
        with np.load(values_path) as archive:
            values = np.asarray(archive["values"], dtype=float)
            if "overlap" in archive.files:
                overlap = np.asarray(archive["overlap"], dtype=float)
    elif payload.get("shape") is not None and "values" not in payload:
        raise PredictionError(f"{source}: prediction declares values but none are on disk")

    prediction = SCFDensityPrediction.from_dict(payload, values=values, overlap=overlap)
    if payload.get("overlap_digest") is not None:
        if prediction.overlap is None:
            raise PredictionError(f"{source}: prediction declares an overlap but none is on disk")
        if array_digest(prediction.overlap) != payload["overlap_digest"]:
            raise PredictionError(f"{source}: overlap digest mismatch")
    expected = payload.get("values_digest")
    if expected is not None and prediction.values is not None:
        actual = array_digest(prediction.values)
        if actual != expected:
            raise PredictionError(f"{source}: values digest mismatch (expected {expected}, got {actual})")
    return prediction


__all__ = [
    "DEFAULT_ELECTRON_TOLERANCE",
    "DENSITY_FORMAT_ATOMIC_POPULATION",
    "DENSITY_FORMAT_DENSITY_MATRIX",
    "PREDICTION_VERSION",
    "SUPPORTED_DENSITY_FORMATS",
    "PredictionError",
    "PredictionProvenance",
    "SCFDensityPrediction",
    "array_digest",
    "read_prediction",
    "utc_timestamp",
    "write_prediction",
]
