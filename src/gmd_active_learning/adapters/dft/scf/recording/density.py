# Ported from PHAL (hotspot_al.cp2k.density_recording) at 96263b6; MIT License, see NOTICE.
"""Persist the converged electronic density CP2K produced for one task.

Layout, alongside the SCF trajectory::

    runtime/cp2k/scf/<task-id>/
    ├── scf_history.json
    └── density/
        ├── density_matrix.npy       (spin_channels, nao, nao)
        ├── overlap_matrix.npy       (nao, nao), when CP2K printed it
        └── density_metadata.json

Recording is off by default (``cp2k.scf_recording.record_density``).  It is not
free: asking CP2K to print AO matrices adds an O(nao^2) text block to every
output, which for a few hundred basis functions is megabytes per task.  Turning
it on is a deliberate decision to collect training data, not something a
labeling campaign should pay for by accident.

Nothing here computes anything.  The density is CP2K's; this module transcribes
it and records enough context to know what it is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gmd_active_learning.adapters.dft.scf.ao_matrices import (
    AOMatrixError,
    electron_count,
    parse_density_matrix,
    parse_overlap_matrix,
)

DENSITY_DIR_NAME = "density"
DENSITY_MATRIX_NAME = "density_matrix.npy"
OVERLAP_MATRIX_NAME = "overlap_matrix.npy"
DENSITY_METADATA_NAME = "density_metadata.json"

#: What ``density_matrix.npy`` contains. Consumers check this before trusting
#: the array's shape, so a future change to the layout cannot be misread.
MATRIX_FORMAT_AO_SPIN = "ao-density-matrix-spin-major-v1"

#: The basis the recorded matrix lives in. AO basis functions overlap, so the
#: electron count is Tr(P S), not Tr(P). Recorded explicitly so a consumer
#: never has to infer it from the format string.
REPRESENTATION_AO_NONORTHOGONAL = "ao-nonorthogonal"

#: Electron-count agreement required of a recorded density, relative to the
#: valence electron count implied by the calculation.
DEFAULT_ELECTRON_TOLERANCE = 1.0e-3


class DensityRecordingError(RuntimeError):
    """Raised when a density was requested but cannot be recorded."""


@dataclass(frozen=True, slots=True)
class DensityArtifacts:
    """Where a recorded density landed, and what it says about itself."""

    directory: Path
    density_path: Path
    overlap_path: Path | None
    metadata_path: Path
    n_ao: int
    spin_channels: int
    electron_count: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "density_matrix_path": str(self.density_path),
            "overlap_path": None if self.overlap_path is None else str(self.overlap_path),
            "metadata_path": str(self.metadata_path),
            "n_ao": self.n_ao,
            "spin_channels": self.spin_channels,
            "electron_count": self.electron_count,
            "matrix_format": MATRIX_FORMAT_AO_SPIN,
            "representation": REPRESENTATION_AO_NONORTHOGONAL,
            "has_overlap": self.overlap_path is not None,
            "electron_count_method": "trace_ps" if self.overlap_path is not None else "trace_p",
        }


def density_recording_enabled(config: Mapping[str, Any]) -> bool:
    """Whether this configuration asks for density artifacts.

    Both switches must be on: density recording is an extension of SCF
    recording, not an independent one.
    """

    recording = _recording_config(config)
    return bool(recording.get("enabled", False)) and bool(recording.get("record_density", False))


def record_density(
    output_text: str,
    *,
    output_root: str | Path,
    task_id: str,
    config: Mapping[str, Any],
    electronic_state: Mapping[str, Any] | None = None,
    cp2k_version: str | None = None,
    scf_converged: bool = False,
    strict: bool = False,
) -> DensityArtifacts | None:
    """Extract and persist the density printed in ``output_text``.

    Returns ``None`` when recording is disabled or when CP2K printed no
    density.  With ``strict=True`` a missing or unusable density raises
    instead, which is what a data-collection run wants and what a labeling run
    does not.
    """

    if not density_recording_enabled(config):
        return None

    try:
        density = parse_density_matrix(output_text)
        overlap = parse_overlap_matrix(output_text)
    except AOMatrixError as exc:
        if strict:
            raise DensityRecordingError(f"{task_id}: {exc}") from exc
        return None

    if not np.isfinite(density).all():
        message = f"{task_id}: recorded density contains non-finite values"
        if strict:
            raise DensityRecordingError(message)
        return None

    state = dict(electronic_state or {})
    cp2k_cfg = _cp2k_config(config)
    counted = electron_count(density, overlap)

    directory = Path(output_root) / task_id / DENSITY_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)

    density_path = directory / DENSITY_MATRIX_NAME
    np.save(density_path, density)

    overlap_path: Path | None = None
    if overlap is not None:
        overlap_path = directory / OVERLAP_MATRIX_NAME
        np.save(overlap_path, overlap)

    metadata = {
        "cp2k_version": cp2k_version,
        "functional": cp2k_cfg.get("functional"),
        "basis_set": cp2k_cfg.get("basis"),
        "potential": cp2k_cfg.get("potential"),
        "charge": state.get("charge"),
        "multiplicity": state.get("multiplicity"),
        "electron_count": counted,
        "spin_channels": int(density.shape[0]),
        "matrix_format": MATRIX_FORMAT_AO_SPIN,
        "scf_converged": bool(scf_converged),
        # Beyond the required fields: without these a consumer cannot tell an
        # exact Tr(PS) from the Tr(P) approximation, or reconstruct the shape.
        "n_ao": int(density.shape[1]),
        "electron_count_method": "trace_ps" if overlap is not None else "trace_p",
        "has_overlap": overlap is not None,
        # The basis this matrix lives in. AO basis functions are not
        # orthonormal, so Ne = Tr(P S) and the overlap above is not optional
        # for anything that needs the electron count. A consumer that reads
        # "trace_p" is looking at an approximation that is 20-50% low on real
        # systems; see docs/density_representation.md.
        "representation": REPRESENTATION_AO_NONORTHOGONAL,
        "task_id": task_id,
    }
    metadata_path = directory / DENSITY_METADATA_NAME
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return DensityArtifacts(
        directory=directory,
        density_path=density_path,
        overlap_path=overlap_path,
        metadata_path=metadata_path,
        n_ao=int(density.shape[1]),
        spin_channels=int(density.shape[0]),
        electron_count=counted,
    )


def load_matrix_file(path: str | Path) -> np.ndarray:
    """Load a matrix from ``.npy`` or ``.npz``.

    Recording writes ``.npy``; a prediction export writes ``.npz``. Both feed
    the same validation and training code, so both are read here rather than at
    each call site, where one of them would inevitably be forgotten.
    """

    source = Path(path)
    if source.suffix == ".npz":
        with np.load(source) as archive:
            if not archive.files:
                raise DensityRecordingError(f"{source}: archive is empty")
            key = "values" if "values" in archive.files else archive.files[0]
            return np.asarray(archive[key], dtype=float)
    return np.asarray(np.load(source), dtype=float)


def load_density(directory: str | Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Load a recorded density, its overlap, and its metadata."""

    source = Path(directory)
    if source.name != DENSITY_DIR_NAME and (source / DENSITY_DIR_NAME).is_dir():
        source = source / DENSITY_DIR_NAME

    density_path = source / DENSITY_MATRIX_NAME
    metadata_path = source / DENSITY_METADATA_NAME
    if not density_path.is_file():
        raise DensityRecordingError(f"no density matrix at {density_path}")
    if not metadata_path.is_file():
        raise DensityRecordingError(f"no density metadata at {metadata_path}")

    try:
        density = np.load(density_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DensityRecordingError(f"could not read density artifacts in {source}: {exc}") from exc

    declared = str(metadata.get("matrix_format", ""))
    if declared != MATRIX_FORMAT_AO_SPIN:
        raise DensityRecordingError(
            f"{metadata_path}: matrix_format {declared!r} is not {MATRIX_FORMAT_AO_SPIN!r}"
        )

    overlap = None
    overlap_path = source / OVERLAP_MATRIX_NAME
    if overlap_path.is_file():
        try:
            overlap = np.load(overlap_path)
        except (OSError, ValueError) as exc:
            raise DensityRecordingError(f"could not read {overlap_path}: {exc}") from exc

    return np.asarray(density, dtype=float), overlap, metadata


def write_density_artifacts(
    directory: str | Path,
    density: np.ndarray,
    *,
    overlap: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DensityArtifacts:
    """Write density artifacts directly. Used by tests and by converters."""

    matrices = np.asarray(density, dtype=float)
    if matrices.ndim == 2:
        matrices = matrices[None, :, :]
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise DensityRecordingError(
            f"density must be (spin, nao, nao), got shape {np.asarray(density).shape}"
        )

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    density_path = target / DENSITY_MATRIX_NAME
    np.save(density_path, matrices)

    overlap_path: Path | None = None
    if overlap is not None:
        overlap_path = target / OVERLAP_MATRIX_NAME
        np.save(overlap_path, np.asarray(overlap, dtype=float))

    counted = electron_count(matrices, overlap)
    payload = {
        "cp2k_version": None,
        "functional": None,
        "basis_set": None,
        "potential": None,
        "charge": None,
        "multiplicity": None,
        "electron_count": counted,
        "spin_channels": int(matrices.shape[0]),
        "matrix_format": MATRIX_FORMAT_AO_SPIN,
        "scf_converged": False,
        "n_ao": int(matrices.shape[1]),
        "electron_count_method": "trace_ps" if overlap is not None else "trace_p",
        "has_overlap": overlap is not None,
        **dict(metadata or {}),
    }
    metadata_path = target / DENSITY_METADATA_NAME
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return DensityArtifacts(
        directory=target,
        density_path=density_path,
        overlap_path=overlap_path,
        metadata_path=metadata_path,
        n_ao=int(matrices.shape[1]),
        spin_channels=int(matrices.shape[0]),
        electron_count=float(payload["electron_count"]),
    )


def _recording_config(config: Any) -> Mapping[str, Any]:
    from gmd_active_learning.adapters.dft.scf.config import scf_config

    return scf_config(config).recording.model_dump()


def _cp2k_config(config: Any) -> Mapping[str, Any]:
    """CP2K *labeling* settings (basis/potential/functional/charge), read only for provenance."""

    from gmd_active_learning.adapters.dft.scf.config import labeling_cp2k_section

    return labeling_cp2k_section(config)


__all__ = [
    "DEFAULT_ELECTRON_TOLERANCE",
    "DENSITY_DIR_NAME",
    "DENSITY_MATRIX_NAME",
    "DENSITY_METADATA_NAME",
    "MATRIX_FORMAT_AO_SPIN",
    "OVERLAP_MATRIX_NAME",
    "DensityArtifacts",
    "DensityRecordingError",
    "density_recording_enabled",
    "load_density",
    "load_matrix_file",
    "record_density",
    "write_density_artifacts",
]
