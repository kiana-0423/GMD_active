"""NPZ files: the lossless canonical sample archive and GMD_SGT's NPZ training format.

* :func:`write_samples_npz` / :func:`read_samples_npz` -- canonical, pickle-free archive
  (concatenated per-atom arrays + ``n_atoms`` offsets) with a JSON sidecar holding provenance and
  region metadata. This is what the dataset store keeps; round trips are bit-exact.
* :func:`write_sgt_npz` -- GMD_SGT's reader convention (``R Z E F cell pbc stress force_weight
  energy_weight stress_weight split_group``). Missing energies/stresses are NaN, which GMD_SGT
  treats as "no label" (energy only when ``energy_weight == 0``). Structures of different sizes
  are stored as object arrays, which GMD_SGT loads with ``allow_pickle=True``.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from gmd_active_learning.core.exceptions import DataContractError
from gmd_active_learning.datasets.samples import LabeledSample, samples_from_arrays, samples_to_arrays


def _atomic_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return path


def _npz_bytes(arrays: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def sidecar_path(npz_path: str | Path) -> Path:
    path = Path(npz_path)
    return path.with_name(path.stem + ".json")


def write_samples_npz(samples: list[LabeledSample], path: str | Path) -> tuple[Path, Path]:
    """Canonical archive ``<name>.npz`` + ``<name>.json``; both written atomically."""

    arrays, records = samples_to_arrays(samples)
    target = Path(path)
    npz = _atomic_bytes(target, _npz_bytes(arrays))
    sidecar = _atomic_bytes(sidecar_path(target), json.dumps(records, indent=1, sort_keys=True).encode("utf-8"))
    return npz, sidecar


def read_samples_npz(path: str | Path) -> list[LabeledSample]:
    target = Path(path)
    with np.load(target, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    records = json.loads(sidecar_path(target).read_text(encoding="utf-8"))
    return samples_from_arrays(arrays, records)


def _stack(values: list[np.ndarray], dtype: Any) -> np.ndarray:
    shapes = {value.shape for value in values}
    if len(shapes) == 1:
        return np.stack([np.asarray(value, dtype=dtype) for value in values])
    out = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        out[index] = np.asarray(value, dtype=dtype)
    return out


def write_sgt_npz(samples: list[LabeledSample], path: str | Path) -> Path:
    """GMD_SGT NPZ (see :func:`gmd_sgt.data.reader.read_npz`)."""

    if not samples:
        raise DataContractError("cannot export an empty sample list")
    arrays: dict[str, Any] = {
        "R": _stack([s.region.positions for s in samples], float),
        "Z": _stack([s.region.numbers for s in samples], np.int64),
        "E": np.asarray([np.nan if s.energy is None else s.energy for s in samples], dtype=float),
        "F": _stack([s.forces for s in samples], float),
        "cell": np.stack([np.zeros((3, 3)) if s.region.cell is None else s.region.cell for s in samples]),
        "pbc": np.asarray([s.region.pbc for s in samples], dtype=bool),
        "stress": np.stack([np.full((3, 3), np.nan) if s.stress is None else s.stress for s in samples]),
        "force_weight": _stack([s.region.force_weights for s in samples], float),
        "energy_weight": np.asarray([s.energy_weight for s in samples], dtype=float),
        "stress_weight": np.asarray([s.stress_weight for s in samples], dtype=float),
        "split_group": np.asarray([s.split_group for s in samples]),
        "sample_id": np.asarray([s.sample_id for s in samples]),
        "region_code": _stack([s.region.region_codes for s in samples], np.int64),
        "parent_atom_id": _stack([s.region.parent_atom_ids for s in samples], np.int64),
        "image_shift": _stack([s.region.image_shifts for s in samples], np.int64),
    }
    return _atomic_bytes(Path(path), _npz_bytes(arrays))


__all__ = ["read_samples_npz", "sidecar_path", "write_samples_npz", "write_sgt_npz"]
