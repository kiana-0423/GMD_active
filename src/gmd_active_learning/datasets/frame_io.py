# Ported from PHAL (hotspot_al.io.trajectory_reader, hotspot_al.io.extxyz_reader) at 96263b6; MIT License, see NOTICE.
"""Read and write :class:`~gmd_active_learning.contracts.Frame` sequences from ASE-readable files.

Differences from PHAL (ledger IO-001 / IO-003):

* ``Frame.step`` is the source MD step (``info['step']``) when present, otherwise the frame's
  position in the *source file* (PHAL used the position after slicing, so ``index="-1"`` gave
  step 0).
* External forces are indexed by source frame position, so slicing keeps them aligned (PHAL
  attached ``forces[0]`` to the last frame for ``index="-1"``).
* Load errors are raised, not swallowed. LAMMPS dumps are not read here: GMD replaces LAMMPS and
  its adapter owns trajectory parsing.
* :func:`write_frames` writes floats with their shortest exact repr, so a write/read cycle is
  lossless (ASE's writer rounds to 8 decimals).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase.data import chemical_symbols
from ase.io import read
from ase.io.extxyz import key_val_dict_to_str

from gmd_active_learning.contracts import Frame
from gmd_active_learning.core.exceptions import DataContractError, UnsupportedCombinationError

_LAMMPS_SUFFIXES = {".lammpstrj", ".dump"}


def _selected_positions(n_frames: int, index: str | int | slice) -> list[int]:
    positions = list(range(n_frames))
    if isinstance(index, int):
        return [positions[index]]
    if isinstance(index, slice):
        return positions[index]
    text = str(index).strip()
    if ":" not in text:
        return [positions[int(text)]]
    parts = [int(part) if part.strip() else None for part in text.split(":")]
    return positions[slice(*parts)]


def load_external_forces(path: str | Path) -> np.ndarray:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        forces = np.load(source, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as data:
            key = "forces" if "forces" in data.files else data.files[0]
            forces = data[key]
    elif suffix in {".txt", ".dat"}:
        forces = np.loadtxt(source)
    elif suffix == ".csv":
        forces = np.loadtxt(source, delimiter=",")
    else:
        raise DataContractError(f"unsupported external force file {source}")
    return np.asarray(forces, dtype=float)


def read_frames(
    path: str | Path,
    *,
    run_id: str,
    index: str | int | slice = ":",
    external_forces: str | Path | np.ndarray | None = None,
    format: str | None = None,
) -> list[Frame]:
    source = Path(path)
    if source.suffix.lower() in _LAMMPS_SUFFIXES:
        raise UnsupportedCombinationError("LAMMPS dumps are not read by GMD_active; use the GMD adapter's trajectory reader")
    images = read(str(source), index=":", format=format)
    if not isinstance(images, list):
        images = [images]
    chosen = _selected_positions(len(images), index)
    forces_all = None
    if external_forces is not None:
        forces_all = external_forces if isinstance(external_forces, np.ndarray) else load_external_forces(external_forces)
        forces_all = np.asarray(forces_all, dtype=float)
        if forces_all.ndim == 2:
            if len(chosen) != 1:
                raise DataContractError("2-D external forces describe one frame; select exactly one frame")
            forces_all = {chosen[0]: forces_all}  # type: ignore[assignment]
        elif forces_all.ndim == 3:
            if forces_all.shape[0] != len(images):
                raise DataContractError(f"external forces hold {forces_all.shape[0]} frames, the file {len(images)}")
            forces_all = {k: forces_all[k] for k in chosen}  # type: ignore[assignment]
        else:
            raise DataContractError(f"external forces must be 2-D or 3-D, got shape {forces_all.shape}")
    frames = []
    for position in chosen:
        atoms = images[position]
        forces = None if forces_all is None else forces_all[position]  # type: ignore[index]
        if forces is None and atoms.calc is not None and "forces" in getattr(atoms.calc, "results", {}):
            forces = atoms.calc.results["forces"]
        energy = atoms.info.get("energy")
        if energy is None and atoms.calc is not None:
            energy = atoms.calc.results.get("energy")
        velocities = atoms.arrays.get("velocities")
        frames.append(
            Frame.from_atoms(
                atoms,
                run_id=str(atoms.info.get("run_id", run_id)),
                step=int(atoms.info.get("step", position)),
                forces=forces,
                energy=energy,
                velocities=velocities,
                model_version=atoms.info.get("model_version"),
                metadata={"source_index": position, "source_file": source.name},
            )
        )
    return frames


def write_frames(frames: Sequence[Frame] | Iterable[Frame], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for frame in frames:
        properties = "species:S:1:pos:R:3:atom_id:I:1"
        if frame.forces is not None:
            properties += ":forces:R:3"
        if frame.velocities is not None:
            properties += ":velocities:R:3"
        header = []
        if frame.cell is not None:
            header.append('Lattice="' + " ".join(repr(float(v)) for v in frame.cell.reshape(9)) + '"')
        header.append(f"Properties={properties}")
        info: dict[str, Any] = {"run_id": frame.run_id, "step": int(frame.step)}
        if frame.time_fs is not None:
            info["time_fs"] = float(frame.time_fs)
        if frame.energy is not None:
            info["energy"] = float(frame.energy)
        if frame.model_version is not None:
            info["model_version"] = frame.model_version
        info["pbc"] = np.asarray(frame.pbc, dtype=bool)
        header.append(key_val_dict_to_str(info))
        lines = [str(frame.n_atoms), " ".join(header)]
        for k in range(frame.n_atoms):
            values = [chemical_symbols[int(frame.numbers[k])], *(repr(float(v)) for v in frame.positions[k]), str(int(frame.atom_ids[k]))]
            if frame.forces is not None:
                values += [repr(float(v)) for v in frame.forces[k]]
            if frame.velocities is not None:
                values += [repr(float(v)) for v in frame.velocities[k]]
            lines.append(" ".join(values))
        chunks.append("\n".join(lines) + "\n")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
        handle.write("".join(chunks))
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, target)
    return target


__all__ = ["load_external_forces", "read_frames", "write_frames"]
