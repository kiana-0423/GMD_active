"""The MD frame contract delivered by MD adapters (GMD) to monitors and extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols

from gmd_active_learning.core.exceptions import DataContractError

UNITS: dict[str, str] = {
    "length": "Angstrom",
    "energy": "eV",
    "force": "eV/Angstrom",
    "stress": "eV/Angstrom^3",
    "time": "fs",
}


def _as_float_array(name: str, value: Any, shape: tuple[int | None, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != len(shape) or any(expected is not None and got != expected for got, expected in zip(array.shape, shape)):
        raise DataContractError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.all(np.isfinite(array)):
        raise DataContractError(f"{name} contains non-finite values")
    return array


@dataclass(slots=True)
class Frame:
    """One MD frame with explicit identity, units and provenance.

    ``atom_ids`` are stable across the whole run (they survive any reordering by the engine);
    ``numbers`` are atomic numbers. ``forces`` are the forces that drove the dynamics (the
    deployed model's forces for ML runs). ``committee_forces`` is ``None`` when no committee
    was evaluated; that is distinct from a committee that agrees perfectly.
    """

    run_id: str
    step: int
    atom_ids: np.ndarray
    numbers: np.ndarray
    positions: np.ndarray
    cell: np.ndarray | None
    pbc: tuple[bool, bool, bool]
    forces: np.ndarray | None = None
    time_fs: float | None = None
    energy: float | None = None
    velocities: np.ndarray | None = None
    committee_forces: np.ndarray | None = None
    model_version: str | None = None
    model_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.step = int(self.step)
        self.atom_ids = np.asarray(self.atom_ids, dtype=np.int64)
        self.numbers = np.asarray(self.numbers, dtype=np.int64)
        n_atoms = len(self.numbers)
        if self.atom_ids.shape != (n_atoms,):
            raise DataContractError(f"atom_ids shape {self.atom_ids.shape} != ({n_atoms},)")
        if len(np.unique(self.atom_ids)) != n_atoms:
            raise DataContractError("atom_ids must be unique")
        if np.any(self.numbers <= 0) or np.any(self.numbers >= len(chemical_symbols)):
            raise DataContractError("numbers must be valid atomic numbers (unknown species are rejected)")
        self.positions = _as_float_array("positions", self.positions, (n_atoms, 3))
        self.pbc = tuple(bool(flag) for flag in self.pbc)  # type: ignore[assignment]
        if len(self.pbc) != 3:
            raise DataContractError("pbc must have three flags")
        if self.cell is not None:
            self.cell = _as_float_array("cell", self.cell, (3, 3))
        if any(self.pbc) and (self.cell is None or abs(np.linalg.det(self.cell)) < 1.0e-12):
            raise DataContractError("a periodic frame needs a non-degenerate cell")
        if self.forces is not None:
            self.forces = _as_float_array("forces", self.forces, (n_atoms, 3))
        if self.velocities is not None:
            self.velocities = _as_float_array("velocities", self.velocities, (n_atoms, 3))
        if self.committee_forces is not None:
            self.committee_forces = _as_float_array("committee_forces", self.committee_forces, (None, n_atoms, 3))
        if self.energy is not None and not np.isfinite(self.energy):
            raise DataContractError("energy must be finite")

    @property
    def n_atoms(self) -> int:
        return int(len(self.numbers))

    @property
    def symbols(self) -> list[str]:
        return [chemical_symbols[int(number)] for number in self.numbers]

    def index_of(self, atom_ids: Any) -> np.ndarray:
        """Frame indices of the given stable atom IDs."""

        lookup = {int(atom_id): index for index, atom_id in enumerate(self.atom_ids)}
        try:
            return np.asarray([lookup[int(atom_id)] for atom_id in np.atleast_1d(atom_ids)], dtype=np.int64)
        except KeyError as exc:
            raise DataContractError(f"atom id {exc.args[0]} is not in frame {self.run_id}:{self.step}") from exc

    def to_atoms(self) -> Atoms:
        atoms = Atoms(numbers=self.numbers, positions=self.positions, pbc=self.pbc)
        if self.cell is not None:
            atoms.set_cell(self.cell)
        atoms.arrays["atom_id"] = self.atom_ids.copy()
        if self.forces is not None:
            atoms.arrays["forces"] = self.forces.copy()
        atoms.info.update({"run_id": self.run_id, "step": self.step})
        if self.time_fs is not None:
            atoms.info["time_fs"] = self.time_fs
        if self.energy is not None:
            atoms.info["energy"] = self.energy
        if self.model_version is not None:
            atoms.info["model_version"] = self.model_version
        return atoms

    @classmethod
    def from_atoms(
        cls,
        atoms: Atoms,
        *,
        run_id: str,
        step: int | None = None,
        forces: np.ndarray | None = None,
        **kwargs: Any,
    ) -> Frame:
        if forces is None:
            if "forces" in atoms.arrays:
                forces = atoms.arrays["forces"]
            elif atoms.calc is not None and "forces" in getattr(atoms.calc, "results", {}):
                forces = atoms.calc.results["forces"]
        atom_ids = atoms.arrays.get("atom_id", atoms.arrays.get("id", np.arange(len(atoms))))
        cell = atoms.cell.array.copy() if atoms.cell.rank == 3 else None
        energy = kwargs.pop("energy", atoms.info.get("energy"))
        return cls(
            run_id=run_id,
            step=int(atoms.info.get("step", 0) if step is None else step),
            atom_ids=np.asarray(atom_ids),
            numbers=atoms.get_atomic_numbers(),
            positions=atoms.get_positions(),
            cell=cell,
            pbc=tuple(bool(flag) for flag in atoms.pbc),  # type: ignore[arg-type]
            forces=forces,
            energy=energy,
            time_fs=kwargs.pop("time_fs", atoms.info.get("time_fs")),
            **kwargs,
        )


def symbols_to_numbers(symbols: list[str]) -> np.ndarray:
    try:
        return np.asarray([atomic_numbers[symbol] for symbol in symbols], dtype=np.int64)
    except KeyError as exc:
        raise DataContractError(f"unknown chemical symbol {exc.args[0]!r}") from exc
