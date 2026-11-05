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
    #: extra per-atom arrays (first axis = atoms), e.g. ASE ``arrays`` beyond positions/numbers/forces
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    #: ASE constraints as ``constraint.todict()`` mappings (e.g. FixAtoms), restored by ``to_atoms``
    constraints: list[dict[str, Any]] = field(default_factory=list)
    #: free-form per-frame info (ASE ``atoms.info`` beyond the contract's own keys)
    info: dict[str, Any] = field(default_factory=dict)

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
        arrays = {}
        for name, value in dict(self.arrays).items():
            array = np.asarray(value)
            if array.ndim == 0 or array.shape[0] != n_atoms:
                raise DataContractError(f"per-atom array {name!r} has shape {array.shape}, expected ({n_atoms}, ...)")
            arrays[str(name)] = array
        self.arrays = arrays
        self.constraints = [dict(item) for item in self.constraints]
        for item in self.constraints:
            if "name" not in item:
                raise DataContractError(f"constraint {item} is not an ASE todict() mapping (no 'name')")

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
        from ase.constraints import dict2constraint

        atoms = Atoms(numbers=self.numbers, positions=self.positions, pbc=self.pbc)
        if self.cell is not None:
            atoms.set_cell(self.cell)
        atoms.info.update(self.info)
        for name, array in self.arrays.items():
            atoms.arrays[name] = np.array(array, copy=True)
        if self.velocities is not None and "momenta" not in self.arrays:
            atoms.set_velocities(self.velocities)
        if self.constraints:
            atoms.set_constraint([dict2constraint(dict(item)) for item in self.constraints])
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
        if "arrays" not in kwargs:
            kwargs["arrays"] = {name: np.array(value, copy=True) for name, value in atoms.arrays.items() if name not in _OWN_ARRAYS}
        if "constraints" not in kwargs:
            kwargs["constraints"] = _constraints_to_dicts(atoms.constraints)
        if "info" not in kwargs:
            kwargs["info"] = {key: value for key, value in atoms.info.items() if key not in _OWN_INFO}
        if "velocities" not in kwargs:
            if "vel" in atoms.arrays:  # GMD frames: Properties=...:vel:R:3 (Angstrom/fs)
                kwargs["velocities"] = np.asarray(atoms.arrays["vel"], dtype=float)
            elif "momenta" in atoms.arrays:
                kwargs["velocities"] = atoms.get_velocities()
        energy = kwargs.pop("energy", atoms.info.get("energy"))
        if energy is None and atoms.calc is not None:
            # ASE's extxyz reader moves per-frame energy/forces into a SinglePointCalculator.
            energy = getattr(atoms.calc, "results", {}).get("energy")
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


    # ------------------------------------------------------------------ legacy GMD_active MDFrame
    def to_md_frame(self) -> Any:
        """Legacy ``core.data_types.MDFrame`` (shared fields only; committee -> ensemble_forces)."""

        from gmd_active_learning.core.data_types import MDFrame

        if self.forces is None:
            raise DataContractError("MDFrame requires forces")
        metadata = {**self.metadata, "run_id": self.run_id, "atom_ids": self.atom_ids.tolist()}
        return MDFrame(step=self.step, positions=self.positions.copy(), symbols=self.symbols, forces=self.forces.copy(),
                       cell=None if self.cell is None else self.cell.copy(), pbc=tuple(self.pbc), energy=self.energy,
                       ensemble_forces=None if self.committee_forces is None else self.committee_forces.copy(), metadata=metadata)

    @classmethod
    def from_md_frame(cls, md_frame: Any, *, run_id: str | None = None) -> Frame:
        metadata = dict(md_frame.metadata)
        run = run_id or metadata.pop("run_id", None) or "legacy"
        metadata.pop("run_id", None)
        atom_ids = metadata.pop("atom_ids", None)
        return cls(run_id=run, step=md_frame.step, atom_ids=np.arange(len(md_frame.symbols)) if atom_ids is None else np.asarray(atom_ids),
                   numbers=symbols_to_numbers(list(md_frame.symbols)), positions=md_frame.positions, cell=md_frame.cell, pbc=md_frame.pbc,
                   forces=md_frame.forces, energy=md_frame.energy, committee_forces=md_frame.ensemble_forces, metadata=metadata)


#: ASE arrays/info keys the contract carries in dedicated fields.
_OWN_ARRAYS = frozenset({"numbers", "positions", "atom_id", "id", "forces", "momenta", "vel"})
_OWN_INFO = frozenset({"run_id", "step", "time_fs", "energy", "model_version"})


def _constraints_to_dicts(constraints: Any) -> list[dict[str, Any]]:
    out = []
    for constraint in constraints or []:
        try:
            out.append(constraint.todict())
        except AttributeError as exc:  # PHAL dropped these with a warning; refusing is safer
            raise DataContractError(f"constraint {type(constraint).__name__} cannot be serialised (no todict())") from exc
    return out


def symbols_to_numbers(symbols: list[str]) -> np.ndarray:
    try:
        return np.asarray([atomic_numbers[symbol] for symbol in symbols], dtype=np.int64)
    except KeyError as exc:
        raise DataContractError(f"unknown chemical symbol {exc.args[0]!r}") from exc
