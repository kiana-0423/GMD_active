"""Extracted-region contract: a labelable sub-system with full parent mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols

from gmd_active_learning.core.exceptions import DataContractError


class RegionRole(str, Enum):
    CORE = "core"
    INNER_BUFFER = "inner_buffer"
    OUTER_BUFFER = "outer_buffer"
    BOUNDARY = "boundary"  # also block mode's frozen shell
    H_CAP = "h_cap"
    UNASSIGNED = "unassigned"


# Integer codes are PHAL-compatible (training/region_codes.py) so historical datasets stay readable.
REGION_CODES: dict[RegionRole, int] = {
    RegionRole.CORE: 0,
    RegionRole.INNER_BUFFER: 1,
    RegionRole.OUTER_BUFFER: 2,
    RegionRole.BOUNDARY: 3,
    RegionRole.H_CAP: 4,
    RegionRole.UNASSIGNED: -1,
}


@dataclass(slots=True)
class ExtractedRegion:
    """Atoms cut out of a parent frame around a hotspot.

    Every per-atom array has exactly ``n_atoms`` entries; adding cap atoms extends all of them
    together (see :meth:`with_caps`). ``parent_atom_ids`` holds the stable ID of each atom in
    the parent frame (-1 for cap atoms); ``image_shifts`` holds the integer lattice vector that
    was added to the parent position, so ``position = parent_position + image_shift @ cell``.
    ``energy_is_parent_energy`` is always False: a region's DFT energy describes the extracted
    (and possibly capped) system only.
    """

    region_id: str
    mode: str
    run_id: str
    parent_step: int
    event_id: str | None
    numbers: np.ndarray
    positions: np.ndarray
    cell: np.ndarray | None
    pbc: tuple[bool, bool, bool]
    parent_atom_ids: np.ndarray
    image_shifts: np.ndarray
    roles: list[RegionRole]
    force_weights: np.ndarray
    frozen: np.ndarray
    hotspot_atom_ids: list[int]
    charge: int = 0
    multiplicity: int | None = None
    model_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.numbers = np.asarray(self.numbers, dtype=np.int64)
        n_atoms = len(self.numbers)
        self.positions = np.asarray(self.positions, dtype=float).reshape(n_atoms, 3)
        self.parent_atom_ids = np.asarray(self.parent_atom_ids, dtype=np.int64)
        self.image_shifts = np.asarray(self.image_shifts, dtype=np.int64).reshape(n_atoms, 3)
        self.force_weights = np.asarray(self.force_weights, dtype=float)
        self.frozen = np.asarray(self.frozen, dtype=bool)
        self.roles = [RegionRole(role) for role in self.roles]
        self.pbc = tuple(bool(flag) for flag in self.pbc)  # type: ignore[assignment]
        if self.cell is not None:
            self.cell = np.asarray(self.cell, dtype=float).reshape(3, 3)
        for name, size in (
            ("parent_atom_ids", len(self.parent_atom_ids)),
            ("roles", len(self.roles)),
            ("force_weights", len(self.force_weights)),
            ("frozen", len(self.frozen)),
        ):
            if size != n_atoms:
                raise DataContractError(f"region {self.region_id}: {name} has {size} entries for {n_atoms} atoms")
        if np.any(self.force_weights < 0) or not np.all(np.isfinite(self.force_weights)):
            raise DataContractError(f"region {self.region_id}: force weights must be finite and >= 0")
        if any(self.pbc) and self.cell is None:
            raise DataContractError(f"region {self.region_id}: periodic region needs a cell")
        caps = np.asarray([role is RegionRole.H_CAP for role in self.roles])
        if np.any(self.parent_atom_ids[caps] != -1) or np.any(self.parent_atom_ids[~caps] < 0):
            raise DataContractError(f"region {self.region_id}: cap atoms (and only cap atoms) have parent id -1")

    @property
    def n_atoms(self) -> int:
        return int(len(self.numbers))

    @property
    def region_codes(self) -> np.ndarray:
        return np.asarray([REGION_CODES[role] for role in self.roles], dtype=np.int64)

    def indices_with(self, *roles: RegionRole) -> np.ndarray:
        return np.asarray([index for index, role in enumerate(self.roles) if role in roles], dtype=np.int64)

    def with_caps(
        self,
        cap_positions: np.ndarray,
        cap_metadata: list[dict[str, Any]],
        *,
        cap_numbers: np.ndarray | None = None,
    ) -> ExtractedRegion:
        """Return a copy with cap atoms appended; every per-atom array grows together."""

        cap_positions = np.asarray(cap_positions, dtype=float).reshape(-1, 3)
        n_caps = len(cap_positions)
        numbers = np.full(n_caps, 1, dtype=np.int64) if cap_numbers is None else np.asarray(cap_numbers, dtype=np.int64)
        metadata = dict(self.metadata)
        metadata["caps"] = [*metadata.get("caps", []), *cap_metadata]
        return ExtractedRegion(
            region_id=self.region_id,
            mode=self.mode,
            run_id=self.run_id,
            parent_step=self.parent_step,
            event_id=self.event_id,
            numbers=np.concatenate([self.numbers, numbers]),
            positions=np.vstack([self.positions, cap_positions]),
            cell=self.cell,
            pbc=self.pbc,
            parent_atom_ids=np.concatenate([self.parent_atom_ids, np.full(n_caps, -1, dtype=np.int64)]),
            image_shifts=np.vstack([self.image_shifts, np.zeros((n_caps, 3), dtype=np.int64)]),
            roles=[*self.roles, *([RegionRole.H_CAP] * n_caps)],
            force_weights=np.concatenate([self.force_weights, np.zeros(n_caps)]),
            frozen=np.concatenate([self.frozen, np.zeros(n_caps, dtype=bool)]),
            hotspot_atom_ids=list(self.hotspot_atom_ids),
            charge=self.charge,
            multiplicity=self.multiplicity,
            model_version=self.model_version,
            metadata=metadata,
        )

    def to_atoms(self) -> Atoms:
        atoms = Atoms(numbers=self.numbers, positions=self.positions, pbc=self.pbc)
        if self.cell is not None:
            atoms.set_cell(self.cell)
        atoms.arrays["parent_atom_id"] = self.parent_atom_ids.copy()
        atoms.arrays["image_shift"] = self.image_shifts.copy()
        atoms.arrays["region_code"] = self.region_codes
        atoms.arrays["force_weight"] = self.force_weights.copy()
        atoms.arrays["frozen"] = self.frozen.astype(np.int64)
        atoms.info.update(
            {
                "region_id": self.region_id,
                "extraction_mode": self.mode,
                "run_id": self.run_id,
                "parent_step": self.parent_step,
                "event_id": self.event_id or "",
                "charge": self.charge,
            }
        )
        if self.multiplicity is not None:
            atoms.info["multiplicity"] = self.multiplicity
        return atoms

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "mode": self.mode,
            "run_id": self.run_id,
            "parent_step": self.parent_step,
            "event_id": self.event_id,
            "symbols": [chemical_symbols[int(number)] for number in self.numbers],
            "numbers": self.numbers.tolist(),
            "positions": self.positions.tolist(),
            "cell": None if self.cell is None else self.cell.tolist(),
            "pbc": list(self.pbc),
            "parent_atom_ids": self.parent_atom_ids.tolist(),
            "image_shifts": self.image_shifts.tolist(),
            "roles": [role.value for role in self.roles],
            "force_weights": self.force_weights.tolist(),
            "frozen": self.frozen.tolist(),
            "hotspot_atom_ids": list(map(int, self.hotspot_atom_ids)),
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "model_version": self.model_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractedRegion:
        return cls(
            region_id=data["region_id"],
            mode=data["mode"],
            run_id=data["run_id"],
            parent_step=int(data["parent_step"]),
            event_id=data.get("event_id"),
            numbers=np.asarray(data["numbers"]),
            positions=np.asarray(data["positions"]),
            cell=None if data.get("cell") is None else np.asarray(data["cell"]),
            pbc=tuple(data["pbc"]),  # type: ignore[arg-type]
            parent_atom_ids=np.asarray(data["parent_atom_ids"]),
            image_shifts=np.asarray(data["image_shifts"]),
            roles=[RegionRole(role) for role in data["roles"]],
            force_weights=np.asarray(data["force_weights"]),
            frozen=np.asarray(data["frozen"]),
            hotspot_atom_ids=list(data.get("hotspot_atom_ids", [])),
            charge=int(data.get("charge", 0)),
            multiplicity=data.get("multiplicity"),
            model_version=data.get("model_version"),
            metadata=dict(data.get("metadata", {})),
        )
