# Ported from PHAL (hotspot_al.monitor.neighbor_utils) at 96263b6; MIT License, see NOTICE.
"""Shared per-frame neighbour list for the atom-wise monitors.

PHAL wrapped ``ase.neighborlist.NeighborList`` and queried it atom by atom; the coordination
and close-contact monitors then built *another* ASE neighbour list for large systems. Here one
``ase.neighborlist.neighbor_list`` call per frame produces every pair within the largest monitor
cutoff, and all monitors (rmin, coordination, local LJ) read from it.

Convention (identical to PHAL): ``vectors[k] = r_j + shift - r_i`` points FROM atom ``i`` TO its
neighbour ``j``. Every periodic image of ``j`` inside the cutoff is a separate pair, which is what a
pair potential sums over. Distances below ``MIN_PAIR_DISTANCE`` are dropped (coincident images).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from gmd_active_learning.core.exceptions import DataContractError

logger = logging.getLogger(__name__)

MIN_PAIR_DISTANCE = 1.0e-8


@dataclass(frozen=True, slots=True)
class NeighborPairs:
    """All directed pairs (i -> j) within ``cutoff`` of one frame, sorted by ``first``."""

    n_atoms: int
    cutoff: float
    first: np.ndarray  # (P,) int
    second: np.ndarray  # (P,) int
    vectors: np.ndarray  # (P, 3) r_j - r_i (minimum image / explicit image)
    distances: np.ndarray  # (P,)
    offsets: np.ndarray  # (N + 1,) pairs of atom i are first[offsets[i]:offsets[i + 1]]

    def neighbors_of(self, index: int, sub_cutoff: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Neighbour indices, vectors (i -> j) and distances of atom ``index`` within ``sub_cutoff``."""

        start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        indices = self.second[start:stop]
        vectors = self.vectors[start:stop]
        distances = self.distances[start:stop]
        if sub_cutoff is not None and sub_cutoff < self.cutoff:
            mask = distances <= sub_cutoff
            return indices[mask], vectors[mask], distances[mask]
        return indices, vectors, distances

    def within(self, sub_cutoff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(first, second, vectors, distances)`` restricted to ``distance <= sub_cutoff``."""

        if sub_cutoff >= self.cutoff:
            return self.first, self.second, self.vectors, self.distances
        mask = self.distances <= sub_cutoff
        return self.first[mask], self.second[mask], self.vectors[mask], self.distances[mask]


def build_neighbor_pairs(
    positions: np.ndarray,
    cell: np.ndarray | None,
    pbc: tuple[bool, bool, bool] | np.ndarray,
    cutoff: float,
) -> NeighborPairs:
    """Every directed pair within ``cutoff`` (Angstrom), periodic images included."""

    if cutoff <= 0.0:
        raise DataContractError(f"neighbour cutoff must be positive, got {cutoff}")
    positions = np.asarray(positions, dtype=float)
    n_atoms = len(positions)
    pbc_flags = tuple(bool(flag) for flag in np.broadcast_to(np.asarray(pbc, dtype=bool), 3))
    if any(pbc_flags) and cell is None:
        raise DataContractError("periodic neighbour search needs a cell")
    atoms = Atoms(numbers=np.ones(n_atoms, dtype=int), positions=positions, pbc=pbc_flags)
    if cell is not None:
        atoms.set_cell(np.asarray(cell, dtype=float))
    if n_atoms == 0:
        empty_int = np.zeros(0, dtype=np.int64)
        return NeighborPairs(0, float(cutoff), empty_int, empty_int, np.zeros((0, 3)), np.zeros(0), np.zeros(1, dtype=np.int64))
    first, second, distances, vectors = neighbor_list("ijdD", atoms, float(cutoff))
    keep = distances > MIN_PAIR_DISTANCE
    first, second, distances, vectors = first[keep], second[keep], distances[keep], vectors[keep]
    order = np.argsort(first, kind="stable")
    first, second, distances, vectors = first[order], second[order], distances[order], vectors[order]
    offsets = np.searchsorted(first, np.arange(n_atoms + 1), side="left").astype(np.int64)
    return NeighborPairs(
        n_atoms=n_atoms,
        cutoff=float(cutoff),
        first=first.astype(np.int64),
        second=second.astype(np.int64),
        vectors=np.asarray(vectors, dtype=float),
        distances=np.asarray(distances, dtype=float),
        offsets=offsets,
    )


class MonitorNeighbors:
    """Neighbour list shared by the light (rmin, coordination) and physics (LJ) monitors.

    ``cutoff = max(lj_cutoff, coordination_cutoff)``. Call :meth:`rebuild` once per frame; the
    pairs are recomputed from scratch (PHAL used skin 0, which also rebuilt every frame).
    """

    def __init__(self, *, lj_cutoff: float = 6.0, coordination_cutoff: float | None = None) -> None:
        self.lj_cutoff = float(lj_cutoff)
        self.coordination_cutoff = float(coordination_cutoff if coordination_cutoff is not None else lj_cutoff)
        if self.lj_cutoff <= 0.0 or self.coordination_cutoff <= 0.0:
            raise DataContractError("neighbour cutoffs must be positive")
        self.cutoff = max(self.lj_cutoff, self.coordination_cutoff)
        self._pairs: NeighborPairs | None = None
        self.builds = 0

    @property
    def n_atoms(self) -> int:
        return 0 if self._pairs is None else self._pairs.n_atoms

    @property
    def pairs(self) -> NeighborPairs:
        if self._pairs is None:
            raise DataContractError("neighbour list used before rebuild()")
        return self._pairs

    def rebuild(
        self,
        positions: np.ndarray,
        cell: np.ndarray | None,
        pbc: tuple[bool, bool, bool] | np.ndarray,
    ) -> NeighborPairs:
        """Recompute all pairs for a new frame (any atom count)."""

        self._pairs = build_neighbor_pairs(positions, cell, pbc, self.cutoff)
        self.builds += 1
        logger.debug("rebuilt monitor neighbour list: %d atoms, %d pairs", self._pairs.n_atoms, len(self._pairs.first))
        return self._pairs

    def get_displacements(self, index: int, sub_cutoff: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """PHAL-compatible accessor: neighbour indices, vectors (i -> j) and distances."""

        return self.pairs.neighbors_of(index, self.cutoff if sub_cutoff is None else float(sub_cutoff))


__all__ = ["MIN_PAIR_DISTANCE", "MonitorNeighbors", "NeighborPairs", "build_neighbor_pairs"]
