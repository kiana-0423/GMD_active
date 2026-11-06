"""Exact geometry of n-alkanes: all-trans construction and rigid internal-coordinate changes.

Atom order is fixed: carbons ``0..n-1`` along the chain, then the hydrogens grouped by carbon
(three on C0, two on every inner carbon, three on C(n-1)). Atom IDs equal indices, so a perturbation
``location`` (carbon index) is also an atom index.

Every perturbation here is a rigid rotation of one connected side of the molecule, so bond lengths
are preserved exactly (to floating-point rounding):

- torsion about bond (k, k+1): the side containing k+1 rotates about the k->k+1 axis; every angle is
  preserved and every dihedral X-k-(k+1)-Y changes by the same amount;
- bend at vertex k: the side containing k+1 rotates about the axis through k perpendicular to the
  (k-1, k, k+1) plane; the C-C-C angle at k and the two H-C(k)-C(k+1) angles change, nothing else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property

import numpy as np

LOGGER = logging.getLogger(__name__)

TETRAHEDRAL_DEG = float(np.degrees(np.arccos(-1.0 / 3.0)))


@dataclass(frozen=True)
class AlkaneTopology:
    """Connectivity of C(n)H(2n+2) in the fixed atom order."""

    n_carbons: int

    @property
    def n_atoms(self) -> int:
        return 3 * self.n_carbons + 2

    @cached_property
    def numbers(self) -> np.ndarray:
        return np.asarray([6] * self.n_carbons + [1] * (2 * self.n_carbons + 2), dtype=np.int64)

    @cached_property
    def hydrogens_of(self) -> list[list[int]]:
        """Hydrogen indices bonded to each carbon."""

        result: list[list[int]] = []
        index = self.n_carbons
        for carbon in range(self.n_carbons):
            count = 3 if carbon in (0, self.n_carbons - 1) else 2
            if self.n_carbons == 1:
                count = 4
            result.append(list(range(index, index + count)))
            index += count
        return result

    @cached_property
    def carbon_of(self) -> dict[int, int]:
        return {h: c for c, hs in enumerate(self.hydrogens_of) for h in hs}

    @cached_property
    def bonds(self) -> list[tuple[int, int]]:
        pairs = [(c, c + 1) for c in range(self.n_carbons - 1)]
        pairs += [(c, h) for c, hs in enumerate(self.hydrogens_of) for h in hs]
        return sorted(tuple(sorted(pair)) for pair in pairs)  # type: ignore[misc]

    @cached_property
    def adjacency(self) -> list[list[int]]:
        adjacency: list[list[int]] = [[] for _ in range(self.n_atoms)]
        for i, j in self.bonds:
            adjacency[i].append(j)
            adjacency[j].append(i)
        return [sorted(item) for item in adjacency]

    @cached_property
    def angles(self) -> list[tuple[int, int, int]]:
        """Bonded triples ``(i, center, j)`` with ``i < j``."""

        triples = []
        for center, neighbours in enumerate(self.adjacency):
            for a in range(len(neighbours)):
                for b in range(a + 1, len(neighbours)):
                    triples.append((neighbours[a], center, neighbours[b]))
        return triples

    @cached_property
    def graph_distances(self) -> np.ndarray:
        """All-pairs bond-count distances (BFS; the molecule is a tree)."""

        n = self.n_atoms
        distances = np.full((n, n), -1, dtype=np.int64)
        for source in range(n):
            distances[source, source] = 0
            frontier = [source]
            while frontier:
                nxt = []
                for node in frontier:
                    for neighbour in self.adjacency[node]:
                        if distances[source, neighbour] < 0:
                            distances[source, neighbour] = distances[source, node] + 1
                            nxt.append(neighbour)
                frontier = nxt
        return distances

    def side(self, keep: int, moving: int) -> list[int]:
        """Atoms connected to ``moving`` once the bond ``keep``-``moving`` is removed (incl. ``moving``)."""

        if moving not in self.adjacency[keep]:
            raise ValueError(f"atoms {keep} and {moving} are not bonded")
        seen = {moving}
        stack = [moving]
        while stack:
            node = stack.pop()
            for neighbour in self.adjacency[node]:
                if node == moving and neighbour == keep:
                    continue
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return sorted(seen)

    def site_atoms(self, carbons: list[int]) -> list[int]:
        """The given carbons plus their hydrogens, sorted."""

        atoms = set(carbons)
        for carbon in carbons:
            atoms.update(self.hydrogens_of[carbon])
        return sorted(atoms)

    @property
    def formula(self) -> str:
        return f"C{self.n_carbons}H{2 * self.n_carbons + 2}"


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        raise ValueError("zero-length vector")
    return vector / norm


def build_all_trans(n_carbons: int, *, cc_bond: float = 1.53, ch_bond: float = 1.09, angle_deg: float = TETRAHEDRAL_DEG) -> np.ndarray:
    """Positions of all-trans C(n)H(2n+2): planar zigzag backbone in the xy plane, staggered CH3 ends.

    With ``angle_deg`` tetrahedral, every C-C-C, H-C-C and H-C-H angle equals it exactly.
    """

    if n_carbons < 2:
        raise ValueError("an alkane chain needs at least two carbons")
    topology = AlkaneTopology(n_carbons)
    theta = np.radians(angle_deg)
    half = 0.5 * theta
    positions = np.zeros((topology.n_atoms, 3))
    for c in range(n_carbons):
        positions[c] = [c * cc_bond * np.sin(half), (c % 2) * cc_bond * np.cos(half), 0.0]
    normal = np.array([0.0, 0.0, 1.0])
    for c in range(n_carbons):
        hydrogens = topology.hydrogens_of[c]
        if 0 < c < n_carbons - 1:
            n1 = _unit(positions[c - 1] - positions[c])
            n2 = _unit(positions[c + 1] - positions[c])
            bisector = -_unit(n1 + n2)
            # Tetrahedral: the two H lie in the plane spanned by the bisector and the plane normal.
            # Angle H-C-H = theta exactly requires the half angle theta/2 from the bisector, which also
            # makes H-C-C = theta when C-C-C = theta.
            for sign, h in zip((1.0, -1.0), hydrogens):
                positions[h] = positions[c] + ch_bond * (np.cos(half) * bisector + sign * np.sin(half) * normal)
        else:
            neighbour = c + 1 if c == 0 else c - 1
            axis = _unit(positions[neighbour] - positions[c])
            # Staggered methyl: one H anti to the next carbon along the chain (in plane).
            further = neighbour + (1 if c == 0 else -1)
            if 0 <= further < n_carbons:
                reference = positions[further] - positions[neighbour]
            else:
                reference = np.array([0.0, 1.0, 0.0])
            perp = reference - np.dot(reference, axis) * axis
            e1 = -_unit(perp)  # anti to the reference direction
            e2 = np.cross(axis, e1)
            for k, h in enumerate(hydrogens):
                phi = 2.0 * np.pi * k / 3.0
                direction = np.cos(theta) * axis + np.sin(theta) * (np.cos(phi) * e1 + np.sin(phi) * e2)
                positions[h] = positions[c] + ch_bond * direction
    return positions


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Right-handed rotation by ``angle_rad`` about the unit vector along ``axis`` (Rodrigues)."""

    k = _unit(np.asarray(axis, dtype=float))
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle_rad) * kx + (1.0 - np.cos(angle_rad)) * (kx @ kx)


def rotate_atoms(positions: np.ndarray, atoms: list[int], origin: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    out = np.array(positions, dtype=float, copy=True)
    matrix = rotation_matrix(axis, angle_rad)
    out[atoms] = (out[atoms] - origin) @ matrix.T + origin
    return out


def distance(positions: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(positions[j] - positions[i]))


def angle_deg(positions: np.ndarray, i: int, center: int, j: int) -> float:
    a = positions[i] - positions[center]
    b = positions[j] - positions[center]
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def dihedral_deg(positions: np.ndarray, a: int, b: int, c: int, d: int) -> float:
    """IUPAC dihedral in (-180, 180]; trans = 180."""

    b0 = positions[a] - positions[b]
    b1 = _unit(positions[c] - positions[b])
    b2 = positions[d] - positions[c]
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return float(np.degrees(np.arctan2(y, x)))


def wrap_deg(value: float) -> float:
    """Wrap to (-180, 180]."""

    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def apply_torsion(positions: np.ndarray, topology: AlkaneTopology, bond: tuple[int, int], delta_deg: float) -> tuple[np.ndarray, list[int]]:
    """Rotate the side of ``bond[1]`` about the bond axis by ``delta_deg``; returns (positions, moved atoms).

    Every dihedral ``X-bond[0]-bond[1]-Y`` changes by exactly ``delta_deg`` (right-hand rule about
    ``bond[0] -> bond[1]``).
    """

    keep, moving = bond
    side = topology.side(keep, moving)
    origin = positions[moving]
    axis = positions[moving] - positions[keep]
    return rotate_atoms(positions, side, origin, axis, np.radians(delta_deg)), side


def apply_bend(positions: np.ndarray, topology: AlkaneTopology, vertex: int, delta_deg: float) -> tuple[np.ndarray, list[int]]:
    """Open (``delta_deg > 0``) or close the C(v-1)-C(v)-C(v+1) angle by rotating the C(v+1) side."""

    if not 0 < vertex < topology.n_carbons - 1:
        raise ValueError(f"bend vertex {vertex} must be an inner carbon")
    previous, following = vertex - 1, vertex + 1
    side = topology.side(vertex, following)
    a = positions[previous] - positions[vertex]
    b = positions[following] - positions[vertex]
    normal = np.cross(a, b)  # rotating b about +normal increases the angle from a to b
    return rotate_atoms(positions, side, positions[vertex], normal, np.radians(delta_deg)), side


def gaussian_displacement(n_atoms: int, amplitude: float, seed: int) -> np.ndarray:
    """Seeded isotropic Gaussian displacements (standard deviation ``amplitude`` per component)."""

    rng = np.random.default_rng(int(seed))
    return rng.normal(0.0, float(amplitude), size=(n_atoms, 3))


__all__ = [
    "TETRAHEDRAL_DEG",
    "AlkaneTopology",
    "angle_deg",
    "apply_bend",
    "apply_torsion",
    "build_all_trans",
    "dihedral_deg",
    "distance",
    "gaussian_displacement",
    "rotate_atoms",
    "rotation_matrix",
    "wrap_deg",
]
