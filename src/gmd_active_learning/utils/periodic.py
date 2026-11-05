# Ported from PHAL (hotspot_al.utils.periodic) at 96263b6; MIT License, see NOTICE.
"""Periodic boundary condition helpers."""

from __future__ import annotations

import numpy as np


def as_cell_matrix(cell: np.ndarray | None) -> np.ndarray | None:
    """Normalize a cell description to a 3x3 matrix."""

    if cell is None:
        return None
    matrix = np.asarray(cell, dtype=float)
    if matrix.shape == (3,):
        return np.diag(matrix)
    if matrix.shape != (3, 3):
        msg = f"Unsupported cell shape: {matrix.shape}"
        raise ValueError(msg)
    return matrix


def _is_effectively_orthorhombic(cell_matrix: np.ndarray, tolerance: float = 1.0e-10) -> bool:
    """Whether the cell is diagonal to within ``tolerance``."""

    off_diagonal = cell_matrix - np.diag(np.diag(cell_matrix))
    return bool(np.all(np.abs(off_diagonal) <= tolerance))


def _triclinic_mic(displacements: np.ndarray, cell_matrix: np.ndarray, pbc_mask: np.ndarray) -> np.ndarray:
    """Exact minimum image for skewed cells, including partially periodic ones.

    Only the periodic cell vectors generate translations. The displacement is projected onto
    the (Minkowski-reduced) periodic sub-lattice by least squares, rounded, and refined over
    two shells of neighbouring translations; the component orthogonal to the sub-lattice is
    unaffected by any translation.

    PHAL decomposed the vector in the full 3-D basis, which mixes the non-periodic vector into
    the periodic coordinates and returns non-minimal images for skewed cells with partial PBC
    (docs/migration/findings.md F-34). ``ase.geometry.find_mic`` is not used because its
    shortcut accepts wrong answers for strongly skewed cells.
    """

    from ase.geometry import minkowski_reduce

    vectors = np.atleast_2d(np.asarray(displacements, dtype=float))
    pbc = tuple(bool(flag) for flag in pbc_mask)
    reduced_cell, _transform = minkowski_reduce(cell_matrix, pbc=pbc)
    periodic_vectors = reduced_cell[np.asarray(pbc_mask, dtype=bool)]

    coefficients = vectors @ np.linalg.pinv(periodic_vectors)
    wrapped = vectors - np.round(coefficients) @ periodic_vectors

    shells = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    grid = np.meshgrid(*([shells] * len(periodic_vectors)), indexing="ij")
    offsets = np.stack([axis.ravel() for axis in grid], axis=1) @ periodic_vectors

    candidates = wrapped[:, None, :] + offsets[None, :, :]
    best = np.argmin(np.einsum("ijk,ijk->ij", candidates, candidates), axis=1)
    return candidates[np.arange(len(candidates)), best]


def minimum_image_vectors(
    displacements: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
) -> np.ndarray:
    """Reduce displacement vectors of any shape ``(..., 3)`` to the minimum image.

    This is the single implementation every caller should use.  Hand-rolled
    fractional rounding at the call site silently returns non-minimal vectors in
    skewed cells, so the logic lives here once.
    """

    vectors = np.asarray(displacements, dtype=float)
    cell_matrix = as_cell_matrix(cell)
    pbc_mask = np.broadcast_to(np.asarray(pbc, dtype=bool), 3)
    if cell_matrix is None or not np.any(pbc_mask):
        return vectors

    original_shape = vectors.shape
    flat = vectors.reshape(-1, 3)
    if _is_effectively_orthorhombic(cell_matrix):
        fractional = flat @ np.linalg.inv(cell_matrix)
        fractional[:, pbc_mask] -= np.round(fractional[:, pbc_mask])
        return (fractional @ cell_matrix).reshape(original_shape)
    return _triclinic_mic(flat, cell_matrix, pbc_mask).reshape(original_shape)


def mic_displacement(
    source: np.ndarray,
    target: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
) -> np.ndarray:
    """Return the minimum-image displacement from ``source`` to ``target``."""

    displacement = np.asarray(target, dtype=float) - np.asarray(source, dtype=float)
    return minimum_image_vectors(displacement, cell=cell, pbc=pbc)


def mic_displacements_from_reference(
    reference: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
) -> np.ndarray:
    """Return minimum-image displacements from one reference to many positions."""

    reference = np.asarray(reference, dtype=float)
    positions = np.asarray(positions, dtype=float)
    return minimum_image_vectors(positions - reference, cell=cell, pbc=pbc)


def mic_distance(
    source: np.ndarray,
    target: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
) -> float:
    """Return the minimum-image distance between two points."""

    return float(np.linalg.norm(mic_displacement(source, target, cell=cell, pbc=pbc)))
