# Ported from PHAL (hotspot_al.cp2k.ao_matrix_parser) at 96263b6; MIT License, see NOTICE.
"""Parse the AO matrices CP2K prints under ``&DFT &PRINT &AO_MATRICES``.

CP2K writes a full matrix as a series of *column blocks*: a header line of
column indices, an optional line of basis-function labels, then one line per
row holding that row's entries for those columns.  A wide matrix therefore
appears as several blocks that must be stitched back together horizontally.

The parser is deliberately tolerant about the row-label columns, which differ
between CP2K versions (some print ``row atom element shell``, some fewer
fields).  What it is *not* tolerant about is arithmetic: a block whose row
count or column count disagrees with what was already read raises rather than
being padded, because a silently mis-stitched density matrix would be accepted
by every downstream check and be wrong.
"""

from __future__ import annotations

import re
from typing import Iterator, Sequence

import numpy as np

#: Matrices this module knows how to locate, mapped to their CP2K headings.
DENSITY_TITLES: tuple[str, ...] = ("DENSITY MATRIX",)
OVERLAP_TITLES: tuple[str, ...] = ("OVERLAP MATRIX",)

_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?")
_INTEGER_ROW = re.compile(r"^\s*\d+(?:\s+\d+)*\s*$")
_SPIN_SUFFIX = re.compile(r"\bFOR\s+(ALPHA|BETA)\s+SPIN\b", re.IGNORECASE)


class AOMatrixError(ValueError):
    """Raised when a printed AO matrix cannot be reconstructed."""


def parse_ao_matrices(text: str, titles: Sequence[str]) -> list[np.ndarray]:
    """Return every matrix printed under one of ``titles``, in output order.

    A spin-polarised calculation prints the density twice (alpha then beta), so
    the caller receives a list rather than a single array.
    """

    return [matrix for _spin, matrix in parse_ao_matrices_with_spin(text, titles)]


def parse_ao_matrices_with_spin(
    text: str,
    titles: Sequence[str],
) -> list[tuple[str | None, np.ndarray]]:
    """Return ``(spin, matrix)`` pairs; ``spin`` is ``"alpha"``/``"beta"``/``None``."""

    lines = text.splitlines()
    results: list[tuple[str | None, np.ndarray]] = []
    for start, spin in _title_positions(lines, titles):
        matrix = _parse_blocks(lines, start + 1)
        if matrix is not None:
            results.append((spin, matrix))
    return results


def parse_density_matrix(text: str) -> np.ndarray:
    """Return the density matrix as ``(spin_channels, nao, nao)``.

    A restricted calculation prints one matrix and yields one channel; an
    unrestricted calculation prints alpha and beta and yields two.
    """

    found = parse_ao_matrices_with_spin(text, DENSITY_TITLES)
    if not found:
        raise AOMatrixError(
            "no density matrix found in the CP2K output; enable "
            "&DFT &PRINT &AO_MATRICES DENSITY .TRUE."
        )

    # CP2K reprints the matrix at every SCF step when the print level is high,
    # so the converged density is the last one of each spin channel.
    alpha = [matrix for spin, matrix in found if spin in (None, "alpha")]
    beta = [matrix for spin, matrix in found if spin == "beta"]
    channels = [alpha[-1]] if alpha else []
    if beta:
        channels.append(beta[-1])
    if not channels:  # pragma: no cover - defensive
        raise AOMatrixError("density matrix blocks were located but none could be read")

    _require_square_and_consistent(channels, "density")
    return np.stack(channels, axis=0)


def parse_density_trajectory(text: str) -> list[np.ndarray]:
    """Return every density printed during the run, in SCF order.

    Each element is ``(spin_channels, nao, nao)``, one per printed block --
    which, with ``&AO_MATRICES DENSITY .TRUE.``, means one per SCF iteration
    plus a final post-SCF print. An unrestricted run prints alpha then beta
    within each iteration, and those are paired back into a single frame here.

    The last element is the converged density, i.e. the one
    :func:`parse_density_matrix` returns.
    """

    found = parse_ao_matrices_with_spin(text, DENSITY_TITLES)
    if not found:
        raise AOMatrixError(
            "no density matrix found in the CP2K output; enable "
            "&DFT &PRINT &AO_MATRICES DENSITY .TRUE."
        )

    frames: list[list[np.ndarray]] = []
    for spin, matrix in found:
        # A restricted run labels nothing; an unrestricted one alternates
        # alpha/beta, and a beta block always continues the frame the
        # preceding alpha opened.
        if spin == "beta" and frames and len(frames[-1]) == 1:
            frames[-1].append(matrix)
        else:
            frames.append([matrix])

    channel_counts = {len(frame) for frame in frames}
    if len(channel_counts) != 1:
        raise AOMatrixError(
            f"inconsistent spin channels across the trajectory: saw {sorted(channel_counts)} "
            "matrices per iteration; the output is probably truncated"
        )

    stacked = [np.stack(frame, axis=0) for frame in frames]
    _require_square_and_consistent([channel for frame in stacked for channel in frame], "density")
    return stacked


def parse_overlap_matrix(text: str) -> np.ndarray | None:
    """Return the AO overlap matrix, or ``None`` when it was not printed.

    The overlap is what makes the electron count checkable: the number of
    electrons is ``Tr(P S)``, not ``Tr(P)``, in a non-orthogonal basis.
    """

    found = parse_ao_matrices(text, OVERLAP_TITLES)
    if not found:
        return None
    matrix = found[-1]
    _require_square_and_consistent([matrix], "overlap")
    return matrix


def electron_count(density: np.ndarray, overlap: np.ndarray | None) -> float:
    """Return ``Tr(P S)`` summed over spin channels, or ``Tr(P)`` without ``S``.

    Falling back to ``Tr(P)`` is an approximation and is only meaningful for an
    orthonormal basis, so callers that care record which one they used.
    """

    matrices = np.asarray(density, dtype=float)
    if matrices.ndim == 2:
        matrices = matrices[None, :, :]
    if overlap is None:
        return float(np.trace(matrices, axis1=1, axis2=2).sum())
    weights = np.asarray(overlap, dtype=float)
    if weights.shape != matrices.shape[1:]:
        raise AOMatrixError(
            f"overlap shape {weights.shape} does not match density shape {matrices.shape[1:]}"
        )
    # Tr(P S) == sum(P * S^T); S is symmetric, so the transpose is free.
    return float(sum(np.sum(channel * weights) for channel in matrices))


def _title_positions(lines: Sequence[str], titles: Sequence[str]) -> Iterator[tuple[int, str | None]]:
    upper_titles = [title.upper() for title in titles]
    for index, line in enumerate(lines):
        stripped = line.strip().upper()
        if not any(stripped.startswith(title) for title in upper_titles):
            continue
        spin_match = _SPIN_SUFFIX.search(stripped)
        spin = spin_match.group(1).lower() if spin_match is not None else None
        yield index, spin


def _parse_blocks(lines: Sequence[str], start: int) -> np.ndarray | None:
    """Stitch the column blocks that follow a matrix heading."""

    rows: dict[int, dict[int, float]] = {}
    columns: list[int] = []
    seen_any = False
    blank_run = 0

    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            # Two consecutive blank lines end a matrix; one separates blocks.
            blank_run += 1
            if seen_any and blank_run >= 2:
                break
            continue
        blank_run = 0

        if _INTEGER_ROW.match(line):
            columns = [int(value) for value in stripped.split()]
            continue

        if not columns:
            # Text before the first column header (units, labels) is skipped;
            # a heading for a different matrix ends this one.
            if _looks_like_a_new_section(stripped):
                break
            continue

        values = _NUMBER.findall(line)
        leading = re.match(r"^\s*(\d+)\s", line)
        if leading is None or len(values) < len(columns) + 1:
            if seen_any and _looks_like_a_new_section(stripped):
                break
            continue

        row_index = int(leading.group(1))
        tail = [_to_float(value) for value in values[-len(columns) :]]
        entries = rows.setdefault(row_index, {})
        for column, value in zip(columns, tail, strict=True):
            entries[column] = value
        seen_any = True

    if not rows:
        return None
    return _assemble(rows)


def _assemble(rows: dict[int, dict[int, float]]) -> np.ndarray:
    row_indices = sorted(rows)
    column_indices = sorted({column for entries in rows.values() for column in entries})
    matrix = np.zeros((len(row_indices), len(column_indices)), dtype=float)

    row_position = {index: position for position, index in enumerate(row_indices)}
    column_position = {index: position for position, index in enumerate(column_indices)}

    filled = 0
    for row_index, entries in rows.items():
        for column_index, value in entries.items():
            matrix[row_position[row_index], column_position[column_index]] = value
            filled += 1

    if filled != matrix.size:
        raise AOMatrixError(
            f"printed matrix is incomplete: {filled} of {matrix.size} entries were found "
            f"({len(row_indices)} rows x {len(column_indices)} columns)"
        )
    return matrix


def _require_square_and_consistent(matrices: Sequence[np.ndarray], label: str) -> None:
    for matrix in matrices:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise AOMatrixError(f"{label} matrix is not square: shape {matrix.shape}")
    shapes = {matrix.shape for matrix in matrices}
    if len(shapes) > 1:
        raise AOMatrixError(f"{label} spin channels have different shapes: {sorted(shapes)}")


def _looks_like_a_new_section(stripped: str) -> bool:
    if stripped.startswith("*"):
        return True
    upper = stripped.upper()
    return any(
        marker in upper
        for marker in ("MATRIX", "ENERGY|", "SCF ", "FORCE_EVAL", "ATOMIC FORCES", "MULLIKEN")
    )


def _to_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


__all__ = [
    "DENSITY_TITLES",
    "OVERLAP_TITLES",
    "AOMatrixError",
    "electron_count",
    "parse_ao_matrices",
    "parse_ao_matrices_with_spin",
    "parse_density_matrix",
    "parse_overlap_matrix",
]
