# Ported from PHAL (tests.fake_backends.fake_density) at 96263b6; MIT License, see NOTICE.
"""Fake CP2K AO-matrix output and density artifacts for offline tests.

The printed-matrix helpers reproduce CP2K's blocked column layout, including
the case that actually breaks naive parsers: a matrix wider than one block, so
the blocks must be stitched back together horizontally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

#: CP2K prints this many columns per block before wrapping.
COLUMNS_PER_BLOCK = 4

_SHELL_LABELS = ("1s", "2s", "2px", "2py", "2pz", "3s", "3px", "3py", "3pz", "3d")


def symmetric_density(n_ao: int, *, electrons: float, seed: int = 0) -> np.ndarray:
    """Return a symmetric, positive, correctly normalised single-channel density."""

    rng = np.random.default_rng(seed)
    root = rng.normal(size=(n_ao, n_ao))
    matrix = root @ root.T
    matrix += n_ao * np.eye(n_ao)
    matrix *= electrons / np.trace(matrix)
    return matrix[None, :, :]


def spin_polarized_density(n_ao: int, *, alpha: float, beta: float, seed: int = 0) -> np.ndarray:
    """Return a two-channel density with the requested per-channel populations."""

    first = symmetric_density(n_ao, electrons=alpha, seed=seed)[0]
    second = symmetric_density(n_ao, electrons=beta, seed=seed + 1)[0]
    return np.stack([first, second], axis=0)


def format_ao_matrix(matrix: np.ndarray, title: str, *, symbols: Sequence[str] | None = None) -> str:
    """Render one matrix the way CP2K prints it under ``&AO_MATRICES``."""

    rows, columns = matrix.shape
    labels = list(symbols or ["H"] * rows)
    lines = [f" {title}", ""]

    for start in range(0, columns, COLUMNS_PER_BLOCK):
        block = list(range(start, min(start + COLUMNS_PER_BLOCK, columns)))
        lines.append("        " + "".join(f"{index + 1:>21d}" for index in block))
        lines.append("        " + "".join(f"{_SHELL_LABELS[index % len(_SHELL_LABELS)]:>21}" for index in block))
        for row in range(rows):
            prefix = (
                f"{row + 1:>7d}{row + 1:>5d} {labels[row % len(labels)]:<3}"
                f"{_SHELL_LABELS[row % len(_SHELL_LABELS)]:<6}"
            )
            values = "".join(f"{matrix[row, column]:>21.12f}" for column in block)
            lines.append(prefix + values)
        lines.append("")

    lines.append("")
    return "\n".join(lines)


def fake_cp2k_density_output(
    density: np.ndarray,
    *,
    overlap: np.ndarray | None = None,
    converged: bool = True,
    version: str = "CP2K version TEST-2026.1",
) -> str:
    """Assemble a CP2K output carrying an SCF trajectory and printed matrices."""

    matrices = np.asarray(density, dtype=float)
    if matrices.ndim == 2:
        matrices = matrices[None, :, :]

    parts = [f" CP2K| version string: {version}", " Source code revision git:deadbeef", ""]

    if overlap is not None:
        parts.append(format_ao_matrix(np.asarray(overlap, dtype=float), "OVERLAP MATRIX"))

    if matrices.shape[0] == 1:
        parts.append(format_ao_matrix(matrices[0], "DENSITY MATRIX"))
    else:
        parts.append(format_ao_matrix(matrices[0], "DENSITY MATRIX FOR ALPHA SPIN"))
        parts.append(format_ao_matrix(matrices[1], "DENSITY MATRIX FOR BETA SPIN"))

    parts.extend(
        [
            " SCF ITERATION 1 RESIDUAL 1.0E-2 ENERGY -10.000000",
            " SCF ITERATION 2 RESIDUAL 8.0E-7 ENERGY -10.260000",
            " *** SCF run converged in 2 steps ***" if converged else " *** SCF run NOT converged ***",
            " ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -10.260000",
            "",
        ]
    )
    return "\n".join(parts)


def write_fake_density_artifacts(
    task_dir: str | Path,
    density: np.ndarray,
    *,
    overlap: np.ndarray | None = None,
    **metadata,
) -> Path:
    """Write a ``density/`` directory the dataset layer can ingest."""

    from gmd_active_learning.adapters.dft.scf.recording.density import DENSITY_DIR_NAME, write_density_artifacts

    directory = Path(task_dir) / DENSITY_DIR_NAME
    write_density_artifacts(directory, density, overlap=overlap, metadata=metadata)
    return directory


__all__ = [
    "COLUMNS_PER_BLOCK",
    "fake_cp2k_density_output",
    "format_ao_matrix",
    "spin_polarized_density",
    "symmetric_density",
    "write_fake_density_artifacts",
]
