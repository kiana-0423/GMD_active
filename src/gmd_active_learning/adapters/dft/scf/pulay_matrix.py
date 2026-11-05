# Ported from PHAL (hotspot_al.cp2k.pulay_matrix) at 96263b6; MIT License, see NOTICE.
"""Read the Pulay matrices CP2K recorded during an SCF run.

The ML mixer is fed ``mixing_store%pulay_matrix`` at run time, so that is what
a model has to be trained on. It cannot be reconstructed from recorded
densities: CP2K builds it from g-space residuals, weights them by the Kerker
factor and again by ``special_metric``, sums across ranks, and scales by
``vol^2``. A reconstruction in the AO basis is a different matrix in a
different space, and a model trained on one and fed the other degrades
convergence -- measured at 11 -> 15 -> 37 SCF steps as more of its proposals
were accepted.

``ml_pulay_recorder.F`` writes one plain-text dump per (iteration, spin);
this module parses them and :mod:`gmd_active_learning.adapters.dft.scf.recording.trajectory` folds them
into ``trajectory/iter_xxxx/pulay_matrix.npy``.

Every dump carries the coefficients CP2K went on to use, which is what makes
the export *checkable*: for the recorded ``B``, the closed form
``c = B^-1 1 / (1^T B^-1 1)`` must reproduce them. :func:`verify_record` is
that check, and it is what distinguishes "this is CP2K's matrix" from "this is
a matrix".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

RECORD_FORMAT = "phal-pulay-matrix-v1"

#: The representation tag every recorded matrix carries. A consumer that finds
#: anything else is looking at a different quantity and must not mix them.
REPRESENTATION_CP2K_PULAY = "cp2k_pulay_metric"
METRIC_GSPACE_KERKER = "gspace_kerker"

#: An AO-space reconstruction, kept as a named value purely so the mismatch can
#: be *detected and reported* rather than silently averaged into a dataset.
REPRESENTATION_AO_RECONSTRUCTED = "ao_frobenius_reconstructed"

RECORD_GLOB = "pulay_*.dat"
_NAME = re.compile(r"pulay_(\d+)_(\d+)\.dat$")


class PulayRecordError(RuntimeError):
    """Raised when a recorded Pulay matrix is missing or unusable."""


@dataclass(frozen=True, slots=True)
class PulayRecord:
    """One recorded ``(B, alpha)`` pair from a single SCF iteration."""

    iteration: int
    spin: int
    b_matrix: np.ndarray
    alpha: np.ndarray
    representation: str = REPRESENTATION_CP2K_PULAY
    metric: str = METRIC_GSPACE_KERKER

    @property
    def depth(self) -> int:
        return int(self.b_matrix.shape[0])


def read_record(path: str | Path) -> PulayRecord:
    """Parse one dump written by ``ml_pulay_recorder.F``."""

    source = Path(path)
    try:
        lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines()]
    except OSError as exc:
        raise PulayRecordError(f"{source}: cannot be read ({exc})") from exc

    lines = [line for line in lines if line]
    if not lines or not lines[0].startswith("PHAL-PULAY-MATRIX "):
        raise PulayRecordError(f"{source}: not a Pulay matrix dump")
    version = lines[0].split(maxsplit=1)[1]
    if version != RECORD_FORMAT:
        raise PulayRecordError(f"{source}: format {version!r}, expected {RECORD_FORMAT!r}")

    header: dict[str, str] = {}
    index = 1
    while index < len(lines) and lines[index] != "B_VALUES":
        parts = lines[index].split(maxsplit=1)
        if len(parts) == 2:
            header[parts[0]] = parts[1]
        index += 1
    if index >= len(lines):
        raise PulayRecordError(f"{source}: no B_VALUES section")

    for key in ("REPRESENTATION", "METRIC", "ITERATION", "SPIN", "NB"):
        if key not in header:
            raise PulayRecordError(f"{source}: header is missing {key}")

    try:
        depth = int(header["NB"])
        iteration = int(header["ITERATION"])
        spin = int(header["SPIN"])
    except ValueError as exc:
        raise PulayRecordError(f"{source}: malformed header ({exc})") from exc
    if depth < 1:
        raise PulayRecordError(f"{source}: NB is {depth}")

    body = lines[index + 1 :]
    try:
        split_at = body.index("ALPHA_VALUES")
    except ValueError as exc:
        raise PulayRecordError(f"{source}: no ALPHA_VALUES section") from exc

    try:
        b_flat = np.asarray([float(value) for value in body[:split_at]], dtype=float)
        alpha = np.asarray([float(value) for value in body[split_at + 1 :]], dtype=float)
    except ValueError as exc:
        raise PulayRecordError(f"{source}: values could not be parsed ({exc})") from exc

    if b_flat.size != depth * depth:
        raise PulayRecordError(f"{source}: expected {depth * depth} B values, found {b_flat.size}")
    if alpha.size != depth:
        raise PulayRecordError(f"{source}: expected {depth} coefficients, found {alpha.size}")

    return PulayRecord(
        iteration=iteration,
        spin=spin,
        # Written column-major to match Fortran storage.
        b_matrix=b_flat.reshape((depth, depth), order="F"),
        alpha=alpha,
        representation=header["REPRESENTATION"],
        metric=header["METRIC"],
    )


def read_records(directory: str | Path, *, spin: int = 1) -> list[PulayRecord]:
    """Read every dump for one spin channel, ordered by SCF iteration."""

    source = Path(directory)
    if not source.is_dir():
        raise PulayRecordError(f"{source}: not a directory")
    records = [read_record(path) for path in sorted(source.glob(RECORD_GLOB))]
    selected = [record for record in records if record.spin == spin]
    return sorted(selected, key=lambda record: record.iteration)


def verify_record(record: PulayRecord, *, tolerance: float = 1.0e-6) -> tuple[bool, float]:
    """Check the exported ``B`` against the coefficients CP2K derived from it.

    Returns ``(ok, max_absolute_deviation)``. The closed form
    ``c = B^-1 1 / (1^T B^-1 1)`` is exactly CP2K's solve, so if the recorded
    matrix is the one CP2K used, it must reproduce the recorded coefficients.

    This is the check that would have caught the AO-reconstruction mistake: a
    reconstructed ``B`` reproduces nothing, because it never produced anything.
    """

    matrix = np.asarray(record.b_matrix, dtype=float)
    if not np.isfinite(matrix).all():
        return False, float("inf")

    ones = np.ones(record.depth, dtype=float)
    inverse = np.linalg.pinv(matrix)
    weights = inverse @ ones
    total = float(weights.sum())
    if not np.isfinite(total) or abs(total) < 1e-30:
        return False, float("inf")

    deviation = float(np.abs(weights / total - record.alpha).max())
    return deviation <= tolerance, deviation


def verify_records(
    records: Sequence[PulayRecord], *, tolerance: float = 1.0e-6
) -> dict[str, object]:
    """Verify a whole run and summarise it for a report."""

    deviations: list[float] = []
    failures: list[int] = []
    for record in records:
        ok, deviation = verify_record(record, tolerance=tolerance)
        deviations.append(deviation)
        if not ok:
            failures.append(record.iteration)

    finite = [value for value in deviations if np.isfinite(value)]
    return {
        "n_records": len(records),
        "n_verified": len(records) - len(failures),
        "failed_iterations": failures,
        "max_deviation": max(finite) if finite else float("inf"),
        "tolerance": tolerance,
        "representation": records[0].representation if records else None,
        "metric": records[0].metric if records else None,
    }


#: Above this condition number the DIIS solve is worth regularising: at 1e8 a
#: float64 solve has already surrendered roughly half its significant digits,
#: and the coefficients are what a model has to reproduce.
DEFAULT_CONDITION_LIMIT = 1.0e8


def b_matrix_diagnostics(
    matrix: np.ndarray, *, condition_limit: float = DEFAULT_CONDITION_LIMIT
) -> dict[str, Any]:
    """Describe how hard one ``B`` is to solve.

    ``B`` is a Gram matrix of SCF residuals, so in exact arithmetic it is
    symmetric and positive semi-definite. It is also near-singular by
    construction late in a run: successive residuals become nearly collinear as
    the SCF converges, which is precisely when a learned model's coefficients
    are most sensitive to its input. Recording the spectrum per iteration means
    that sensitivity is a measured property of the dataset rather than a
    surprise during training.

    ``regularization_shift`` is the diagonal shift ``s`` that would bring
    ``cond(B + sI)`` down to ``condition_limit`` -- reported, not applied.
    Nothing here modifies the matrix CP2K solved.
    """

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise PulayRecordError(f"B must be square, got shape {array.shape}")
    if not np.isfinite(array).all():
        return {"depth": int(array.shape[0]), "finite": False}

    depth = int(array.shape[0])
    # Symmetrise before the spectrum: eigvalsh reads one triangle, so feeding
    # it an asymmetric matrix silently discards the other half. The asymmetry
    # is reported instead, since a large one means the export is suspect.
    symmetric = 0.5 * (array + array.T)
    asymmetry = float(np.abs(array - array.T).max())
    eigenvalues = np.linalg.eigvalsh(symmetric)

    magnitudes = np.abs(eigenvalues)
    largest = float(magnitudes.max())
    smallest = float(magnitudes.min())
    # numpy's own matrix_rank tolerance, stated explicitly so a reader can see
    # what "rank" means here rather than inferring it.
    tolerance = largest * depth * float(np.finfo(float).eps)
    rank = int((magnitudes > tolerance).sum())

    condition = float(largest / smallest) if smallest > 0.0 else float("inf")

    # cond(B + sI) <= limit  <=>  s >= (lmax - limit*lmin)/(limit - 1),
    # using the signed extremes so an indefinite B (numerical noise on a matrix
    # that should be PSD) still yields a shift that makes it positive.
    lo, hi = float(eigenvalues.min()), float(eigenvalues.max())
    shift = 0.0
    if condition_limit > 1.0:
        shift = max(0.0, (hi - condition_limit * lo) / (condition_limit - 1.0))

    return {
        "depth": depth,
        "finite": True,
        "eigenvalues": eigenvalues.tolist(),
        "eigenvalue_min": lo,
        "eigenvalue_max": hi,
        "rank": rank,
        "rank_deficient": rank < depth,
        "rank_tolerance": float(tolerance),
        "condition_number": condition,
        "asymmetry": asymmetry,
        "positive_definite": bool(lo > 0.0),
        "condition_limit": float(condition_limit),
        "regularization_required": bool(
            not np.isfinite(condition) or condition > condition_limit
        ),
        "regularization_shift": float(shift),
        # Relative to the largest eigenvalue, so it is comparable across systems
        # whose B differ by decades in absolute scale.
        "regularization_relative": float(shift / hi) if hi > 0.0 else float("inf"),
        "coefficient_l1": _coefficient_l1(symmetric),
    }


def _coefficient_l1(matrix: np.ndarray) -> float:
    """``||c||_1`` for the DIIS solve, as an amplification measure.

    The constraint is ``sum(c) = 1``, so an L1 norm well above 1 means the
    coefficients are large and opposite-signed: the extrapolation relies on
    cancellation, and small errors in ``c`` do not stay small.
    """

    ones = np.ones(matrix.shape[0], dtype=float)
    weights = np.linalg.pinv(matrix) @ ones
    total = float(weights.sum())
    if not np.isfinite(total) or abs(total) < 1e-30:
        return float("inf")
    return float(np.abs(weights / total).sum())


def require_cp2k_representation(records: Sequence[PulayRecord]) -> None:
    """Refuse a mixed or reconstructed set.

    Training on a mixture of CP2K's metric and an AO reconstruction is the
    failure this whole module exists to prevent, and it is invisible once the
    matrices are stacked into an array.
    """

    representations = {record.representation for record in records}
    metrics = {record.metric for record in records}
    if len(representations) > 1 or len(metrics) > 1:
        raise PulayRecordError(
            f"refusing a mixed dataset: representations {sorted(representations)}, "
            f"metrics {sorted(metrics)}"
        )
    if representations and representations != {REPRESENTATION_CP2K_PULAY}:
        raise PulayRecordError(
            f"representation is {sorted(representations)[0]!r}, expected "
            f"{REPRESENTATION_CP2K_PULAY!r}; an AO-space reconstruction is not the "
            "matrix CP2K solves and a model trained on it will not transfer"
        )
    if metrics and metrics != {METRIC_GSPACE_KERKER}:
        raise PulayRecordError(
            f"metric is {sorted(metrics)[0]!r}, expected {METRIC_GSPACE_KERKER!r}"
        )
