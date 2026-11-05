# Ported from PHAL (hotspot_al.cp2k_ml_scf.density.exchange) at 96263b6; MIT License, see NOTICE.
"""The PHAL -> CP2K exchange format for a predicted initial density.

CP2K reads this file from Fortran, so the format is plain text with a fixed
header and one number per line.  Binary formats (``.npz``, HDF5) would need a
library on the Fortran side; the cost of that dependency is not worth paying
for a file that is written once per calculation.

The format is versioned by :data:`EXCHANGE_PROTOCOL`.  Both sides check it, so
a format change can never be silently misread as the old one::

    PHAL-ML-SCF phal-ml-scf-v2
    FORMAT         phal-atomic-population-v1
    REPRESENTATION ao-nonorthogonal
    STRUCTURE      sha256:...
    MODEL          sha256:...
    ELECTRONS      10.000000000000000
    NSPIN          1
    DIM            3
    BASIS          DZVP-MOLOPT-GTH
    OVERLAP        absent
    VALUES
    3.333333333333333
    ...

A density *matrix* additionally carries ``OVERLAP present`` and an
``OVERLAP_VALUES`` block, because its electron count is ``Tr(P S)`` and CP2K
refuses a matrix whose count it cannot evaluate.  See
``docs/density_representation.md``.

``VALUES`` are written in Fortran (column-major) order so the Fortran reader's
nested loops fill the array correctly without a transpose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gmd_active_learning.adapters.dft.scf.density.prediction import (
    DENSITY_FORMAT_ATOMIC_POPULATION,
    DENSITY_FORMAT_DENSITY_MATRIX,
    PredictionError,
    SCFDensityPrediction,
)

# v2 carries the AO overlap. v1 let CP2K check the electron count as
# Tr(P), which is wrong in a non-orthogonal basis where Ne = Tr(P S);
# see docs/density_representation.md.
EXCHANGE_PROTOCOL = "phal-ml-scf-v2"

#: The basis these matrices live in. AO functions overlap, so Ne = Tr(P S).
REPRESENTATION_AO_NONORTHOGONAL = "ao-nonorthogonal"
EXCHANGE_SUFFIX = ".ml_density"

#: Placeholder for a header field with no value. CP2K's reader cannot tell an
#: empty field from a missing one.
EMPTY_FIELD = "-"

#: Enough digits to round-trip a float64 exactly.
_VALUE_FORMAT = "{:.17g}"


def write_density_exchange(prediction: SCFDensityPrediction, path: str | Path) -> Path:
    """Write ``prediction`` in the format CP2K's ML initial guess reads."""

    if prediction.values is None:
        raise PredictionError("cannot export a prediction that carries no density values")

    values = np.asarray(prediction.values, dtype=float)
    if not np.isfinite(values).all():
        # Writing a NaN would only move the rejection into Fortran, where the
        # message is far less useful.
        raise PredictionError("refusing to export a density with non-finite values")

    if prediction.density_format == DENSITY_FORMAT_ATOMIC_POPULATION:
        if values.ndim != 2:
            raise PredictionError(f"atomic populations must be 2-D, got shape {values.shape}")
        leading_dim = values.shape[1]
        ordered = values.T[:, None, :]  # (dim, 1, nspin)
    elif prediction.density_format == DENSITY_FORMAT_DENSITY_MATRIX:
        if values.ndim != 3 or values.shape[1] != values.shape[2]:
            raise PredictionError(f"density matrices must be square per spin, got shape {values.shape}")
        leading_dim = values.shape[1]
        ordered = np.moveaxis(values, 0, -1)  # (dim, dim, nspin)
    else:  # pragma: no cover - the constructor rejects unknown formats
        raise PredictionError(f"unsupported density_format {prediction.density_format!r}")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # A density matrix without an overlap cannot have its electron count
    # verified by CP2K, which refuses it outright rather than fall back to the
    # Tr(P) approximation. Failing here gives a usable message; failing in
    # Fortran gives a silent fallback in the middle of a run.
    overlap = None
    if prediction.density_format == DENSITY_FORMAT_DENSITY_MATRIX:
        if prediction.overlap is None:
            raise PredictionError(
                "cannot export a density matrix without an overlap: the electron count is "
                "Tr(P S), so CP2K will reject a matrix that carries no S "
                "(see docs/density_representation.md)"
            )
        overlap = np.asarray(prediction.overlap, dtype=float)
        if overlap.shape != (leading_dim, leading_dim):
            raise PredictionError(
                f"overlap shape {overlap.shape} does not match density dimension {leading_dim}"
            )
        if not np.isfinite(overlap).all():
            raise PredictionError("refusing to export a non-finite overlap")

    lines = [
        f"PHAL-ML-SCF {EXCHANGE_PROTOCOL}",
        f"FORMAT {prediction.density_format}",
        f"REPRESENTATION {REPRESENTATION_AO_NONORTHOGONAL}",
        # Never an empty value. The reader is Fortran list-directed input,
        # which does not stop at the end of a record: a bare "STRUCTURE" makes
        # it take the *next* line as the value and every field after it shifts
        # by one, so the file fails to parse with no hint of why.
        f"STRUCTURE {prediction.structure_hash or EMPTY_FIELD}",
        f"MODEL {prediction.model_digest or EMPTY_FIELD}",
        f"ELECTRONS {_VALUE_FORMAT.format(float(prediction.electron_count))}",
        f"NSPIN {int(prediction.spin_channels)}",
        f"DIM {int(leading_dim)}",
        f"BASIS {prediction.metadata.get('basis_set') or 'unknown'}",
        f"OVERLAP {'present' if overlap is not None else 'absent'}",
        "VALUES",
    ]
    # Fortran order so the reader's (i, j, ispin) loop nest matches.
    lines.extend(_VALUE_FORMAT.format(float(value)) for value in ordered.ravel(order="F"))
    if overlap is not None:
        lines.append("OVERLAP_VALUES")
        lines.extend(_VALUE_FORMAT.format(float(value)) for value in overlap.ravel(order="F"))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def read_density_exchange(path: str | Path) -> dict[str, Any]:
    """Read an exchange file back, for tests and for round-trip verification."""

    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise PredictionError(f"{source}: file is empty")

    header: dict[str, str] = {}
    index = 0
    first = lines[0].split()
    if len(first) != 2 or first[0] != "PHAL-ML-SCF":
        raise PredictionError(f"{source}: not an ML-SCF exchange file")
    if first[1] != EXCHANGE_PROTOCOL:
        raise PredictionError(f"{source}: protocol {first[1]!r}, expected {EXCHANGE_PROTOCOL!r}")
    index = 1

    while index < len(lines):
        parts = lines[index].split(maxsplit=1)
        index += 1
        if not parts:
            continue
        if parts[0] == "VALUES":
            break
        if len(parts) != 2:
            raise PredictionError(f"{source}: malformed header line {index}: {lines[index - 1]!r}")
        header[parts[0]] = parts[1].strip()
    else:
        raise PredictionError(f"{source}: header has no VALUES marker")

    for key in (
        "FORMAT",
        "REPRESENTATION",
        "STRUCTURE",
        "MODEL",
        "ELECTRONS",
        "NSPIN",
        "DIM",
        "BASIS",
        "OVERLAP",
    ):
        if key not in header:
            raise PredictionError(f"{source}: header is missing {key}")

    nspin = int(header["NSPIN"])
    leading_dim = int(header["DIM"])
    is_matrix = "density-matrix" in header["FORMAT"]
    columns = leading_dim if is_matrix else 1
    expected = leading_dim * columns * nspin

    body = [line for line in lines[index:] if line.strip()]
    has_overlap = header["OVERLAP"].strip() == "present"
    overlap_lines: list[str] = []
    if has_overlap:
        if "OVERLAP_VALUES" not in body:
            raise PredictionError(f"{source}: OVERLAP is 'present' but OVERLAP_VALUES is missing")
        split_at = body.index("OVERLAP_VALUES")
        overlap_lines = body[split_at + 1 :]
        body = body[:split_at]

    try:
        flat = np.asarray([float(line) for line in body], dtype=float)
    except ValueError as exc:
        raise PredictionError(f"{source}: values could not be parsed ({exc})") from exc
    if flat.size != expected:
        raise PredictionError(f"{source}: expected {expected} values, found {flat.size}")

    overlap = None
    if has_overlap:
        try:
            overlap_flat = np.asarray([float(line) for line in overlap_lines], dtype=float)
        except ValueError as exc:
            raise PredictionError(f"{source}: overlap could not be parsed ({exc})") from exc
        if overlap_flat.size != leading_dim * leading_dim:
            raise PredictionError(
                f"{source}: expected {leading_dim * leading_dim} overlap values, "
                f"found {overlap_flat.size}"
            )
        overlap = overlap_flat.reshape((leading_dim, leading_dim), order="F")

    ordered = flat.reshape((leading_dim, columns, nspin), order="F")
    values = np.moveaxis(ordered, -1, 0)
    if not is_matrix:
        values = values[:, :, 0]

    return {
        "protocol": EXCHANGE_PROTOCOL,
        "density_format": header["FORMAT"],
        "structure_hash": "" if header["STRUCTURE"] == EMPTY_FIELD else header["STRUCTURE"],
        "model_digest": "" if header["MODEL"] == EMPTY_FIELD else header["MODEL"],
        "electron_count": float(header["ELECTRONS"]),
        "spin_channels": nspin,
        "leading_dim": leading_dim,
        "values": values,
        "representation": header["REPRESENTATION"],
        "basis_set": header["BASIS"],
        "overlap": overlap,
    }


def exchange_path_for(input_file: str | Path) -> Path:
    """Where CP2K expects the density for ``input_file``."""

    source = Path(input_file)
    return source.with_suffix(source.suffix + EXCHANGE_SUFFIX)


__all__ = [
    "EXCHANGE_PROTOCOL",
    "EXCHANGE_SUFFIX",
    "exchange_path_for",
    "read_density_exchange",
    "write_density_exchange",
]
