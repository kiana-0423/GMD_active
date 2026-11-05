# Ported from PHAL (hotspot_al.lammps.dump_parser, hotspot_al.io.lammps_reader) at 96263b6; MIT License, see NOTICE.
"""Import legacy LAMMPS ``dump custom`` trajectories as :class:`~gmd_active_learning.contracts.Frame` objects.

GMD_active does not run LAMMPS (GMD is the MD engine). This importer exists so historical PHAL /
LAMMPS trajectories can be monitored, extracted and turned into datasets with the same code paths
as GMD runs (ledger MD-002, IO-002).

Parsing follows PHAL's ``dump_parser`` exactly:

* frames are ``ITEM: TIMESTEP`` / ``NUMBER OF ATOMS`` / ``BOX BOUNDS`` / ``ATOMS`` blocks; anything
  between frames is skipped, a truncated or malformed frame raises :class:`DataContractError`;
* orthogonal (``lo hi``) and triclinic (``lo_bound hi_bound tilt``) bounds become an ASE-style
  lower-triangular cell plus the box origin ``(xlo, ylo, zlo)``;
* ``pbc`` comes from the two-character boundary tokens (``pp``, ``ff``, ``ss``, ``mm``, ``fs``, ``sm``,
  ...): a dimension is periodic only when its token starts with ``p``. A header without three
  boundary tokens (very old dumps) is read as fully periodic, as PHAL did, with a warning;
* atom rows are sorted by LAMMPS ``id``; the ids become ``Frame.atom_ids`` (stable across frames);
* positions from ``x y z`` or ``xu yu zu`` are shifted by ``-origin`` (so they differ from the raw
  dump values by ``-lo``; the origin is recorded in ``metadata["origin"]``), scaled ``xs ys zs`` are
  multiplied by the cell (no shift). Precedence ``x`` > ``xu`` > ``xs`` as in PHAL;
* species from the ``element`` column, else ``type`` through ``type_map``;
* ``fx fy fz`` -> forces, ``vx vy vz`` -> velocities, ``energy = sum(c_pe)``, ``time_fs = step *
  timestep_fs``; ``type``/``q``/``mol`` columns go to ``metadata["types"|"charges"|"molecule_ids"]``.

Deliberate differences from PHAL:

* frames are streamed from the file instead of reading the whole text into memory;
* unknown chemical symbols (``element`` column or ``type_map`` value) raise a clear error instead of
  failing later inside ASE;
* ``units`` makes the LAMMPS unit style explicit. ``metal`` (the default, PHAL's implicit
  assumption) keeps eV, eV/A and A; velocities are converted from A/ps to A/fs. ``real`` converts
  kcal/mol to eV. Any other style is rejected rather than silently mislabelled;
* the LAMMPS 2024+ general-triclinic header ``BOX BOUNDS abc origin`` is rejected explicitly (PHAL
  crashed with an unpacking error).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, Union

import numpy as np
from ase.data import atomic_numbers, chemical_symbols
from ase.units import kcal, mol

from gmd_active_learning.contracts import Frame
from gmd_active_learning.core.exceptions import DataContractError

LOGGER = logging.getLogger(__name__)

DEFAULT_RUN_ID = "lammps-import"
SOURCE_FORMAT = "lammps-dump-custom"

# LAMMPS boundary styles: periodic, fixed, shrink-wrapped, minimum shrink-wrapped.
_BOUNDARY_STYLE_CHARS = frozenset("pfsm")

LammpsUnits = Literal["metal", "real"]
# (energy factor -> eV, force factor -> eV/A, velocity factor -> A/fs)
_UNIT_FACTORS: dict[str, tuple[float, float, float]] = {
    "metal": (1.0, 1.0, 1.0e-3),
    "real": (kcal / mol, kcal / mol, 1.0),
}


class SupportsReadText(Protocol):
    """Minimal interface shared by ``Path`` and in-memory dump shims."""

    def read_text(self, encoding: str = "utf-8") -> str: ...


TypeMap = Union[Mapping[Any, str], Sequence[str]]
DumpSource = Union[str, Path, SupportsReadText]


def normalize_type_map(type_map: TypeMap | None) -> dict[int, str] | None:
    """``{lammps_type: symbol}`` from a mapping (int or int-like keys) or a sequence (index 0 = type 1).

    Every symbol must be a known chemical element; unknown species raise :class:`DataContractError`.
    """

    if type_map is None:
        return None
    if isinstance(type_map, Mapping):
        items = list(type_map.items())
    elif isinstance(type_map, Sequence) and not isinstance(type_map, str):
        items = [(index + 1, symbol) for index, symbol in enumerate(type_map)]
    else:
        raise DataContractError(f"type_map must be a mapping or a sequence of symbols, got {type(type_map).__name__}")
    normalized: dict[int, str] = {}
    for key, symbol in items:
        try:
            lammps_type = int(key)
        except (TypeError, ValueError) as exc:
            raise DataContractError(f"type_map key {key!r} is not an integer LAMMPS type") from exc
        _check_symbol(str(symbol), context=f"type_map[{lammps_type}]")
        normalized[lammps_type] = str(symbol)
    return normalized


def _check_symbol(symbol: str, *, context: str) -> int:
    number = atomic_numbers.get(symbol)
    if number is None or number <= 0 or number >= len(chemical_symbols):
        raise DataContractError(f"unknown species {symbol!r} in {context}; expected a chemical element symbol")
    return int(number)


# ---------------------------------------------------------------------------------------------
# header sections


def _parse_box_bounds(header: str, lines: Sequence[str]) -> tuple[np.ndarray, np.ndarray, tuple[bool, bool, bool]]:
    if len(lines) < 3:
        raise DataContractError("Malformed LAMMPS dump: truncated BOX BOUNDS section.")
    tokens = header.split()[3:]
    if "abc" in tokens or "origin" in tokens:
        raise DataContractError(
            "LAMMPS general-triclinic 'BOX BOUNDS abc origin' dumps are not supported; "
            "re-dump with the restricted triclinic convention"
        )
    # Each dimension has a two-character lo/hi style drawn from p/f/s/m. Matching on the character
    # set (not an explicit list of pairs) keeps "mm", "sm", "ms" -- free-surface slabs -- from being
    # read as periodic, which is the PHAL misparse that tools/audit/audit_artifacts.py checks for.
    boundary_tokens = [token for token in tokens if len(token) == 2 and set(token) <= _BOUNDARY_STYLE_CHARS]
    if len(boundary_tokens) >= 3:
        pbc = tuple(token.startswith("p") for token in boundary_tokens[:3])
    else:
        LOGGER.warning("BOX BOUNDS header %r carries no boundary styles; assuming fully periodic (as PHAL)", header)
        pbc = (True, True, True)

    rows: list[list[float]] = []
    for line in lines[:3]:
        try:
            rows.append([float(value) for value in line.split()])
        except ValueError as exc:
            raise DataContractError(f"Malformed LAMMPS dump: non-numeric BOX BOUNDS line {line!r}.") from exc
    widths = {len(row) for row in rows}
    if widths == {2}:
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = rows
        cell = np.array([[xhi - xlo, 0.0, 0.0], [0.0, yhi - ylo, 0.0], [0.0, 0.0, zhi - zlo]], dtype=float)
        origin = np.array([xlo, ylo, zlo], dtype=float)
        return cell, origin, pbc  # type: ignore[return-value]
    if widths != {3}:
        raise DataContractError("Malformed LAMMPS dump: BOX BOUNDS lines must all have 2 (orthogonal) or 3 (triclinic) values.")
    (xlo_bound, xhi_bound, xy), (ylo_bound, yhi_bound, xz), (zlo_bound, zhi_bound, yz) = rows
    xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
    xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
    ylo = ylo_bound - min(0.0, yz)
    yhi = yhi_bound - max(0.0, yz)
    zlo, zhi = zlo_bound, zhi_bound
    cell = np.array([[xhi - xlo, 0.0, 0.0], [xy, yhi - ylo, 0.0], [xz, yz, zhi - zlo]], dtype=float)
    origin = np.array([xlo, ylo, zlo], dtype=float)
    return cell, origin, pbc  # type: ignore[return-value]


def _positions(fields: Sequence[str], table: dict[str, np.ndarray], cell: np.ndarray, origin: np.ndarray) -> tuple[np.ndarray, str]:
    names = set(fields)
    if {"x", "y", "z"} <= names:
        return np.column_stack([table["x"], table["y"], table["z"]]) - origin, "x"
    if {"xu", "yu", "zu"} <= names:
        return np.column_stack([table["xu"], table["yu"], table["zu"]]) - origin, "xu"
    if {"xs", "ys", "zs"} <= names:
        return np.column_stack([table["xs"], table["ys"], table["zs"]]) @ cell, "xs"
    raise DataContractError("LAMMPS dump must contain x/y/z, xu/yu/zu, or xs/ys/zs coordinates.")


def _numbers(table: dict[str, np.ndarray], type_map: dict[int, str] | None) -> np.ndarray:
    if "element" in table:
        return np.asarray([_check_symbol(str(value), context="the 'element' column") for value in table["element"]], dtype=np.int64)
    if "type" not in table:
        raise DataContractError("LAMMPS dump must contain either 'element' or 'type'.")
    if type_map is None:
        raise DataContractError("A type_map is required when the dump does not contain 'element'.")
    try:
        symbols = [type_map[int(value)] for value in table["type"]]
    except KeyError as exc:
        raise DataContractError(f"LAMMPS type {exc.args[0]} is missing from type_map.") from exc
    return np.asarray([atomic_numbers[symbol] for symbol in symbols], dtype=np.int64)


def _as_int_column(name: str, values: np.ndarray) -> np.ndarray:
    rounded = np.rint(values)
    if not np.all(np.isfinite(values)) or not np.array_equal(rounded, values):
        raise DataContractError(f"Malformed LAMMPS dump: column {name!r} must hold integers.")
    return rounded.astype(np.int64)


def _build_frame(
    *,
    step: int,
    cell: np.ndarray,
    origin: np.ndarray,
    pbc: tuple[bool, bool, bool],
    fields: list[str],
    atom_lines: Sequence[str],
    type_map: dict[int, str] | None,
    timestep_fs: float | None,
    run_id: str,
    units: str,
    source: str | None,
) -> Frame:
    if len(set(fields)) != len(fields):
        raise DataContractError(f"Malformed LAMMPS dump: duplicate column names in ATOMS header {fields}.")
    raw_rows = [line.split() for line in atom_lines]
    for row in raw_rows:
        if len(row) < len(fields):
            raise DataContractError("Malformed LAMMPS dump: atom row has fewer columns than ATOMS header.")
    table: dict[str, np.ndarray] = {}
    for index, name in enumerate(fields):
        column = [row[index] for row in raw_rows]
        if name == "element":
            table[name] = np.asarray(column, dtype=object)
            continue
        try:
            table[name] = np.asarray(column, dtype=float)
        except ValueError as exc:
            raise DataContractError(f"Malformed LAMMPS dump: non-numeric values in column {name!r}.") from exc

    n_atoms = len(raw_rows)
    if "id" in table:
        ids = _as_int_column("id", table["id"])
        order = np.argsort(ids, kind="stable")
        for name, values in list(table.items()):
            table[name] = values[order]
        atom_ids = ids[order]
        if len(np.unique(atom_ids)) != n_atoms:
            raise DataContractError(f"LAMMPS dump step {step}: duplicate atom ids.")
    else:
        LOGGER.warning("LAMMPS dump step %d has no 'id' column; atom ids default to row order 1..N", step)
        atom_ids = np.arange(1, n_atoms + 1, dtype=np.int64)

    energy_factor, force_factor, velocity_factor = _UNIT_FACTORS[units]
    positions, coordinate_style = _positions(fields, table, cell, origin)
    numbers = _numbers(table, type_map)
    names = set(fields)
    forces = np.column_stack([table["fx"], table["fy"], table["fz"]]) * force_factor if {"fx", "fy", "fz"} <= names else None
    velocities = (
        np.column_stack([table["vx"], table["vy"], table["vz"]]) * velocity_factor if {"vx", "vy", "vz"} <= names else None
    )
    energy = float(np.sum(table["c_pe"])) * energy_factor if "c_pe" in table else None

    metadata: dict[str, Any] = {
        "source_format": SOURCE_FORMAT,
        "fields": list(fields),
        "origin": origin.tolist(),
        "coordinate_style": coordinate_style,
        "lammps_units": units,
    }
    if source is not None:
        metadata["source_path"] = source
    if "type" in table:
        metadata["types"] = _as_int_column("type", table["type"]).tolist()
    if "q" in table:
        metadata["charges"] = table["q"].astype(float).tolist()
    if "mol" in table:
        metadata["molecule_ids"] = _as_int_column("mol", table["mol"]).tolist()
    try:
        return Frame(
            run_id=run_id,
            step=step,
            atom_ids=atom_ids,
            numbers=numbers,
            positions=positions,
            cell=cell,
            pbc=pbc,
            forces=forces,
            time_fs=None if timestep_fs is None else step * float(timestep_fs),
            energy=energy,
            velocities=velocities,
            metadata=metadata,
        )
    except DataContractError as exc:
        where = f" in {source}" if source else ""
        raise DataContractError(f"LAMMPS dump step {step}{where}: {exc}") from exc


# ---------------------------------------------------------------------------------------------
# frame scanner


class _Lines:
    """Line cursor over an iterable of lines (no trailing newlines) that supports bounded look-ahead."""

    def __init__(self, lines: Iterable[str]) -> None:
        self._iterator = iter(lines)
        self.line_number = 0

    def next(self) -> str | None:
        try:
            line = next(self._iterator)
        except StopIteration:
            return None
        self.line_number += 1
        return line.rstrip("\r\n")

    def take(self, count: int) -> list[str]:
        out: list[str] = []
        for _ in range(count):
            line = self.next()
            if line is None:
                break
            out.append(line)
        return out


def _iter_frames_from_lines(
    lines: Iterable[str],
    *,
    type_map: dict[int, str] | None,
    timestep_fs: float | None,
    run_id: str,
    units: str,
    source: str | None,
) -> Iterator[Frame]:
    cursor = _Lines(lines)
    while True:
        line = cursor.next()
        if line is None:
            return
        if not line.startswith("ITEM: TIMESTEP"):
            continue
        header = cursor.take(8)
        if len(header) < 8:
            raise DataContractError("Malformed LAMMPS dump: truncated frame header.")
        try:
            step = int(header[0].strip())
        except ValueError as exc:
            raise DataContractError("Malformed LAMMPS dump: invalid timestep.") from exc
        if not header[1].startswith("ITEM: NUMBER OF ATOMS"):
            raise DataContractError("Malformed LAMMPS dump: missing NUMBER OF ATOMS section.")
        try:
            n_atoms = int(header[2].strip())
        except ValueError as exc:
            raise DataContractError("Malformed LAMMPS dump: invalid atom count.") from exc
        if n_atoms < 0:
            raise DataContractError("Malformed LAMMPS dump: invalid atom count.")
        if not header[3].startswith("ITEM: BOX BOUNDS"):
            raise DataContractError("Malformed LAMMPS dump: missing BOX BOUNDS section.")
        cell, origin, pbc = _parse_box_bounds(header[3], header[4:7])
        if not header[7].startswith("ITEM: ATOMS"):
            raise DataContractError("Malformed LAMMPS dump: missing ATOMS section.")
        fields = header[7].split()[2:]
        atom_lines = cursor.take(n_atoms)
        if len(atom_lines) != n_atoms or any(row.startswith("ITEM:") for row in atom_lines):
            raise DataContractError("Malformed LAMMPS dump: truncated atom section.")
        yield _build_frame(
            step=step,
            cell=cell,
            origin=origin,
            pbc=pbc,
            fields=fields,
            atom_lines=atom_lines,
            type_map=type_map,
            timestep_fs=timestep_fs,
            run_id=run_id,
            units=units,
            source=source,
        )


def _check_units(units: str) -> str:
    if units not in _UNIT_FACTORS:
        raise DataContractError(f"unsupported LAMMPS units {units!r}; supported: {sorted(_UNIT_FACTORS)}")
    return units


# ---------------------------------------------------------------------------------------------
# public API


def iter_lammps_dump(
    path: DumpSource,
    *,
    type_map: TypeMap | None = None,
    timestep_fs: float | None = None,
    run_id: str = DEFAULT_RUN_ID,
    units: LammpsUnits = "metal",
) -> Iterator[Frame]:
    """Stream the frames of a LAMMPS custom dump file as :class:`Frame` objects.

    ``path`` is a filesystem path or any object with ``read_text`` (PHAL's in-memory shims). Argument
    errors (unknown species in ``type_map``, unsupported units, missing file) raise immediately;
    malformed frames raise when the iterator reaches them.
    """

    mapping = normalize_type_map(type_map)
    _check_units(units)
    if hasattr(path, "read_text") and not isinstance(path, Path):
        text = path.read_text(encoding="utf-8")  # type: ignore[union-attr]
        return _iter_frames_from_lines(
            text.splitlines(), type_map=mapping, timestep_fs=timestep_fs, run_id=run_id, units=units, source=None
        )
    file_path = Path(path)  # type: ignore[arg-type]
    if not file_path.is_file():
        raise DataContractError(f"LAMMPS dump {file_path} does not exist")
    return _iter_file(file_path, type_map=mapping, timestep_fs=timestep_fs, run_id=run_id, units=units)


def _iter_file(
    file_path: Path, *, type_map: dict[int, str] | None, timestep_fs: float | None, run_id: str, units: str
) -> Iterator[Frame]:
    count = 0
    with file_path.open("r", encoding="utf-8") as handle:
        for frame in _iter_frames_from_lines(
            handle, type_map=type_map, timestep_fs=timestep_fs, run_id=run_id, units=units, source=str(file_path)
        ):
            count += 1
            yield frame
    LOGGER.debug("imported %d frame(s) from %s", count, file_path)


def read_lammps_dump(
    path: DumpSource,
    *,
    type_map: TypeMap | None = None,
    timestep_fs: float | None = None,
    run_id: str = DEFAULT_RUN_ID,
    units: LammpsUnits = "metal",
) -> list[Frame]:
    """Read every frame of a LAMMPS custom dump file."""

    frames = list(iter_lammps_dump(path, type_map=type_map, timestep_fs=timestep_fs, run_id=run_id, units=units))
    LOGGER.info("imported %d LAMMPS frame(s) from %s", len(frames), path)
    return frames


def parse_lammps_dump_frame(
    source: str | Sequence[str],
    *,
    type_map: TypeMap | None = None,
    timestep_fs: float | None = None,
    run_id: str = DEFAULT_RUN_ID,
    units: LammpsUnits = "metal",
    source_path: str | None = None,
) -> Frame:
    """Parse exactly one frame from dump text (a string) or its lines.

    Raises :class:`DataContractError` when the input holds no frame or more than one.
    """

    lines = source.splitlines() if isinstance(source, str) else list(source)
    frames = list(
        _iter_frames_from_lines(
            lines,
            type_map=normalize_type_map(type_map),
            timestep_fs=timestep_fs,
            run_id=run_id,
            units=_check_units(units),
            source=source_path,
        )
    )
    if len(frames) != 1:
        raise DataContractError(f"expected exactly one LAMMPS dump frame, found {len(frames)}")
    return frames[0]


__all__ = [
    "DEFAULT_RUN_ID",
    "SOURCE_FORMAT",
    "SupportsReadText",
    "iter_lammps_dump",
    "normalize_type_map",
    "parse_lammps_dump_frame",
    "read_lammps_dump",
]
