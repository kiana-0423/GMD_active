# Ported from PHAL (hotspot_al.cp2k_ml_scf.structures) at 96263b6; MIT License, see NOTICE.
"""Turn a structure collection into CP2K tasks for Pulay dataset generation.

The first Pulay dataset was nine geometries of two molecules, and a model
trained on it had no reason to generalise: every sample shared one basis, one
element pair and one ``n_ao``. This module is how the collection grows without
hand-writing an input file per structure -- read ``.xyz`` / ``.extxyz`` /
``.cif`` through ASE, render one CP2K input each from a shared template, and
hand the result to :mod:`tools.scf.pulay_dataset`.

Two things are deliberately *not* guessed:

* **Basis and pseudopotential** come from CP2K's own ``BASIS_MOLOPT`` and
  ``POTENTIAL`` data files. The MOLOPT basis alias carries the valence it was
  optimised for (``DZVP-MOLOPT-SR-GTH-q12``), so pairing it with
  ``GTH-PBE-q12`` is a lookup, not a recollection. An element with no usable
  entry raises rather than silently producing an input CP2K will reject after
  it has been queued.
* **Periodicity** comes from the file. A ``.cif`` is periodic; an ``.xyz``
  without a cell is not, and gets a vacuum box. Quietly treating a molecule as
  periodic changes the physics, so the choice is recorded in the sample
  metadata alongside the geometry.

The descriptors written into each task's ``structure`` field are what
``dataset_report.json`` later summarises as geometry diversity -- they exist so
"is this dataset chemically diverse?" is answerable from the report instead of
from a directory listing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from tools.scf.pulay_dataset import PulayTask

#: Placeholders a template may use. A template that uses anything else is
#: rejected: a typo'd placeholder would otherwise reach CP2K verbatim.
PLACEHOLDERS = ("PROJECT", "COORD", "CELL", "KIND", "LABEL", "GUESS")
REQUIRED_PLACEHOLDERS = ("COORD", "KIND")

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Z_]+)\s*\}\}")

#: Basis alias -> valence, e.g. DZVP-MOLOPT-SR-GTH-q12 -> 12.
_BASIS_ALIAS = re.compile(r"^(?P<name>[A-Z0-9-]*MOLOPT[A-Z0-9-]*GTH)-q(?P<q>\d+)$")

#: Preference order for the basis family. SR ("short range") exists for the
#: transition metals; the plain MOLOPT set does not go past the main group.
DEFAULT_BASIS_PREFERENCE = ("DZVP-MOLOPT-SR-GTH", "DZVP-MOLOPT-GTH", "SZV-MOLOPT-SR-GTH")

DEFAULT_POTENTIAL_FAMILY = "GTH-PBE"

#: Padding around a non-periodic molecule, in Angstrom, per side.
DEFAULT_VACUUM = 6.0


class StructureError(RuntimeError):
    """Raised when a structure collection cannot be turned into CP2K tasks."""


@dataclass(frozen=True, slots=True)
class KindSpec:
    """The ``&KIND`` settings for one element."""

    element: str
    basis_set: str
    potential: str

    def to_block(self, indent: str = "    ") -> str:
        return (
            f"{indent}&KIND {self.element}\n"
            f"{indent}  BASIS_SET {self.basis_set}\n"
            f"{indent}  POTENTIAL {self.potential}\n"
            f"{indent}&END KIND"
        )


@dataclass(frozen=True, slots=True)
class Structure:
    """One structure from a collection, with where it came from."""

    label: str
    symbols: tuple[str, ...]
    positions: np.ndarray
    cell: np.ndarray | None
    periodic: bool
    source: Path
    index: int
    info: Mapping[str, Any]

    @property
    def elements(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.symbols)))

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)


# -- reading ------------------------------------------------------------


def read_structures(path: str | Path, *, label: str | None = None) -> list[Structure]:
    """Read every frame in one file.

    ASE decides the format from the suffix, which is why ``.extxyz`` files can
    carry a lattice and ``.xyz`` files generally cannot.
    """

    source = Path(path)
    if not source.is_file():
        raise StructureError(f"{source}: no such structure file")

    try:
        from ase.io import read as ase_read
    except ImportError as exc:  # pragma: no cover - the container pins ase
        raise StructureError(
            "ase is required to read structure collections; it is pinned in "
            "containers/versions.yaml and installed in the cp2k-ml image"
        ) from exc

    try:
        frames = ase_read(str(source), index=":")
    except Exception as exc:  # noqa: BLE001 - ase raises many unrelated types
        raise StructureError(f"{source}: cannot be read as a structure file ({exc})") from exc
    if not isinstance(frames, list):
        frames = [frames]
    if not frames:
        raise StructureError(f"{source}: contains no structures")

    structures: list[Structure] = []
    for index, atoms in enumerate(frames):
        info = dict(getattr(atoms, "info", {}) or {})
        # A name in the file wins over a positional one: an extxyz comment line
        # of `label=rutile-tio2` is the author saying what this frame is.
        name = str(info.get("label") or info.get("name") or "")
        if not name:
            stem = label or source.stem
            name = stem if len(frames) == 1 else f"{stem}-{index:03d}"

        cell = np.asarray(atoms.get_cell(), dtype=float)
        periodic = bool(np.asarray(atoms.get_pbc()).any()) and float(np.abs(cell).max()) > 0.0
        structures.append(
            Structure(
                label=_sanitise(name),
                symbols=tuple(atoms.get_chemical_symbols()),
                positions=np.asarray(atoms.get_positions(), dtype=float),
                cell=cell if float(np.abs(cell).max()) > 0.0 else None,
                periodic=periodic,
                source=source,
                index=index,
                info={key: value for key, value in info.items() if _jsonable(value)},
            )
        )
    return structures


def read_collection(paths: Iterable[str | Path]) -> list[Structure]:
    """Read several files, refusing duplicate labels.

    Duplicates are refused rather than renamed: two samples with the same id
    would overwrite each other's directory, and the loss would show up as a
    short dataset rather than as an error.
    """

    structures: list[Structure] = []
    seen: dict[str, Path] = {}
    for path in paths:
        for structure in read_structures(path):
            if structure.label in seen:
                raise StructureError(
                    f"duplicate structure label {structure.label!r} in {structure.source} "
                    f"and {seen[structure.label]}"
                )
            seen[structure.label] = structure.source
            structures.append(structure)
    if not structures:
        raise StructureError("the structure collection is empty")
    return structures


def _sanitise(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", name.strip()).strip("-")
    return cleaned or "structure"


def _jsonable(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


# -- descriptors --------------------------------------------------------


def descriptors(structure: Structure) -> dict[str, Any]:
    """Summarise one structure for the dataset report.

    ``min_distance`` uses the minimum image convention for periodic cells, so a
    two-atom cell does not look artificially sparse. It is the descriptor worth
    watching: a structure whose atoms are far closer than equilibrium converges
    differently, and a dataset of only equilibrium geometries teaches a mixer
    nothing about the hard cases.
    """

    positions = structure.positions
    counts: dict[str, int] = {}
    for symbol in structure.symbols:
        counts[symbol] = counts.get(symbol, 0) + 1

    volume = None
    if structure.cell is not None:
        volume = abs(float(np.linalg.det(structure.cell)))

    return {
        "label": structure.label,
        "formula": "".join(f"{element}{counts[element]}" for element in sorted(counts)),
        "elements": list(structure.elements),
        "element_counts": dict(sorted(counts.items())),
        "n_atoms": structure.n_atoms,
        "periodic": structure.periodic,
        "cell_volume_A3": volume,
        "number_density_A-3": (structure.n_atoms / volume) if volume else None,
        "min_distance_A": _min_distance(positions, structure.cell if structure.periodic else None),
        "extent_A": (
            float(np.abs(positions - positions.mean(axis=0)).max()) if len(positions) else 0.0
        ),
        "source": str(structure.source),
        "source_index": structure.index,
        **{key: value for key, value in structure.info.items() if key not in {"label", "name"}},
    }


def _min_distance(positions: np.ndarray, cell: np.ndarray | None) -> float | None:
    if len(positions) < 2:
        return None
    deltas = positions[:, None, :] - positions[None, :, :]
    if cell is not None:
        # Minimum image over the 27 neighbouring cells. Enough for the compact
        # cells here, and it does not pretend to handle extreme skew.
        best = np.full(deltas.shape[:2], np.inf)
        for shifts in _cell_shifts(cell):
            best = np.minimum(best, np.linalg.norm(deltas + shifts, axis=-1))
        distances = best
    else:
        distances = np.linalg.norm(deltas, axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def _cell_shifts(cell: np.ndarray) -> Iterable[np.ndarray]:
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                yield np.asarray([i, j, k], dtype=float) @ cell


# -- kinds, from CP2K's own data files ----------------------------------


def resolve_kinds(
    elements: Sequence[str],
    *,
    data_dir: str | Path,
    basis_preference: Sequence[str] = DEFAULT_BASIS_PREFERENCE,
    potential_family: str = DEFAULT_POTENTIAL_FAMILY,
    overrides: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, KindSpec]:
    """Pick a basis and pseudopotential per element from CP2K's data files.

    The pairing is read, not remembered: the MOLOPT alias states the valence it
    was optimised for and the potential file states which valences exist, so a
    mismatched pair -- the kind of thing that fails an hour into a queued job --
    cannot be produced here.
    """

    directory = Path(data_dir)
    basis_file = directory / "BASIS_MOLOPT"
    potential_file = directory / "POTENTIAL"
    for path in (basis_file, potential_file):
        if not path.is_file():
            raise StructureError(
                f"{path}: not found; point --cp2k-data at the CP2K data directory "
                "(the fork's cp2k/data)"
            )

    available_basis = _basis_table(basis_file)
    available_potentials = _potential_table(potential_file)
    override_map = {key: dict(value) for key, value in (overrides or {}).items()}

    kinds: dict[str, KindSpec] = {}
    for element in sorted(set(elements)):
        if element in override_map:
            entry = override_map[element]
            try:
                kinds[element] = KindSpec(element, entry["basis_set"], entry["potential"])
            except KeyError as exc:
                raise StructureError(
                    f"override for {element} needs both basis_set and potential"
                ) from exc
            continue

        entries = available_basis.get(element, {})
        chosen: KindSpec | None = None
        for name in basis_preference:
            valence = entries.get(name)
            if valence is None:
                continue
            potential = f"{potential_family}-q{valence}"
            if potential not in available_potentials.get(element, set()):
                continue
            chosen = KindSpec(element, name, potential)
            break
        if chosen is None:
            raise StructureError(
                f"{element}: no {'/'.join(basis_preference)} basis with a matching "
                f"{potential_family} potential in {directory}; pass an explicit kind override"
            )
        kinds[element] = chosen
    return kinds


def _basis_table(path: Path) -> dict[str, dict[str, int]]:
    """``{element: {basis_name: valence}}`` from ``BASIS_MOLOPT``."""

    table: dict[str, dict[str, int]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        element = fields[0]
        if not re.fullmatch(r"[A-Z][a-z]?", element) or len(fields) < 2:
            continue
        for alias in fields[1:]:
            match = _BASIS_ALIAS.fullmatch(alias)
            if match:
                table.setdefault(element, {})[match.group("name")] = int(match.group("q"))
    return table


def _potential_table(path: Path) -> dict[str, set[str]]:
    table: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        element = fields[0]
        if not re.fullmatch(r"[A-Z][a-z]?", element) or len(fields) < 2:
            continue
        for name in fields[1:]:
            if "-q" in name:
                table.setdefault(element, set()).add(name)
    return table


# -- rendering ----------------------------------------------------------


def render_input(
    template: str,
    structure: Structure,
    kinds: Mapping[str, KindSpec],
    *,
    vacuum: float = DEFAULT_VACUUM,
    guess: str = "ATOMIC",
) -> str:
    """Fill a CP2K input template for one structure.

    ``guess`` fills the optional GUESS placeholder, which is how the
    initial-guess benchmark runs the same system twice -- once from CP2K's
    atomic superposition and once from an ML prediction -- with every other
    setting byte-identical. Anything else differing between the two arms would
    make the comparison meaningless.
    """

    used = set(_PLACEHOLDER.findall(template))
    unknown = used - set(PLACEHOLDERS)
    if unknown:
        raise StructureError(
            f"template uses unknown placeholder(s) {sorted(unknown)}; "
            f"known: {sorted(PLACEHOLDERS)}"
        )
    missing = set(REQUIRED_PLACEHOLDERS) - used
    if missing:
        raise StructureError(f"template is missing required placeholder(s) {sorted(missing)}")

    missing_kinds = sorted(set(structure.symbols) - set(kinds))
    if missing_kinds:
        raise StructureError(f"{structure.label}: no kind defined for {missing_kinds}")

    values = {
        "PROJECT": structure.label,
        "LABEL": structure.label,
        "COORD": _coord_block(structure),
        "CELL": _cell_block(structure, vacuum=vacuum),
        "KIND": "\n".join(kinds[element].to_block() for element in structure.elements),
        "GUESS": guess,
    }
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)


def _coord_block(structure: Structure, indent: str = "      ") -> str:
    return "\n".join(
        f"{indent}{symbol:<3s} {x:14.8f} {y:14.8f} {z:14.8f}"
        for symbol, (x, y, z) in zip(structure.symbols, structure.positions)
    )


def _cell_block(structure: Structure, *, vacuum: float, indent: str = "      ") -> str:
    if structure.periodic and structure.cell is not None:
        rows = "\n".join(
            f"{indent}{axis} {vector[0]:14.8f} {vector[1]:14.8f} {vector[2]:14.8f}"
            for axis, vector in zip("ABC", structure.cell)
        )
        return f"{rows}\n{indent}PERIODIC XYZ"

    # A molecule in a box: large enough that the isolated system does not see
    # its own image, and PERIODIC NONE so CP2K uses the Poisson solver for it.
    span = np.ptp(structure.positions, axis=0) if structure.n_atoms > 1 else np.zeros(3)
    lengths = np.maximum(span + 2.0 * vacuum, 2.0 * vacuum)
    return (
        f"{indent}ABC {lengths[0]:14.8f} {lengths[1]:14.8f} {lengths[2]:14.8f}\n"
        f"{indent}PERIODIC NONE"
    )


# -- tasks --------------------------------------------------------------


def tasks_from_structures(
    structures: Sequence[Structure],
    *,
    template: str,
    input_dir: str | Path,
    kinds: Mapping[str, KindSpec],
    vacuum: float = DEFAULT_VACUUM,
    guess: str = "ATOMIC",
    resources: Mapping[str, Any] | None = None,
) -> list[PulayTask]:
    """Write one CP2K input per structure and return the tasks that run them."""

    directory = Path(input_dir)
    directory.mkdir(parents=True, exist_ok=True)

    tasks: list[PulayTask] = []
    for structure in structures:
        text = render_input(template, structure, kinds, vacuum=vacuum, guess=guess)
        path = directory / f"{structure.label}.inp"
        path.write_text(text, encoding="utf-8")

        detail = descriptors(structure)
        detail["kinds"] = {
            element: {"basis_set": kinds[element].basis_set, "potential": kinds[element].potential}
            for element in structure.elements
        }
        tasks.append(
            PulayTask(
                task_id=structure.label,
                input_path=path.resolve(),
                structure=detail,
                resources=dict(resources or {}),
            )
        )
    return tasks


def tasks_from_plan(
    plan_path: str | Path,
    *,
    input_dir: str | Path,
    data_dir: str | Path,
    root: str | Path | None = None,
    groups: Sequence[str] | None = None,
    **kwargs: Any,
) -> list[PulayTask]:
    """Build tasks from a sampling plan (``configs/cp2k/pulay_sampling_plan.json``).

    A plan pairs each family of systems with the CP2K settings that suit it --
    molecules and transition-metal oxides do not converge under the same SCF
    block -- and records the family on every sample, so the report can say
    whether the dataset is actually diverse or merely large.
    """

    import json

    source = Path(plan_path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StructureError(f"{source}: cannot be read as a sampling plan ({exc})") from exc

    entries = payload.get("groups") if isinstance(payload, Mapping) else None
    if not entries:
        raise StructureError(f"{source}: plan has no 'groups'")

    base = Path(root) if root is not None else source.parent.parent.parent
    selected = set(groups) if groups else None

    tasks: list[PulayTask] = []
    seen: set[str] = set()
    for entry in entries:
        name = str(entry.get("name") or "group")
        if selected is not None and name not in selected:
            continue
        paths = [_resolve(base, item) for item in entry.get("structures", ())]
        if not paths:
            raise StructureError(f"{source}: group {name!r} lists no structures")
        template = _resolve(base, entry["template"])

        group_tasks = tasks_from_collection(
            paths,
            template_path=template,
            input_dir=Path(input_dir) / name,
            data_dir=data_dir,
            **kwargs,
        )
        for task in group_tasks:
            if task.task_id in seen:
                raise StructureError(f"{source}: duplicate task id {task.task_id!r} across groups")
            seen.add(task.task_id)
            detail = dict(task.structure)
            detail["group"] = name
            detail["template"] = str(template)
            tasks.append(
                PulayTask(
                    task_id=task.task_id,
                    input_path=task.input_path,
                    structure=detail,
                    resources=dict(entry.get("resources", {})) or task.resources,
                )
            )
    if not tasks:
        raise StructureError(f"{source}: no groups selected")
    return tasks


def _resolve(base: Path, item: Any) -> Path:
    path = Path(str(item))
    return path if path.is_absolute() else (base / path)


def tasks_from_collection(
    paths: Iterable[str | Path],
    *,
    template_path: str | Path,
    input_dir: str | Path,
    data_dir: str | Path,
    basis_preference: Sequence[str] = DEFAULT_BASIS_PREFERENCE,
    potential_family: str = DEFAULT_POTENTIAL_FAMILY,
    overrides: Mapping[str, Mapping[str, str]] | None = None,
    vacuum: float = DEFAULT_VACUUM,
    resources: Mapping[str, Any] | None = None,
) -> list[PulayTask]:
    """Collection files + one template -> ready-to-run tasks."""

    structures = read_collection(paths)
    template = Path(template_path).read_text(encoding="utf-8")
    elements = sorted({symbol for structure in structures for symbol in structure.symbols})
    kinds = resolve_kinds(
        elements,
        data_dir=data_dir,
        basis_preference=basis_preference,
        potential_family=potential_family,
        overrides=overrides,
    )
    return tasks_from_structures(
        structures,
        template=template,
        input_dir=input_dir,
        kinds=kinds,
        vacuum=vacuum,
        resources=resources,
    )
