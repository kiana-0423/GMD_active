"""Regenerate the SCF structure collections PHAL referenced but never committed (finding F-11).

PHAL's sampling plan (``configs/cp2k/pulay_sampling_plan.json``), docs and tests
reference ``data/structures/pulay_sampling/{small_molecules,periodic,
transition_metal_oxides}.extxyz`` and ``data/structures/benchmark/{molecules,
periodic,unseen}.extxyz``. They were excluded by PHAL's ``.gitignore`` (``*.extxyz``)
and are not recoverable (searched: the GMD_system workspace, the cp2k-ml-scf
repository, PHAL's history at 96263b6). This script rebuilds *replacements*
deterministically from ASE builders with literature lattice constants; see
``docs/scf/structure_provenance.md`` for what is original, what is regenerated
and which PHAL measurements cannot be compared against these files.

Only ``si-diamond`` is known to match the original exactly: the committed
fixture ``tests/fixtures/scf/pulay_sample_periodic/sample.json`` records its
cell volume, minimum distance and extent, and the regenerated frame reproduces
them (tested).

Usage (from the repository root)::

    python -m tools.scf.generate_structures             # write data/structures/...
    python -m tools.scf.generate_structures --check     # verify committed files match
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

GENERATOR_VERSION = "gmd-scf-structures-v1"
#: Seed for the small symmetry-breaking displacements of strained molecules.
SEED = 20260801
#: Amplitude (A) of those displacements.
JITTER_A = 0.02

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path("data") / "structures"


@dataclass(frozen=True)
class Entry:
    """One frame of a collection: a label, a family and a builder."""

    label: str
    family: str
    build: Callable[[], Any]
    note: str = ""
    info: dict[str, Any] = field(default_factory=dict)


def _molecule(name: str) -> Callable[[], Any]:
    def build() -> Any:
        from ase.build import molecule

        atoms = molecule(name)
        atoms.set_cell(np.zeros((3, 3)))
        atoms.set_pbc(False)
        return atoms

    return build


def _strained_molecule(name: str, factor: float, offset: int) -> Callable[[], Any]:
    """Every interatomic distance scaled by ``factor`` about the centroid, plus a seeded jitter."""

    def build() -> Any:
        atoms = _molecule(name)()
        centre = atoms.get_positions().mean(axis=0)
        positions = centre + factor * (atoms.get_positions() - centre)
        rng = np.random.default_rng(SEED + offset)
        positions = positions + rng.uniform(-JITTER_A, JITTER_A, size=positions.shape)
        atoms.set_positions(np.round(positions, 10))
        return atoms

    return build


def _bulk(name: str, structure: str, **kwargs: float) -> Callable[[], Any]:
    def build() -> Any:
        from ase.build import bulk

        return bulk(name, structure, **kwargs)

    return build


def _strained_bulk(name: str, structure: str, factor: float, **kwargs: float) -> Callable[[], Any]:
    def build() -> Any:
        atoms = _bulk(name, structure, **kwargs)()
        atoms.set_cell(atoms.get_cell() * factor, scale_atoms=True)
        return atoms

    return build


def _crystal(symbols: Sequence[str], basis: Sequence[tuple[float, float, float]], spacegroup: int,
             cellpar: Sequence[float], setting: int = 1) -> Callable[[], Any]:
    def build() -> Any:
        from ase.spacegroup import crystal

        return crystal(list(symbols), basis=list(basis), spacegroup=spacegroup, cellpar=list(cellpar),
                       setting=setting)

    return build


#: 12 equilibrium + 5 strained closed-shell molecules = 17 (PHAL docs: "17 closed-shell molecules,
#: equilibrium and strained"; 13 elements / 21 formulas over the accepted 27-system dataset).
SMALL_MOLECULES: tuple[Entry, ...] = (
    Entry("h2", "small_molecules", _molecule("H2")),
    Entry("h2o", "small_molecules", _molecule("H2O")),
    Entry("nh3", "small_molecules", _molecule("NH3")),
    Entry("ch4", "small_molecules", _molecule("CH4")),
    Entry("co2", "small_molecules", _molecule("CO2")),
    Entry("co", "small_molecules", _molecule("CO")),
    Entry("n2", "small_molecules", _molecule("N2")),
    Entry("hf", "small_molecules", _molecule("HF")),
    Entry("lih", "small_molecules", _molecule("LiH")),
    Entry("c2h4", "small_molecules", _molecule("C2H4")),
    Entry("hcn", "small_molecules", _molecule("HCN")),
    Entry("h2co", "small_molecules", _molecule("H2CO")),
    Entry("h2o-strained", "small_molecules", _strained_molecule("H2O", 1.15, 1), "distances x1.15"),
    Entry("nh3-strained", "small_molecules", _strained_molecule("NH3", 1.15, 2), "distances x1.15"),
    Entry("ch4-strained", "small_molecules", _strained_molecule("CH4", 1.15, 3), "distances x1.15"),
    Entry("co2-strained", "small_molecules", _strained_molecule("CO2", 1.10, 4), "distances x1.10"),
    Entry("n2-strained", "small_molecules", _strained_molecule("N2", 1.20, 5), "distances x1.20"),
)

#: The seven periodic main-group systems of PHAL's 12-solid strategy table, si-diamond first
#: (the committed fixture records ``source_index: 0`` for it).
PERIODIC: tuple[Entry, ...] = (
    Entry("si-diamond", "periodic_main_group", _bulk("Si", "diamond", a=5.431), "matches the PHAL fixture"),
    Entry("si-diamond-strained", "periodic_main_group", _strained_bulk("Si", "diamond", 1.03, a=5.431),
          "lattice x1.03 (PHAL's strain is unknown)"),
    Entry("c-diamond", "periodic_main_group", _bulk("C", "diamond", a=3.567)),
    Entry("aln-wurtzite", "periodic_main_group", _bulk("AlN", "wurtzite", a=3.112, c=4.982, u=0.382)),
    Entry("lih-rocksalt", "periodic_main_group", _bulk("LiH", "rocksalt", a=4.083)),
    Entry("mgo-rocksalt", "periodic_main_group", _bulk("MgO", "rocksalt", a=4.212)),
    Entry("nacl-rocksalt", "periodic_main_group", _bulk("NaCl", "rocksalt", a=5.640)),
)

#: Five transition-metal oxides (anatase comes from the committed tio2_anatase.cif).
TRANSITION_METAL_OXIDES: tuple[Entry, ...] = (
    Entry("tio2-rutile", "transition_metal_oxides",
          _crystal(["Ti", "O"], [(0, 0, 0), (0.305, 0.305, 0)], 136, [4.594, 4.594, 2.959, 90, 90, 90])),
    Entry("nio-rocksalt", "transition_metal_oxides", _bulk("NiO", "rocksalt", a=4.177),
          "non-magnetic primitive cell; PHAL's spin setup is unknown"),
    Entry("mno-rocksalt", "transition_metal_oxides", _bulk("MnO", "rocksalt", a=4.445),
          "non-magnetic primitive cell; PHAL's spin setup is unknown"),
    Entry("zno-wurtzite", "transition_metal_oxides", _bulk("ZnO", "wurtzite", a=3.250, c=5.207, u=0.382)),
    Entry("cu2o-cuprite", "transition_metal_oxides",
          _crystal(["Cu", "O"], [(0, 0, 0), (0.25, 0.25, 0.25)], 224, [4.27, 4.27, 4.27, 90, 90, 90], setting=2)),
)

#: Benchmark sets referenced by PHAL's guess-benchmark docs/scripts. "unseen" holds the
#: held-out chemistry of ml_density_failure_analysis.md (h2cchf, hcf3, hcooh, n2h4).
BENCHMARK_MOLECULES: tuple[Entry, ...] = (
    Entry("co2", "benchmark_molecules", _molecule("CO2")),
    Entry("h2o", "benchmark_molecules", _molecule("H2O")),
    Entry("nh3", "benchmark_molecules", _molecule("NH3")),
)
BENCHMARK_PERIODIC: tuple[Entry, ...] = (
    Entry("si", "benchmark_periodic", _bulk("Si", "diamond", a=5.431)),
)
BENCHMARK_UNSEEN: tuple[Entry, ...] = (
    Entry("h2cchf", "benchmark_unseen", _molecule("H2CCHF")),
    Entry("hcf3", "benchmark_unseen", lambda: _hcf3(), "built by hand; not in ASE G2"),
    Entry("hcooh", "benchmark_unseen", _molecule("HCOOH")),
    Entry("n2h4", "benchmark_unseen", _molecule("N2H4")),
)


def _hcf3() -> Any:
    """Fluoroform is not in ASE's G2 set; built from tetrahedral geometry (C-F 1.33 A, C-H 1.09 A)."""

    from ase import Atoms

    directions = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) / np.sqrt(3.0)
    positions = [np.zeros(3), 1.09 * directions[0], 1.33 * directions[1], 1.33 * directions[2], 1.33 * directions[3]]
    atoms = Atoms("CHF3", positions=np.round(np.asarray(positions), 10))
    atoms.set_cell(np.zeros((3, 3)))
    atoms.set_pbc(False)
    return atoms


COLLECTIONS: dict[str, tuple[Entry, ...]] = {
    "pulay_sampling/small_molecules.extxyz": SMALL_MOLECULES,
    "pulay_sampling/periodic.extxyz": PERIODIC,
    "pulay_sampling/transition_metal_oxides.extxyz": TRANSITION_METAL_OXIDES,
    "benchmark/molecules.extxyz": BENCHMARK_MOLECULES,
    "benchmark/periodic.extxyz": BENCHMARK_PERIODIC,
    "benchmark/unseen.extxyz": BENCHMARK_UNSEEN,
}


def build_collection(entries: Sequence[Entry]) -> list[Any]:
    frames = []
    for entry in entries:
        atoms = entry.build()
        atoms.info = {"label": entry.label, "family": entry.family, "generator": GENERATOR_VERSION}
        if entry.note:
            atoms.info["note"] = entry.note
        frames.append(atoms)
    return frames


def write_collections(root: Path) -> list[Path]:
    from ase.io import write

    written = []
    for relative, entries in COLLECTIONS.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        write(str(target), build_collection(entries), format="extxyz")
        written.append(target)
        LOGGER.info("wrote %s (%d frames)", target, len(entries))
    return written


def check_collections(root: Path, *, atol: float = 1e-8) -> list[str]:
    """Differences between the committed files under ``root`` and a fresh regeneration."""

    from ase.io import read

    problems: list[str] = []
    for relative, entries in COLLECTIONS.items():
        target = root / relative
        if not target.is_file():
            problems.append(f"{relative}: missing")
            continue
        committed = read(str(target), index=":")
        fresh = build_collection(entries)
        if len(committed) != len(fresh):
            problems.append(f"{relative}: {len(committed)} frames, expected {len(fresh)}")
            continue
        for old, new in zip(committed, fresh):
            label = new.info["label"]
            if old.info.get("label") != label:
                problems.append(f"{relative}: label {old.info.get('label')!r} != {label!r}")
            elif list(old.numbers) != list(new.numbers):
                problems.append(f"{relative}:{label}: elements differ")
            elif not np.allclose(old.positions, new.positions, atol=atol) or not np.allclose(
                np.asarray(old.cell), np.asarray(new.cell), atol=atol
            ):
                problems.append(f"{relative}:{label}: geometry differs")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT / DEFAULT_ROOT)
    parser.add_argument("--check", action="store_true", help="verify instead of writing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.check:
        problems = check_collections(args.root)
        for problem in problems:
            print(problem)
        return 1 if problems else 0
    for path in write_collections(args.root):
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
