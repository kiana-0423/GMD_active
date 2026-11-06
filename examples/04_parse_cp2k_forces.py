# Ported from PHAL (examples/04_parse_cp2k_forces.py) at 96263b6; MIT License, see NOTICE.
"""Parse forces (and the label verdict) from a CP2K output file.

    python examples/04_parse_cp2k_forces.py                       # real CP2K 2026.2 output (examples/data)
    python examples/04_parse_cp2k_forces.py run/attempts/01-sp/sp.out --n-atoms 42

Prints PHAL's statistics unchanged (shape, per-axis min/max/mean, min/max force norm, eV/A),
then the total energy, SCF convergence, CP2K version and the label validation verdict that the
labeling service applies (convergence marker, no abort, finite values, row count = atom count
when ``--n-atoms`` is given, normal termination). Exit 1 when the output is not a valid label.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from _example_utils import DATA_DIR, require_file

from gmd_active_learning.adapters.dft.cp2k.config import ValidationSettings
from gmd_active_learning.adapters.dft.cp2k.parse import parse_cp2k_forces, parse_cp2k_output_file
from gmd_active_learning.adapters.dft.cp2k.validation import validation_reasons
from gmd_active_learning.core.exceptions import LabelingError


def force_statistics(forces: np.ndarray) -> dict[str, object]:
    """The statistics PHAL's example printed, as values."""

    norms = np.linalg.norm(forces, axis=1)
    return {
        "shape": forces.shape,
        "min": np.min(forces, axis=0),
        "max": np.max(forces, axis=0),
        "mean": np.mean(forces, axis=0),
        "norm_min": float(np.min(norms)),
        "norm_max": float(np.max(norms)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", type=Path, default=DATA_DIR / "ch3_capped_sp.out", help="CP2K .out file")
    parser.add_argument("--n-atoms", type=int, default=None, help="expected atom count (default: the parsed row count)")
    args = parser.parse_args(argv)
    require_file(parser, args.output, "CP2K output file")

    try:
        forces = parse_cp2k_forces(args.output)
    except LabelingError as exc:
        print(f"no forces: {exc}")
    else:
        stats = force_statistics(forces)
        for key in ("shape", "min", "max", "mean", "norm_min", "norm_max"):
            print(stats[key] if key == "shape" else f"{key}: {stats[key]}")

    output = parse_cp2k_output_file(args.output)
    n_atoms = args.n_atoms if args.n_atoms is not None else output.n_force_rows
    reasons = validation_reasons(output, n_atoms=n_atoms, settings=ValidationSettings())
    print(f"energy_eV: {output.energy}")
    print(f"scf_converged: {output.scf_converged} scf_steps: {output.scf_steps} cp2k_version: {output.version}")
    print("label: VALID" if not reasons else "label: INVALID (" + "; ".join(reasons) + ")")
    return 0 if not reasons else 1


if __name__ == "__main__":
    sys.exit(main())
