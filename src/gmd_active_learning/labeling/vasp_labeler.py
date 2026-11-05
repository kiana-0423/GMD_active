"""UNVALIDATED VASP input writer, kept as is for the legacy workflow.

These INCAR/KPOINTS files are generic placeholders: no functional, cutoff, smearing, spin or k-point
choice has been validated, no POTCAR is assembled, nothing is submitted and no output parser exists.
CP2K is GMD_active's labeling engine; do not train on labels produced from these inputs without a
separate validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from gmd_active_learning.labeling.base_labeler import BaseLabeler
from gmd_active_learning.labeling.job_writer import write_job_script
from gmd_active_learning.utils.io import ensure_dir
from gmd_active_learning.utils.structure_io import read_extxyz, write_poscar


LOGGER = logging.getLogger(__name__)

VALIDATED = False


class VASPLabeler(BaseLabeler):
    """UNVALIDATED: writes placeholder VASP inputs (see the module docstring)."""

    validated = VALIDATED

    def generate(self, candidate_paths: list[Path], config: dict[str, Any]) -> list[Path]:
        LOGGER.warning("VASPLabeler writes UNVALIDATED placeholder inputs; CP2K is the validated labeling engine")
        work_dir = ensure_dir(config["work_dir"])
        generated: list[Path] = []
        for candidate in candidate_paths:
            job_dir = ensure_dir(work_dir / candidate.name)
            atoms = read_extxyz(candidate / "structure.extxyz")
            write_poscar(job_dir / "POSCAR", atoms.symbols, atoms.positions, atoms.cell)
            (job_dir / "INCAR").write_text("PREC = Accurate\nNSW = 0\n", encoding="utf-8")
            (job_dir / "KPOINTS").write_text("Automatic mesh\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
            (job_dir / "POTCAR.placeholder").write_text("Provide POTCAR externally.\n", encoding="utf-8")
            (job_dir / "UNVALIDATED.txt").write_text(
                "These VASP inputs are unvalidated placeholders (gmd_active_learning.labeling.vasp_labeler).\n",
                encoding="utf-8",
            )
            write_job_script(job_dir, config.get("scheduler", "slurm"), "vasp_std")
            generated.append(job_dir)
        return generated
