from __future__ import annotations

from pathlib import Path
from typing import Any

from gmd_active_learning.labeling.base_labeler import BaseLabeler
from gmd_active_learning.labeling.job_writer import write_job_script
from gmd_active_learning.utils.io import ensure_dir
from gmd_active_learning.utils.structure_io import read_extxyz, write_xyz


class CP2KLabeler(BaseLabeler):
    def generate(self, candidate_paths: list[Path], config: dict[str, Any]) -> list[Path]:
        work_dir = ensure_dir(config["work_dir"])
        generated: list[Path] = []
        for candidate in candidate_paths:
            job_dir = ensure_dir(work_dir / candidate.name)
            atoms = read_extxyz(candidate / "structure.extxyz")
            write_xyz(job_dir / "structure.xyz", atoms.symbols, atoms.positions)
            cp2k_input = "&GLOBAL\n  PROJECT gmd_al\n&END GLOBAL\n&FORCE_EVAL\n&END FORCE_EVAL\n"
            (job_dir / "cp2k.inp").write_text(cp2k_input, encoding="utf-8")
            write_job_script(job_dir, config.get("scheduler", "slurm"), config.get("cp2k_command", "cp2k.popt") + " -i cp2k.inp")
            generated.append(job_dir)
        return generated
