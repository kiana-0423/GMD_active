"""Legacy batch entry point: write real CP2K inputs and a job script for candidate directories.

Kept for the original ``ActiveLearningWorkflow`` (``labeling.generate(candidate_dirs, config)``). It now
uses the CP2K adapter instead of the former empty ``&FORCE_EVAL`` placeholder. Each candidate's
``structure.extxyz`` is labeled as a whole (every atom is a core atom, its cell/PBC taken from the
file). For extracted regions with caps, retries and persistent state use
:class:`gmd_active_learning.labeling.service.LabelingService`.

Legacy config keys: ``work_dir``, ``scheduler`` (slurm|pbs|local), ``cp2k_command`` (executable), plus
optional ``cp2k`` (a ``cp2k`` section mapping) and ``resources``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read

from gmd_active_learning.adapters.dft.cp2k.command import build_cp2k_command, cp2k_environment
from gmd_active_learning.adapters.dft.cp2k.config import CP2KConfig
from gmd_active_learning.adapters.dft.cp2k.input import write_cp2k_inputs
from gmd_active_learning.contracts.ids import stable_id
from gmd_active_learning.contracts.regions import ExtractedRegion, RegionRole
from gmd_active_learning.labeling.base_labeler import BaseLabeler
from gmd_active_learning.labeling.job_writer import write_job_script
from gmd_active_learning.utils.io import ensure_dir

LOGGER = logging.getLogger(__name__)


def region_from_structure(path: str | Path, *, region_id: str | None = None) -> ExtractedRegion:
    """Wrap a whole structure file as an :class:`ExtractedRegion` (all atoms core)."""

    atoms = read(path)
    n_atoms = len(atoms)
    cell = np.asarray(atoms.cell.array, dtype=float)
    has_cell = abs(float(np.linalg.det(cell))) > 1.0e-8
    pbc = tuple(bool(flag) for flag in atoms.pbc) if has_cell else (False, False, False)
    return ExtractedRegion(
        region_id=region_id or stable_id("structure", Path(path).resolve().as_posix()),
        mode="whole_structure",
        run_id=str(atoms.info.get("run_id", "legacy")),
        parent_step=int(atoms.info.get("step", 0)),
        event_id=None,
        numbers=atoms.numbers,
        positions=atoms.positions,
        cell=cell if has_cell else None,
        pbc=pbc,  # type: ignore[arg-type]
        parent_atom_ids=np.arange(n_atoms),
        image_shifts=np.zeros((n_atoms, 3), dtype=int),
        roles=[RegionRole.CORE] * n_atoms,
        force_weights=np.ones(n_atoms),
        frozen=np.zeros(n_atoms, dtype=bool),
        hotspot_atom_ids=[],
        charge=int(atoms.info.get("charge", 0)),
        multiplicity=atoms.info.get("multiplicity"),
    )


class CP2KLabeler(BaseLabeler):
    def generate(self, candidate_paths: list[Path], config: dict[str, Any]) -> list[Path]:
        work_dir = ensure_dir(config["work_dir"])
        section = dict(config.get("cp2k", {}))
        if config.get("cp2k_command"):
            section.setdefault("executable", str(config["cp2k_command"]))
        cp2k = CP2KConfig.model_validate(section)
        scheduler = str(config.get("scheduler", "slurm"))
        generated: list[Path] = []
        for candidate in candidate_paths:
            candidate = Path(candidate)
            job_dir = ensure_dir(work_dir / candidate.name)
            region = region_from_structure(candidate / "structure.extxyz", region_id=candidate.name)
            written = write_cp2k_inputs(region, job_dir, cp2k, job_name="cp2k")
            single_point = written["single_point_input"]
            command = build_cp2k_command(cp2k, single_point.name, f"{single_point.stem}.out", resources=cp2k.resources)
            write_job_script(
                job_dir,
                scheduler,
                command,
                resources=config.get("resources", cp2k.resources),
                environment=cp2k_environment(cp2k),
                job_name=f"gmd_{candidate.name}",
            )
            LOGGER.info("wrote CP2K inputs for %s in %s", candidate.name, job_dir)
            generated.append(job_dir)
        return generated
