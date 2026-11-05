"""Legacy batch writers: CP2KLabeler now writes real inputs; job scripts are typed per scheduler."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from gmd_active_learning.core.exceptions import ConfigurationError
from gmd_active_learning.labeling.cp2k_labeler import CP2KLabeler, region_from_structure
from gmd_active_learning.labeling.job_writer import write_job_script
from gmd_active_learning.labeling.vasp_labeler import VASPLabeler

STRUCTURES = Path(__file__).resolve().parents[1] / "fixtures" / "structures"


def _candidate(root: Path, name: str, atoms: Atoms) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    write(directory / "structure.extxyz", atoms, format="extxyz")
    return directory


def test_cp2k_labeler_writes_a_real_input_and_typed_job_script(tmp_path: Path) -> None:
    from ase.io import read

    candidate = _candidate(tmp_path / "candidates", "cand_000", read(STRUCTURES / "h2.xyz"))
    config = {"work_dir": str(tmp_path / "jobs"), "scheduler": "pbs", "cp2k_command": "cp2k.psmp", "resources": {"walltime": "00:10:00", "queue": "debug"}}
    (job_dir,) = CP2KLabeler().generate([candidate], config)
    text = (job_dir / "cp2k_sp.inp").read_text()
    assert "RUN_TYPE ENERGY_FORCE" in text and "PERIODIC NONE" in text and "&KIND H" in text
    script = (job_dir / "submit.pbs").read_text()
    assert "#PBS -l walltime=00:10:00" in script and "#PBS -q debug" in script and "SBATCH" not in script
    assert "cp2k.psmp -i cp2k_sp.inp -o cp2k_sp.out" in script


def test_region_from_periodic_structure_keeps_cell_and_pbc(tmp_path: Path) -> None:
    atoms = Atoms("H2", positions=[[2.13, 2.5, 2.5], [2.87, 2.5, 2.5]], cell=np.eye(3) * 5.0, pbc=True)
    candidate = _candidate(tmp_path, "periodic", atoms)
    region = region_from_structure(candidate / "structure.extxyz")
    assert region.pbc == (True, True, True) and np.allclose(region.cell, np.eye(3) * 5.0)
    (job_dir,) = CP2KLabeler().generate([candidate], {"work_dir": str(tmp_path / "jobs"), "scheduler": "slurm"})
    assert "PERIODIC XYZ" in (job_dir / "cp2k_sp.inp").read_text()
    assert (job_dir / "submit.slurm").read_text().startswith("#!/bin/bash\n#SBATCH --job-name=gmd_periodic")


def test_job_writer_local_and_unknown(tmp_path: Path) -> None:
    path = write_job_script(tmp_path / "j", "local", "echo hi")
    assert path.name == "submit.sh" and ".gmd_exit_code" in path.read_text()
    with pytest.raises(ConfigurationError, match="unknown scheduler"):
        write_job_script(tmp_path / "k", "lsf", "echo hi")


def test_vasp_labeler_is_marked_unvalidated(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from ase.io import read

    candidate = _candidate(tmp_path / "candidates", "cand_000", read(STRUCTURES / "h2.xyz"))
    with caplog.at_level(logging.WARNING):
        (job_dir,) = VASPLabeler().generate([candidate], {"work_dir": str(tmp_path / "jobs"), "scheduler": "slurm"})
    assert VASPLabeler.validated is False and "UNVALIDATED" in caplog.text
    assert (job_dir / "UNVALIDATED.txt").exists() and (job_dir / "INCAR").read_text() == "PREC = Accurate\nNSW = 0\n"
