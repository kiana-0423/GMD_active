"""SCF-021: all 21 exchange vectors written by GMD's writer are byte-identical to PHAL's.

``data/exchange_vectors_phal/`` holds the output of cp2k-ml-scf's ``tests/generate_exchange_vectors.py``
run with PHAL 96263b6 as the writer: GMD_active not importable, ``PYTHONPATH=ref/lammps_monitor/src``,
``--fixtures tests/fixtures/scf``. The test re-runs the generator with GMD_active's writer and compares
every file. It needs the sibling cp2k-ml-scf checkout (``CP2K_ML_SCF_PATH``) and skips without it.
The Fortran reader's verdicts on these vectors need the Linux/aarch64 cp2k-ml-scf binary and are not
checked here.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FROZEN = HERE / "data" / "exchange_vectors_phal"
SIBLING = Path(os.environ.get("CP2K_ML_SCF_PATH", REPO.parent / "cp2k-ml-scf"))
GENERATOR = SIBLING / "tests" / "generate_exchange_vectors.py"
REF = REPO / "ref" / "lammps_monitor"


def _generate(output: Path, pythonpath: Path) -> None:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update(PYTHONPATH=str(pythonpath), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, str(GENERATOR), "--fixtures", str(REPO / "tests" / "fixtures" / "scf"), "--output", str(output)],
                   cwd=REPO, env=env, check=True, capture_output=True, text=True, timeout=300)


def _files(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def test_frozen_capture_has_21_vectors() -> None:
    names = [name for name in _files(FROZEN) if name != "expected.json"]
    assert len(names) == 21 and "expected.json" in _files(FROZEN)


@pytest.mark.skipif(not GENERATOR.is_file(), reason="cp2k-ml-scf checkout not found (set CP2K_ML_SCF_PATH)")
def test_gmd_writer_reproduces_phal_vectors_byte_for_byte(tmp_path: Path) -> None:
    _generate(tmp_path / "gmd", REPO / "src")
    assert _files(tmp_path / "gmd") == _files(FROZEN)


@pytest.mark.skipif(not (GENERATOR.is_file() and (REF / "src" / "hotspot_al").is_dir()), reason="needs cp2k-ml-scf and ref/")
def test_frozen_capture_is_what_phal_writes(tmp_path: Path) -> None:
    _generate(tmp_path / "phal", REF / "src")
    assert _files(tmp_path / "phal") == _files(FROZEN)
