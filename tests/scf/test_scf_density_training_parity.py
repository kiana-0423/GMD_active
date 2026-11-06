"""SCF-026: on the test fixtures with a fixed seed, GMD's density GNN training reproduces PHAL's.

``data/density_training_parity/phal_short_run.json`` was produced by PHAL 96263b6
(``hotspot_al.cp2k_ml_scf.training.train_density.train_density_matrix_model``). The run is 3 epochs,
seed 0, hidden 16, 2 interactions, on PHAL's own ``identifiable_examples`` fixtures (3 water + H2,
seeded fake densities): ``python tests/scf/density_training_run.py phal`` from ref/lammps_monitor. The
GMD run uses the ported fixtures. The fitted AO basis and valence table must be equal, and the loss
trajectory must agree at rtol 1e-5. The 27-system fit in density_representation.md needs the lost
dataset and is not reproducible.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from tests.scf.density_training_run import run  # noqa: E402

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "data" / "density_training_parity" / "phal_short_run.json"
REF = HERE.parents[1] / "ref" / "lammps_monitor"


def _assert_same_run(ours: dict, reference: dict) -> None:
    assert ours["basis"] == reference["basis"]
    assert ours["valence"] == reference["valence"]
    assert [(m["epoch"], m["split"]) for m in ours["history"]] == [(m["epoch"], m["split"]) for m in reference["history"]]
    for mine, theirs in zip(ours["history"], reference["history"], strict=True):
        for key in ("loss", "density_mae", "density_rmse", "electron_mae"):
            assert mine[key] == pytest.approx(theirs[key], rel=1e-5, abs=1e-6), (mine["epoch"], key)


def test_short_training_run_matches_phal(tmp_path: Path) -> None:
    reference = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert len(reference["history"]) == 3 and reference["valence"] is not None
    _assert_same_run(run("gmd", tmp_path), reference)


@pytest.mark.skipif(not (REF / "src" / "hotspot_al").is_dir(), reason="ref/lammps_monitor not present")
def test_frozen_phal_run_is_current() -> None:
    env = {**os.environ, "PYTHONPATH": f"{REF / 'src'}{os.pathsep}{REF}", "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run([sys.executable, str(HERE / "density_training_run.py"), "phal"], cwd=REF, env=env,
                               capture_output=True, text=True, check=True, timeout=600)
    _assert_same_run(json.loads(completed.stdout.strip().splitlines()[-1]), json.loads(FROZEN.read_text(encoding="utf-8")))
