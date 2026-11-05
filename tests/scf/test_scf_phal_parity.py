"""Optional byte-level parity with PHAL @ 96263b6 (skipped when ref/lammps_monitor is absent).

Checks the cross-repository contracts: the density exchange bytes (cp2k-ml-scf's
Fortran reader was validated against PHAL's writer), the SCF-history parse and
the structure hash embedded in exchange files.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
PHAL_SRC = REPO / "ref" / "lammps_monitor" / "src"
FIXTURES = REPO / "tests" / "fixtures" / "scf"

pytestmark = pytest.mark.skipif(not (PHAL_SRC / "hotspot_al").is_dir(), reason="ref/lammps_monitor is absent")


@pytest.fixture(scope="module")
def phal():
    sys.path.insert(0, str(PHAL_SRC))
    try:
        yield {
            "exchange": importlib.import_module("hotspot_al.cp2k_ml_scf.density.exchange"),
            "prediction": importlib.import_module("hotspot_al.cp2k_ml_scf.density.prediction"),
            "schema": importlib.import_module("hotspot_al.cp2k_ml_scf.schema"),
            "history": importlib.import_module("hotspot_al.cp2k.scf_history"),
        }
    finally:
        sys.path.remove(str(PHAL_SRC))


@pytest.mark.parametrize("name", ["h2", "h2o"])
def test_exchange_files_are_byte_identical(tmp_path: Path, phal, name: str) -> None:
    from gmd_active_learning.adapters.dft.scf.density.exchange import write_density_exchange
    from gmd_active_learning.adapters.dft.scf.density.prediction import SCFDensityPrediction

    data = np.load(FIXTURES / f"{name}_scf_trajectory.npz")
    kwargs = dict(
        model_name="replay", model_version="1", model_digest="sha256:abc", structure_hash="h" * 64,
        electron_count=float(data["n_electrons"]), spin_channels=1,
        density_format="phal-density-matrix-v1", values=data["densities"][-1], overlap=data["overlap"],
        metadata={"basis_set": "DZVP-MOLOPT-GTH"}, timestamp="2026-08-04T00:00:00+00:00",
    )
    ours = write_density_exchange(SCFDensityPrediction(**kwargs), tmp_path / "ours.ml_density")
    theirs = phal["exchange"].write_density_exchange(
        phal["prediction"].SCFDensityPrediction(**kwargs), tmp_path / "theirs.ml_density"
    )

    assert ours.read_bytes() == theirs.read_bytes()


def test_structure_hash_matches(phal) -> None:
    from gmd_active_learning.adapters.dft.scf import schema

    args = dict(
        atomic_numbers=(8, 1, 1),
        positions=((0.0, 0.0, 0.0), (0.9572, 0.0, 0.0), (-0.24, 0.9266, 0.0)),
        cell=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
    )
    state = dict(charge=0, multiplicity=1, n_electrons=10, spin_polarized=False, spin_channels=1)

    ours = schema.structure_hash(schema.SCFStructure(**args), schema.SCFElectronicState(**state))
    theirs = phal["schema"].structure_hash(
        phal["schema"].SCFStructure(**args), phal["schema"].SCFElectronicState(**state)
    )

    assert ours == theirs


def test_scf_history_parse_matches_on_a_real_trajectory_output(phal) -> None:
    from gmd_active_learning.adapters.dft.scf.scf_history import parse_scf_history
    from tests.scf.fakes.fake_cp2k import fake_cp2k_scf_output

    text = fake_cp2k_scf_output() + " SCF run NOT converged\n"
    for sample in (fake_cp2k_scf_output(), text):
        assert parse_scf_history(sample).to_dict() == phal["history"].parse_scf_history(sample).to_dict()
