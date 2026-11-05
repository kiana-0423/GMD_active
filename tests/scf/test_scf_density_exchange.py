# Ported from PHAL (tests.scf.test_scf_density_exchange) at 96263b6; MIT License, see NOTICE.
"""Tests for the PHAL -> CP2K density exchange format and the ML-SCF backend hookup."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from gmd_active_learning.adapters.dft.scf.density import DummyDensityPredictor
from gmd_active_learning.adapters.dft.scf.density.exchange import (
    EXCHANGE_PROTOCOL,
    exchange_path_for,
    read_density_exchange,
    write_density_exchange,
)
from gmd_active_learning.adapters.dft.scf.density.prediction import (
    DENSITY_FORMAT_DENSITY_MATRIX,
    PredictionError,
    SCFDensityPrediction,
)

from tests.scf.test_scf_dataset import water_state, water_structure


def water_prediction() -> SCFDensityPrediction:
    return DummyDensityPredictor().load_model().predict(water_structure(), water_state())


# -- format -------------------------------------------------------------


def test_exchange_header_matches_the_documented_layout(tmp_path: Path) -> None:
    path = write_density_exchange(water_prediction(), tmp_path / "guess.ml_density")
    lines = path.read_text(encoding="utf-8").splitlines()

    assert lines[0] == f"PHAL-ML-SCF {EXCHANGE_PROTOCOL}"
    assert lines[1].startswith("FORMAT phal-atomic-population")
    assert lines[2] == "REPRESENTATION ao-nonorthogonal"
    assert lines[3].startswith("STRUCTURE sha256:")
    assert lines[4].startswith("MODEL sha256:")
    assert lines[5].startswith("ELECTRONS ")
    assert lines[6] == "NSPIN 1"
    assert lines[7] == "DIM 3"
    assert lines[8].startswith("BASIS ")
    # Atomic populations need no overlap: there the count is the plain sum.
    assert lines[9] == "OVERLAP absent"
    assert lines[10] == "VALUES"
    assert len(lines) == 11 + 3


def test_populations_round_trip(tmp_path: Path) -> None:
    prediction = water_prediction()

    path = write_density_exchange(prediction, tmp_path / "guess.ml_density")
    restored = read_density_exchange(path)

    assert restored["spin_channels"] == 1
    assert restored["leading_dim"] == 3
    assert np.allclose(restored["values"], prediction.values)
    assert restored["electron_count"] == pytest.approx(8.0)  # valence count (F-23)
    assert restored["structure_hash"] == prediction.structure_hash


def test_open_shell_populations_round_trip(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.schema import SCFElectronicState, SCFStructure

    structure = SCFStructure(
        atomic_numbers=(8, 1),
        positions=((0.0, 0.0, 0.0), (0.97, 0.0, 0.0)),
        cell=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
    )
    state = SCFElectronicState(
        charge=0, multiplicity=2, n_electrons=9, spin_polarized=True, spin_channels=2
    )
    prediction = DummyDensityPredictor().load_model().predict(structure, state)

    restored = read_density_exchange(write_density_exchange(prediction, tmp_path / "g.ml_density"))

    assert restored["spin_channels"] == 2
    assert np.allclose(restored["values"], prediction.values)


def test_density_matrix_round_trip_preserves_indices(tmp_path: Path) -> None:
    matrix = np.asarray([[[1.0, 0.5, 0.0], [0.5, 2.0, 0.25], [0.0, 0.25, 3.0]]])
    prediction = SCFDensityPrediction(
        model_name="test",
        model_version="1",
        model_digest="sha256:0",
        structure_hash="sha256:x",
        electron_count=6.0,
        spin_channels=1,
        density_format=DENSITY_FORMAT_DENSITY_MATRIX,
        values=matrix,
        overlap=np.eye(3),
    )

    restored = read_density_exchange(write_density_exchange(prediction, tmp_path / "g.ml_density"))

    assert restored["values"].shape == (1, 3, 3)
    assert np.allclose(restored["values"], matrix)


def test_values_are_written_in_fortran_order(tmp_path: Path) -> None:
    """The Fortran reader's (i, j, spin) loop nest requires column-major order."""

    matrix = np.asarray([[[1.0, 2.0], [3.0, 4.0]]])
    prediction = SCFDensityPrediction(
        model_name="t", model_version="1", model_digest="sha256:0", structure_hash="sha256:x",
        electron_count=5.0, spin_channels=1, density_format=DENSITY_FORMAT_DENSITY_MATRIX, values=matrix,
        overlap=np.eye(2),
    )

    path = write_density_exchange(prediction, tmp_path / "g.ml_density")
    body = path.read_text(encoding="utf-8").splitlines()
    values = [float(line) for line in body[body.index("VALUES") + 1 : body.index("OVERLAP_VALUES")]]

    # Column-major over (i, j): (1,1) (2,1) (1,2) (2,2) -> 1, 3, 2, 4
    assert values == [1.0, 3.0, 2.0, 4.0]


def test_non_finite_values_are_refused_before_writing(tmp_path: Path) -> None:
    prediction = water_prediction()
    broken = np.array(prediction.values)
    broken[0, 0] = np.nan

    with pytest.raises(PredictionError, match="non-finite"):
        write_density_exchange(
            SCFDensityPrediction(
                model_name="t", model_version="1", model_digest="sha256:0",
                structure_hash="sha256:x", electron_count=10.0, spin_channels=1,
                density_format=prediction.density_format, values=broken,
            ),
            tmp_path / "g.ml_density",
        )


def test_prediction_without_values_cannot_be_exported(tmp_path: Path) -> None:
    prediction = SCFDensityPrediction(
        model_name="t", model_version="1", model_digest="sha256:0", structure_hash="sha256:x",
        electron_count=10.0, spin_channels=1, density_format="phal-atomic-population-v1",
    )

    with pytest.raises(PredictionError, match="no density values"):
        write_density_exchange(prediction, tmp_path / "g.ml_density")


def test_reading_a_foreign_protocol_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "g.ml_density"
    path.write_text("PHAL-ML-SCF phal-ml-scf-v0\nVALUES\n", encoding="utf-8")

    with pytest.raises(PredictionError, match="protocol"):
        read_density_exchange(path)


def test_reading_a_truncated_file_fails_loudly(tmp_path: Path) -> None:
    path = write_density_exchange(water_prediction(), tmp_path / "g.ml_density")
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(PredictionError, match="expected 3 values"):
        read_density_exchange(path)


def test_exchange_path_sits_next_to_the_input() -> None:
    assert exchange_path_for(Path("/tmp/job.inp")).name == "job.inp.ml_density"


# The backend hookup tests (PHAL's CP2KMLSCFBackend subclass) are rewritten
# for the composable wrapper in tests/scf/test_scf_ml_backend.py.


# -- the density representation contract --------------------------------


def test_density_matrix_without_an_overlap_is_refused(tmp_path: Path) -> None:
    """Ne = Tr(P S). Without S, CP2K cannot check the count and will refuse the
    file; failing here gives a message instead of a silent mid-run fallback."""

    matrix = np.asarray([[[2.0, 0.5, 0.0], [0.5, 2.0, 0.5], [0.0, 0.5, 2.0]]])
    prediction = SCFDensityPrediction(
        model_name="t",
        model_version="1",
        model_digest="sha256:0",
        structure_hash="sha256:x",
        electron_count=6.0,
        spin_channels=1,
        density_format=DENSITY_FORMAT_DENSITY_MATRIX,
        values=matrix,
        overlap=None,
    )

    with pytest.raises(PredictionError, match="Tr\\(P S\\)"):
        write_density_exchange(prediction, tmp_path / "g.ml_density")


def test_overlap_round_trips_and_is_not_the_identity(tmp_path: Path) -> None:
    overlap = np.asarray([[1.0, 0.3, 0.1], [0.3, 1.0, 0.3], [0.1, 0.3, 1.0]])
    matrix = np.asarray([[[2.0, 0.5, 0.0], [0.5, 2.0, 0.5], [0.0, 0.5, 2.0]]])
    # The count CP2K will compute is Tr(P S), so declare that, not Tr(P).
    electrons = float((matrix[0] * overlap).sum())
    prediction = SCFDensityPrediction(
        model_name="t",
        model_version="1",
        model_digest="sha256:0",
        structure_hash="sha256:x",
        electron_count=electrons,
        spin_channels=1,
        density_format=DENSITY_FORMAT_DENSITY_MATRIX,
        values=matrix,
        overlap=overlap,
    )

    restored = read_density_exchange(write_density_exchange(prediction, tmp_path / "g.ml_density"))

    assert restored["representation"] == "ao-nonorthogonal"
    assert np.allclose(restored["overlap"], overlap)
    # The whole point: the two traces disagree, and the declared count is Tr(PS).
    assert not np.isclose(float(np.trace(matrix[0])), electrons)
    assert np.isclose(float((restored["values"][0] * restored["overlap"]).sum()), electrons)
