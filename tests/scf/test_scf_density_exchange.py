# Ported from PHAL (tests.scf.test_scf_density_exchange) at 96263b6; MIT License, see NOTICE.
"""Tests for the PHAL -> CP2K density exchange format and the ML-SCF backend hookup."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from gmd_active_learning.adapters.dft.scf.cp2k_ml_backend import CP2KMLSCFBackend
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
from hotspot_al.models import ExtractedRegion
from gmd_active_learning.adapters.dft.scf.config import load_config

from tests.scf.test_scf_dataset import water_state, water_structure


def water_prediction() -> SCFDensityPrediction:
    return DummyDensityPredictor().load_model().predict(water_structure(), water_state())


def water_region() -> ExtractedRegion:
    atoms = Atoms("OH2", positions=water_structure().positions, cell=[10, 10, 10], pbc=True)
    return ExtractedRegion(
        atoms=atoms,
        original_indices=[0, 1, 2],
        core_indices=[0],
        inner_buffer_indices=[1],
        outer_buffer_indices=[2],
        boundary_indices=[],
        h_cap_indices=[],
        hotspot_indices=[0],
        region_labels=["core", "inner_buffer", "outer_buffer"],
        mask_weights=np.ones(3),
    )


def ml_config(tmp_path: Path) -> dict:
    config = load_config("configs/default.yaml")
    config["backend"]["dft"]["engine"] = "cp2k_ml_scf"
    config["cp2k"]["ml_scf"] = {
        "enabled": True,
        "predictor": {"type": "dummy", "options": {}},
        "prediction_dir": str(tmp_path / "predictions"),
    }
    config["cp2k_ml_scf"] = {
        "executable": "cp2k.psmp",
        "model_path": str(tmp_path / "model.pt"),
        "model_digest": None,
        "runtime_dir": str(tmp_path / "runtime"),
        "fallback_to_standard_scf": True,
        "require_final_scf_convergence": True,
        "controller_protocol": "phal-ml-scf-v1",
        "environment": {},
    }
    return config


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
    assert restored["electron_count"] == pytest.approx(10.0)
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


# -- backend hookup -----------------------------------------------------


def test_backend_writes_a_density_guess_next_to_the_input(tmp_path: Path) -> None:
    backend = CP2KMLSCFBackend(config=ml_config(tmp_path))

    written = backend.prepare_inputs(water_region(), tmp_path / "task", task_id="task-0001")

    density_keys = [key for key in written if key.endswith("_ml_density")]
    assert density_keys
    density_path = Path(written[density_keys[0]])
    assert density_path.is_file()
    assert read_density_exchange(density_path)["electron_count"] == pytest.approx(10.0)


def test_backend_records_the_prediction_provenance(tmp_path: Path) -> None:
    backend = CP2KMLSCFBackend(config=ml_config(tmp_path))

    backend.prepare_inputs(water_region(), tmp_path / "task", task_id="task-0001")

    log = tmp_path / "predictions" / "predictions.jsonl"
    assert log.is_file()
    import json

    entry = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[0])
    assert entry["task_id"] == "task-0001"
    assert entry["predictor_type"] == "dummy"
    assert entry["density_file"].endswith(".ml_density")


def test_backend_exports_the_density_path_in_the_environment(tmp_path: Path) -> None:
    config = ml_config(tmp_path)
    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")
    backend = CP2KMLSCFBackend(config=config)
    written = backend.prepare_inputs(water_region(), tmp_path / "task", task_id="task-0001")
    input_file = Path(next(path for key, path in written.items() if key.endswith("_input")))

    request = backend.execution_request(input_file, output_file=tmp_path / "task" / "job.out")

    assert request.environment["PHAL_ML_SCF_ENABLED"] == "1"
    assert Path(request.environment["PHAL_ML_SCF_DENSITY"]).is_file()
    assert request.metadata["ml_density_file"].endswith(".ml_density")


def test_no_density_path_is_exported_when_none_was_written(tmp_path: Path) -> None:
    config = ml_config(tmp_path)
    config["cp2k"]["ml_scf"]["enabled"] = False
    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")
    backend = CP2KMLSCFBackend(config=config)
    written = backend.prepare_inputs(water_region(), tmp_path / "task", task_id="task-0001")
    input_file = Path(next(path for key, path in written.items() if key.endswith("_input")))

    request = backend.execution_request(input_file, output_file=tmp_path / "task" / "job.out")

    assert not any(key.endswith("_ml_density") for key in written)
    assert "PHAL_ML_SCF_DENSITY" not in request.environment
    assert request.metadata["ml_density_file"] is None


def test_a_broken_predictor_does_not_break_input_preparation(tmp_path: Path) -> None:
    config = ml_config(tmp_path)
    config["cp2k"]["ml_scf"]["predictor"]["type"] = "does-not-exist"
    backend = CP2KMLSCFBackend(config=config)

    written = backend.prepare_inputs(water_region(), tmp_path / "task", task_id="task-0001")

    # The CP2K input is still produced; only the optional guess is missing.
    assert any(key.endswith("_input") for key in written)
    assert not any(key.endswith("_ml_density") for key in written)


def cp2k_input_text(written: dict) -> str:
    return Path(next(path for key, path in written.items() if key.endswith("_input"))).read_text(
        encoding="utf-8"
    )


def test_scf_guess_ml_is_written_only_when_a_density_exists(tmp_path: Path) -> None:
    with_guess = CP2KMLSCFBackend(config=ml_config(tmp_path)).prepare_inputs(
        water_region(), tmp_path / "with", task_id="t1"
    )

    disabled = ml_config(tmp_path)
    disabled["cp2k"]["ml_scf"]["enabled"] = False
    without_guess = CP2KMLSCFBackend(config=disabled).prepare_inputs(
        water_region(), tmp_path / "without", task_id="t2"
    )

    assert "SCF_GUESS ML" in cp2k_input_text(with_guess)
    assert "SCF_GUESS ATOMIC" not in cp2k_input_text(with_guess)
    # No density means no reliance on a CP2K-side fallback.
    assert "SCF_GUESS ATOMIC" in cp2k_input_text(without_guess)
    assert "SCF_GUESS ML" not in cp2k_input_text(without_guess)


def test_mixing_method_ml_is_opt_in(tmp_path: Path) -> None:
    default_input = cp2k_input_text(
        CP2KMLSCFBackend(config=ml_config(tmp_path)).prepare_inputs(
            water_region(), tmp_path / "pulay", task_id="t1"
        )
    )

    config = ml_config(tmp_path)
    config["cp2k"]["ml_scf"]["mixing"] = {"method": "ml", "fallback_method": "pulay", "history_length": 6}
    ml_input = cp2k_input_text(
        CP2KMLSCFBackend(config=config).prepare_inputs(water_region(), tmp_path / "ml", task_id="t2")
    )

    assert "METHOD ML" not in default_input
    assert "&MIXING" not in default_input
    assert "METHOD ML" in ml_input
    assert "NBUFFER 6" in ml_input


def test_eps_scf_is_never_touched_by_the_ml_rewrite(tmp_path: Path) -> None:
    config = ml_config(tmp_path)
    config["cp2k"]["ml_scf"]["mixing"] = {"method": "ml"}
    text = cp2k_input_text(
        CP2KMLSCFBackend(config=config).prepare_inputs(water_region(), tmp_path / "task", task_id="t")
    )

    # Convergence stays CP2K's decision, unchanged by any ML keyword.
    assert f"EPS_SCF {config['cp2k']['scf_eps']}" in text
    assert f"MAX_SCF {config['cp2k']['max_scf']}" in text


def test_the_manifest_digest_matches_the_final_input(tmp_path: Path) -> None:
    import json

    from hotspot_al.cp2k.cp2k_result import sha256_file

    written = CP2KMLSCFBackend(config=ml_config(tmp_path)).prepare_inputs(
        water_region(), tmp_path / "task", task_id="t"
    )
    manifest = Path(next(path for key, path in written.items() if key.endswith("_ml_scf_manifest")))
    input_file = Path(next(path for key, path in written.items() if key.endswith("_input")))

    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["input_digest"] == sha256_file(input_file)


def test_the_manifest_carries_capability_provenance(tmp_path: Path) -> None:
    import json

    backend = CP2KMLSCFBackend(config=ml_config(tmp_path))
    written = backend.prepare_inputs(water_region(), tmp_path / "task", task_id="task-0001")
    manifest = Path(next(path for key, path in written.items() if key.endswith("_ml_scf_manifest")))

    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["protocol"] == "phal-ml-scf-v1"
    for key in ("cp2k_version", "cp2k_commit", "ml_model_digest", "ml_scf_enabled"):
        assert key in payload["provenance"]


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
