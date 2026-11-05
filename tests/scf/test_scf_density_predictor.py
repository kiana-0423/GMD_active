# Ported from PHAL (tests.scf.test_scf_density_predictor) at 96263b6; MIT License, see NOTICE.
"""Offline tests for the Stage 2 ML density prediction framework."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gmd_active_learning.adapters.dft.scf.density import (
    DENSITY_FORMAT_ATOMIC_POPULATION,
    DENSITY_FORMAT_DENSITY_MATRIX,
    DummyDensityPredictor,
    PredictionError,
    SCFDensityPrediction,
    SCFDensityPredictor,
    available_predictors,
    create_predictor,
    ml_scf_enabled,
    predictor_spec,
    read_prediction,
    record_prediction,
    register_predictor,
    resolve_predictor_class,
    write_prediction,
)
from gmd_active_learning.adapters.dft.scf.density.registry import (
    PredictorNotFoundError,
    unregister_predictor,
    write_predictor_manifest,
)
from gmd_active_learning.adapters.dft.scf.schema import SCFElectronicState, SCFStructure

from tests.scf.test_scf_dataset import water_state, water_structure


def open_shell_state() -> SCFElectronicState:
    # OH radical: 8 + 1 = 9 electrons, doublet.
    return SCFElectronicState(
        charge=0, multiplicity=2, n_electrons=9, spin_polarized=True, spin_channels=2
    )


def open_shell_structure() -> SCFStructure:
    return SCFStructure(
        atomic_numbers=(8, 1),
        positions=((0.0, 0.0, 0.0), (0.97, 0.0, 0.0)),
        cell=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
    )


# -- dummy predictor ----------------------------------------------------


def test_dummy_predictor_conserves_the_electron_count() -> None:
    predictor = DummyDensityPredictor().load_model()

    prediction = predictor.predict(water_structure(), water_state())

    assert prediction.values.shape == (1, 3)
    # Deliberate deviation from PHAL (10 = sum(Z), finding F-23): the density
    # holds the GTH valence count, 6 + 1 + 1 = 8 for water.
    assert prediction.values.sum() == pytest.approx(8.0)
    assert prediction.electron_count == pytest.approx(8.0)
    assert prediction.density_format == DENSITY_FORMAT_ATOMIC_POPULATION
    assert predictor.validate_prediction(prediction, water_state(), structure=water_structure()).valid


def test_dummy_predictor_splits_spin_channels_by_multiplicity() -> None:
    predictor = DummyDensityPredictor().load_model()

    prediction = predictor.predict(open_shell_structure(), open_shell_state())

    alpha, beta = prediction.values
    assert prediction.values.shape == (2, 2)
    # Valence count 6 + 1 = 7 (PHAL used the all-electron 9; finding F-23).
    assert alpha.sum() == pytest.approx(4.0)
    assert beta.sum() == pytest.approx(3.0)
    assert prediction.values.sum() == pytest.approx(7.0)


def test_dummy_predictor_is_deterministic() -> None:
    first = DummyDensityPredictor().load_model().predict(water_structure(), water_state())
    second = DummyDensityPredictor().load_model().predict(water_structure(), water_state())

    assert np.array_equal(first.values, second.values)
    assert first.model_digest == second.model_digest
    assert first.structure_hash == second.structure_hash


def test_dummy_predictor_loads_lazily_on_first_prediction() -> None:
    predictor = DummyDensityPredictor()

    assert not predictor.is_loaded
    predictor.predict(water_structure(), water_state())
    assert predictor.is_loaded


def test_dummy_predictor_rejects_an_empty_structure() -> None:
    empty = SCFStructure(
        atomic_numbers=(),
        positions=(),
        cell=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
    )
    state = SCFElectronicState(charge=0, multiplicity=1, n_electrons=0, spin_polarized=False)

    with pytest.raises(PredictionError, match="no atoms"):
        DummyDensityPredictor().predict(empty, state)


def test_valence_weighting_changes_the_distribution() -> None:
    valence = DummyDensityPredictor().predict(water_structure(), water_state())
    nuclear = DummyDensityPredictor(
        config={"options": {"use_valence_electrons": False}}
    ).predict(water_structure(), water_state())

    assert not np.allclose(valence.values, nuclear.values)
    assert nuclear.values.sum() == pytest.approx(10.0)
    assert valence.model_digest != nuclear.model_digest


# -- validation ---------------------------------------------------------


def base_prediction(**overrides: Any) -> SCFDensityPrediction:
    fields: dict[str, Any] = {
        "model_name": "test",
        "model_version": "1",
        "model_digest": "sha256:0",
        "structure_hash": "sha256:not-the-water-hash",
        "electron_count": 10.0,
        "spin_channels": 1,
        "density_format": DENSITY_FORMAT_ATOMIC_POPULATION,
        "values": np.full((1, 3), 10.0 / 3.0),
    }
    fields.update(overrides)
    return SCFDensityPrediction(**fields)


def test_validation_rejects_non_finite_values() -> None:
    predictor = DummyDensityPredictor()
    values = np.full((1, 3), 10.0 / 3.0)
    values[0, 1] = np.nan

    result = predictor.validate_prediction(base_prediction(values=values), water_state())

    assert not result.valid
    assert any("non-finite" in reason for reason in result.reasons)


def test_validation_rejects_a_wrong_electron_count() -> None:
    predictor = DummyDensityPredictor()

    result = predictor.validate_prediction(
        base_prediction(values=np.full((1, 3), 4.0)), water_state()
    )

    assert not result.valid
    assert any("integrates to" in reason for reason in result.reasons)


def test_validation_rejects_a_spin_channel_mismatch() -> None:
    predictor = DummyDensityPredictor()

    result = predictor.validate_prediction(
        base_prediction(spin_channels=2, values=np.full((2, 3), 10.0 / 6.0)),
        water_state(),
    )

    assert not result.valid
    assert any("spin channel" in reason for reason in result.reasons)


def test_validation_rejects_negative_populations() -> None:
    predictor = DummyDensityPredictor()
    values = np.asarray([[6.0, 6.0, -2.0]])

    result = predictor.validate_prediction(base_prediction(values=values), water_state())

    assert not result.valid
    assert any("non-negative" in reason for reason in result.reasons)


def test_validation_rejects_a_structure_hash_mismatch() -> None:
    predictor = DummyDensityPredictor()

    result = predictor.validate_prediction(
        base_prediction(), water_state(), structure=water_structure()
    )

    assert not result.valid
    assert any("structure_hash" in reason for reason in result.reasons)


def test_validation_rejects_an_asymmetric_density_matrix() -> None:
    predictor = DummyDensityPredictor()
    matrix = np.zeros((1, 3, 3))
    np.fill_diagonal(matrix[0], 10.0 / 3.0)
    matrix[0, 0, 1] = 0.5

    result = predictor.validate_prediction(
        base_prediction(density_format=DENSITY_FORMAT_DENSITY_MATRIX, values=matrix),
        water_state(),
    )

    assert not result.valid
    assert any("symmetric" in reason for reason in result.reasons)


def test_validation_accepts_a_symmetric_density_matrix() -> None:
    predictor = DummyDensityPredictor()
    matrix = np.zeros((1, 3, 3))
    np.fill_diagonal(matrix[0], 10.0 / 3.0)

    result = predictor.validate_prediction(
        base_prediction(
            density_format=DENSITY_FORMAT_DENSITY_MATRIX,
            values=matrix,
            # An orthonormal overlap, which is what makes this matrix's
            # electron count equal its trace. Ne = Tr(P S) in general.
            overlap=np.eye(3),
        ),
        water_state(),
    )

    assert result.valid, result.reasons


def test_validation_rejects_a_density_matrix_with_no_overlap() -> None:
    """Without S the count Tr(P S) is not evaluable, so it is not guessed."""

    predictor = DummyDensityPredictor()
    matrix = np.zeros((1, 3, 3))
    np.fill_diagonal(matrix[0], 10.0 / 3.0)

    result = predictor.validate_prediction(
        base_prediction(density_format=DENSITY_FORMAT_DENSITY_MATRIX, values=matrix, overlap=None),
        water_state(),
    )

    assert not result.valid
    assert any("Tr(P S)" in reason for reason in result.reasons)


def test_validation_uses_the_overlap_not_the_bare_trace() -> None:
    """A non-orthonormal S must change the verdict, or it is not being used."""

    predictor = DummyDensityPredictor()
    matrix = np.zeros((1, 3, 3))
    np.fill_diagonal(matrix[0], 10.0 / 3.0)
    # Tr(P) is still 10, but Tr(P S) is 15 with this overlap, so a prediction
    # declaring 10 electrons is now wrong and must be rejected.
    overlap = np.eye(3) * 1.5

    result = predictor.validate_prediction(
        base_prediction(
            density_format=DENSITY_FORMAT_DENSITY_MATRIX, values=matrix, overlap=overlap
        ),
        water_state(),
    )

    assert not result.valid
    assert any("integrates to" in reason for reason in result.reasons)


def test_validation_rejects_a_prediction_without_values() -> None:
    predictor = DummyDensityPredictor()

    result = predictor.validate_prediction(base_prediction(values=None), water_state())

    assert not result.valid
    assert "prediction carries no density values" in result.reasons


def test_unsupported_density_format_is_rejected_at_construction() -> None:
    with pytest.raises(PredictionError, match="unsupported density_format"):
        base_prediction(density_format="cube")


# -- persistence and provenance -----------------------------------------


def test_prediction_round_trips_through_disk(tmp_path: Path) -> None:
    predictor = DummyDensityPredictor().load_model()
    prediction = predictor.predict(water_structure(), water_state())

    path = predictor.save_prediction(prediction, tmp_path / "guess.json")
    restored = read_prediction(path)

    assert restored.model_digest == prediction.model_digest
    assert restored.structure_hash == prediction.structure_hash
    assert restored.spin_channels == prediction.spin_channels
    assert np.allclose(restored.values, prediction.values)
    assert restored.output_path == path


def test_written_prediction_records_full_provenance(tmp_path: Path) -> None:
    prediction = DummyDensityPredictor().load_model().predict(water_structure(), water_state())

    path = write_prediction(prediction, tmp_path / "guess.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert set(payload["provenance"]) == {
        "model_digest",
        "structure_hash",
        "timestamp",
        "predictor_type",
    }
    assert payload["provenance"]["predictor_type"] == "dummy"
    assert payload["provenance"]["timestamp"].endswith("+00:00")
    assert payload["model_name"] == "dummy-atomic-population"


def test_tampered_values_are_detected_on_read(tmp_path: Path) -> None:
    prediction = DummyDensityPredictor().load_model().predict(water_structure(), water_state())
    path = write_prediction(prediction, tmp_path / "guess.json")
    values_path = Path(json.loads(path.read_text(encoding="utf-8"))["values_path"])
    np.savez_compressed(values_path, values=np.zeros((1, 3)), density_format=np.asarray("x"))

    with pytest.raises(PredictionError, match="digest mismatch"):
        read_prediction(path)


def test_record_prediction_appends_to_the_provenance_log(tmp_path: Path) -> None:
    predictor = DummyDensityPredictor().load_model()
    prediction = predictor.predict(water_structure(), water_state())

    record_prediction(prediction, output_dir=tmp_path, task_id="task-0001")
    record_prediction(prediction, output_dir=tmp_path, task_id="task-0002")

    lines = (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert entry["task_id"] == "task-0001"
    assert entry["predictor_type"] == "dummy"
    assert entry["model_digest"] == prediction.model_digest
    assert entry["structure_hash"] == prediction.structure_hash
    assert "timestamp" in entry


def test_predictor_manifest_records_the_model_identity(tmp_path: Path) -> None:
    predictor = DummyDensityPredictor().load_model()

    path = write_predictor_manifest(predictor, tmp_path / "manifest.json", extra={"task": "t"})
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["predictor_type"] == "dummy"
    assert payload["model_digest"] == predictor.model_digest
    assert payload["task"] == "t"


# -- registry and configuration -----------------------------------------


def test_dummy_is_registered_by_default() -> None:
    assert "dummy" in available_predictors()
    assert resolve_predictor_class("dummy") is DummyDensityPredictor
    assert resolve_predictor_class("DUMMY") is DummyDensityPredictor


def test_create_predictor_reads_the_config_block() -> None:
    config = {"cp2k": {"ml_scf": {"predictor": {"type": "dummy", "options": {}}}}}

    predictor = create_predictor(config)

    assert isinstance(predictor, DummyDensityPredictor)
    assert predictor.is_loaded


def test_predictor_spec_defaults_to_dummy() -> None:
    spec = predictor_spec({})

    assert spec.type == "dummy"
    assert spec.model_path is None
    assert spec.options == {}


def test_unknown_predictor_type_is_reported_clearly() -> None:
    with pytest.raises(PredictorNotFoundError, match="unknown predictor"):
        create_predictor({"cp2k": {"ml_scf": {"predictor": {"type": "nope"}}}})


def test_register_predictor_rejects_a_non_predictor() -> None:
    with pytest.raises(TypeError):
        register_predictor("bogus", dict)  # type: ignore[arg-type]


def test_registered_predictor_can_be_created_and_removed() -> None:
    class Recording(DummyDensityPredictor):
        predictor_type = "recording"

    register_predictor("recording", Recording)
    try:
        assert isinstance(create_predictor({"cp2k": {"ml_scf": {"predictor": {"type": "recording"}}}}), Recording)
    finally:
        unregister_predictor("recording")
    assert "recording" not in available_predictors()


def test_dotted_path_resolves_an_out_of_tree_predictor() -> None:
    resolved = resolve_predictor_class(
        "gmd_active_learning.adapters.dft.scf.density.predictor:DummyDensityPredictor"
    )

    assert resolved is DummyDensityPredictor


def test_ml_scf_is_disabled_for_the_standard_engine() -> None:
    assert not ml_scf_enabled({"backend": {"dft": {"engine": "cp2k"}}})
    assert ml_scf_enabled({"backend": {"dft": {"engine": "cp2k_ml_scf"}}})
    assert not ml_scf_enabled(
        {"backend": {"dft": {"engine": "cp2k_ml_scf"}}, "cp2k": {"ml_scf": {"enabled": False}}}
    )


def test_abstract_predictor_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        SCFDensityPredictor()  # type: ignore[abstract]
