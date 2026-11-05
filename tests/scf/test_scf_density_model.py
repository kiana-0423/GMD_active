# Ported from PHAL (tests.scf.test_scf_density_model) at 96263b6; MIT License, see NOTICE.
"""Stage 10 tests: the AO density-matrix graph model.

The physical constraints are the point of these tests. Hermiticity and electron
conservation are architectural guarantees here, not things a loss encourages,
so they are asserted on an *untrained* model with random weights -- if they
only held after training, they would not be guarantees.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gmd_active_learning.adapters.dft.scf.schema import SCFElectronicState, SCFStructure
from tools.scf.training.dataset_loader import TARGET_DENSITY_MATRIX, build_examples
from gmd_active_learning.adapters.dft.scf.density.gnn_model import (
    AOBasis,
    DensityGNNConfig,
    DensityModelError,
    LossWeights,
    batch_to_tensors,
    build_density_model,
    channel_electron_counts,
    collate_density_batch,
    density_loss,
    forward_density,
    predicted_electron_counts,
    symmetrize,
)

from tests.scf.fakes.fake_density import spin_polarized_density, symmetric_density
from tests.scf.test_scf_dataset import make_sample

torch = pytest.importorskip("torch")

WATER_BASIS = AOBasis(ao_per_element={1: 5, 8: 13}, basis_set="TEST-DZVP")
WATER_NAO = 23  # 13 (O) + 5 + 5 (H)


def water_structure() -> SCFStructure:
    return SCFStructure(
        atomic_numbers=(8, 1, 1),
        positions=((0.0, 0.0, 0.0), (0.9572, 0.0, 0.0), (-0.24, 0.9266, 0.0)),
        cell=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
    )


def water_state() -> SCFElectronicState:
    return SCFElectronicState(charge=0, multiplicity=1, n_electrons=10, spin_polarized=False)


def water_sample(tmp_path: Path, name: str = "task-0001", *, seed: int = 0, channels: int = 1):
    from gmd_active_learning.adapters.dft.scf.dataset import discover_density_artifacts

    from tests.scf.fakes.fake_density import write_fake_density_artifacts

    directory = tmp_path / name
    if channels == 1:
        density = symmetric_density(WATER_NAO, electrons=8.0, seed=seed)
    else:
        density = spin_polarized_density(WATER_NAO, alpha=4.5, beta=3.5, seed=seed)
    write_fake_density_artifacts(directory, density, overlap=np.eye(WATER_NAO))
    return make_sample(name, artifacts=discover_density_artifacts(directory))


def water_examples(tmp_path: Path, count: int = 4):
    samples = [water_sample(tmp_path, f"task-{index:04d}", seed=index) for index in range(count)]
    return build_examples(samples, target=TARGET_DENSITY_MATRIX)


def hydrogen_sample(tmp_path: Path, name: str = "h2", *, seed: int = 100):
    """A second stoichiometry, so the AO basis is identifiable from the data."""

    from gmd_active_learning.adapters.dft.scf.dataset import discover_density_artifacts

    from tests.scf.fakes.fake_density import write_fake_density_artifacts

    directory = tmp_path / name
    n_ao = WATER_BASIS.n_ao((1, 1))
    write_fake_density_artifacts(
        directory, symmetric_density(n_ao, electrons=2.0, seed=seed), overlap=np.eye(n_ao)
    )
    return make_sample(
        name,
        artifacts=discover_density_artifacts(directory),
        structure=SCFStructure(
            atomic_numbers=(1, 1),
            positions=((0.0, 0.0, 0.0), (0.74, 0.0, 0.0)),
            cell=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
        ),
        electronic_state=SCFElectronicState(
            charge=0, multiplicity=1, n_electrons=2, spin_polarized=False
        ),
    )


def identifiable_examples(tmp_path: Path, count: int = 3):
    """Water plus H2: two independent compositions determine both AO counts."""

    samples = [water_sample(tmp_path, f"task-{index:04d}", seed=index) for index in range(count)]
    samples.append(hydrogen_sample(tmp_path))
    return build_examples(samples, target=TARGET_DENSITY_MATRIX)


def tiny_config() -> DensityGNNConfig:
    return DensityGNNConfig(
        hidden_size=16, n_interactions=2, n_radial_basis=8, max_ao_per_atom=WATER_BASIS.max_ao_per_atom
    )


# -- AO basis -----------------------------------------------------------


def test_ao_basis_counts_orbitals_per_system() -> None:
    assert WATER_BASIS.n_ao([8, 1, 1]) == WATER_NAO
    assert WATER_BASIS.max_ao_per_atom == 13


def test_ao_layout_maps_every_orbital_to_its_atom() -> None:
    atom_of, slot = WATER_BASIS.ao_layout([8, 1, 1])

    assert atom_of.tolist() == [0] * 13 + [1] * 5 + [2] * 5
    assert slot.tolist() == list(range(13)) + list(range(5)) + list(range(5))


def test_unknown_element_is_reported_not_guessed() -> None:
    with pytest.raises(DensityModelError, match="no AO count known"):
        WATER_BASIS.count(26)


def test_basis_is_fitted_from_recorded_dimensions() -> None:
    compositions = [
        ([8, 1, 1], 23),
        ([1, 1], 10),
        ([8, 8], 26),
    ]

    basis = AOBasis.fit(compositions)

    assert basis.ao_per_element == {1: 5, 8: 13}


def test_an_inconsistent_dataset_is_refused() -> None:
    # Mixed basis sets: no single per-element count reproduces both.
    compositions = [([8, 1, 1], 23), ([8, 1, 1], 41), ([1, 1], 10)]

    with pytest.raises(DensityModelError, match="no consistent per-element AO count"):
        AOBasis.fit(compositions)


def test_a_single_stoichiometry_cannot_determine_the_counts() -> None:
    """H2O alone admits both (H=5, O=13) and (H=9, O=5); both give nao=23.

    A fit that picked one would put every block boundary in the wrong place
    while still reproducing the recorded dimension, so it must be refused.
    """

    with pytest.raises(DensityModelError, match="does not determine per-element AO counts"):
        AOBasis.fit([([8, 1, 1], 23), ([8, 1, 1], 23)])


def test_fitting_with_no_systems_is_refused() -> None:
    with pytest.raises(DensityModelError, match="zero systems"):
        AOBasis.fit([])


def test_known_basis_table_is_available() -> None:
    basis = AOBasis.for_basis_set("DZVP-MOLOPT-SR-GTH")

    assert basis.count(1) == 5
    assert basis.count(8) == 13


def test_unknown_basis_set_name_is_reported() -> None:
    with pytest.raises(DensityModelError, match="no AO table"):
        AOBasis.for_basis_set("MADE-UP-BASIS")


def test_basis_round_trips_through_json() -> None:
    restored = AOBasis.from_dict(json.loads(json.dumps(WATER_BASIS.to_dict())))

    assert restored.ao_per_element == WATER_BASIS.ao_per_element
    assert restored.basis_set == "TEST-DZVP"


# -- collation ----------------------------------------------------------


def test_collate_pads_and_masks(tmp_path: Path) -> None:
    examples = water_examples(tmp_path, count=2)

    batch = collate_density_batch(list(examples), WATER_BASIS)

    assert batch.density_target.shape == (2, 1, WATER_NAO, WATER_NAO)
    assert batch.ao_mask.sum() == 2 * WATER_NAO
    assert batch.atom_mask.sum() == 6
    assert batch.has_overlap.tolist() == [1.0, 1.0]


def test_collate_rejects_a_basis_that_disagrees_with_the_density(tmp_path: Path) -> None:
    examples = water_examples(tmp_path, count=1)
    wrong = AOBasis(ao_per_element={1: 4, 8: 13})

    with pytest.raises(DensityModelError, match="implies .* orbitals"):
        collate_density_batch(list(examples), wrong)


def test_collate_rejects_an_empty_batch() -> None:
    with pytest.raises(DensityModelError, match="empty batch"):
        collate_density_batch([], WATER_BASIS)


def test_channel_electron_counts_follow_the_multiplicity() -> None:
    assert channel_electron_counts(10.0, 1, 1) == (10.0,)
    assert channel_electron_counts(9.0, 2, 2) == (5.0, 4.0)


# -- physical constraints -----------------------------------------------


def test_output_is_hermitian_with_untrained_weights(tmp_path: Path) -> None:
    """Hermiticity is architectural, so it must hold before any training."""

    batch = collate_density_batch(list(water_examples(tmp_path, count=3)), WATER_BASIS)
    tensors = batch_to_tensors(batch, torch, torch.device("cpu"))
    model = build_density_model(tiny_config())

    predicted = forward_density(model, tensors)

    asymmetry = float((predicted - predicted.transpose(-1, -2)).abs().max().detach())
    assert asymmetry == pytest.approx(0.0, abs=1e-12)


def test_electrons_are_conserved_with_untrained_weights(tmp_path: Path) -> None:
    batch = collate_density_batch(list(water_examples(tmp_path, count=3)), WATER_BASIS)
    tensors = batch_to_tensors(batch, torch, torch.device("cpu"))
    model = build_density_model(tiny_config())

    predicted = forward_density(model, tensors)
    counts = predicted_electron_counts(predicted, tensors["overlap"]).detach().numpy()

    assert np.allclose(counts, tensors["channel_electrons"].numpy(), atol=1e-4)


def test_electrons_are_conserved_against_a_non_trivial_overlap(tmp_path: Path) -> None:
    examples = list(water_examples(tmp_path, count=1))
    overlap = np.eye(WATER_NAO) + 0.05 * (np.ones((WATER_NAO, WATER_NAO)) - np.eye(WATER_NAO))
    overlap = 0.5 * (overlap + overlap.T)
    object.__setattr__(examples[0], "overlap", overlap)

    batch = collate_density_batch(examples, WATER_BASIS)
    tensors = batch_to_tensors(batch, torch, torch.device("cpu"))
    predicted = forward_density(build_density_model(tiny_config()), tensors)

    counts = predicted_electron_counts(predicted, tensors["overlap"]).detach().numpy()
    assert np.allclose(counts, tensors["channel_electrons"].numpy(), atol=1e-3)


def test_output_is_finite(tmp_path: Path) -> None:
    batch = collate_density_batch(list(water_examples(tmp_path, count=2)), WATER_BASIS)
    tensors = batch_to_tensors(batch, torch, torch.device("cpu"))

    predicted = forward_density(build_density_model(tiny_config()), tensors)

    assert torch.isfinite(predicted).all()


def test_two_spin_channels_are_normalised_independently(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.dataset import discover_density_artifacts

    from tests.scf.fakes.fake_density import write_fake_density_artifacts

    directory = tmp_path / "oh"
    n_ao = WATER_BASIS.n_ao([8, 1])
    write_fake_density_artifacts(
        directory, spin_polarized_density(n_ao, alpha=5.0, beta=4.0), overlap=np.eye(n_ao)
    )
    sample = make_sample(
        "oh",
        artifacts=discover_density_artifacts(directory),
        structure=SCFStructure(
            atomic_numbers=(8, 1),
            positions=((0.0, 0.0, 0.0), (0.97, 0.0, 0.0)),
            cell=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
        ),
        electronic_state=SCFElectronicState(
            charge=0, multiplicity=2, n_electrons=9, spin_polarized=True, spin_channels=2
        ),
    )
    examples = build_examples([sample], target=TARGET_DENSITY_MATRIX)
    assert len(examples) == 1, examples.rejected

    batch = collate_density_batch(list(examples), WATER_BASIS)
    tensors = batch_to_tensors(batch, torch, torch.device("cpu"))
    predicted = forward_density(build_density_model(tiny_config()), tensors)
    counts = predicted_electron_counts(predicted, tensors["overlap"]).detach().numpy()

    assert predicted.shape[1] == 2
    # 9 valence electrons, doublet -> 5 alpha, 4 beta.
    assert counts[0, 0] == pytest.approx(5.0, abs=1e-3)
    assert counts[0, 1] == pytest.approx(4.0, abs=1e-3)


def test_padded_orbitals_stay_exactly_zero(tmp_path: Path) -> None:
    small = build_examples([water_sample(tmp_path, "small")], target=TARGET_DENSITY_MATRIX)[0]

    from gmd_active_learning.adapters.dft.scf.dataset import discover_density_artifacts

    from tests.scf.fakes.fake_density import write_fake_density_artifacts

    directory = tmp_path / "big"
    big_numbers = (8, 8, 1, 1, 1, 1)
    n_ao = WATER_BASIS.n_ao(big_numbers)
    write_fake_density_artifacts(directory, symmetric_density(n_ao, electrons=16.0), overlap=np.eye(n_ao))
    big_sample = make_sample(
        "big",
        artifacts=discover_density_artifacts(directory),
        structure=SCFStructure(
            atomic_numbers=big_numbers,
            positions=tuple((float(index), 0.0, 0.0) for index in range(6)),
            cell=((12.0, 0.0, 0.0), (0.0, 12.0, 0.0), (0.0, 0.0, 12.0)),
        ),
        electronic_state=SCFElectronicState(
            charge=0, multiplicity=1, n_electrons=20, spin_polarized=False
        ),
    )
    big = build_examples([big_sample], target=TARGET_DENSITY_MATRIX)[0]

    batch = collate_density_batch([small, big], WATER_BASIS)
    tensors = batch_to_tensors(batch, torch, torch.device("cpu"))
    predicted = forward_density(build_density_model(tiny_config()), tensors).detach().numpy()

    assert predicted[0, :, WATER_NAO:, :].max() == pytest.approx(0.0)
    assert predicted[0, :, :, WATER_NAO:].max() == pytest.approx(0.0)


def test_symmetrize_is_exact() -> None:
    matrix = torch.randn(2, 1, 5, 5)

    result = symmetrize(matrix)

    assert torch.equal(result, result.transpose(-1, -2))


# -- loss ---------------------------------------------------------------


def test_loss_reports_every_term(tmp_path: Path) -> None:
    batch = collate_density_batch(list(water_examples(tmp_path, count=2)), WATER_BASIS)
    tensors = batch_to_tensors(batch, torch, torch.device("cpu"))
    predicted = forward_density(build_density_model(tiny_config()), tensors)

    total, breakdown = density_loss(predicted, tensors, LossWeights())

    assert set(breakdown) == {"density", "electron", "energy", "idempotency", "total"}
    assert breakdown["density"] > 0.0
    # The output is normalised, so the electron term should be ~0.
    assert breakdown["electron"] == pytest.approx(0.0, abs=1e-6)
    assert float(total.detach()) == pytest.approx(breakdown["total"])


def test_a_perfect_prediction_has_zero_density_loss(tmp_path: Path) -> None:
    batch = collate_density_batch(list(water_examples(tmp_path, count=2)), WATER_BASIS)
    tensors = batch_to_tensors(batch, torch, torch.device("cpu"))

    _total, breakdown = density_loss(tensors["density_target"], tensors, LossWeights())

    assert breakdown["density"] == pytest.approx(0.0, abs=1e-10)


def test_weighting_the_energy_term_without_a_hamiltonian_is_refused(tmp_path: Path) -> None:
    batch = collate_density_batch(list(water_examples(tmp_path, count=1)), WATER_BASIS)
    tensors = batch_to_tensors(batch, torch, torch.device("cpu"))
    predicted = forward_density(build_density_model(tiny_config()), tensors)

    with pytest.raises(DensityModelError, match="no core Hamiltonian"):
        density_loss(predicted, tensors, LossWeights(energy=1.0))


def test_the_energy_term_is_zero_for_a_perfect_prediction(tmp_path: Path) -> None:
    batch = collate_density_batch(list(water_examples(tmp_path, count=1)), WATER_BASIS)
    tensors = batch_to_tensors(batch, torch, torch.device("cpu"))
    hamiltonian = torch.randn(1, WATER_NAO, WATER_NAO)
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.transpose(-1, -2))

    _total, breakdown = density_loss(
        tensors["density_target"], tensors, LossWeights(energy=1.0), hamiltonian=hamiltonian
    )

    assert breakdown["energy"] == pytest.approx(0.0, abs=1e-8)


def test_the_idempotency_term_is_available(tmp_path: Path) -> None:
    batch = collate_density_batch(list(water_examples(tmp_path, count=1)), WATER_BASIS)
    tensors = batch_to_tensors(batch, torch, torch.device("cpu"))
    predicted = forward_density(build_density_model(tiny_config()), tensors)

    _total, breakdown = density_loss(predicted, tensors, LossWeights(idempotency=1.0))

    assert breakdown["idempotency"] > 0.0


# -- training and serving -----------------------------------------------


def test_training_runs_and_serves_a_valid_prediction(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.density.gnn_predictor import AODensityMatrixPredictor
    from tools.scf.training.density_gnn import (
        DensityTrainingConfig,
        train_density_matrix_model,
    )

    train_set = identifiable_examples(tmp_path / "train", count=3)
    validation_set = water_examples(tmp_path / "val", count=2)
    checkpoint = tmp_path / "density_model.pt"

    result = train_density_matrix_model(
        train_set,
        validation_set,
        checkpoint,
        config=DensityTrainingConfig(epochs=2, batch_size=2, hidden_size=16, n_interactions=2),
    )

    assert checkpoint.is_file()
    assert result.best_epoch >= 1
    assert result.selected_on == "validation"
    assert result.basis.ao_per_element == {1: 5, 8: 13}
    assert all(metrics.max_asymmetry == pytest.approx(0.0, abs=1e-10) for metrics in result.history)

    sidecar = json.loads(checkpoint.with_suffix(".pt.json").read_text(encoding="utf-8"))
    assert sidecar["model_digest"].startswith("sha256:")
    assert sidecar["basis"]["ao_per_element"] == {"1": 5, "8": 13}

    predictor = AODensityMatrixPredictor(config={"model_path": str(checkpoint)}).load_model()
    prediction = predictor.predict(water_structure(), water_state())

    assert prediction.values.shape == (1, WATER_NAO, WATER_NAO)
    assert prediction.density_format == "phal-density-matrix-v1"
    assert np.allclose(prediction.values, np.swapaxes(prediction.values, 1, 2), atol=1e-10)

    # Predicted without an overlap, so it is normalised to Tr(P) -- and is
    # therefore *rejected*, because Tr(P) is not the electron count in a
    # non-orthogonal AO basis. This is the contract working, not a regression:
    # a guess is only deployable when S is supplied, which in practice means
    # in-process from CP2K. See docs/ml_scf_architecture.md.
    verdict = predictor.validate_prediction(
        prediction, water_state(), structure=water_structure()
    )
    assert not verdict.valid
    assert "Tr(P S)" in verdict.reasons[0]


def test_prediction_uses_the_supplied_overlap(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.density.gnn_predictor import AODensityMatrixPredictor
    from tools.scf.training.density_gnn import (
        DensityTrainingConfig,
        train_density_matrix_model,
    )

    checkpoint = tmp_path / "m.pt"
    train_density_matrix_model(
        water_examples(tmp_path / "train", count=2),
        None,
        checkpoint,
        basis=WATER_BASIS,
        config=DensityTrainingConfig(epochs=1, batch_size=2, hidden_size=16, n_interactions=2),
    )
    predictor = AODensityMatrixPredictor(config={"model_path": str(checkpoint)}).load_model()

    overlap = np.eye(WATER_NAO) + 0.03 * (np.ones((WATER_NAO, WATER_NAO)) - np.eye(WATER_NAO))
    overlap = 0.5 * (overlap + overlap.T)
    prediction = predictor.predict(water_structure(), water_state(), overlap=overlap)

    counted = float(np.sum(prediction.values[0] * overlap))
    # Deliberate deviation from PHAL (10 = sum(Z), finding F-23): normalised to
    # the valence count the training data carried (8 for water), and the
    # overlap now travels with the prediction so it can be exported.
    assert counted == pytest.approx(8.0, abs=1e-3)
    assert prediction.metadata["normalization"] == "trace_ps"
    assert prediction.overlap is not None
    assert np.allclose(prediction.overlap, overlap)


def test_prediction_rejects_a_mis_shaped_overlap(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.density.prediction import PredictionError
    from gmd_active_learning.adapters.dft.scf.density.gnn_predictor import AODensityMatrixPredictor
    from tools.scf.training.density_gnn import (
        DensityTrainingConfig,
        train_density_matrix_model,
    )

    checkpoint = tmp_path / "m.pt"
    train_density_matrix_model(
        water_examples(tmp_path / "train", count=2),
        None,
        checkpoint,
        basis=WATER_BASIS,
        config=DensityTrainingConfig(epochs=1, batch_size=2, hidden_size=16, n_interactions=2),
    )
    predictor = AODensityMatrixPredictor(config={"model_path": str(checkpoint)}).load_model()

    with pytest.raises(PredictionError, match="overlap has shape"):
        predictor.predict(water_structure(), water_state(), overlap=np.eye(5))


def test_training_refuses_the_wrong_target(tmp_path: Path) -> None:
    from tools.scf.training.dataset_loader import DataError, build_examples
    from tools.scf.training.density_gnn import train_density_matrix_model

    population_set = build_examples([make_sample()])

    with pytest.raises(DataError, match="density training needs target"):
        train_density_matrix_model(population_set, None, tmp_path / "m.pt")


def test_training_without_validation_says_what_it_selected_on(tmp_path: Path) -> None:
    from tools.scf.training.density_gnn import (
        DensityTrainingConfig,
        train_density_matrix_model,
    )

    result = train_density_matrix_model(
        water_examples(tmp_path / "train", count=2),
        None,
        tmp_path / "m.pt",
        basis=WATER_BASIS,
        config=DensityTrainingConfig(epochs=1, batch_size=2, hidden_size=16, n_interactions=2),
    )

    assert result.selected_on == "train"


def test_a_checkpoint_from_a_foreign_version_is_refused(tmp_path: Path) -> None:
    from tools.scf.training.density_gnn import (
        DensityTrainingConfig,
        load_density_checkpoint,
        train_density_matrix_model,
    )

    checkpoint = tmp_path / "m.pt"
    train_density_matrix_model(
        water_examples(tmp_path / "train", count=2),
        None,
        checkpoint,
        basis=WATER_BASIS,
        config=DensityTrainingConfig(epochs=1, batch_size=2, hidden_size=16, n_interactions=2),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["checkpoint_version"] = "something-else"
    torch.save(payload, checkpoint)

    with pytest.raises(DensityModelError, match="checkpoint_version"):
        load_density_checkpoint(checkpoint)


def test_density_gnn_predictor_is_registered() -> None:
    from gmd_active_learning.adapters.dft.scf.density import available_predictors

    assert "density_gnn" in available_predictors()


# -- evaluation ---------------------------------------------------------


def test_atomic_guess_is_block_diagonal_and_conserving() -> None:
    from gmd_active_learning.adapters.dft.scf.density.gnn_predictor import atomic_guess_density

    guess = atomic_guess_density(water_structure(), water_state(), WATER_BASIS)

    assert guess.shape == (1, WATER_NAO, WATER_NAO)
    assert np.trace(guess[0]) == pytest.approx(10.0)
    # Off-site blocks are zero: an atomic guess has no interatomic density.
    assert guess[0, 0, 13] == pytest.approx(0.0)


def test_compare_guesses_scores_atomic_against_ml(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.density.gnn_predictor import AODensityMatrixPredictor, compare_guesses
    from tools.scf.training.density_gnn import (
        DensityTrainingConfig,
        train_density_matrix_model,
    )

    train_set = water_examples(tmp_path / "train", count=4)
    checkpoint = tmp_path / "m.pt"
    train_density_matrix_model(
        train_set,
        None,
        checkpoint,
        basis=WATER_BASIS,
        config=DensityTrainingConfig(epochs=2, batch_size=2, hidden_size=16, n_interactions=2),
    )
    predictor = AODensityMatrixPredictor(config={"model_path": str(checkpoint)}).load_model()
    structures = {
        example.sample_id: (water_structure(), water_state()) for example in train_set
    }

    comparison = compare_guesses(
        predictor,
        list(train_set),
        structures=structures,
        scf_iterations={"a": (24, 15)},
        energies={"a": (-17.0, -17.0)},
        convergence={"a": (True, True)},
    )

    assert comparison.n_systems == 4
    assert comparison.failures == ()
    assert np.isfinite(comparison.atomic_density_mae)
    assert np.isfinite(comparison.ml_density_mae)
    assert comparison.scf_iterations_atomic == pytest.approx(24.0)
    assert comparison.scf_iterations_ml == pytest.approx(15.0)
    assert comparison.energy_difference == pytest.approx(0.0)
    assert comparison.convergence_rate_ml == pytest.approx(1.0)
    payload = comparison.to_dict()
    assert "improved" in payload


def test_compare_guesses_reports_missing_measurements_as_none(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.density.gnn_predictor import AODensityMatrixPredictor, compare_guesses
    from tools.scf.training.density_gnn import (
        DensityTrainingConfig,
        train_density_matrix_model,
    )

    train_set = water_examples(tmp_path / "train", count=2)
    checkpoint = tmp_path / "m.pt"
    train_density_matrix_model(
        train_set,
        None,
        checkpoint,
        basis=WATER_BASIS,
        config=DensityTrainingConfig(epochs=1, batch_size=2, hidden_size=16, n_interactions=2),
    )
    predictor = AODensityMatrixPredictor(config={"model_path": str(checkpoint)}).load_model()

    comparison = compare_guesses(
        predictor,
        list(train_set),
        structures={example.sample_id: (water_structure(), water_state()) for example in train_set},
    )

    # Not measured is reported as None, never as zero.
    assert comparison.scf_iterations_atomic is None
    assert comparison.energy_difference is None
    assert comparison.convergence_rate_ml is None


def test_compare_guesses_requires_something_to_compare(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.density.gnn_predictor import AODensityMatrixPredictor, compare_guesses
    from tools.scf.training.density_gnn import (
        DensityTrainingConfig,
        train_density_matrix_model,
    )

    train_set = water_examples(tmp_path / "train", count=2)
    checkpoint = tmp_path / "m.pt"
    train_density_matrix_model(
        train_set,
        None,
        checkpoint,
        basis=WATER_BASIS,
        config=DensityTrainingConfig(epochs=1, batch_size=2, hidden_size=16, n_interactions=2),
    )
    predictor = AODensityMatrixPredictor(config={"model_path": str(checkpoint)}).load_model()

    with pytest.raises(DensityModelError, match="no example could be compared"):
        compare_guesses(predictor, list(train_set), structures={})


def test_density_error_rejects_a_shape_mismatch() -> None:
    from gmd_active_learning.adapters.dft.scf.density.gnn_predictor import density_error

    with pytest.raises(DensityModelError, match="shape mismatch"):
        density_error(np.zeros((1, 3, 3)), np.zeros((1, 4, 4)))


def test_electron_error_uses_the_overlap() -> None:
    from gmd_active_learning.adapters.dft.scf.density.gnn_predictor import electron_error

    density = np.asarray([[[2.0, 1.0], [1.0, 2.0]]])
    overlap = np.asarray([[1.0, 0.5], [0.5, 1.0]])

    assert electron_error(density, None, 4.0) == pytest.approx(0.0)
    assert electron_error(density, overlap, 5.0) == pytest.approx(0.0)
