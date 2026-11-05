# Ported from PHAL (tests.scf.test_scf_training) at 96263b6; MIT License, see NOTICE.
"""Offline tests for the Stage 3 ML-SCF training framework."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gmd_active_learning.adapters.dft.scf.dataset import SCFDataset
from gmd_active_learning.adapters.dft.scf.schema import (
    SCFDensityArtifacts,
    SCFElectronicState,
    SCFIterationRecord,
    SCFStructure,
    SCFTrajectory,
)
from tools.scf.training.dataset_loader import (
    TARGET_ATOMIC_POPULATION,
    TARGET_DENSITY_MATRIX,
    DataError,
    Normalization,
    atomic_population_target,
    build_examples,
    iterate_batches,
    load_split,
    pad_batch,
)
from tools.scf.benchmark.evaluate import (
    EvaluationError,
    SCFRun,
    compare_densities,
    compare_scf_runs,
    iteration_reduction,
)

from tests.scf.test_scf_dataset import (
    converged_trajectory,
    make_sample,
    water_state,
    water_structure,
)

torch = pytest.importorskip("torch")


def jittered_sample(index: int):
    shift = 0.01 * index
    return make_sample(
        f"task-{index:04d}",
        structure=SCFStructure(
            atomic_numbers=(8, 1, 1),
            positions=(
                (0.0, 0.0, 0.0),
                (0.9572 + shift, 0.0, 0.0),
                (-0.24, 0.9266 - shift, 0.0),
            ),
            cell=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
        ),
        trajectory=SCFTrajectory(
            iterations=tuple(
                SCFIterationRecord(step=step, residual=10.0 ** (-step), energy=-17.0 - 0.01 * step)
                for step in range(1, 3 + index % 4)
            ),
            eps_scf=1.0e-6,
            max_scf=100,
            converged=True,
            initial_guess="ATOMIC",
        ),
    )


# -- loader -------------------------------------------------------------


def test_build_examples_produces_conserving_targets() -> None:
    examples = build_examples([make_sample()])

    assert len(examples) == 1
    example = examples[0]
    assert example.density_target.shape == (1, 3)
    assert example.density_target.sum() == pytest.approx(10.0)
    assert example.n_scf_iterations == 3
    assert example.density_format.startswith("phal-atomic-population")


def test_build_examples_drops_non_converged_runs_by_default() -> None:
    stalled = make_sample(
        "stalled",
        trajectory=SCFTrajectory(
            iterations=(SCFIterationRecord(step=1, residual=1.0, energy=-1.0),),
            eps_scf=1.0e-6,
            max_scf=100,
            converged=False,
            initial_guess="ATOMIC",
        ),
    )

    strict = build_examples([stalled])
    permissive = build_examples([stalled], require_converged=False)

    assert len(strict) == 0
    assert strict.rejected == ["stalled: SCF did not converge"]
    assert len(permissive) == 1


def test_build_examples_reports_invalid_samples_instead_of_crashing() -> None:
    broken = make_sample(
        "broken",
        electronic_state=SCFElectronicState(
            charge=0, multiplicity=1, n_electrons=99, spin_polarized=False
        ),
    )

    examples = build_examples([broken, make_sample("good")])

    assert [example.sample_id for example in examples] == ["good"]
    assert "broken" in examples.rejected[0]


def test_density_matrix_target_requires_an_artifact() -> None:
    examples = build_examples([make_sample()], target=TARGET_DENSITY_MATRIX)

    assert len(examples) == 0
    assert "no density matrix is recorded" in examples.rejected[0]


def test_density_matrix_target_reads_an_npz_artifact(tmp_path: Path) -> None:
    """A prediction export writes .npz; recording writes .npy. Both must load."""

    matrix = np.zeros((1, 4, 4))
    np.fill_diagonal(matrix[0], 2.5)
    path = tmp_path / "dm.npz"
    np.savez_compressed(path, values=matrix)
    # Deviation from PHAL: an overlap is now required (Tr(P S) is never
    # approximated by Tr(P)); the identity overlap keeps the same count.
    overlap_path = tmp_path / "overlap.npy"
    np.save(overlap_path, np.eye(4))
    sample = make_sample(
        artifacts=SCFDensityArtifacts(
            density_matrix_path=str(path),
            density_format="phal-density-matrix-v1",
            density_electron_count=10.0,
            overlap_path=str(overlap_path),
        )
    )

    examples = build_examples([sample], target=TARGET_DENSITY_MATRIX)

    assert len(examples) == 1, examples.rejected
    assert examples[0].density_target.shape == (1, 4, 4)


def test_unsupported_target_is_rejected() -> None:
    with pytest.raises(DataError, match="unsupported target"):
        build_examples([make_sample()], target="cube")


def test_atomic_population_target_matches_the_spin_channels() -> None:
    open_shell = make_sample(
        structure=SCFStructure(
            atomic_numbers=(8, 1),
            positions=((0.0, 0.0, 0.0), (0.97, 0.0, 0.0)),
            cell=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
        ),
        electronic_state=SCFElectronicState(
            charge=0, multiplicity=2, n_electrons=9, spin_polarized=True, spin_channels=2
        ),
    )

    target = atomic_population_target(open_shell)

    assert target.shape == (2, 2)
    assert target[0].sum() == pytest.approx(5.0)
    assert target[1].sum() == pytest.approx(4.0)


def test_load_split_reads_a_dataset_directory(tmp_path: Path) -> None:
    dataset = SCFDataset.create(tmp_path / "ds")
    for index in range(3):
        dataset.add_sample(jittered_sample(index), split="train")

    examples = load_split(dataset.root, "train")

    assert len(examples) == 3


def test_normalization_survives_constant_features() -> None:
    matrix = np.tile(np.asarray([1.0, 2.0, 3.0]), (5, 1))

    normalization = Normalization.fit(matrix)
    applied = normalization.apply(matrix)

    assert np.isfinite(applied).all()
    assert np.allclose(applied, 0.0)


def test_normalization_round_trips_through_json() -> None:
    normalization = Normalization.fit(np.random.default_rng(0).normal(size=(8, 4)))

    restored = Normalization.from_dict(json.loads(json.dumps(normalization.to_dict())))

    assert np.allclose(restored.mean, normalization.mean)
    assert np.allclose(restored.scale, normalization.scale)


def test_pad_batch_masks_padded_atoms() -> None:
    small = build_examples([make_sample("small")])[0]
    big = build_examples(
        [
            make_sample(
                "big",
                structure=SCFStructure(
                    atomic_numbers=(8, 1, 1, 8, 1, 1),
                    positions=tuple((float(i), 0.0, 0.0) for i in range(6)),
                    cell=((12.0, 0.0, 0.0), (0.0, 12.0, 0.0), (0.0, 0.0, 12.0)),
                ),
                electronic_state=SCFElectronicState(
                    charge=0, multiplicity=1, n_electrons=20, spin_polarized=False
                ),
            )
        ]
    )[0]

    batch = pad_batch([small, big])

    assert batch["per_atom_features"].shape[1] == 6
    assert batch["atom_mask"][0].tolist() == [1, 1, 1, 0, 0, 0]
    assert batch["density_target"][0, 0, 3:].tolist() == [0.0, 0.0, 0.0]


def test_pad_batch_rejects_an_empty_batch() -> None:
    with pytest.raises(DataError, match="zero examples"):
        pad_batch([])


def test_iterate_batches_is_seeded() -> None:
    examples = build_examples([jittered_sample(index) for index in range(6)])

    first = [batch["n_scf_iterations"].tolist() for batch in iterate_batches(examples, 2, shuffle=True, seed=7)]
    second = [batch["n_scf_iterations"].tolist() for batch in iterate_batches(examples, 2, shuffle=True, seed=7)]

    assert first == second


# -- model and training -------------------------------------------------


def test_model_conserves_electrons_by_construction() -> None:
    from tools.scf.experimental.atomic_population_mlp.model import ModelConfig, build_model

    examples = build_examples([jittered_sample(index) for index in range(4)])
    batch = pad_batch(list(examples))
    model = build_model(ModelConfig.from_features(hidden_size=16, n_layers=2))

    output = model(
        torch.as_tensor(batch["system_features"]),
        torch.as_tensor(batch["per_atom_features"]),
        torch.as_tensor(batch["atom_mask"]),
    )
    fractions = output["fractions"].detach().numpy()

    assert np.allclose(fractions.sum(axis=-1), 1.0, atol=1e-5)
    assert np.isfinite(fractions).all()


def test_model_gives_padded_atoms_zero_share() -> None:
    from tools.scf.experimental.atomic_population_mlp.model import ModelConfig, build_model

    small = build_examples([make_sample("small")])[0]
    big = build_examples(
        [
            make_sample(
                "big",
                structure=SCFStructure(
                    atomic_numbers=(8, 1, 1, 8, 1, 1),
                    positions=tuple((float(i), 0.0, 0.0) for i in range(6)),
                    cell=((12.0, 0.0, 0.0), (0.0, 12.0, 0.0), (0.0, 0.0, 12.0)),
                ),
                electronic_state=SCFElectronicState(
                    charge=0, multiplicity=1, n_electrons=20, spin_polarized=False
                ),
            )
        ]
    )[0]
    batch = pad_batch([small, big])
    model = build_model(ModelConfig.from_features(hidden_size=16, n_layers=2))

    output = model(
        torch.as_tensor(batch["system_features"]),
        torch.as_tensor(batch["per_atom_features"]),
        torch.as_tensor(batch["atom_mask"]),
    )
    fractions = output["fractions"].detach().numpy()

    assert np.allclose(fractions[0, :, 3:], 0.0)


def test_training_runs_and_writes_a_usable_checkpoint(tmp_path: Path) -> None:
    from tools.scf.experimental.atomic_population_mlp.model import TorchDensityPredictor
    from tools.scf.experimental.atomic_population_mlp.train import TrainingConfig, train_density_model

    train_set = build_examples([jittered_sample(index) for index in range(8)])
    validation_set = build_examples([jittered_sample(index) for index in range(8, 11)])
    checkpoint = tmp_path / "model.pt"

    result = train_density_model(
        train_set,
        validation_set,
        checkpoint,
        config=TrainingConfig(epochs=3, batch_size=4, hidden_size=16, n_layers=2),
    )

    assert checkpoint.is_file()
    assert result.best_epoch >= 1
    assert len(result.history) == 6
    assert {metrics.split for metrics in result.history} == {"train", "validation"}
    assert np.isfinite(result.best_validation_density_mae)

    sidecar = json.loads(checkpoint.with_suffix(".pt.json").read_text(encoding="utf-8"))
    assert sidecar["model_digest"].startswith("sha256:")
    assert sidecar["feature_version"].startswith("phal-scf-features")

    predictor = TorchDensityPredictor(config={"model_path": str(checkpoint)}).load_model()
    prediction = predictor.predict(water_structure(), water_state())

    assert prediction.values.shape == (1, 3)
    # Valence count (8), not PHAL's sum(Z) = 10 (finding F-23).
    assert prediction.values.sum() == pytest.approx(8.0, abs=1e-4)
    assert predictor.validate_prediction(prediction, water_state(), structure=water_structure()).valid
    assert prediction.model_digest == sidecar["model_digest"]
    assert prediction.predictor_type == "torch"


def test_training_refuses_an_empty_split(tmp_path: Path) -> None:
    from tools.scf.training.dataset_loader import SCFExampleSet
    from tools.scf.experimental.atomic_population_mlp.train import train_density_model

    with pytest.raises(DataError, match="empty training split"):
        train_density_model(SCFExampleSet(), None, tmp_path / "model.pt")


def test_torch_predictor_requires_a_model_path() -> None:
    from gmd_active_learning.adapters.dft.scf.density.prediction import PredictionError
    from tools.scf.experimental.atomic_population_mlp.model import TorchDensityPredictor

    with pytest.raises(PredictionError, match="model_path"):
        TorchDensityPredictor(config={})


def test_checkpoint_from_a_different_feature_version_is_rejected(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.density.prediction import PredictionError
    from tools.scf.experimental.atomic_population_mlp.model import load_checkpoint

    payload = torch.load(_train_tiny_checkpoint(tmp_path), map_location="cpu", weights_only=False)
    payload["feature_version"] = "phal-scf-features-v0"
    stale = tmp_path / "stale.pt"
    torch.save(payload, stale)

    with pytest.raises(PredictionError, match="feature_version"):
        load_checkpoint(stale)


def test_torch_predictor_is_registered_when_torch_is_available() -> None:
    from gmd_active_learning.adapters.dft.scf.density import available_predictors

    assert "torch" in available_predictors()


def _train_tiny_checkpoint(tmp_path: Path) -> Path:
    from tools.scf.experimental.atomic_population_mlp.train import TrainingConfig, train_density_model

    checkpoint = tmp_path / "tiny.pt"
    train_density_model(
        build_examples([jittered_sample(index) for index in range(4)]),
        None,
        checkpoint,
        config=TrainingConfig(epochs=1, batch_size=2, hidden_size=8, n_layers=2),
    )
    return checkpoint


# -- evaluation ---------------------------------------------------------


def test_compare_densities_reports_elementwise_error() -> None:
    metrics = compare_densities(np.asarray([[1.0, 2.0]]), np.asarray([[1.5, 2.0]]))

    assert metrics["density_mae"] == pytest.approx(0.25)
    assert metrics["density_max_error"] == pytest.approx(0.5)
    assert metrics["electron_error"] == pytest.approx(0.5)


def test_compare_densities_rejects_a_shape_mismatch() -> None:
    with pytest.raises(EvaluationError, match="shape mismatch"):
        compare_densities(np.zeros((1, 2)), np.zeros((1, 3)))


def make_runs(label: str, iterations: dict[str, int], **kwargs) -> list[SCFRun]:
    return [
        SCFRun(label=label, system_id=system_id, scf_iterations=count, converged=True, **kwargs)
        for system_id, count in iterations.items()
    ]


def test_compare_scf_runs_reports_the_iteration_saving() -> None:
    baseline = make_runs("atomic", {"a": 20, "b": 30}, wall_time_seconds=10.0, final_energy=-17.0)
    candidate = make_runs("ml", {"a": 12, "b": 18}, wall_time_seconds=7.0, final_energy=-17.0)

    comparison = compare_scf_runs(baseline, candidate)

    assert comparison.n_systems == 2
    assert comparison.baseline_iterations == pytest.approx(25.0)
    assert comparison.ml_iterations == pytest.approx(15.0)
    assert comparison.iteration_reduction == pytest.approx(10.0)
    assert comparison.speedup == pytest.approx(25.0 / 15.0)
    assert comparison.runtime_speedup == pytest.approx(20.0 / 14.0)
    assert comparison.energy_error == pytest.approx(0.0)
    assert comparison.converged


def test_compare_scf_runs_excludes_non_converged_candidates() -> None:
    baseline = make_runs("atomic", {"a": 20, "b": 30})
    candidate = [
        SCFRun(label="ml", system_id="a", scf_iterations=12, converged=True),
        SCFRun(label="ml", system_id="b", scf_iterations=100, converged=False),
    ]

    comparison = compare_scf_runs(baseline, candidate)

    assert comparison.n_systems == 1
    assert comparison.ml_iterations == pytest.approx(12.0)
    assert comparison.candidate_success_rate == pytest.approx(0.5)
    assert not comparison.converged


def test_compare_scf_runs_reports_the_fallback_rate() -> None:
    baseline = make_runs("atomic", {"a": 20, "b": 20})
    candidate = [
        SCFRun(label="ml", system_id="a", scf_iterations=15, converged=True, fallback_used=True),
        SCFRun(label="ml", system_id="b", scf_iterations=14, converged=True),
    ]

    comparison = compare_scf_runs(baseline, candidate)

    assert comparison.fallback_rate == pytest.approx(0.5)


def test_compare_scf_runs_requires_shared_systems() -> None:
    with pytest.raises(EvaluationError, match="share no system ids"):
        compare_scf_runs(make_runs("atomic", {"a": 10}), make_runs("ml", {"z": 10}))


def test_compare_scf_runs_requires_a_comparable_pair() -> None:
    baseline = make_runs("atomic", {"a": 10})
    candidate = [SCFRun(label="ml", system_id="a", scf_iterations=100, converged=False)]

    with pytest.raises(EvaluationError, match="nothing can be compared"):
        compare_scf_runs(baseline, candidate)


def test_iteration_reduction_summarises_paired_counts() -> None:
    summary = iteration_reduction([20, 30, 10], [15, 35, 5])

    # (20-15) + (30-35) + (10-5) = 5 saved iterations over 3 systems.
    assert summary["mean_reduction"] == pytest.approx(5.0 / 3.0)
    assert summary["fraction_improved"] == pytest.approx(2 / 3)
    assert summary["fraction_worse"] == pytest.approx(1 / 3)


def test_iteration_reduction_rejects_unpaired_input() -> None:
    with pytest.raises(EvaluationError, match="lengths differ"):
        iteration_reduction([1, 2], [1])


def test_scf_run_round_trips_through_json() -> None:
    run = SCFRun(
        label="ml",
        system_id="a",
        scf_iterations=12,
        converged=True,
        wall_time_seconds=3.5,
        final_energy=-17.2,
        fallback_used=True,
    )

    restored = SCFRun.from_dict(json.loads(json.dumps(run.to_dict())))

    assert restored == run


def test_converged_trajectory_helper_is_shared_with_stage_one() -> None:
    # Guards against the Stage 1 fixtures drifting away from Stage 3's loader.
    assert converged_trajectory().n_iterations == 3
