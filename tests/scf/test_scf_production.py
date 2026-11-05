# Ported from PHAL (tests.scf.test_scf_production) at 96263b6; MIT License, see NOTICE.
"""Stage 12-14 tests: mixer, deployment, benchmark acceptance, production fallback."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.scf.benchmark.modes import (
    MODE_ML_FULL,
    MODE_ML_GUESS,
    MODE_NATIVE,
    MODE_RESTART,
    AcceptanceCriteria,
    BenchmarkError,
    run_benchmark,
    write_benchmark_results,
)
from tools.scf.benchmark.evaluate import SCFRun

torch = pytest.importorskip("torch")


# -- Stage 12: the neural mixer -----------------------------------------


def test_mixer_coefficients_sum_to_one() -> None:
    """Affine combinations are what make the proposal electron-conserving."""

    from tools.scf.experimental.neural_mixer.model import MixerConfig, build_mixer_model

    model = build_mixer_model(MixerConfig(hidden_size=16))
    history = torch.randn(4, 1, 12)
    residual = torch.randn(4, 1, 12) * 0.01

    with torch.no_grad():
        features = model.features(history, residual)
        logits = model.coefficient_head(model.encoder(features)).squeeze(-1)
        coefficients = torch.softmax(logits, dim=0)

    assert float(coefficients.sum()) == pytest.approx(1.0, abs=1e-6)


def test_mixer_conserves_electrons_when_the_history_does() -> None:
    from tools.scf.experimental.neural_mixer.model import MixerConfig, build_mixer_model

    model = build_mixer_model(MixerConfig(hidden_size=16))
    # Every history entry holds exactly 10 electrons; the residuals sum to zero.
    history = torch.full((4, 1, 20), 10.0 / 20)
    residual = torch.randn(4, 1, 20)
    residual = residual - residual.mean(dim=2, keepdim=True)

    with torch.no_grad():
        proposal = model(history, residual)

    assert float(proposal.sum()) == pytest.approx(10.0, abs=1e-4)


def test_mixer_output_has_the_shape_cp2k_expects() -> None:
    from tools.scf.experimental.neural_mixer.model import MixerConfig, build_mixer_model

    model = build_mixer_model(MixerConfig(hidden_size=8))

    with torch.no_grad():
        output = model(torch.randn(5, 2, 9), torch.randn(5, 2, 9))

    assert tuple(output.shape) == (2, 9)


def test_pulay_reference_reproduces_a_zero_residual_combination() -> None:
    from tools.scf.experimental.neural_mixer.model import pulay_reference

    # Two residuals that cancel at equal weights; DIIS should find them.
    residual = torch.stack(
        [torch.full((1, 6), 1.0, dtype=torch.float64), torch.full((1, 6), -1.0, dtype=torch.float64)]
    )
    history = torch.stack(
        [torch.zeros((1, 6), dtype=torch.float64), torch.full((1, 6), 2.0, dtype=torch.float64)]
    )

    result = pulay_reference(history, residual)

    assert np.allclose(result.numpy(), 1.0, atol=1e-6)


def test_mixer_checkpoint_round_trips(tmp_path: Path) -> None:
    from tools.scf.experimental.neural_mixer.model import (
        MixerConfig,
        build_mixer_model,
        load_mixer_checkpoint,
        save_mixer_checkpoint,
    )

    config = MixerConfig(hidden_size=8, max_history=5)
    model = build_mixer_model(config)
    save_mixer_checkpoint(tmp_path / "mixer.pt", model=model, config=config)

    restored, restored_config = load_mixer_checkpoint(tmp_path / "mixer.pt")

    assert restored_config.max_history == 5
    history, residual = torch.randn(3, 1, 7), torch.randn(3, 1, 7)
    with torch.no_grad():
        assert torch.allclose(restored(history, residual), model(history, residual))


def test_a_foreign_mixer_checkpoint_is_refused(tmp_path: Path) -> None:
    from tools.scf.experimental.neural_mixer.model import (
        MixerConfig,
        MixerModelError,
        build_mixer_model,
        load_mixer_checkpoint,
        save_mixer_checkpoint,
    )

    config = MixerConfig(hidden_size=8)
    save_mixer_checkpoint(tmp_path / "m.pt", model=build_mixer_model(config), config=config)
    payload = torch.load(tmp_path / "m.pt", map_location="cpu", weights_only=False)
    payload["checkpoint_version"] = "other"
    torch.save(payload, tmp_path / "m.pt")

    with pytest.raises(MixerModelError, match="checkpoint_version"):
        load_mixer_checkpoint(tmp_path / "m.pt")


# -- Stage 12: deployment -----------------------------------------------


def test_mixer_exports_to_torchscript_and_answers_cp2ks_call(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.deploy import MIXER_MODEL_ENV, verify_deployed
    from tools.scf.experimental.neural_mixer.export import export_mixer
    from tools.scf.experimental.neural_mixer.model import (
        MixerConfig,
        build_mixer_model,
        save_mixer_checkpoint,
    )

    config = MixerConfig(hidden_size=8)
    save_mixer_checkpoint(tmp_path / "mixer.pt", model=build_mixer_model(config), config=config)

    deployed = export_mixer(tmp_path / "mixer.pt", tmp_path / "mixer.ts.pt", experimental=True)

    assert deployed.kind == "mixer"
    assert deployed.environment_variable == MIXER_MODEL_ENV
    assert deployed.digest.startswith("sha256:")
    manifest = json.loads((tmp_path / "mixer.ts.pt.deploy.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "mixer"

    verification = verify_deployed(deployed.path, "mixer")
    assert verification["output_shape"] == [1, 12]


def test_exporting_a_mixer_without_the_experimental_opt_in_is_refused(tmp_path: Path) -> None:
    """Neural mixing is a negative result; it must not be wired in by habit.

    The failure mode is invisible from outside -- the run still converges, just
    slower -- so nothing else would catch it.
    """

    from gmd_active_learning.adapters.dft.scf.deploy import DeployError
    from tools.scf.experimental.neural_mixer.export import export_mixer
    from tools.scf.experimental.neural_mixer.model import (
        MixerConfig,
        build_mixer_model,
        save_mixer_checkpoint,
    )

    config = MixerConfig(hidden_size=8)
    save_mixer_checkpoint(tmp_path / "mixer.pt", model=build_mixer_model(config), config=config)

    with pytest.raises(DeployError, match="not a production backend"):
        export_mixer(tmp_path / "mixer.pt", tmp_path / "mixer.ts.pt")


def test_density_model_exports_in_cp2ks_fortran_layout(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.deploy import GUESS_MODEL_ENV, export_density_guess, verify_deployed
    from tools.scf.training.density_gnn import (
        DensityTrainingConfig,
        train_density_matrix_model,
    )

    from tests.scf.test_scf_density_model import identifiable_examples

    checkpoint = tmp_path / "density.pt"
    train_density_matrix_model(
        identifiable_examples(tmp_path / "data", count=2),
        None,
        checkpoint,
        config=DensityTrainingConfig(epochs=1, batch_size=2, hidden_size=16, n_interactions=2),
    )

    deployed = export_density_guess(checkpoint, tmp_path / "guess.ts.pt")
    verification = verify_deployed(deployed.path, "density_guess")

    assert deployed.environment_variable == GUESS_MODEL_ENV
    # CP2K reads (nao, nao, nspin); water in this basis has 23 orbitals.
    assert verification["output_shape"] == [23, 23, 1]


def test_a_deployed_guess_is_hermitian_and_conserving(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.deploy import export_density_guess
    from tools.scf.training.density_gnn import (
        DensityTrainingConfig,
        train_density_matrix_model,
    )

    from tests.scf.test_scf_density_model import identifiable_examples

    checkpoint = tmp_path / "density.pt"
    train_density_matrix_model(
        identifiable_examples(tmp_path / "data", count=2),
        None,
        checkpoint,
        config=DensityTrainingConfig(epochs=1, batch_size=2, hidden_size=16, n_interactions=2),
    )
    deployed = export_density_guess(checkpoint, tmp_path / "guess.ts.pt")

    module = torch.jit.load(str(deployed.path))
    # A deliberately non-orthonormal overlap: an identity here would let a
    # model normalised to Tr(P) pass the electron-count check it must fail.
    n_ao = 23
    probe = torch.full((n_ao, n_ao), 0.05, dtype=torch.float64)
    probe.fill_diagonal_(1.0)
    overlap = probe.unsqueeze(0)
    with torch.no_grad():
        output = module(
            torch.as_tensor([[8, 1, 1]], dtype=torch.long),
            torch.as_tensor(
                np.asarray([[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]]), dtype=torch.float64
            ),
            torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0,
            torch.zeros(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            overlap,
        )
    matrix = output.numpy()[:, :, 0]

    assert np.allclose(matrix, matrix.T, atol=1e-10)
    # Tr(P S), not Tr(P): with a non-orthonormal S the two differ, which is the
    # point of the probe overlap above.
    #
    # 8, not 10. This assertion previously read 10.0 -- sum(Z) for water -- and
    # passed, because the exported wrapper normalised to the nuclear charge.
    # The training fixtures always integrated to 8, so the test encoded the bug
    # rather than catching it.
    assert float((matrix * overlap.numpy()[0]).sum()) == pytest.approx(8.0, abs=1e-3)
    assert np.isfinite(matrix).all()


def test_a_deployed_guess_normalises_to_valence_not_nuclear_charge(tmp_path: Path) -> None:
    """H2O under GTH: sum(Z) = 10, but the density holds 8 electrons.

    A model normalised to 10 is 25% heavy and CP2K rejects it against a 1e-3
    tolerance -- which presents as "the ML guess failed" rather than "the model
    was normalised against the wrong quantity", and the run silently uses the
    atomic guess.
    """

    from gmd_active_learning.adapters.dft.scf.deploy import export_density_guess
    from tools.scf.training.density_gnn import (
        DensityTrainingConfig,
        train_density_matrix_model,
    )

    from tests.scf.test_scf_density_model import identifiable_examples

    checkpoint = tmp_path / "density.pt"
    train_density_matrix_model(
        identifiable_examples(tmp_path / "data", count=2),
        None,
        checkpoint,
        config=DensityTrainingConfig(epochs=1, batch_size=2, hidden_size=16, n_interactions=2),
    )
    deployed = export_density_guess(checkpoint, tmp_path / "guess.ts.pt")

    manifest = json.loads((tmp_path / "guess.ts.pt.deploy.json").read_text(encoding="utf-8"))
    valence = manifest["metadata"]["valence_per_element"]
    assert manifest["metadata"]["normalization"] == "valence_electrons"
    assert valence["8"] == 6, "oxygen keeps 6 valence electrons under GTH, not 8"
    assert valence["1"] == 1

    n_ao = 23
    probe = torch.full((n_ao, n_ao), 0.05, dtype=torch.float64)
    probe.fill_diagonal_(1.0)
    module = torch.jit.load(str(deployed.path))
    with torch.no_grad():
        output = module(
            torch.as_tensor([[8, 1, 1]], dtype=torch.long),
            torch.as_tensor(
                np.asarray([[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]]),
                dtype=torch.float64,
            ),
            torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0,
            torch.zeros(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            probe.unsqueeze(0),
        )

    trace_ps = float((output.to(torch.float64) * probe.unsqueeze(-1)).sum())
    assert sum([8, 1, 1]) == 10
    assert trace_ps == pytest.approx(8.0, abs=1e-3)
    assert trace_ps != pytest.approx(10.0, abs=1e-3)


def test_a_checkpoint_without_a_valence_table_cannot_be_deployed(tmp_path: Path) -> None:
    """The only fallback would be sum(Z), which is the bug."""

    from gmd_active_learning.adapters.dft.scf.deploy import DeployError, export_density_guess
    from tools.scf.training.density_gnn import (
        DensityTrainingConfig,
        train_density_matrix_model,
    )

    from tests.scf.test_scf_density_model import identifiable_examples

    checkpoint = tmp_path / "density.pt"
    train_density_matrix_model(
        identifiable_examples(tmp_path / "data", count=2),
        None,
        checkpoint,
        config=DensityTrainingConfig(epochs=1, batch_size=2, hidden_size=16, n_interactions=2),
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["valence"] = None
    torch.save(payload, checkpoint)

    with pytest.raises(DeployError, match="no valence electron table"):
        export_density_guess(checkpoint, tmp_path / "guess.ts.pt")


def test_a_nuclear_charge_table_is_refused_at_export(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.deploy import DeployError, export_density_guess
    from tools.scf.training.density_gnn import (
        DensityTrainingConfig,
        train_density_matrix_model,
    )

    from tests.scf.test_scf_density_model import identifiable_examples

    checkpoint = tmp_path / "density.pt"
    train_density_matrix_model(
        identifiable_examples(tmp_path / "data", count=2),
        None,
        checkpoint,
        config=DensityTrainingConfig(epochs=1, batch_size=2, hidden_size=16, n_interactions=2),
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Z for every element: an all-electron table, wrong for these runs.
    payload["valence"] = {"valence_per_element": {"1": 1, "8": 8}, "potential_family": None}
    torch.save(payload, checkpoint)

    with pytest.raises(DeployError, match="equals the nuclear charge"):
        export_density_guess(checkpoint, tmp_path / "guess.ts.pt")


def test_verify_refuses_a_model_deployed_before_electron_counts_were_pinned(
    tmp_path: Path,
) -> None:
    """An old manifest has no valence table, so its model normalises to sum(Z)."""

    from gmd_active_learning.adapters.dft.scf.deploy import DeployError, export_density_guess, verify_deployed
    from tools.scf.training.density_gnn import (
        DensityTrainingConfig,
        train_density_matrix_model,
    )

    from tests.scf.test_scf_density_model import identifiable_examples

    checkpoint = tmp_path / "density.pt"
    train_density_matrix_model(
        identifiable_examples(tmp_path / "data", count=2),
        None,
        checkpoint,
        config=DensityTrainingConfig(epochs=1, batch_size=2, hidden_size=16, n_interactions=2),
    )
    deployed = export_density_guess(checkpoint, tmp_path / "guess.ts.pt")

    manifest_path = tmp_path / "guess.ts.pt.deploy.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["metadata"]["valence_per_element"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DeployError, match="normalises to sum"):
        verify_deployed(deployed.path, "density_guess")


def test_verify_rejects_a_missing_deployed_model(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.deploy import DeployError, verify_deployed

    with pytest.raises(DeployError, match="not found"):
        verify_deployed(tmp_path / "absent.pt", "mixer")


def test_deployment_environment_names_both_models(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.deploy import deployment_environment

    environment = deployment_environment(guess_model="/g.pt", mixer_model="/m.pt")

    assert environment == {
        "PHAL_ML_SCF_GUESS_MODEL": "/g.pt",
        "PHAL_ML_SCF_MIXER_MODEL": "/m.pt",
        # The master switch travels with the models. Without it CP2K honours
        # SCF_GUESS ML by falling back to ATOMIC and reporting it: a correct
        # run, a working model, and no acceleration.
        "PHAL_ML_SCF_ENABLED": "1",
    }


def test_deployment_environment_is_empty_when_nothing_is_deployed() -> None:
    """No model means no switch: enabling ML with nothing to run is a fallback."""

    from gmd_active_learning.adapters.dft.scf.deploy import deployment_environment

    assert deployment_environment() == {}


# -- Stage 13: acceptance criteria --------------------------------------


def run(label: str, system: str, iterations: int, **kwargs) -> SCFRun:
    fields = {
        "converged": True,
        "wall_time_seconds": float(iterations),
        "final_energy": -17.0,
        "forces": np.asarray([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
        "stress": np.eye(3) * 2.0,
    }
    fields.update(kwargs)
    return SCFRun(label=label, system_id=system, scf_iterations=iterations, **fields)


def four_modes() -> list[SCFRun]:
    return [
        run(MODE_NATIVE, "a", 24),
        run(MODE_RESTART, "a", 18),
        run(MODE_ML_GUESS, "a", 15),
        run(MODE_ML_FULL, "a", 11),
    ]


def test_every_mode_gets_an_accepted_verdict() -> None:
    report = run_benchmark(four_modes())

    modes = {result.mode: result for result in report.results}
    assert set(modes) == {MODE_NATIVE, MODE_RESTART, MODE_ML_GUESS, MODE_ML_FULL}
    assert all(result.accepted for result in report.results)
    assert report.accepted_modes == (MODE_NATIVE, MODE_RESTART, MODE_ML_GUESS, MODE_ML_FULL)


def test_results_carry_the_contract_keys() -> None:
    payload = run_benchmark(four_modes()).to_dict()

    for row in payload["results"]:
        for key in ("mode", "iterations", "runtime", "energy_error", "force_error", "speedup", "accepted"):
            assert key in row, key


def test_a_fast_but_wrong_mode_is_rejected() -> None:
    """The whole point of the acceptance criteria."""

    runs = [run(MODE_NATIVE, "a", 24), run(MODE_ML_FULL, "a", 8, final_energy=-16.9)]

    report = run_benchmark(runs)
    ml = next(result for result in report.results if result.mode == MODE_ML_FULL)

    assert ml.speedup == pytest.approx(3.0)
    assert not ml.accepted
    assert any("energy differs" in reason for reason in ml.rejection_reasons)


def test_a_force_difference_rejects_a_mode() -> None:
    runs = [
        run(MODE_NATIVE, "a", 24),
        run(MODE_ML_FULL, "a", 10, forces=np.asarray([[0.1, 0.0, 0.0], [-0.2, 0.0, 0.0]])),
    ]

    report = run_benchmark(runs)
    ml = next(result for result in report.results if result.mode == MODE_ML_FULL)

    assert ml.force_error == pytest.approx(0.1)
    assert not ml.accepted
    assert any("forces differ" in reason for reason in ml.rejection_reasons)


def test_a_stress_difference_rejects_a_mode() -> None:
    runs = [run(MODE_NATIVE, "a", 24), run(MODE_ML_FULL, "a", 10, stress=np.eye(3) * 2.5)]

    report = run_benchmark(runs)
    ml = next(result for result in report.results if result.mode == MODE_ML_FULL)

    assert ml.stress_error == pytest.approx(0.5)
    assert not ml.accepted


def test_a_non_converged_mode_is_rejected() -> None:
    runs = [run(MODE_NATIVE, "a", 24), run(MODE_ML_FULL, "a", 99, converged=False)]

    report = run_benchmark(runs)
    ml = next(result for result in report.results if result.mode == MODE_ML_FULL)

    assert not ml.accepted
    assert any("did not converge" in reason for reason in ml.rejection_reasons)


def test_a_fallback_rate_cap_can_reject_a_mode() -> None:
    runs = [
        run(MODE_NATIVE, "a", 24),
        run(MODE_NATIVE, "b", 24),
        run(MODE_ML_FULL, "a", 12, fallback_used=True),
        run(MODE_ML_FULL, "b", 12),
    ]

    report = run_benchmark(runs, criteria=AcceptanceCriteria(max_fallback_rate=0.25))
    ml = next(result for result in report.results if result.mode == MODE_ML_FULL)

    assert ml.fallback_rate == pytest.approx(0.5)
    assert not ml.accepted
    assert any("fallback rate" in reason for reason in ml.rejection_reasons)


def test_looser_tolerances_can_accept_a_mode() -> None:
    runs = [run(MODE_NATIVE, "a", 24), run(MODE_ML_FULL, "a", 10, final_energy=-17.0 + 1e-5)]

    strict = run_benchmark(runs)
    lenient = run_benchmark(runs, criteria=AcceptanceCriteria(energy_tolerance=1e-3))

    assert not next(r for r in strict.results if r.mode == MODE_ML_FULL).accepted
    assert next(r for r in lenient.results if r.mode == MODE_ML_FULL).accepted


def test_density_difference_is_measured_when_recorded(tmp_path: Path) -> None:
    from tests.scf.fakes.fake_density import symmetric_density

    native_path = tmp_path / "native.npy"
    ml_path = tmp_path / "ml.npy"
    density = symmetric_density(4, electrons=8.0)
    np.save(native_path, density)
    np.save(ml_path, density + 0.25)

    runs = [
        run(MODE_NATIVE, "a", 24, density_matrix_path=str(native_path)),
        run(MODE_ML_FULL, "a", 12, density_matrix_path=str(ml_path)),
    ]

    report = run_benchmark(runs)
    ml = next(result for result in report.results if result.mode == MODE_ML_FULL)

    assert ml.density_error == pytest.approx(0.25)


def test_mismatched_force_shapes_are_reported_not_ignored() -> None:
    runs = [
        run(MODE_NATIVE, "a", 24),
        run(MODE_ML_FULL, "a", 12, forces=np.zeros((3, 3))),
    ]

    with pytest.raises(BenchmarkError, match="cannot compare arrays"):
        run_benchmark(runs)


def test_benchmark_results_json_is_written(tmp_path: Path) -> None:
    report = run_benchmark(four_modes())

    path = write_benchmark_results(report, tmp_path)

    assert path.name == "benchmark_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["acceptance_criteria"]["energy_tolerance"] == pytest.approx(1e-6)
    assert len(payload["results"]) == 4


def test_the_cli_can_fail_on_a_rejected_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from gmd_active_learning.adapters.dft.scf.cli import main

    runs = [run(MODE_NATIVE, "a", 24), run(MODE_ML_FULL, "a", 8, final_energy=-16.0)]
    runs_path = tmp_path / "runs.json"
    runs_path.write_text(json.dumps([entry.to_dict() for entry in runs]), encoding="utf-8")

    exit_code = main(
        [
            "scf", "benchmark",
            "--runs", str(runs_path),
            "--results-dir", str(tmp_path / "out"),
            "--require-accepted",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "REJECT" in output
    assert (tmp_path / "out" / "benchmark_results.json").is_file()


# -- Stage 14: production fallback and provenance -----------------------
# Rewritten for the composable ML-SCF wrapper in tests/scf/test_scf_ml_backend.py.
