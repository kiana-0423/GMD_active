# Ported from PHAL (tests.scf.test_scf_pulay_mixer) at 96263b6; MIT License, see NOTICE.
"""Tests for ML Pulay coefficient prediction and its training data.

The fixtures are real CP2K v2026.2 SCF trajectories. What is asserted here is
the property that makes coefficient prediction safe: because the update is an
affine combination of densities that each satisfy Tr(P S) = Ne, any coefficient
vector summing to one conserves electrons exactly -- so the model cannot break
conservation even when it is wrong, and only "is this an improvement?" has to
be checked at run time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gmd_active_learning.adapters.dft.scf.ao_matrices import electron_count
from tools.scf.negative_results.pulay_mixer import (
    DEFAULT_MAX_HISTORY,
    MixerSample,
    PulayMixerError,
    build_b_matrix,
    native_coefficients,
    residual_norm,
    samples_from_trajectory,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "scf"
REAL_SYSTEMS = [("h2", 2.0), ("h2o", 8.0)]


def load_fixture(name: str):
    data = np.load(FIXTURES / f"{name}_scf_trajectory.npz")
    return data["densities"], data["overlap"], float(data["n_electrons"])


# -- the safety property -------------------------------------------------


@pytest.mark.parametrize("name,electrons", REAL_SYSTEMS)
def test_any_affine_combination_conserves_electrons(name, electrons):
    """This is why coefficients are predicted instead of a density.

    Tr(P S) is linear in P, so for coefficients summing to one the combination
    carries exactly the electron count every history entry carries -- for any
    coefficients at all, including ones a broken model produced.
    """

    densities, overlap, expected = load_fixture(name)
    history = densities[:6]

    rng = np.random.default_rng(0)
    for _ in range(20):
        raw = rng.normal(size=len(history))
        coefficients = raw / raw.sum()  # the only constraint
        combined = np.tensordot(coefficients, history, axes=(0, 0))

        assert electron_count(combined, overlap) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("name,electrons", REAL_SYSTEMS)
def test_affine_combination_stays_hermitian(name, electrons):
    densities, _overlap, _expected = load_fixture(name)
    history = densities[:6]

    rng = np.random.default_rng(1)
    raw = rng.normal(size=len(history))
    combined = np.tensordot(raw / raw.sum(), history, axes=(0, 0))

    assert float(np.abs(combined - np.swapaxes(combined, 1, 2)).max()) < 1e-8


def test_coefficients_that_do_not_sum_to_one_break_conservation():
    """The converse, so the constraint is not merely decorative."""

    densities, overlap, expected = load_fixture("h2")
    history = densities[:4]

    coefficients = np.array([0.5, 0.5, 0.5, 0.5])  # sums to 2
    combined = np.tensordot(coefficients, history, axes=(0, 0))

    assert electron_count(combined, overlap) == pytest.approx(2.0 * expected, abs=1e-6)


# -- the native solve ----------------------------------------------------


def test_native_coefficients_sum_to_one():
    rng = np.random.default_rng(2)
    raw = rng.normal(size=(5, 5))
    b = raw @ raw.T + 5.0 * np.eye(5)

    coefficients = native_coefficients(b)

    assert coefficients.sum() == pytest.approx(1.0)


def test_native_coefficients_minimise_the_diis_residual():
    """Native DIIS is the constrained minimiser, so nothing beats it on c^T B c.

    This is what the run-time gate compares against, and it is why a rejected
    proposal costs nothing: the fallback is already optimal in this metric.
    """

    rng = np.random.default_rng(3)
    raw = rng.normal(size=(6, 6))
    b = raw @ raw.T + 6.0 * np.eye(6)
    best = residual_norm(b, native_coefficients(b))

    for _ in range(200):
        candidate = rng.normal(size=6)
        candidate = candidate / candidate.sum()
        assert residual_norm(b, candidate) >= best - 1e-9


def test_native_coefficients_reject_a_singular_b():
    with pytest.raises(PulayMixerError, match="singular"):
        native_coefficients(np.zeros((3, 3)))


def test_residual_norm_reports_nan_for_a_negative_form():
    # A non-Gram matrix can produce a negative quadratic form; that is "no
    # estimate", not a residual of zero.
    b = np.array([[-1.0, 0.0], [0.0, -1.0]])

    assert np.isnan(residual_norm(b, np.array([0.5, 0.5])))


# -- training samples ----------------------------------------------------


@pytest.mark.parametrize("name,electrons", REAL_SYSTEMS)
def test_samples_are_generated_from_a_real_trajectory(name, electrons):
    densities, overlap, _expected = load_fixture(name)

    samples = samples_from_trajectory(densities, overlap, system=name)

    assert samples
    for sample in samples:
        assert isinstance(sample, MixerSample)
        depth = sample.target.size
        assert 2 <= depth <= DEFAULT_MAX_HISTORY
        assert sample.b_matrix.shape == (depth, depth)
        assert np.isfinite(sample.b_matrix).all()
        assert np.isfinite(sample.target).all()


@pytest.mark.parametrize("name,electrons", REAL_SYSTEMS)
def test_training_targets_sum_to_one(name, electrons):
    """A target that did not sum to one would train toward a rejected vector."""

    densities, overlap, _expected = load_fixture(name)

    for sample in samples_from_trajectory(densities, overlap):
        assert sample.target.sum() == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("name,electrons", REAL_SYSTEMS)
def test_training_targets_conserve_electrons_when_applied(name, electrons):
    densities, overlap, expected = load_fixture(name)

    for sample in samples_from_trajectory(densities, overlap):
        depth = sample.target.size
        start = sample.iteration - depth
        history = densities[start : start + depth]
        combined = np.tensordot(sample.target, history, axes=(0, 0))

        assert electron_count(combined, overlap) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("name,electrons", REAL_SYSTEMS)
def test_training_targets_beat_the_last_iterate(name, electrons):
    """The target must be a better answer than "keep the newest density".

    Otherwise there is nothing for the model to learn: the trivial coefficient
    vector (0, ..., 0, 1) would already be optimal.
    """

    densities, overlap, _expected = load_fixture(name)
    converged = densities[-1]

    improved = 0
    total = 0
    for sample in samples_from_trajectory(densities, overlap):
        depth = sample.target.size
        start = sample.iteration - depth
        history = densities[start : start + depth]

        target_error = np.abs(np.tensordot(sample.target, history, axes=(0, 0)) - converged).max()
        latest_error = np.abs(history[-1] - converged).max()
        total += 1
        if target_error <= latest_error + 1e-12:
            improved += 1

    assert total > 0
    # The fitted combination is at least as close as the newest iterate almost
    # everywhere; it is a least-squares fit toward the converged density.
    assert improved >= total - 1


def test_iteration_ordering_is_monotonic():
    densities, overlap, _expected = load_fixture("h2o")

    samples = samples_from_trajectory(densities, overlap)

    iterations = [sample.iteration for sample in samples]
    assert iterations == sorted(iterations)
    assert len(set(iterations)) == len(iterations)


def test_history_window_is_capped():
    densities, overlap, _expected = load_fixture("h2o")

    samples = samples_from_trajectory(densities, overlap, max_history=3)

    assert samples
    assert max(sample.target.size for sample in samples) == 3


def test_a_trajectory_too_short_to_learn_from_yields_nothing():
    densities, overlap, _expected = load_fixture("h2")

    assert samples_from_trajectory(densities[:2], overlap) == []


def test_b_matrix_is_symmetric_and_positive_semidefinite():
    densities, _overlap, _expected = load_fixture("h2o")
    residuals = [density - densities[-1] for density in densities[:5]]

    b = build_b_matrix(residuals)

    assert np.allclose(b, b.T)
    assert float(np.linalg.eigvalsh(b).min()) > -1e-8


def test_b_matrix_rejects_an_empty_history():
    with pytest.raises(PulayMixerError, match="empty"):
        build_b_matrix([])


# -- what is and is not learnable ---------------------------------------
#
# See docs/ml_pulay_objective.md. These pin the two facts that determine the
# whole training design, because both are easy to undo by accident.


def test_minimising_the_diis_form_is_exactly_native_diis():
    """`min c^T B c s.t. sum(c)=1` has a closed form, and it is what CP2K does.

    So it is not a training target: a model fitted to it can only reproduce
    native Pulay.
    """

    rng = np.random.default_rng(4)
    for depth in (3, 5, 8):
        raw = rng.normal(size=(depth, depth))
        b = raw @ raw.T + depth * np.eye(depth)

        ones = np.ones(depth)
        inverse = np.linalg.pinv(b)
        closed_form = (inverse @ ones) / (ones @ inverse @ ones)

        assert np.allclose(closed_form, native_coefficients(b), atol=1e-9)


def test_hindsight_residuals_make_the_target_native_diis():
    """Why B must be built from successive differences, not from the answer.

    With r_i = P_i - P_final and sum(c) = 1, `c^T B c` *is* the squared
    distance to the converged density -- so the "optimal" target collapses onto
    native DIIS and there is nothing left to learn.
    """

    densities, _overlap, _electrons = load_fixture("h2o")
    converged = densities[-1]
    window = list(range(2, 8))

    hindsight_b = build_b_matrix([densities[i] - converged for i in window])
    history = np.stack([densities[i].ravel() for i in window], axis=0)

    from tools.scf.negative_results.pulay_mixer import _constrained_least_squares

    diis = native_coefficients(hindsight_b)
    least_squares = _constrained_least_squares(history, converged.ravel())

    # cond(B) is 2.5e10 for this window, so how many digits the two solves share
    # depends on the BLAS underneath: 1.7e-5 apart on macOS/Accelerate, 1.4e-4
    # in the container. Coefficients of order 4, so that is ~3e-5 relative.
    # The claim being tested is that the two objectives coincide, and the check
    # that carries it is on the density they mix to -- which is well
    # conditioned and agrees to 2.6e-10 on both.
    assert np.allclose(diis, least_squares, rtol=0.0, atol=1e-3)
    mixed_diis = (diis[:, None] * history).sum(axis=0)
    mixed_least_squares = (least_squares[:, None] * history).sum(axis=0)
    assert np.allclose(mixed_diis, mixed_least_squares, atol=1e-8)


def test_online_residuals_leave_real_headroom():
    """The converse: with the B a running SCF can actually form, native DIIS is
    well away from the optimum, and that gap is the training signal."""

    from tools.scf.negative_results.ml_pulay_model import (
        density_error,
        training_samples,
    )

    densities, _overlap, _electrons = load_fixture("h2o")
    samples = training_samples(densities, system="h2o")

    assert samples
    native_errors = [density_error(native_coefficients(s.b_online), s) for s in samples]
    oracle_errors = [density_error(s.target, s) for s in samples]

    assert np.mean(native_errors) > 1.5 * np.mean(oracle_errors)


def test_the_default_gate_would_reject_the_optimum():
    """The safety gate forbids exactly the improvement being trained for.

    Deploying a trained mixer therefore requires raising
    PHAL_ML_SCF_PULAY_TOLERANCE deliberately; it is not an oversight that it
    does nothing at the default.
    """

    from tools.scf.negative_results.ml_pulay_model import training_samples

    densities, _overlap, _electrons = load_fixture("h2o")
    rejected = 0
    for sample in training_samples(densities):
        native = residual_norm(sample.b_online, native_coefficients(sample.b_online))
        optimal = residual_norm(sample.b_online, sample.target)
        if optimal > 1.000001 * native:
            rejected += 1

    assert rejected > 0


def test_model_constraints_hold_for_any_output():
    """sum(c)=1, finite, and bounded magnitude are architectural."""

    torch = pytest.importorskip("torch")
    from tools.scf.negative_results.ml_pulay_model import DEFAULT_L1_LIMIT, build_model

    model = build_model(torch)
    rng = np.random.default_rng(5)
    for depth in (2, 4, 8):
        raw = rng.normal(size=(depth, depth))
        b = torch.tensor(raw @ raw.T + depth * np.eye(depth), dtype=torch.float64).unsqueeze(0)

        c = model(b, torch.tensor([[3.0]], dtype=torch.float64))

        assert c.numel() == depth
        assert bool(torch.isfinite(c).all())
        assert float(c.sum()) == pytest.approx(1.0, abs=1e-9)
        # The cap applies before normalisation, so allow the normalising factor.
        assert float(c.abs().sum()) <= DEFAULT_L1_LIMIT * 1.01 / max(abs(float(c.sum())), 1e-9)
