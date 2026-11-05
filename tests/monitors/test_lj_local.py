"""Local per-atom LJ residual (MON-006, MON-007, MON-016): correctness on physical (ASE) LJ forces."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones

from gmd_active_learning.monitors.lj_local import (
    LocalLJEstimator,
    compute_local_lj_residuals,
    fit_local_lj_force_linear,
    fit_local_lj_force_nonlinear,
    lj_force_basis,
)
from gmd_active_learning.monitors.lj_projection import fit_lj_projection
from gmd_active_learning.utils.neighbors import build_neighbor_pairs

EPSILON, SIGMA, CUTOFF = 0.0104, 3.40, 8.0


def _argon_cluster(n_atoms: int = 18, seed: int = 3) -> Atoms:
    rng = np.random.default_rng(seed)
    positions: list[np.ndarray] = []
    while len(positions) < n_atoms:
        candidate = rng.uniform(0.0, 9.0, 3)
        if all(np.linalg.norm(candidate - other) > 3.3 for other in positions):
            positions.append(candidate)
    atoms = Atoms(f"Ar{n_atoms}", positions=positions)
    # rc beyond every pair and smooth=False: forces are the exact 12-6 gradient, F = -dE/dx.
    atoms.calc = LennardJones(epsilon=EPSILON, sigma=SIGMA, rc=50.0, smooth=False)
    return atoms


def _pairs(atoms: Atoms):
    return build_neighbor_pairs(atoms.positions, None, False, 50.0)


def _phal_basis(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PHAL's (inverted) basis, re-implemented independently: a = sum 2 d / r^14, b = sum -d / r^8."""

    distances = np.linalg.norm(vectors, axis=1)
    return np.sum(2.0 * vectors / distances[:, None] ** 14, axis=0), np.sum(-vectors / distances[:, None] ** 8, axis=0)


def test_ase_forces_are_minus_gradient() -> None:
    atoms = _argon_cluster()
    forces = atoms.get_forces()
    step, index = 1.0e-5, 4
    plus, minus = atoms.copy(), atoms.copy()
    plus.calc = minus.calc = None
    plus.positions[index, 0] += step
    minus.positions[index, 0] -= step
    plus.calc = LennardJones(epsilon=EPSILON, sigma=SIGMA, rc=50.0, smooth=False)
    minus.calc = LennardJones(epsilon=EPSILON, sigma=SIGMA, rc=50.0, smooth=False)
    numerical = -(plus.get_potential_energy() - minus.get_potential_energy()) / (2 * step)
    assert forces[index, 0] == pytest.approx(numerical, rel=1e-6, abs=1e-9)


def test_exact_lj_forces_fit_with_zero_residual_and_recover_parameters() -> None:
    atoms = _argon_cluster()
    residuals, fits = compute_local_lj_residuals(_pairs(atoms), atoms.get_forces(), cutoff=50.0)
    assert all(fit.valid and fit.evaluated for fit in fits)
    assert np.max(residuals) < 1e-8
    for fit in fits:
        assert fit.epsilon == pytest.approx(EPSILON, rel=1e-6)
        assert fit.sigma == pytest.approx(SIGMA, rel=1e-6)


def test_negated_forces_are_rejected_f01() -> None:
    """The mirror image (what PHAL accepted) must now be invalid with residual 1.0."""

    atoms = _argon_cluster()
    residuals, fits = compute_local_lj_residuals(_pairs(atoms), -atoms.get_forces(), cutoff=50.0)
    assert not any(fit.valid for fit in fits)
    assert np.all(residuals == 1.0)


def test_corrected_fit_on_f_equals_phal_fit_on_minus_f() -> None:
    """corrected(F) == PHAL(-F): same parameters, same residual, for random forces too."""

    rng = np.random.default_rng(8)
    for _ in range(50):
        vectors = rng.normal(0.0, 1.5, size=(6, 3))
        vectors *= (2.2 + rng.uniform(0.0, 1.0, size=(6, 1))) / np.linalg.norm(vectors, axis=1, keepdims=True)
        force = rng.normal(size=3)
        a, b = lj_force_basis(vectors)
        pa, pb = _phal_basis(vectors)
        np.testing.assert_allclose(a, -pa, rtol=1e-14)
        np.testing.assert_allclose(b, -pb, rtol=1e-14)
        mine = fit_local_lj_force_linear(vectors, force)
        # PHAL fitted F ~ alpha*pa + beta*pb; with -F that is alpha*a + beta*b.
        design = np.column_stack((pa, pb))
        scale = np.linalg.norm(design, axis=0)
        solution, *_ = np.linalg.lstsq(design / scale, -force, rcond=None)
        alpha, beta = solution / scale
        if mine.valid:
            sigma = (alpha / beta) ** (1.0 / 6.0)
            assert mine.sigma == pytest.approx(sigma, rel=1e-10)
            phal_residual = np.linalg.norm(alpha * pa + beta * pb + force) / (np.linalg.norm(force) + 1e-8)
            assert mine.residual == pytest.approx(phal_residual, rel=1e-10, abs=1e-14)
        else:
            assert alpha <= 0.0 or beta <= 0.0 or not (0.2 <= (alpha / beta) ** (1 / 6) <= 10.0)


def test_lazy_mask_fits_only_masked_atoms() -> None:
    atoms = _argon_cluster()
    mask = np.zeros(len(atoms), dtype=bool)
    mask[[1, 5]] = True
    residuals, fits = LocalLJEstimator(cutoff=50.0)(_pairs(atoms), atoms.get_forces(), mask)
    assert [index for index, fit in enumerate(fits) if fit.evaluated] == [1, 5]
    assert all(residuals[index] == 0.0 and not fits[index].valid for index in range(len(atoms)) if not mask[index])


def test_collinear_and_ill_conditioned_geometries_are_invalid() -> None:
    collinear = fit_local_lj_force_linear(np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), np.array([1.0, 0.0, 0.0]))
    assert not collinear.valid and collinear.residual == 1.0
    nearly = np.array([[1.0, 0.0, 0.0], [1.01, 1e-4, 0.0], [1.02, -1e-4, 0.0], [1.03, 2e-4, 0.0]])
    assert not fit_local_lj_force_linear(nearly, np.array([1.0, 0.0, 0.0])).valid
    assert not fit_local_lj_force_linear(np.zeros((0, 3)), np.ones(3)).valid


def test_nonlinear_reference_estimator_agrees_with_linear_on_exact_forces() -> None:
    atoms = _argon_cluster()
    pairs = _pairs(atoms)
    forces = atoms.get_forces()
    for index in (0, 3, 7):
        _indices, vectors, _distances = pairs.neighbors_of(index)
        linear = fit_local_lj_force_linear(vectors, forces[index])
        nonlinear = fit_local_lj_force_nonlinear(vectors, forces[index])
        assert nonlinear.valid
        assert nonlinear.epsilon == pytest.approx(linear.epsilon, rel=1e-5)
        assert nonlinear.sigma == pytest.approx(linear.sigma, rel=1e-5)
    residuals, fits = LocalLJEstimator(estimator="nonlinear", cutoff=50.0)(pairs, forces, np.arange(len(atoms)) < 2)
    assert sum(fit.evaluated for fit in fits) == 2 and np.all(residuals[:2] < 1e-4)


def test_both_projection_modes_accept_physical_lj_forces_mon016() -> None:
    atoms = _argon_cluster(n_atoms=10)
    forces = atoms.get_forces()
    frame = {"positions": atoms.positions, "symbols": atoms.get_chemical_symbols(), "ml_forces": forces, "cell": None, "pbc": (False,) * 3}
    global_fit = fit_lj_projection("global_pair", frames=[frame], cutoff=50.0, ridge_lambda=1e-8)
    assert global_fit.is_physical and global_fit.relative_force_residual < 1e-6
    params = global_fit.pair_params_epsilon_sigma["Ar-Ar"]
    assert params["epsilon"] == pytest.approx(EPSILON, rel=1e-5)
    assert params["sigma"] == pytest.approx(SIGMA, rel=1e-5)
    residuals, fits = fit_lj_projection("local_atom", pairs=_pairs(atoms), forces=forces, cutoff=50.0)
    assert all(fit.valid for fit in fits) and np.max(residuals) < 1e-6
    with pytest.raises(ValueError):
        fit_lj_projection("bogus")
