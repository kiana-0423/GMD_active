"""Independent correctness tests for the LJ projection fitter.

Expected values come from ASE's Lennard-Jones calculator (forces = -dE/dx, checked here by
finite differences), not from a hand-written force expression that could share the fitter's
sign convention. See docs/migration/findings.md F-01.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones

from gmd_active_learning.monitors.lj_projection import LJProjectionFitter

EPSILON, SIGMA = 0.0104, 3.40  # Ar, eV and Angstrom


def _ar_cluster(seed: int = 0) -> Atoms:
    rng = np.random.default_rng(seed)
    grid = np.array([[i, j, k] for i in range(3) for j in range(3) for k in range(2)], dtype=float) * 3.8
    atoms = Atoms("Ar" * len(grid), positions=grid + rng.normal(0.0, 0.15, grid.shape))
    atoms.calc = LennardJones(epsilon=EPSILON, sigma=SIGMA, rc=50.0, smooth=False)
    return atoms


def test_ase_reference_forces_are_negative_energy_gradients() -> None:
    atoms = _ar_cluster()
    forces = atoms.get_forces()
    h = 1.0e-5
    for atom, axis in [(0, 0), (5, 1), (11, 2)]:
        shifted = []
        for sign in (1.0, -1.0):
            probe = atoms.copy()
            probe.positions[atom, axis] += sign * h
            probe.calc = LennardJones(epsilon=EPSILON, sigma=SIGMA, rc=50.0, smooth=False)
            shifted.append(probe.get_potential_energy())
        assert -(shifted[0] - shifted[1]) / (2.0 * h) == pytest.approx(forces[atom, axis], abs=1.0e-7)


def test_exact_lj_forces_recover_physical_parameters() -> None:
    atoms = _ar_cluster()
    fitter = LJProjectionFitter(cutoff=50.0, ridge_lambda=0.0)
    result = fitter.fit_single_frame(atoms.get_positions(), atoms.get_chemical_symbols(), atoms.get_forces())
    params = result.pair_params_ab["Ar-Ar"]
    assert params["A"] == pytest.approx(4.0 * EPSILON * SIGMA**12, rel=1.0e-8)
    assert params["B"] == pytest.approx(4.0 * EPSILON * SIGMA**6, rel=1.0e-8)
    assert result.is_physical
    assert result.relative_force_residual < 1.0e-10
    sigma_eps = result.pair_params_epsilon_sigma["Ar-Ar"]
    assert sigma_eps["sigma"] == pytest.approx(SIGMA, rel=1.0e-8)
    assert sigma_eps["epsilon"] == pytest.approx(EPSILON, rel=1.0e-8)


def test_inverted_forces_are_flagged_nonphysical() -> None:
    """The mirror image of a physical LJ force field must be rejected (the PHAL-era defect accepted it)."""

    atoms = _ar_cluster()
    fitter = LJProjectionFitter(cutoff=50.0, ridge_lambda=0.0)
    result = fitter.fit_single_frame(atoms.get_positions(), atoms.get_chemical_symbols(), -atoms.get_forces())
    assert result.is_physical is False
    assert any("nonphysical" in warning for warning in result.warnings)


def test_non_lj_forces_leave_a_large_residual() -> None:
    atoms = _ar_cluster()
    rng = np.random.default_rng(1)
    random_forces = rng.normal(0.0, 0.05, size=(len(atoms), 3))
    fitter = LJProjectionFitter(cutoff=50.0, ridge_lambda=0.0)
    result = fitter.fit_single_frame(atoms.get_positions(), atoms.get_chemical_symbols(), random_forces)
    assert result.relative_force_residual > 0.5


def test_two_species_pairs_are_fitted_independently() -> None:
    """Mixed Ar/Kr cluster with per-pair parameters built from ASE-convention pair forces."""

    rng = np.random.default_rng(2)
    grid = np.array([[i, j, k] for i in range(3) for j in range(3) for k in range(3)], dtype=float) * 3.9
    positions = grid + rng.normal(0.0, 0.2, grid.shape)
    symbols = ["Ar" if index % 2 else "Kr" for index in range(len(positions))]
    params = {"Ar-Ar": (0.0104, 3.40), "Ar-Kr": (0.0120, 3.55), "Kr-Kr": (0.0140, 3.65)}
    energy = 0.0
    forces = np.zeros_like(positions)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            eps, sig = params["-".join(sorted((symbols[i], symbols[j])))]
            d = positions[j] - positions[i]
            r = float(np.linalg.norm(d))
            energy += 4 * eps * ((sig / r) ** 12 - (sig / r) ** 6)
            dv_dr = 4 * eps * (-12 * sig**12 / r**13 + 6 * sig**6 / r**7)
            forces[i] += dv_dr * d / r  # -dV/dr_i, with dr/dr_i = -d/r
            forces[j] -= dv_dr * d / r
    # Guard the hand-built reference with an energy finite difference.
    h = 1.0e-6

    def total_energy(pos: np.ndarray) -> float:
        total = 0.0
        for a in range(len(pos)):
            for b in range(a + 1, len(pos)):
                eps, sig = params["-".join(sorted((symbols[a], symbols[b])))]
                r = float(np.linalg.norm(pos[b] - pos[a]))
                total += 4 * eps * ((sig / r) ** 12 - (sig / r) ** 6)
        return total

    plus, minus = positions.copy(), positions.copy()
    plus[3, 1] += h
    minus[3, 1] -= h
    assert -(total_energy(plus) - total_energy(minus)) / (2 * h) == pytest.approx(forces[3, 1], abs=1.0e-7)

    result = LJProjectionFitter(cutoff=50.0, ridge_lambda=0.0).fit_single_frame(positions, symbols, forces)
    assert result.is_physical
    for pair, (eps, sig) in params.items():
        fitted = result.pair_params_epsilon_sigma[pair]
        assert fitted["epsilon"] == pytest.approx(eps, rel=1.0e-6)
        assert fitted["sigma"] == pytest.approx(sig, rel=1.0e-6)


@pytest.mark.parametrize("ridge_lambda", [1.0e-8, 1.0e-6, 1.0e-4])
def test_configured_ridge_does_not_bias_a_well_posed_fit(ridge_lambda: float) -> None:
    """Regression: an unscaled ridge collapsed A (99275 -> 8e-4) at the shipped lambda=1e-6."""

    atoms = _ar_cluster()
    result = LJProjectionFitter(cutoff=50.0, ridge_lambda=ridge_lambda).fit_single_frame(
        atoms.get_positions(), atoms.get_chemical_symbols(), atoms.get_forces()
    )
    assert result.is_physical
    assert result.pair_params_epsilon_sigma["Ar-Ar"]["sigma"] == pytest.approx(SIGMA, rel=1.0e-3)
    assert result.pair_params_epsilon_sigma["Ar-Ar"]["epsilon"] == pytest.approx(EPSILON, rel=1.0e-2)
