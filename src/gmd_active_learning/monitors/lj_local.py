# Ported from PHAL (hotspot_al.monitor.lj_residual_fast, hotspot_al.monitor.lj_residual) at 96263b6;
# MIT License, see NOTICE.
"""Per-atom local Lennard-Jones projection residual (``mode=local_atom``).

For atom i with neighbours j (``d_ij = r_j - r_i``, within ``lj_cutoff``) the physical 12-6 LJ
force is

    F_i = sum_j 24 eps (-2 sigma^12 / r^14 + sigma^6 / r^8) d_ij = alpha * a_i + beta * b_i,
    a_i = sum_j -2 d_ij / r^14,   b_i = sum_j d_ij / r^8,   alpha = 24 eps sigma^12,  beta = 24 eps sigma^6.

The fit solves the 3x2 least-squares problem for (alpha, beta) on column-normalised design and
reports the dimensionless residual |alpha a + beta b - F_i| / (|F_i| + delta). A fit is valid only
with rank 2, condition number <= ``condition_limit``, alpha > 0, beta > 0 and sigma inside
[sigma_min, sigma_max]; invalid fits report residual 1.0 (as PHAL).

SIGN FIX (docs/migration/findings.md F-01): PHAL used a = sum 2 d / r^14 and b = sum -d / r^8, i.e.
it fitted the NEGATED force; physically correct LJ forces were all rejected (residual 1.0).
``corrected(F) == PHAL(-F)`` exactly, which is how the parity tests compare against the fixtures.

With 2 parameters per atom and 3 force components, one residual degree of freedom remains: the
residual measures the force component normal to span{a, b} plus the sign/sigma acceptance. It
cannot detect a wrong pair *strength* (see MON-006 notes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from gmd_active_learning.core.exceptions import DataContractError
from gmd_active_learning.utils.neighbors import NeighborPairs

logger = logging.getLogger(__name__)

DESIGN_CONDITION_LIMIT = 1.0e4
SIGMA_MIN = 0.2
SIGMA_MAX = 10.0
INVALID_RESIDUAL = 1.0


@dataclass(frozen=True, slots=True)
class LocalLJFit:
    """Fit result for one atom. ``evaluated=False`` marks atoms skipped by lazy evaluation."""

    epsilon: float
    sigma: float
    residual: float
    valid: bool
    evaluated: bool = True


def invalid_fit() -> LocalLJFit:
    return LocalLJFit(epsilon=0.0, sigma=0.0, residual=INVALID_RESIDUAL, valid=False)


def skipped_fit() -> LocalLJFit:
    return LocalLJFit(epsilon=0.0, sigma=0.0, residual=0.0, valid=False, evaluated=False)


def lj_force_basis(displacement_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(a, b) with F_i = alpha * a + beta * b for d = r_j - r_i (F = -grad E)."""

    vectors = np.asarray(displacement_vectors, dtype=float)
    distances = np.linalg.norm(vectors, axis=1)
    a_vector = np.sum(-2.0 * vectors / distances[:, None] ** 14, axis=0)
    b_vector = np.sum(vectors / distances[:, None] ** 8, axis=0)
    return a_vector, b_vector


def lj_pair_force(displacement: np.ndarray, epsilon: float, sigma: float) -> np.ndarray:
    """Force on atom i from neighbour j at ``displacement = r_j - r_i`` (F = -grad E)."""

    radius = float(np.linalg.norm(displacement))
    if radius < 1.0e-8:
        return np.zeros(3, dtype=float)
    prefactor = 24.0 * epsilon * (-(2.0 * sigma**12 / radius**14) + (sigma**6 / radius**8))
    return prefactor * np.asarray(displacement, dtype=float)


def fit_local_lj_force_linear(
    displacement_vectors: np.ndarray,
    target_force: np.ndarray,
    *,
    delta: float = 1.0e-8,
    condition_limit: float = DESIGN_CONDITION_LIMIT,
    sigma_min: float = SIGMA_MIN,
    sigma_max: float = SIGMA_MAX,
) -> LocalLJFit:
    """Linear (alpha, beta) least-squares projection of one atom's force onto the LJ basis."""

    vectors = np.asarray(displacement_vectors, dtype=float)
    force = np.asarray(target_force, dtype=float)
    if len(vectors) == 0:
        return invalid_fit()
    distances = np.linalg.norm(vectors, axis=1)
    mask = distances > 1.0e-8
    if not np.any(mask):
        return invalid_fit()
    a_vector, b_vector = lj_force_basis(vectors[mask])

    design = np.column_stack((a_vector, b_vector))
    column_scale = np.linalg.norm(design, axis=0)
    if not np.all(np.isfinite(column_scale)) or np.any(column_scale <= 0.0):
        return invalid_fit()
    normalized = design / column_scale
    try:
        solution, _sum_sq, rank, singular = np.linalg.lstsq(normalized, force, rcond=None)
    except np.linalg.LinAlgError:
        return invalid_fit()
    if rank < 2 or len(singular) < 2 or singular[-1] <= 0.0:
        return invalid_fit()
    condition = float(singular[0] / singular[-1])
    if not np.isfinite(condition) or condition > condition_limit:
        return invalid_fit()

    alpha, beta = solution / column_scale
    if alpha <= 0.0 or beta <= 0.0:
        return invalid_fit()
    sigma = float((alpha / beta) ** (1.0 / 6.0))
    if not np.isfinite(sigma) or sigma < sigma_min or sigma > sigma_max:
        return invalid_fit()
    epsilon = float(beta / (24.0 * sigma**6))
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        return invalid_fit()
    predicted = alpha * a_vector + beta * b_vector
    residual = float(np.linalg.norm(predicted - force) / (np.linalg.norm(force) + delta))
    return LocalLJFit(epsilon=epsilon, sigma=sigma, residual=residual, valid=True)


def fit_local_lj_force_nonlinear(
    displacement_vectors: np.ndarray,
    target_force: np.ndarray,
    *,
    delta: float = 1.0e-8,
    max_nfev: int = 400,
    tolerance: float = 1.0e-12,
) -> LocalLJFit:
    """Bounded nonlinear (log eps, log sigma) reference fit (scipy). Diagnostic estimator only.

    Its acceptance rules and residual scale differ from the linear estimator, so thresholds tuned
    for one do not transfer to the other. Invalid fits report ``max(residual, 1.0)``.
    """

    from scipy.optimize import least_squares

    vectors = np.asarray(displacement_vectors, dtype=float)
    force = np.asarray(target_force, dtype=float)
    if len(vectors) == 0:
        return invalid_fit()
    distances = np.linalg.norm(vectors, axis=1)
    positive = distances[distances > 1.0e-8]
    if len(positive) == 0:
        return invalid_fit()
    vectors = vectors[distances > 1.0e-8]
    sigma0 = max(float(np.median(positive) / (2.0 ** (1.0 / 6.0))), 0.5)
    epsilon0 = max(float(np.linalg.norm(force) / (len(vectors) + 1.0)), 1.0e-3)
    min_distance = float(np.min(positive))
    upper_sigma = max(min_distance * 1.25, 0.6)

    def residuals(log_params: np.ndarray) -> np.ndarray:
        epsilon = float(np.exp(log_params[0]))
        sigma = float(np.exp(log_params[1]))
        predicted = np.sum([lj_pair_force(vector, epsilon, sigma) for vector in vectors], axis=0)
        return predicted - force

    result = least_squares(
        residuals,
        x0=np.log([epsilon0, min(sigma0, upper_sigma)]),
        bounds=(np.log([1.0e-8, 0.2]), np.log([1.0e4, upper_sigma])),
        max_nfev=max_nfev,
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance,
    )
    epsilon, sigma = np.exp(result.x)
    residual = float(np.linalg.norm(result.fun) / (np.linalg.norm(force) + delta))
    physically_sane = bool(epsilon > 0.0 and 0.2 <= sigma <= upper_sigma and min_distance > 0.5)
    valid = physically_sane and bool(result.success or residual < 1.0)
    if not valid:
        residual = max(residual, INVALID_RESIDUAL)
    return LocalLJFit(epsilon=float(epsilon), sigma=float(sigma), residual=residual, valid=valid)


@dataclass(frozen=True, slots=True)
class LocalLJEstimator:
    """Configured per-atom estimator (``metrics.lj_residual`` settings)."""

    estimator: Literal["linear", "nonlinear"] = "linear"
    cutoff: float = 6.0
    delta: float = 1.0e-8
    condition_limit: float = DESIGN_CONDITION_LIMIT
    sigma_min: float = SIGMA_MIN
    sigma_max: float = SIGMA_MAX
    nonlinear_max_nfev: int = 400

    def fit_atom(self, displacement_vectors: np.ndarray, force: np.ndarray) -> LocalLJFit:
        if self.estimator == "linear":
            return fit_local_lj_force_linear(
                displacement_vectors,
                force,
                delta=self.delta,
                condition_limit=self.condition_limit,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
            )
        return fit_local_lj_force_nonlinear(displacement_vectors, force, delta=self.delta, max_nfev=self.nonlinear_max_nfev)

    def __call__(
        self,
        pairs: NeighborPairs,
        forces: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[LocalLJFit]]:
        """Residuals for the atoms in ``mask`` (all atoms when None); skipped atoms carry 0.0 / evaluated=False."""

        forces = np.asarray(forces, dtype=float)
        if forces.shape != (pairs.n_atoms, 3):
            raise DataContractError(f"forces shape {forces.shape} does not match {pairs.n_atoms} atoms")
        selected = np.ones(pairs.n_atoms, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if selected.shape != (pairs.n_atoms,):
            raise DataContractError("LJ evaluation mask length must match the number of atoms")
        residuals = np.zeros(pairs.n_atoms, dtype=float)
        fits: list[LocalLJFit] = []
        for index in range(pairs.n_atoms):
            if not selected[index]:
                fits.append(skipped_fit())
                continue
            _indices, vectors, _distances = pairs.neighbors_of(index, self.cutoff)
            fit = self.fit_atom(vectors, forces[index])
            residuals[index] = fit.residual
            fits.append(fit)
        return residuals, fits


def compute_local_lj_residuals(
    pairs: NeighborPairs,
    forces: np.ndarray,
    *,
    cutoff: float,
    mask: np.ndarray | None = None,
    estimator: Literal["linear", "nonlinear"] = "linear",
) -> tuple[np.ndarray, list[LocalLJFit]]:
    """Convenience wrapper around :class:`LocalLJEstimator`."""

    return LocalLJEstimator(estimator=estimator, cutoff=cutoff)(pairs, forces, mask)


__all__ = [
    "DESIGN_CONDITION_LIMIT",
    "INVALID_RESIDUAL",
    "LocalLJEstimator",
    "LocalLJFit",
    "compute_local_lj_residuals",
    "fit_local_lj_force_linear",
    "fit_local_lj_force_nonlinear",
    "invalid_fit",
    "lj_force_basis",
    "lj_pair_force",
    "skipped_fit",
]
