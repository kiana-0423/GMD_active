"""The NumPy loss oracles themselves (independent of GMD_SGT)."""

from __future__ import annotations

import numpy as np

from reference.masked_force_mse import masked_force_mse, weighted_force_loss


def test_masked_force_mse_reduces_to_plain_mse_and_ignores_masked_atoms() -> None:
    rng = np.random.default_rng(0)
    predicted, target = rng.normal(size=(6, 3)), rng.normal(size=(6, 3))
    plain = float(np.mean(np.sum((predicted - target) ** 2, axis=1)))
    assert np.isclose(masked_force_mse(predicted, target, np.ones(6)), plain, rtol=1e-8)
    mask = np.array([1.0, 0.3, 0.0, 0.0, 1.0, 0.0])
    perturbed = target.copy()
    perturbed[mask == 0] += 100.0
    assert masked_force_mse(predicted, target, mask) == masked_force_mse(predicted, perturbed, mask)


def test_weighted_loss_relations() -> None:
    rng = np.random.default_rng(1)
    predicted, target = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    weights = np.array([1.0, 0.3, 0.0, 2.0, 0.0])
    mse = weighted_force_loss(predicted, target, weights, "mse")
    assert np.isclose(3.0 * mse, masked_force_mse(predicted, target, weights, epsilon=0.0))
    assert np.isclose(weighted_force_loss(predicted, target, weights, "rmse"), np.sqrt(mse))
    target[2] = np.nan
    assert np.isfinite(weighted_force_loss(predicted, target, weights, "mae"))
    assert weighted_force_loss(predicted, target, np.zeros(5), "rmse") == 0.0
