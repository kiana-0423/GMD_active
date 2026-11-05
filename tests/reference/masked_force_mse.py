# Ported from PHAL (hotspot_al.training.loss_mask) at 96263b6; MIT License, see NOTICE.
"""NumPy reference implementations of mask-weighted force losses (test oracles, ledger TRN-003).

Training happens in GMD_SGT (``gmd_sgt.training.loss.EnergyForceLoss``); these functions are the
independent definitions its weighted force term is checked against.
"""

from __future__ import annotations

import numpy as np


def masked_force_mse(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray, epsilon: float = 1.0e-8) -> float:
    """PHAL's definition: ``sum_i w_i |F_pred_i - F_ref_i|^2 / (sum_i w_i + eps)`` (per-atom squared norm)."""

    predicted = np.asarray(predicted, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.asarray(mask, dtype=float).reshape(-1, 1)
    squared = np.sum((predicted - target) ** 2, axis=1, keepdims=True)
    return float(np.sum(weights * squared) / (np.sum(weights) + epsilon))


def weighted_force_loss(predicted: np.ndarray, target: np.ndarray, weights: np.ndarray, kind: str = "rmse", huber_delta: float = 0.01) -> float:
    """Component-normalised weighted force loss, the definition GMD_SGT documents.

    ``mae``/``huber``: ``sum_i w_i sum_c l(dF_ic) / (3 sum_i w_i)``; ``mse``: same with ``l = d^2``;
    ``rmse``: the square root of the ``mse`` value. Atoms with ``w_i = 0`` are excluded entirely
    (their targets may even be non-finite); a zero total weight gives exactly 0.
    """

    weights = np.asarray(weights, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float)
    target = np.asarray(target, dtype=float)
    keep = weights > 0
    if not np.any(keep):
        return 0.0
    diff = predicted[keep] - target[keep]
    w = weights[keep][:, None]
    if kind in ("mse", "rmse"):
        per = diff**2
    elif kind == "mae":
        per = np.abs(diff)
    elif kind == "huber":
        absolute = np.abs(diff)
        per = np.where(absolute <= huber_delta, 0.5 * diff**2, huber_delta * (absolute - 0.5 * huber_delta))
    else:
        raise ValueError(f"unknown loss kind {kind!r}")
    value = float(np.sum(w * per) / (3.0 * np.sum(weights[keep])))
    return float(np.sqrt(max(value, 1.0e-12))) if kind == "rmse" else value


__all__ = ["masked_force_mse", "weighted_force_loss"]
