# Ported from PHAL (hotspot_al.cp2k_ml_scf.training.ml_pulay_model) at 96263b6; MIT License, see NOTICE.
"""Learn DIIS coefficients that beat native Pulay on the *true* SCF residual.

Read ``docs/ml_pulay_objective.md`` before changing anything here. The short
version, because it determines the whole design:

``min c^T B c  subject to  sum(c) = 1`` has the closed form
``c = B^-1 1 / (1^T B^-1 1)``, which is exactly what CP2K computes. Training
toward that target can only reproduce native DIIS. There is nothing to learn in
it.

What *is* learnable is the gap between two different ``B`` matrices:

* **online** ``B_ij = <P_{i+1} - P_i, P_{j+1} - P_j>`` -- successive
  differences, which is all an SCF has while it is running. Native DIIS
  minimises the quadratic form built from this.
* **hindsight** ``B_ij = <P_i - P_final, P_j - P_final>`` -- distance to the
  converged density, available only after the fact. With ``sum(c) = 1`` its
  quadratic form *is* ``||sum_i c_i P_i - P_final||^2``, so minimising it is
  literally minimising the distance to the answer.

Native DIIS is optimal for the first and, on recorded trajectories, between
1.8x and 10^7x away from optimal for the second. That gap is the training
signal: map the online ``B`` (which a running SCF has) onto the coefficients the
hindsight solve would have chosen (which it does not).

The constraint ``sum(c) = 1`` is imposed by the architecture rather than the
loss, so electron conservation holds for any output at all -- see
``docs/density_representation.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DEFAULT_MAX_HISTORY = 8

#: Bound on ``sum |c_i|``. DIIS coefficients are legitimately negative and can
#: legitimately exceed one in magnitude -- that is how it extrapolates -- but an
#: unbounded combination of nearly-collinear densities amplifies noise without
#: limit. This is the standard DIIS safeguard and is what §3's "reasonable
#: magnitude" means operationally.
DEFAULT_L1_LIMIT = 12.0


class MLPulayError(RuntimeError):
    """Raised when a model cannot be built, trained, or exported."""


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One SCF iteration, framed as a learning problem.

    ``b_online`` is what the model sees; ``target`` is what it should have
    said; ``history`` and ``converged`` let the loss be evaluated as an actual
    density error rather than a coefficient error.
    """

    b_online: np.ndarray
    target: np.ndarray
    history: np.ndarray
    converged: np.ndarray
    residual_norms: np.ndarray
    iteration: int
    system: str = ""

    @property
    def depth(self) -> int:
        return int(self.target.size)


@dataclass
class TrainingReport:
    """Everything the report JSON needs, recorded as training proceeds."""

    epochs: int = 0
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    baseline_train_loss: float = float("nan")
    baseline_validation_loss: float = float("nan")
    n_train: int = 0
    n_validation: int = 0
    systems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        final_train = self.train_loss[-1] if self.train_loss else float("nan")
        final_validation = self.validation_loss[-1] if self.validation_loss else float("nan")
        return {
            "epochs": self.epochs,
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "systems": self.systems,
            "train_loss_final": final_train,
            "validation_loss_final": final_validation,
            "train_loss_curve": self.train_loss,
            "validation_loss_curve": self.validation_loss,
            # Native DIIS scored with the same loss. The only number that says
            # whether the model learned anything: below 1.0 it beats native.
            "baseline_train_loss": self.baseline_train_loss,
            "baseline_validation_loss": self.baseline_validation_loss,
            "train_loss_ratio_vs_native": _ratio(final_train, self.baseline_train_loss),
            "validation_loss_ratio_vs_native": _ratio(final_validation, self.baseline_validation_loss),
        }


def _ratio(value: float, baseline: float) -> float:
    if not np.isfinite(value) or not np.isfinite(baseline) or baseline == 0.0:
        return float("nan")
    return float(value / baseline)


# -- data ---------------------------------------------------------------


def online_b_matrix(densities: Sequence[np.ndarray]) -> np.ndarray:
    """``B`` from AO-space successive differences.

    .. deprecated:: superseded by recorded CP2K matrices.

       This is **not** the matrix CP2K solves. CP2K builds ``B`` from g-space
       residuals, Kerker-weighted, weighted again by ``special_metric``, and
       scaled by ``vol^2``. A model trained on this reconstruction and fed the
       real thing at run time degraded convergence from 11 to 37 SCF steps.

       Kept only so the two can be *compared* -- see
       ``compare_b_matrices`` -- never to train on. Use
       :func:`samples_from_records`.
    """

    frames = [np.asarray(density, dtype=float) for density in densities]
    if len(frames) < 2:
        raise MLPulayError("need at least two densities to form an online residual")
    residuals = [(frames[i + 1] - frames[i]).ravel() for i in range(len(frames) - 1)]
    depth = len(residuals)
    matrix = np.empty((depth, depth), dtype=float)
    for i in range(depth):
        for j in range(i, depth):
            value = float(residuals[i] @ residuals[j])
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def hindsight_target(history: np.ndarray, converged: np.ndarray) -> np.ndarray:
    """``argmin ||sum c_i P_i - P_final||`` subject to ``sum(c) = 1``.

    Solved as a KKT system. This is the label: the coefficients an oracle with
    the converged density would have chosen at this iteration.
    """

    flat = history.reshape(history.shape[0], -1)
    depth = flat.shape[0]
    gram = flat @ flat.T
    rhs = flat @ converged.ravel()

    kkt = np.zeros((depth + 1, depth + 1), dtype=float)
    kkt[:depth, :depth] = gram
    kkt[depth, :depth] = 1.0
    kkt[:depth, depth] = 1.0
    right = np.zeros(depth + 1, dtype=float)
    right[:depth] = rhs
    right[depth] = 1.0

    solution, *_ = np.linalg.lstsq(kkt, right, rcond=None)
    coefficients = solution[:depth]
    total = coefficients.sum()
    if not np.isfinite(total) or abs(total) < 1e-12:
        fallback = np.zeros(depth, dtype=float)
        fallback[-1] = 1.0
        return fallback
    return coefficients / total


def training_samples(
    densities: Sequence[np.ndarray],
    *,
    max_history: int = DEFAULT_MAX_HISTORY,
    system: str = "",
) -> list[TrainingSample]:
    """Turn one recorded trajectory into learning problems.

    For each iteration the model sees the online ``B`` over the preceding
    window and must produce the hindsight-optimal coefficients for that window.
    """

    frames = [np.asarray(density, dtype=float) for density in densities]
    if len(frames) < 4:
        return []
    converged = frames[-1]

    samples: list[TrainingSample] = []
    # The window must end before the final frame: including it would leak the
    # answer into the input, and the model would learn to read it off.
    for end in range(2, len(frames) - 1):
        start = max(0, end - max_history)
        window = list(range(start, end))
        if len(window) < 2:
            continue

        history = np.stack([frames[i] for i in window], axis=0)
        # Successive differences need one frame past the window end, which is
        # available online because it is the iterate the SCF just produced.
        b_online = online_b_matrix([frames[i] for i in window] + [frames[end]])
        b_online = b_online[: len(window), : len(window)]
        target = hindsight_target(history, converged)
        norms = np.array(
            [float(np.linalg.norm(frames[i + 1] - frames[i])) for i in window], dtype=float
        )
        samples.append(
            TrainingSample(
                b_online=b_online,
                target=target,
                history=history,
                converged=converged,
                residual_norms=norms,
                iteration=end,
                system=system,
            )
        )
    return samples


def density_error(coefficients: np.ndarray, sample: TrainingSample) -> float:
    """Relative distance from ``sum c_i P_i`` to the converged density.

    This is the quantity the model is trained to reduce, and the one the
    report calls the loss. It is a real density error, not a proxy.
    """

    combined = np.tensordot(np.asarray(coefficients, dtype=float), sample.history, axes=(0, 0))
    scale = float(np.linalg.norm(sample.converged))
    if scale <= 0.0:
        return float("nan")
    return float(np.linalg.norm(combined - sample.converged) / scale)


# -- features -----------------------------------------------------------


def features_from_b(b_matrix: np.ndarray, iteration: int, max_history: int) -> np.ndarray:
    """Pad and scale ``B`` into a fixed-length feature vector.

    ``B``'s magnitude is the residual norm squared and spans many orders of
    magnitude over an SCF, while only its *shape* says anything about the
    coefficients -- so it is normalised. The log of the scale is kept as its
    own feature, because how converged the SCF is does matter.
    """

    matrix = np.asarray(b_matrix, dtype=float)
    depth = matrix.shape[0]
    if depth > max_history:
        raise MLPulayError(f"history depth {depth} exceeds max_history {max_history}")

    scale = float(np.abs(matrix).max())
    normalised = matrix / scale if scale > 0.0 else matrix

    padded = np.zeros((max_history, max_history), dtype=float)
    padded[:depth, :depth] = normalised

    mask = np.zeros(max_history, dtype=float)
    mask[:depth] = 1.0

    log_scale = float(np.log10(scale)) if scale > 0.0 else -30.0
    return np.concatenate(
        [padded.ravel(), mask, [log_scale / 30.0, float(iteration) / 50.0, depth / max_history]]
    )


def feature_size(max_history: int) -> int:
    return max_history * max_history + max_history + 3


# -- model --------------------------------------------------------------


def build_model(torch: Any, max_history: int = DEFAULT_MAX_HISTORY, l1_limit: float = DEFAULT_L1_LIMIT) -> Any:
    """The network CP2K's shim calls: ``(B, iteration) -> c``.

    Three constraints are enforced by the architecture, not hoped for:

    * ``sum(c) = 1``     -- divide by the sum, so electron conservation is
      structural for any output whatsoever;
    * finite             -- a non-finite sum falls back to ``(0, ..., 0, 1)``,
      i.e. "keep the newest density", which is always valid;
    * bounded magnitude  -- ``sum |c_i|`` is capped, then renormalised, so a
      wild extrapolation is damped toward the newest iterate instead of being
      emitted and rejected downstream.
    """

    nn = torch.nn
    from torch import Tensor

    class MLPulayCoefficients(nn.Module):
        def __init__(self, depth: int, limit: float) -> None:
            super().__init__()
            self.max_history = depth
            self.l1_limit = limit
            size = depth * depth + depth + 3
            self.net = nn.Sequential(
                nn.Linear(size, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, depth),
            )

        def forward(self, b_matrix: Tensor, iteration: Tensor) -> Tensor:
            b = b_matrix.to(torch.float64)
            if b.dim() == 3:
                b = b[0]
            nb = b.shape[0]

            scale = b.abs().max()
            normalised = b / scale if float(scale) > 0.0 else b

            padded = torch.zeros(
                (self.max_history, self.max_history), dtype=torch.float64, device=b.device
            )
            padded[:nb, :nb] = normalised
            mask = torch.zeros(self.max_history, dtype=torch.float64, device=b.device)
            mask[:nb] = 1.0

            log_scale = torch.log10(scale + 1e-300) / 30.0
            step = iteration.to(torch.float64).reshape(-1)[:1] / 50.0
            depth_feature = torch.full((1,), float(nb) / self.max_history, dtype=torch.float64, device=b.device)

            features = torch.cat(
                [padded.reshape(-1), mask, log_scale.reshape(1), step, depth_feature], dim=0
            )
            raw = self.net(features.to(torch.float32)).to(torch.float64)[:nb]

            # Bounded magnitude, before normalisation so the cap is meaningful.
            l1 = raw.abs().sum()
            if float(l1) > self.l1_limit:
                raw = raw * (self.l1_limit / l1)

            total = raw.sum()
            if torch.abs(total) < 1e-12 or not bool(torch.isfinite(total)):
                fallback = torch.zeros(nb, dtype=torch.float64, device=b.device)
                fallback[nb - 1] = 1.0
                return fallback
            return raw / total

    return MLPulayCoefficients(max_history, l1_limit)


# -- training -----------------------------------------------------------


def train(
    torch: Any,
    model: Any,
    train_samples: Sequence[TrainingSample],
    validation_samples: Sequence[TrainingSample],
    *,
    epochs: int = 400,
    learning_rate: float = 1.0e-3,
    report: TrainingReport | None = None,
) -> TrainingReport:
    """Minimise the real density error, not a coefficient distance.

    The loss is ``|| sum_i c_i P_i - P_final || / ||P_final||`` -- the thing the
    mixer is supposed to reduce. Training on coefficient MSE instead would
    weight every coefficient equally, and DIIS coefficients differing in the
    fourth decimal can land in very different places.
    """

    from tools.scf.negative_results.pulay_mixer import native_coefficients

    if not train_samples:
        raise MLPulayError("no training samples")

    result = report or TrainingReport()
    result.epochs = epochs
    result.n_train = len(train_samples)
    result.n_validation = len(validation_samples)
    result.systems = sorted({sample.system for sample in train_samples if sample.system})

    def prepared(samples: Sequence[TrainingSample]) -> list[tuple[Any, Any, Any, Any]]:
        prepared_samples = []
        for sample in samples:
            b = torch.tensor(sample.b_online, dtype=torch.float64).unsqueeze(0)
            step = torch.tensor([[float(sample.iteration)]], dtype=torch.float64)
            history = torch.tensor(sample.history.reshape(sample.depth, -1), dtype=torch.float64)
            converged = torch.tensor(sample.converged.ravel(), dtype=torch.float64)
            prepared_samples.append((b, step, history, converged))
        return prepared_samples

    train_tensors = prepared(train_samples)
    validation_tensors = prepared(validation_samples)

    def batch_loss(tensors: Sequence[tuple[Any, Any, Any, Any]]) -> Any:
        total = None
        for b, step, history, converged in tensors:
            coefficients = model(b, step)
            combined = coefficients @ history
            error = torch.linalg.vector_norm(combined - converged) / torch.linalg.vector_norm(converged)
            total = error if total is None else total + error
        return total / max(len(tensors), 1)

    # Native DIIS scored with the same loss, so the curves have a reference.
    def baseline(samples: Sequence[TrainingSample]) -> float:
        if not samples:
            return float("nan")
        errors = []
        for sample in samples:
            try:
                coefficients = native_coefficients(sample.b_online)
            except Exception:  # noqa: BLE001 - a singular B scores as no answer
                continue
            errors.append(density_error(coefficients, sample))
        return float(np.mean(errors)) if errors else float("nan")

    result.baseline_train_loss = baseline(train_samples)
    result.baseline_validation_loss = baseline(validation_samples)

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        model.train()
        optimiser.zero_grad()
        loss = batch_loss(train_tensors)
        loss.backward()
        optimiser.step()
        result.train_loss.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            if validation_tensors:
                result.validation_loss.append(float(batch_loss(validation_tensors).detach()))
            else:
                result.validation_loss.append(float("nan"))

    return result


# -- training on CP2K's own matrices ------------------------------------


def samples_from_records(
    records: Sequence[Any],
    densities: Sequence[np.ndarray],
    *,
    system: str = "",
) -> list[TrainingSample]:
    """Build training samples from *recorded* CP2K Pulay matrices.

    This is the supported path. ``records`` are :class:`PulayRecord` objects
    read from a run's ``PHAL_ML_SCF_PULAY_RECORD_DIR``; ``densities`` is the
    matching density trajectory, which supplies the hindsight target.

    The input is CP2K's own ``B`` -- the exact matrix the mixer is handed at
    run time -- so there is no train/serve mismatch to explain away.
    """

    from gmd_active_learning.adapters.dft.scf.pulay_matrix import require_cp2k_representation

    require_cp2k_representation(records)

    frames = [np.asarray(density, dtype=float) for density in densities]
    if len(frames) < 3:
        return []
    converged = frames[-1]

    samples: list[TrainingSample] = []
    for record in records:
        depth = record.depth
        # CP2K's mixing call n uses the n most recent iterates, capped at the
        # buffer depth. The window ends at the iterate that call produced.
        end = int(record.iteration)
        start = end - depth
        if start < 0 or end > len(frames) - 1 or depth < 2:
            continue

        history = np.stack(frames[start:end], axis=0)
        if history.shape[0] != depth:
            continue

        samples.append(
            TrainingSample(
                b_online=np.asarray(record.b_matrix, dtype=float),
                target=hindsight_target(history, converged),
                history=history,
                converged=converged,
                residual_norms=np.asarray(record.alpha, dtype=float),
                iteration=end,
                system=system,
            )
        )
    return samples


def compare_b_matrices(record: Any, densities: Sequence[np.ndarray]) -> dict[str, float]:
    """Quantify how far an AO reconstruction is from CP2K's own matrix.

    Both are normalised by their largest element first, because only the shape
    matters to a model whose input is scale-normalised. What remains is the
    part a reconstruction cannot recover.
    """

    depth = record.depth
    end = int(record.iteration)
    start = end - depth
    frames = [np.asarray(density, dtype=float) for density in densities]
    if start < 0 or end >= len(frames):
        return {}

    reconstructed = online_b_matrix(frames[start : end + 1])[:depth, :depth]
    actual = np.asarray(record.b_matrix, dtype=float)

    def unit(matrix: np.ndarray) -> np.ndarray:
        scale = float(np.abs(matrix).max())
        return matrix / scale if scale > 0.0 else matrix

    a, b = unit(actual), unit(reconstructed)
    from tools.scf.negative_results.pulay_mixer import native_coefficients

    try:
        coefficient_error = float(
            np.abs(native_coefficients(actual) - native_coefficients(reconstructed)).max()
        )
    except Exception:  # noqa: BLE001 - a singular reconstruction is itself the answer
        coefficient_error = float("inf")

    return {
        "shape_max_abs_difference": float(np.abs(a - b).max()),
        "shape_relative_frobenius": float(
            np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300)
        ),
        "coefficient_max_abs_difference": coefficient_error,
    }
