# Ported from PHAL (hotspot_al.cp2k_ml_scf.training.pulay_mixer) at 96263b6; MIT License, see NOTICE.
"""ML Pulay: predict the DIIS coefficient vector from the residual-overlap matrix.

CP2K's Pulay mixing solves

    minimise  c^T B c   subject to  sum_i c_i = 1,      B_ij = <r_i | r_j>,

and forms the next input density as ``sum_i c_i (rho_in(i) + beta r_i)``. This
module learns to produce ``c`` directly from ``B``.

Predicting *coefficients* rather than a density is what makes the method safe:

* electron conservation is structural -- an affine combination of densities
  that each carry the right electron count carries it too, so nothing has to be
  renormalised and the model cannot invent charge;
* Hermiticity is structural for the same reason;
* the residual of a candidate is ``c^T B c``, computable in ``nb^2`` work, so
  "is this an improvement?" is answered exactly within the DIIS model without a
  Kohn-Sham build.

The model is also basis-independent: ``B`` is ``nb x nb`` where ``nb`` is the
history depth (at most ``NBUFFER``, typically 8), so one model serves any system
regardless of ``nao``.

The exported TorchScript module takes ``(B, iteration)`` and returns ``c``; the
constraint ``sum(c) = 1`` is imposed by the architecture, not learned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DEFAULT_MAX_HISTORY = 8


class PulayMixerError(RuntimeError):
    """Raised when a mixer model cannot be built, trained, or exported."""


def native_coefficients(b_matrix: np.ndarray) -> np.ndarray:
    """Reproduce CP2K's own DIIS solve, for reference and for training targets.

    This is ``qs_gspace_mixing.F``'s computation written out: invert ``B``, sum
    its columns, normalise. Kept here so a candidate can be scored against the
    thing it has to beat without running CP2K.
    """

    matrix = np.asarray(b_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise PulayMixerError(f"B must be square, got shape {matrix.shape}")

    inverse = np.linalg.pinv(matrix)
    weights = inverse.sum(axis=0)
    total = weights.sum()
    if not np.isfinite(total) or abs(total) < 1e-30:
        raise PulayMixerError("B is singular; the native DIIS solve has no solution")
    return weights / total


def residual_norm(b_matrix: np.ndarray, coefficients: np.ndarray) -> float:
    """The DIIS residual of a coefficient vector: ``sqrt(c^T B c)``.

    This is the quantity the Fortran gate compares against native Pulay's, and
    the only thing about a proposal that is not guaranteed by construction.
    """

    matrix = np.asarray(b_matrix, dtype=float)
    vector = np.asarray(coefficients, dtype=float)
    value = float(vector @ matrix @ vector)
    if not np.isfinite(value) or value < 0.0:
        return float("nan")
    return float(np.sqrt(value))


@dataclass(frozen=True, slots=True)
class MixerSample:
    """One training example: a B matrix and the coefficients it should give."""

    b_matrix: np.ndarray
    target: np.ndarray
    iteration: int
    system: str = ""

    def __post_init__(self) -> None:
        if self.b_matrix.shape != (self.target.size, self.target.size):
            raise PulayMixerError(
                f"B {self.b_matrix.shape} does not match target length {self.target.size}"
            )


def build_b_matrix(residuals: Sequence[np.ndarray]) -> np.ndarray:
    """Assemble ``B_ij = <r_i | r_j>`` from a residual history.

    CP2K builds this in g-space with a metric; here it is the plain Frobenius
    inner product of AO-space residuals, which is what a recorded trajectory
    provides. The learned map is from the *shape* of B to the coefficients, and
    that shape is what both metrics share.
    """

    history = [np.asarray(residual, dtype=float).ravel() for residual in residuals]
    depth = len(history)
    if depth == 0:
        raise PulayMixerError("cannot build B from an empty residual history")
    matrix = np.empty((depth, depth), dtype=float)
    for i in range(depth):
        for j in range(i, depth):
            value = float(history[i] @ history[j])
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def samples_from_trajectory(
    densities: Sequence[np.ndarray],
    overlap: np.ndarray,
    *,
    max_history: int = DEFAULT_MAX_HISTORY,
    system: str = "",
) -> list[MixerSample]:
    """Turn one recorded SCF trajectory into training samples.

    For each iteration ``n`` the input is the history available *before* that
    iteration, and the target is the coefficient vector whose combination lands
    closest to the converged density -- which is the "optimal next density from
    the converged trajectory" a mixer should have produced.

    The target is solved as a constrained least-squares problem

        minimise  || sum_i c_i P_i - P_final ||   subject to  sum_i c_i = 1,

    so the target itself conserves electrons by construction and the model is
    never trained toward a vector that CP2K would reject.
    """

    frames = [np.asarray(density, dtype=float) for density in densities]
    if len(frames) < 3:
        return []
    converged = frames[-1]
    # The residual of iterate k is how far it sits from the answer. Using the
    # converged density makes these exact rather than the finite-difference
    # approximation an online mixer has to work with; that is the point of
    # training on completed trajectories.
    residuals = [frame - converged for frame in frames[:-1]]

    samples: list[MixerSample] = []
    for end in range(2, len(residuals) + 1):
        start = max(0, end - max_history)
        window = residuals[start:end]
        depth = len(window)
        if depth < 2:
            continue

        b_matrix = build_b_matrix(window)
        history = np.stack([frames[start + offset].ravel() for offset in range(depth)], axis=0)
        target = _constrained_least_squares(history, converged.ravel())
        samples.append(
            MixerSample(b_matrix=b_matrix, target=target, iteration=end, system=system)
        )
    return samples


def _constrained_least_squares(history: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve ``min ||c^T H - t||`` subject to ``sum(c) = 1`` via a KKT system."""

    depth = history.shape[0]
    gram = history @ history.T
    rhs = history @ target

    kkt = np.zeros((depth + 1, depth + 1), dtype=float)
    kkt[:depth, :depth] = gram
    kkt[depth, :depth] = 1.0
    kkt[:depth, depth] = 1.0
    right = np.zeros(depth + 1, dtype=float)
    right[:depth] = rhs
    right[depth] = 1.0

    solution, *_ = np.linalg.lstsq(kkt, right, rcond=None)
    coefficients = solution[:depth]
    # lstsq on a singular KKT system can drift off the constraint; re-impose it
    # rather than emit a target the Fortran gate would reject.
    total = coefficients.sum()
    if not np.isfinite(total) or abs(total) < 1e-12:
        uniform = np.zeros(depth, dtype=float)
        uniform[-1] = 1.0
        return uniform
    return coefficients / total


# -- the model ----------------------------------------------------------


def build_mixer_module(torch: Any, max_history: int = DEFAULT_MAX_HISTORY) -> Any:
    """Build the TorchScript-able module CP2K's shim calls.

    Signature: ``forward(b_matrix, iteration) -> coefficients``, with
    ``b_matrix`` of shape ``(1, nb, nb)`` and the result of length ``nb``.

    ``nb`` varies from iteration to iteration (the history fills up), so the
    module is written to accept any ``nb <= max_history`` rather than a fixed
    size. The output is projected onto ``sum(c) = 1`` as the last operation, so
    the constraint holds whatever the network does.
    """

    nn = torch.nn
    from torch import Tensor

    class PulayCoefficientModel(nn.Module):
        def __init__(self, depth: int) -> None:
            super().__init__()
            self.max_history = depth
            # The input is B rescaled to be scale-free, flattened and padded.
            self.net = nn.Sequential(
                nn.Linear(depth * depth + 1, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, depth),
            )

        def forward(self, b_matrix: Tensor, iteration: Tensor) -> Tensor:
            b = b_matrix.to(torch.float64)
            if b.dim() == 3:
                b = b[0]
            nb = b.shape[0]

            # B's scale is the residual norm squared and spans orders of
            # magnitude over an SCF. Only its shape carries information about
            # the coefficients, so it is normalised away.
            scale = b.abs().max()
            if float(scale) > 0.0:
                b = b / scale

            padded = torch.zeros(
                (self.max_history, self.max_history), dtype=torch.float64, device=b.device
            )
            padded[:nb, :nb] = b

            features = torch.cat(
                [padded.reshape(-1), iteration.to(torch.float64).reshape(-1)[:1]], dim=0
            )
            raw = self.net(features.to(torch.float32)).to(torch.float64)[:nb]

            # sum(c) = 1 by construction. A softmax would also normalise but
            # would force every coefficient positive, and DIIS coefficients are
            # legitimately negative -- that is how it extrapolates.
            total = raw.sum()
            if torch.abs(total) < 1e-12:
                fallback = torch.zeros(nb, dtype=torch.float64, device=b.device)
                fallback[nb - 1] = 1.0
                return fallback
            return raw / total

    return PulayCoefficientModel(max_history)


def build_reference_module(torch: Any) -> Any:
    """A module that reproduces CP2K's own DIIS solve exactly.

    Not a learned model: it exists to exercise the whole path -- shim, gate,
    density update -- with a proposal that is known-good, so that an accepted
    prediction can be observed end to end before any training has happened. A
    model that cannot beat this is not worth deploying.
    """

    nn = torch.nn
    from torch import Tensor

    class NativeDIIS(nn.Module):
        def forward(self, b_matrix: Tensor, iteration: Tensor) -> Tensor:
            b = b_matrix.to(torch.float64)
            if b.dim() == 3:
                b = b[0]
            inverse = torch.linalg.pinv(b)
            weights = inverse.sum(dim=0)
            total = weights.sum()
            nb = b.shape[0]
            if torch.abs(total) < 1e-30:
                fallback = torch.zeros(nb, dtype=torch.float64, device=b.device)
                fallback[nb - 1] = 1.0
                return fallback
            return weights / total

    return NativeDIIS()


def export_mixer(module: Any, torch: Any, path: str | Path, *, experimental: bool = False) -> Path:
    """Script a mixer module and check it on a probe before writing it.

    Deliberate deviation from PHAL (finding F-23): PHAL's copy of this function
    scripted and saved a mixer without the ``experimental=True`` opt-in that
    ``deploy.export_mixer`` enforces, which contradicted
    ``docs/ml_pulay_objective.md`` ("cannot be wired into a production
    configuration by accident"). Both export routes now require the opt-in.
    """

    if not experimental:
        raise PulayMixerError(
            "export_mixer is experimental and a documented negative result: native DIIS "
            "already solves min c^T B c exactly, so a learned mixer cannot beat it "
            "(docs/scf/ml_pulay_objective.md). Pass experimental=True to export one anyway."
        )

    try:
        scripted = torch.jit.script(module)
    except Exception as exc:  # noqa: BLE001 - reported as an export failure
        raise PulayMixerError(f"the mixer could not be scripted: {exc}") from exc

    rng = np.random.default_rng(0)
    for nb in (2, 4, DEFAULT_MAX_HISTORY):
        raw = rng.normal(size=(nb, nb))
        probe = torch.tensor(raw @ raw.T + nb * np.eye(nb), dtype=torch.float64).unsqueeze(0)
        output = scripted(probe, torch.tensor([[float(nb)]], dtype=torch.float64))
        if output.numel() != nb:
            raise PulayMixerError(f"the mixer returned {output.numel()} coefficients for nb={nb}")
        if not bool(torch.isfinite(output).all()):
            raise PulayMixerError(f"the mixer produced non-finite coefficients for nb={nb}")
        total = float(output.sum())
        if abs(total - 1.0) > 1e-6:
            raise PulayMixerError(
                f"the mixer's coefficients sum to {total:.6f}, not 1, for nb={nb}; "
                "electron conservation depends on that sum and CP2K will reject it"
            )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(target))
    return target
