# Ported from PHAL (hotspot_al.cp2k_ml_scf.training.mixer_model) at 96263b6; MIT License, see NOTICE.
"""The neural density mixer CP2K calls each SCF iteration (Stage 12).

The mixer sees the density history and the residual history and proposes the
next input density.  It replaces the Pulay/Broyden *update rule* and nothing
else: the Hamiltonian, the KS solver and the ``EPS_SCF`` test are untouched.

The architecture is a learned DIIS.  Classical DIIS solves a small linear
system for coefficients that minimise the norm of the extrapolated residual;
this model predicts those coefficients instead, from features of the residual
history.  Two consequences follow, and both are deliberate:

* The proposal is always an affine combination of densities CP2K itself
  produced.  A network free to emit arbitrary matrix elements could invent a
  density that is nowhere near the SCF manifold; this one cannot.
* Coefficients summing to one make the proposal electron-conserving whenever
  the history is, which is what lets the Fortran gate accept it.

The model is exported to TorchScript because CP2K calls it in-process through
``phal_ml_torch.cpp``; it therefore must be scriptable, which rules out Python
control flow that depends on tensor values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from gmd_active_learning.adapters.dft.scf._torch import require_torch

MIXER_ARCHITECTURE = "phal-scf-neural-mixer-v1"
MIXER_CHECKPOINT_VERSION = "phal-scf-mixer-checkpoint-v1"


class MixerModelError(RuntimeError):
    """Raised when a mixer model cannot be built, trained, or exported."""


@dataclass(frozen=True, slots=True)
class MixerConfig:
    """Architecture of the neural mixer."""

    max_history: int = 8
    hidden_size: int = 32
    n_layers: int = 2
    #: Features per history entry: residual norm, log residual norm, overlap
    #: with the newest residual, density-change norm, recency.
    n_features: int = 5
    architecture: str = MIXER_ARCHITECTURE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MixerConfig":
        known = {key: payload[key] for key in payload if key in cls.__slots__}
        return cls(**known)


def build_mixer_model(config: MixerConfig) -> Any:
    """Construct the neural mixer described by ``config``.

    The returned module's ``forward`` takes ``(density_history,
    residual_history)`` shaped ``(nhist, nspin, ndim)`` -- exactly what
    ``phal_ml_mix`` hands over -- and returns one ``(nspin, ndim)`` proposal.
    """

    torch = require_torch()
    nn = torch.nn
    # TorchScript resolves annotations from the enclosing scope, so the tensor
    # type must be a real name here rather than the module-level `Any` alias.
    from torch import Tensor

    class NeuralMixer(nn.Module):
        """Predict DIIS-like coefficients over the density history."""

        def __init__(self, model_config: MixerConfig) -> None:
            super().__init__()
            self.max_history = model_config.max_history
            hidden = model_config.hidden_size

            layers: list[Any] = [nn.Linear(model_config.n_features, hidden), nn.SiLU()]
            for _ in range(max(0, model_config.n_layers - 1)):
                layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
            self.encoder = nn.Sequential(*layers)
            self.coefficient_head = nn.Linear(hidden, 1)
            # A learned damping on the residual admixture, kept small at
            # initialisation so an untrained mixer behaves like plain
            # extrapolation rather than a random kick.
            self.damping = nn.Parameter(torch.tensor(0.05))

        def features(self, density_history: Tensor, residual_history: Tensor) -> Tensor:
            nhist = residual_history.shape[0]
            flat_residual = residual_history.reshape(nhist, -1)
            flat_density = density_history.reshape(nhist, -1)

            norms = torch.linalg.norm(flat_residual, dim=1)
            newest = flat_residual[nhist - 1]
            newest_norm = torch.linalg.norm(newest).clamp(min=1e-30)
            overlap = (flat_residual @ newest) / (norms.clamp(min=1e-30) * newest_norm)
            change = torch.linalg.norm(flat_density - flat_density[nhist - 1], dim=1)
            recency = torch.arange(
                nhist, dtype=density_history.dtype, device=density_history.device
            ) / float(max(nhist - 1, 1))

            return torch.stack(
                [
                    norms / norms.max().clamp(min=1e-30),
                    torch.log1p(norms),
                    overlap,
                    change / change.max().clamp(min=1e-30),
                    recency,
                ],
                dim=1,
            )

        def forward(self, density_history: Tensor, residual_history: Tensor) -> Tensor:
            features = self.features(density_history, residual_history)
            logits = self.coefficient_head(self.encoder(features)).squeeze(-1)
            # Softmax coefficients sum to one, which is what makes the
            # proposal an affine combination of densities CP2K produced and
            # therefore electron-conserving by construction.
            coefficients = torch.softmax(logits, dim=0)

            nhist = density_history.shape[0]
            weights = coefficients.view(nhist, 1, 1)
            extrapolated = (density_history * weights).sum(dim=0)
            correction = (residual_history * weights).sum(dim=0)
            return extrapolated + self.damping * correction

    return NeuralMixer(config)


def save_mixer_checkpoint(
    path: str | Path,
    *,
    model: Any,
    config: MixerConfig,
    metrics: Mapping[str, Any] | None = None,
) -> Path:
    """Write a mixer checkpoint for later export or retraining."""

    torch = require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": MIXER_CHECKPOINT_VERSION,
            "model_config": config.to_dict(),
            "state_dict": model.state_dict(),
            "metrics": dict(metrics or {}),
        },
        target,
    )
    return target


def load_mixer_checkpoint(path: str | Path) -> tuple[Any, MixerConfig]:
    """Load a mixer checkpoint, refusing an unsupported version."""

    torch = require_torch()
    source = Path(path)
    if not source.is_file():
        raise MixerModelError(f"mixer checkpoint not found: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)

    version = str(payload.get("checkpoint_version", ""))
    if version != MIXER_CHECKPOINT_VERSION:
        raise MixerModelError(
            f"{source}: checkpoint_version {version!r} is not supported "
            f"(expected {MIXER_CHECKPOINT_VERSION!r})"
        )

    config = MixerConfig.from_dict(payload["model_config"])
    model = build_mixer_model(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config


def pulay_reference(density_history: Any, residual_history: Any) -> Any:
    """Classical DIIS/Pulay extrapolation, the baseline the mixer must beat.

    Provided so a mixer can be trained against, and compared with, the thing it
    replaces rather than against nothing.
    """

    torch = require_torch()

    nhist = residual_history.shape[0]
    flat = residual_history.reshape(nhist, -1)
    overlap = flat @ flat.T
    overlap = overlap + 1.0e-10 * torch.eye(nhist, dtype=overlap.dtype) * overlap.diagonal().max().clamp(min=1.0)

    # Solve the constrained problem min ||sum c_i r_i|| subject to sum c_i = 1
    # via the standard bordered system.
    ones = torch.ones((nhist, 1), dtype=overlap.dtype)
    top = torch.cat([overlap, ones], dim=1)
    bottom = torch.cat([ones.T, torch.zeros((1, 1), dtype=overlap.dtype)], dim=1)
    system = torch.cat([top, bottom], dim=0)
    rhs = torch.zeros((nhist + 1, 1), dtype=overlap.dtype)
    rhs[nhist, 0] = 1.0

    try:
        solution = torch.linalg.solve(system, rhs)
    except Exception as exc:  # pragma: no cover - singular history
        raise MixerModelError(f"the Pulay reference system is singular: {exc}") from exc

    coefficients = solution[:nhist, 0].view(nhist, 1, 1)
    return (density_history * coefficients).sum(dim=0)


__all__ = [
    "MIXER_ARCHITECTURE",
    "MIXER_CHECKPOINT_VERSION",
    "MixerConfig",
    "MixerModelError",
    "build_mixer_model",
    "load_mixer_checkpoint",
    "pulay_reference",
    "save_mixer_checkpoint",
]
