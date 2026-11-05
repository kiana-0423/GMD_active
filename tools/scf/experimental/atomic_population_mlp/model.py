# Ported from PHAL (hotspot_al.cp2k_ml_scf.training.models) at 96263b6; MIT License, see NOTICE.
"""PyTorch models that predict an initial density representation.

The model predicts *how the electrons are distributed*, not how many there are.
Its density head emits a masked softmax over atoms, which is then multiplied by
the electron count CP2K asked for.  Electron conservation is therefore exact by
construction rather than something a loss term has to encourage -- a model that
lost half an electron would produce an initial guess CP2K must reject, so this
is the one property worth building into the architecture.

An auxiliary head predicts the SCF iteration count.  It is not used by CP2K; it
exists so a trained model can estimate, before a calculation runs, whether the
system is one where acceleration is likely to pay off.

Torch is an optional dependency.  Importing this module without it raises a
clear error rather than failing deep inside a training loop.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gmd_active_learning.adapters.dft.scf.density.predictor import SCFDensityPredictor, expected_density_electrons
from gmd_active_learning.adapters.dft.scf.density.prediction import (
    DENSITY_FORMAT_ATOMIC_POPULATION,
    PredictionError,
    SCFDensityPrediction,
)
from gmd_active_learning.adapters.dft.scf.features import feature_metadata, featurize
from gmd_active_learning.adapters.dft.scf.schema import SCFElectronicState, SCFStructure
from gmd_active_learning.adapters.dft.scf.schema import structure_hash as compute_structure_hash
from tools.scf.training.dataset_loader import Normalization
from gmd_active_learning.adapters.dft.scf._torch import (  # re-exported for PHAL compatibility
    TorchUnavailableError,
    model_digest,
    require_torch,
    torch_available,
)

MODEL_ARCHITECTURE = "phal-scf-density-mlp-v1"
CHECKPOINT_VERSION = "phal-scf-checkpoint-v1"
MAX_SPIN_CHANNELS = 2


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture hyper-parameters, stored with every checkpoint."""

    n_system_features: int
    n_per_atom_features: int
    hidden_size: int = 64
    n_layers: int = 3
    dropout: float = 0.0
    max_spin_channels: int = MAX_SPIN_CHANNELS
    architecture: str = MODEL_ARCHITECTURE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelConfig":
        known = {key: payload[key] for key in payload if key in cls.__slots__}
        return cls(**known)

    @classmethod
    def from_features(cls, **overrides: Any) -> "ModelConfig":
        metadata = feature_metadata()
        return cls(
            n_system_features=int(metadata["n_system_features"]),
            n_per_atom_features=int(metadata["n_per_atom_features"]),
            **overrides,
        )


def build_model(config: ModelConfig) -> Any:
    """Construct the density model described by ``config``."""

    torch = require_torch()
    nn = torch.nn

    class SCFDensityModel(nn.Module):
        """Per-atom density fractions plus an SCF-iteration estimate."""

        def __init__(self, model_config: ModelConfig) -> None:
            super().__init__()
            self.config = model_config
            hidden = model_config.hidden_size

            self.system_encoder = nn.Sequential(
                nn.Linear(model_config.n_system_features, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            self.atom_encoder = nn.Sequential(
                nn.Linear(model_config.n_per_atom_features, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )

            layers: list[Any] = []
            for _ in range(max(1, model_config.n_layers - 1)):
                layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
                if model_config.dropout > 0.0:
                    layers.append(nn.Dropout(model_config.dropout))
            self.trunk = nn.Sequential(*layers)
            self.density_head = nn.Linear(hidden, model_config.max_spin_channels)
            self.iteration_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

        def forward(
            self,
            system_features: Any,
            per_atom_features: Any,
            atom_mask: Any,
        ) -> dict[str, Any]:
            system = self.system_encoder(system_features).unsqueeze(1)
            atoms = self.atom_encoder(per_atom_features)
            joined = self.trunk(atoms + system)

            logits = self.density_head(joined).transpose(1, 2)  # (B, channels, atoms)
            mask = atom_mask.unsqueeze(1)
            # Padded atoms must receive exactly zero share, so they are pushed
            # to -inf before the softmax rather than merely down-weighted.
            masked = logits.masked_fill(mask <= 0, float("-inf"))
            fractions = torch.softmax(masked, dim=-1)
            fractions = torch.nan_to_num(fractions, nan=0.0) * mask

            pooled = (joined * atom_mask.unsqueeze(-1)).sum(dim=1) / atom_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            log_iterations = self.iteration_head(pooled).squeeze(-1)
            return {"fractions": fractions, "log_iterations": log_iterations}

    return SCFDensityModel(config)


def channel_electron_counts(
    electron_count: float,
    multiplicity: int,
    spin_channels: int,
) -> tuple[float, ...]:
    """Split an electron count across spin channels for the given multiplicity."""

    if spin_channels == 1:
        return (float(electron_count),)
    unpaired = float(multiplicity - 1)
    return ((electron_count + unpaired) / 2.0, (electron_count - unpaired) / 2.0)


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Everything needed to reproduce a prediction from a trained model."""

    model_config: ModelConfig
    state_dict: Mapping[str, Any]
    normalization: Normalization
    feature_version: str
    target: str
    model_name: str
    model_version: str
    metrics: Mapping[str, Any]
    metadata: Mapping[str, Any]


def save_checkpoint(
    path: str | Path,
    *,
    model: Any,
    model_config: ModelConfig,
    normalization: Normalization,
    target: str,
    model_name: str,
    model_version: str,
    metrics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a checkpoint plus a sidecar JSON describing it."""

    torch = require_torch()
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    from gmd_active_learning.adapters.dft.scf.features import FEATURE_VERSION

    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_config": model_config.to_dict(),
        "state_dict": model.state_dict(),
        "normalization": normalization.to_dict(),
        "feature_version": FEATURE_VERSION,
        "target": target,
        "model_name": model_name,
        "model_version": model_version,
        "metrics": dict(metrics or {}),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, target_path)

    sidecar = target_path.with_suffix(target_path.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in {"state_dict", "normalization"}
            }
            | {"model_digest": model_digest(target_path)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target_path


def load_checkpoint(path: str | Path) -> Checkpoint:
    """Read a checkpoint, refusing one built from a different feature layout."""

    torch = require_torch()
    source = Path(path)
    if not source.is_file():
        raise PredictionError(f"model checkpoint not found: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise PredictionError(f"{source}: checkpoint is not a mapping")

    version = str(payload.get("checkpoint_version", ""))
    if version != CHECKPOINT_VERSION:
        raise PredictionError(
            f"{source}: checkpoint_version {version!r} is not supported (expected {CHECKPOINT_VERSION!r})"
        )

    from gmd_active_learning.adapters.dft.scf.features import FEATURE_VERSION

    feature_version = str(payload.get("feature_version", ""))
    if feature_version != FEATURE_VERSION:
        raise PredictionError(
            f"{source}: was trained on feature_version {feature_version!r}, "
            f"this build produces {FEATURE_VERSION!r}; retrain before use"
        )

    return Checkpoint(
        model_config=ModelConfig.from_dict(payload["model_config"]),
        state_dict=payload["state_dict"],
        normalization=Normalization.from_dict(payload["normalization"]),
        feature_version=feature_version,
        target=str(payload.get("target", "atomic_population")),
        model_name=str(payload.get("model_name", "phal-scf-density")),
        model_version=str(payload.get("model_version", "0")),
        metrics=dict(payload.get("metrics", {})),
        metadata=dict(payload.get("metadata", {})),
    )


class TorchDensityPredictor(SCFDensityPredictor):
    """Serve a trained checkpoint through the Stage 2 predictor contract."""

    predictor_type = "torch"
    density_format = DENSITY_FORMAT_ATOMIC_POPULATION

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config=config)
        model_path = self.config.get("model_path")
        if not model_path:
            options = self.config.get("options", {})
            model_path = options.get("model_path") if isinstance(options, Mapping) else None
        if not model_path:
            raise PredictionError(
                "the torch predictor requires cp2k.ml_scf.predictor.model_path to point at a checkpoint"
            )
        self.model_path = Path(str(model_path))
        self._model: Any | None = None
        self._checkpoint: Checkpoint | None = None
        self._digest: str | None = None

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return self._checkpoint.model_name if self._checkpoint else "phal-scf-density"

    @property
    def model_version(self) -> str:  # type: ignore[override]
        return self._checkpoint.model_version if self._checkpoint else "0"

    @property
    def model_digest(self) -> str:
        if self._digest is None:
            self._digest = model_digest(self.model_path)
        return self._digest

    def load_model(self) -> "TorchDensityPredictor":
        if self._loaded:
            return self
        torch = require_torch()
        checkpoint = load_checkpoint(self.model_path)
        model = build_model(checkpoint.model_config)
        model.load_state_dict(checkpoint.state_dict)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model
        self._checkpoint = checkpoint
        self._digest = model_digest(self.model_path)
        self._loaded = True
        del torch
        return self

    def predict(
        self,
        structure: SCFStructure,
        electronic_state: SCFElectronicState,
    ) -> SCFDensityPrediction:
        if not self._loaded:
            self.load_model()
        assert self._model is not None and self._checkpoint is not None  # for type checkers
        torch = require_torch()

        if structure.n_atoms == 0:
            raise PredictionError("cannot predict a density for a structure with no atoms")

        features = featurize(structure, electronic_state)
        normalized = self._checkpoint.normalization.apply(features.flat()[None, :])

        with torch.no_grad():
            output = self._model(
                torch.as_tensor(normalized, dtype=torch.float32),
                torch.as_tensor(features.per_atom[None, ...], dtype=torch.float32),
                torch.ones((1, structure.n_atoms), dtype=torch.float32),
            )
            fractions = output["fractions"][0].cpu().numpy()
            predicted_iterations = float(np.expm1(float(output["log_iterations"][0].cpu())))

        channels = electronic_state.spin_channels
        # Deliberate fix (finding F-23): distribute the GTH *valence* count, not
        # the all-electron sum(Z) - charge.
        target_electrons = expected_density_electrons(structure, electronic_state)
        counts = channel_electron_counts(
            target_electrons, electronic_state.multiplicity, channels
        )
        values = np.vstack(
            [fractions[index] * counts[index] for index in range(channels)]
        )

        return SCFDensityPrediction(
            model_name=self.model_name,
            model_version=self.model_version,
            model_digest=self.model_digest,
            structure_hash=compute_structure_hash(structure, electronic_state),
            electron_count=float(target_electrons),
            spin_channels=channels,
            density_format=self.density_format,
            values=values,
            predictor_type=self.predictor_type,
            metadata={
                "model_path": str(self.model_path),
                "target": self._checkpoint.target,
                "predicted_scf_iterations": round(max(predicted_iterations, 0.0), 3),
            },
        )


__all__ = [
    "CHECKPOINT_VERSION",
    "MAX_SPIN_CHANNELS",
    "MODEL_ARCHITECTURE",
    "Checkpoint",
    "ModelConfig",
    "TorchDensityPredictor",
    "TorchUnavailableError",
    "build_model",
    "channel_electron_counts",
    "load_checkpoint",
    "model_digest",
    "require_torch",
    "save_checkpoint",
    "torch_available",
]
