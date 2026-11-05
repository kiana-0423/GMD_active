# Ported from PHAL (hotspot_al.cp2k_ml_scf.training.train) at 96263b6; MIT License, see NOTICE.
"""Training loop for ML-SCF density models.

The loop is intentionally small and dependency-light: one optimizer, one
validation pass per epoch, best-checkpoint selection on validation density
error.  Everything that matters for reproducibility -- feature version, split
counts, seed, metrics per epoch -- is written next to the checkpoint, because a
model whose provenance is unknown cannot be trusted to seed an SCF.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tools.scf.training.dataset_loader import (
    TARGET_ATOMIC_POPULATION,
    DataError,
    Normalization,
    SCFExampleSet,
    iterate_batches,
    load_split,
)
from tools.scf.experimental.atomic_population_mlp.model import (
    ModelConfig,
    build_model,
    require_torch,
    save_checkpoint,
)
from gmd_active_learning.utils.atomic_io import write_json_atomic


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyper-parameters for one training run."""

    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    iteration_loss_weight: float = 0.1
    seed: int = 0
    hidden_size: int = 64
    n_layers: int = 3
    dropout: float = 0.0
    target: str = TARGET_ATOMIC_POPULATION
    model_name: str = "phal-scf-density"
    model_version: str = "1"
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass(slots=True)
class EpochMetrics:
    """Metrics for one epoch on one split."""

    epoch: int
    split: str
    loss: float
    density_mae: float
    density_rmse: float
    electron_error: float
    iteration_mae: float

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass(slots=True)
class TrainingResult:
    """What a training run produced."""

    checkpoint_path: Path
    history: list[EpochMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_validation_density_mae: float = float("inf")
    train_summary: Mapping[str, Any] = field(default_factory=dict)
    validation_summary: Mapping[str, Any] = field(default_factory=dict)
    wall_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "best_epoch": self.best_epoch,
            "best_validation_density_mae": self.best_validation_density_mae,
            "history": [metrics.to_dict() for metrics in self.history],
            "train": dict(self.train_summary),
            "validation": dict(self.validation_summary),
            "wall_time_seconds": round(self.wall_time_seconds, 3),
        }


def train_density_model(
    train_set: SCFExampleSet,
    validation_set: SCFExampleSet | None,
    output_path: str | Path,
    *,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train a density model and write the best checkpoint to ``output_path``."""

    settings = config or TrainingConfig()
    if len(train_set) == 0:
        raise DataError("cannot train on an empty training split")

    torch = require_torch()
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)

    normalization = Normalization.fit(train_set.system_matrix())
    model_config = ModelConfig.from_features(
        hidden_size=settings.hidden_size,
        n_layers=settings.n_layers,
        dropout=settings.dropout,
    )
    device = torch.device(settings.device)
    model = build_model(model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )

    checkpoint_path = Path(output_path)
    result = TrainingResult(
        checkpoint_path=checkpoint_path,
        train_summary=train_set.summary(),
        validation_summary=validation_set.summary() if validation_set else {},
    )

    started = time.perf_counter()
    for epoch in range(1, settings.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_set,
            settings,
            normalization,
            device=device,
            optimizer=optimizer,
            epoch=epoch,
            split="train",
            torch=torch,
        )
        result.history.append(train_metrics)

        if validation_set is not None and len(validation_set) > 0:
            validation_metrics = _run_epoch(
                model,
                validation_set,
                settings,
                normalization,
                device=device,
                optimizer=None,
                epoch=epoch,
                split="validation",
                torch=torch,
            )
            result.history.append(validation_metrics)
            selection = validation_metrics.density_mae
        else:
            selection = train_metrics.density_mae

        if selection < result.best_validation_density_mae:
            result.best_validation_density_mae = selection
            result.best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                model=model,
                model_config=model_config,
                normalization=normalization,
                target=train_set.target,
                model_name=settings.model_name,
                model_version=settings.model_version,
                metrics={"selection_density_mae": selection, "epoch": epoch},
                metadata={
                    "training_config": settings.to_dict(),
                    "train": train_set.summary(),
                    "validation": validation_set.summary() if validation_set else {},
                },
            )

    result.wall_time_seconds = time.perf_counter() - started

    if result.best_epoch == 0:  # pragma: no cover - epochs <= 0 is rejected below
        raise DataError("training produced no checkpoint; check that epochs is positive")

    write_json_atomic(checkpoint_path.with_suffix(checkpoint_path.suffix + ".history.json"), result.to_dict())
    return result


def train_from_dataset(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    config: TrainingConfig | None = None,
    require_converged: bool = True,
) -> TrainingResult:
    """Load the train/validation splits from disk and train on them."""

    settings = config or TrainingConfig()
    train_set = load_split(
        dataset_root, "train", target=settings.target, require_converged=require_converged
    )
    validation_set = load_split(
        dataset_root, "validation", target=settings.target, require_converged=require_converged
    )
    return train_density_model(train_set, validation_set, output_path, config=settings)


def _run_epoch(
    model: Any,
    examples: SCFExampleSet,
    settings: TrainingConfig,
    normalization: Normalization,
    *,
    device: Any,
    optimizer: Any | None,
    epoch: int,
    split: str,
    torch: Any,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)

    totals = {
        "loss": 0.0,
        "abs_error": 0.0,
        "squared_error": 0.0,
        "electron_error": 0.0,
        "iteration_error": 0.0,
    }
    n_entries = 0.0
    n_examples = 0

    for batch in iterate_batches(
        examples, settings.batch_size, shuffle=training, seed=settings.seed + epoch
    ):
        system = torch.as_tensor(
            normalization.apply(batch["system_features"]), dtype=torch.float32, device=device
        )
        per_atom = torch.as_tensor(batch["per_atom_features"], dtype=torch.float32, device=device)
        atom_mask = torch.as_tensor(batch["atom_mask"], dtype=torch.float32, device=device)
        target = torch.as_tensor(batch["density_target"], dtype=torch.float32, device=device)
        channel_mask = torch.as_tensor(batch["channel_mask"], dtype=torch.float32, device=device)
        iterations = torch.as_tensor(batch["n_scf_iterations"], dtype=torch.float32, device=device)

        with torch.set_grad_enabled(training):
            output = model(system, per_atom, atom_mask)
            # The target already carries the electron count, so the predicted
            # fractions are rescaled to the per-channel electron totals before
            # they are compared.
            channel_totals = target.sum(dim=-1, keepdim=True)
            predicted = output["fractions"] * channel_totals

            mask = atom_mask.unsqueeze(1) * channel_mask.unsqueeze(-1)
            n_masked = mask.sum().clamp(min=1.0)
            density_loss = (((predicted - target) ** 2) * mask).sum() / n_masked
            iteration_loss = torch.nn.functional.mse_loss(
                output["log_iterations"], torch.log1p(iterations)
            )
            loss = density_loss + settings.iteration_loss_weight * iteration_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        with torch.no_grad():
            errors = (predicted - target).abs() * mask
            totals["loss"] += float(loss) * float(mask.shape[0])
            totals["abs_error"] += float(errors.sum())
            totals["squared_error"] += float((((predicted - target) ** 2) * mask).sum())
            totals["electron_error"] += float(
                (predicted.sum(dim=(1, 2)) - target.sum(dim=(1, 2))).abs().sum()
            )
            predicted_iterations = torch.expm1(output["log_iterations"])
            totals["iteration_error"] += float((predicted_iterations - iterations).abs().sum())
            n_entries += float(mask.sum())
            n_examples += int(mask.shape[0])

    scale = max(n_entries, 1.0)
    count = max(n_examples, 1)
    return EpochMetrics(
        epoch=epoch,
        split=split,
        loss=totals["loss"] / count,
        density_mae=totals["abs_error"] / scale,
        density_rmse=float(np.sqrt(totals["squared_error"] / scale)),
        electron_error=totals["electron_error"] / count,
        iteration_mae=totals["iteration_error"] / count,
    )


def write_training_report(result: TrainingResult, path: str | Path) -> Path:
    """Write a standalone JSON report for a completed training run."""

    return write_json_atomic(path, result.to_dict())


def load_training_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "EpochMetrics",
    "TrainingConfig",
    "TrainingResult",
    "load_training_report",
    "train_density_model",
    "train_from_dataset",
    "write_training_report",
]
