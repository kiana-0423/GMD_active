# Ported from PHAL (hotspot_al.cp2k_ml_scf.training.train_density) at 96263b6; MIT License, see NOTICE.
"""Train the AO density-matrix model (Stage 10).

The AO basis is not a hyper-parameter: it is derived from the dataset with
:meth:`AOBasis.fit` and stored in the checkpoint.  A model served against a
different basis would produce a matrix of the wrong size, so the checkpoint
carries the basis it was fitted to and inference refuses a mismatch.

Checkpoint selection is on validation density error.  A run with no validation
split falls back to training error and says so in the report, because silently
selecting on training loss is how an overfit checkpoint gets deployed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.scf.training.dataset_loader import (
    TARGET_DENSITY_MATRIX,
    DataError,
    SCFExampleSet,
    load_split,
)
from gmd_active_learning.adapters.dft.scf.density.gnn_model import (
    AOBasis,
    DensityGNNConfig,
    DensityModelError,
    LossWeights,
    batch_to_tensors,
    build_density_model,
    collate_density_batch,
    density_loss,
    forward_density,
    predicted_electron_counts,
)
from gmd_active_learning.adapters.dft.scf.density.gnn_model import (
    DENSITY_CHECKPOINT_VERSION,
    ValenceTable,
    load_density_checkpoint,
)
from gmd_active_learning.adapters.dft.scf._torch import model_digest, require_torch
from gmd_active_learning.utils.atomic_io import write_json_atomic



@dataclass(frozen=True, slots=True)
class DensityTrainingConfig:
    """Hyper-parameters for one density-model training run."""

    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    seed: int = 0
    hidden_size: int = 64
    n_interactions: int = 3
    n_radial_basis: int = 24
    cutoff: float = 6.0
    loss_weights: LossWeights = field(default_factory=LossWeights)
    model_name: str = "phal-scf-density-gnn"
    model_version: str = "1"
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["loss_weights"] = self.loss_weights.to_dict()
        return payload


@dataclass(slots=True)
class DensityEpochMetrics:
    """Per-epoch metrics on one split."""

    epoch: int
    split: str
    loss: float
    density_mae: float
    density_rmse: float
    electron_mae: float
    max_asymmetry: float
    breakdown: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["breakdown"] = dict(self.breakdown)
        return payload


@dataclass(slots=True)
class DensityTrainingResult:
    """What a density training run produced."""

    checkpoint_path: Path
    basis: AOBasis
    valence: ValenceTable | None = None
    #: Why no valence table could be fitted, when none could. Kept so the
    #: reason surfaces at training time rather than at deployment time.
    valence_error: str | None = None
    history: list[DensityEpochMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_density_mae: float = float("inf")
    selected_on: str = "validation"
    train_summary: Mapping[str, Any] = field(default_factory=dict)
    validation_summary: Mapping[str, Any] = field(default_factory=dict)
    wall_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "basis": self.basis.to_dict(),
            "valence": self.valence.to_dict() if self.valence else None,
            "valence_error": self.valence_error,
            "best_epoch": self.best_epoch,
            "best_density_mae": self.best_density_mae,
            "selected_on": self.selected_on,
            "history": [metrics.to_dict() for metrics in self.history],
            "train": dict(self.train_summary),
            "validation": dict(self.validation_summary),
            "wall_time_seconds": round(self.wall_time_seconds, 3),
        }


def fit_basis(*example_sets: SCFExampleSet, basis_set: str | None = None) -> AOBasis:
    """Derive the AO basis from every example that carries a density."""

    compositions: list[tuple[Sequence[int], int]] = []
    for examples in example_sets:
        for example in examples:
            if example.n_ao is None:
                continue
            compositions.append((example.atomic_numbers, int(example.n_ao)))
    if not compositions:
        raise DataError("no density-matrix examples are available to fit an AO basis")
    return AOBasis.fit(compositions, basis_set=basis_set)


def fit_valence(*example_sets: SCFExampleSet, potential_family: str | None = None) -> ValenceTable:
    """Derive per-element valence electron counts from the training data.

    The recorded electron count is ``Tr(P S)`` of a converged pseudopotential
    density, so this is the number CP2K will hold a prediction to. Fitting it
    from the same examples the model is trained on means the model and its
    normalisation can never disagree about what basis and potential they are
    for.
    """

    compositions: list[tuple[Sequence[int], float]] = []
    charges: list[float] = []
    for examples in example_sets:
        for example in examples:
            if not example.atomic_numbers:
                continue
            compositions.append((example.atomic_numbers, float(example.electron_count)))
            charges.append(float(example.charge))
    if not compositions:
        raise DataError("no examples carry both a composition and an electron count")
    return ValenceTable.fit(compositions, charges=charges, potential_family=potential_family)


def train_density_matrix_model(
    train_set: SCFExampleSet,
    validation_set: SCFExampleSet | None,
    output_path: str | Path,
    *,
    config: DensityTrainingConfig | None = None,
    basis: AOBasis | None = None,
    valence: ValenceTable | None = None,
) -> DensityTrainingResult:
    """Train the density GNN and write the best checkpoint to ``output_path``."""

    settings = config or DensityTrainingConfig()
    if len(train_set) == 0:
        raise DataError("cannot train on an empty training split")
    if train_set.target != TARGET_DENSITY_MATRIX:
        raise DataError(
            f"density training needs target={TARGET_DENSITY_MATRIX!r}, got {train_set.target!r}"
        )

    torch = require_torch()
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)

    resolved_basis = basis or fit_basis(train_set, validation_set or SCFExampleSet())

    # The valence table is needed to *deploy*, not to train: a model can be
    # fitted on one stoichiometry, where the per-element electron counts are
    # not identifiable. Training therefore records the failure and continues;
    # export refuses, because a deployed model with no electron counts would
    # have to normalise to sum(Z), which is the bug this table exists to fix.
    resolved_valence = valence
    valence_error: str | None = None
    if resolved_valence is None:
        try:
            resolved_valence = fit_valence(train_set, validation_set or SCFExampleSet())
        except (DataError, DensityModelError) as exc:
            resolved_valence = None
            valence_error = str(exc)
    model_config = DensityGNNConfig(
        hidden_size=settings.hidden_size,
        n_interactions=settings.n_interactions,
        n_radial_basis=settings.n_radial_basis,
        cutoff=settings.cutoff,
        max_ao_per_atom=resolved_basis.max_ao_per_atom,
    )

    device = torch.device(settings.device)
    model = build_density_model(model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )

    checkpoint_path = Path(output_path)
    has_validation = validation_set is not None and len(validation_set) > 0
    result = DensityTrainingResult(
        checkpoint_path=checkpoint_path,
        basis=resolved_basis,
        valence=resolved_valence,
        valence_error=valence_error,
        selected_on="validation" if has_validation else "train",
        train_summary=train_set.summary(),
        validation_summary=validation_set.summary() if validation_set else {},
    )

    started = time.perf_counter()
    for epoch in range(1, settings.epochs + 1):
        train_metrics = _run_epoch(
            model, train_set, resolved_basis, settings, torch, device, optimizer, epoch, "train"
        )
        result.history.append(train_metrics)

        selection = train_metrics.density_mae
        if has_validation:
            validation_metrics = _run_epoch(
                model, validation_set, resolved_basis, settings, torch, device, None, epoch, "validation"
            )
            result.history.append(validation_metrics)
            selection = validation_metrics.density_mae

        if selection < result.best_density_mae:
            result.best_density_mae = selection
            result.best_epoch = epoch
            save_density_checkpoint(
                checkpoint_path,
                model=model,
                model_config=model_config,
                basis=resolved_basis,
                valence=resolved_valence,
                settings=settings,
                metrics={"selection_density_mae": selection, "epoch": epoch, "selected_on": result.selected_on},
            )

    result.wall_time_seconds = time.perf_counter() - started
    if result.best_epoch == 0:
        raise DataError("training produced no checkpoint; check that epochs is positive")

    write_json_atomic(checkpoint_path.with_suffix(checkpoint_path.suffix + ".history.json"), result.to_dict())
    return result


def train_density_from_dataset(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    config: DensityTrainingConfig | None = None,
) -> DensityTrainingResult:
    """Load the density-matrix splits from disk and train on them."""

    train_set = load_split(dataset_root, "train", target=TARGET_DENSITY_MATRIX)
    validation_set = load_split(dataset_root, "validation", target=TARGET_DENSITY_MATRIX)
    return train_density_matrix_model(train_set, validation_set, output_path, config=config)


def save_density_checkpoint(
    path: str | Path,
    *,
    model: Any,
    model_config: DensityGNNConfig,
    basis: AOBasis,
    settings: DensityTrainingConfig,
    valence: ValenceTable | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> Path:
    """Write a density checkpoint plus a readable sidecar."""

    torch = require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint_version": DENSITY_CHECKPOINT_VERSION,
        "model_config": model_config.to_dict(),
        "state_dict": model.state_dict(),
        "basis": basis.to_dict(),
        # The electron counts a prediction must reproduce. Without them the
        # export has to guess, and the only available guess -- the nuclear
        # charge -- is wrong for every pseudopotential calculation.
        "valence": valence.to_dict() if valence else None,
        "target": TARGET_DENSITY_MATRIX,
        "model_name": settings.model_name,
        "model_version": settings.model_version,
        "training_config": settings.to_dict(),
        "metrics": dict(metrics or {}),
    }
    torch.save(payload, target)

    sidecar = target.with_suffix(target.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {key: value for key, value in payload.items() if key != "state_dict"}
            | {"model_digest": model_digest(target)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def _run_epoch(
    model: Any,
    examples: SCFExampleSet,
    basis: AOBasis,
    settings: DensityTrainingConfig,
    torch: Any,
    device: Any,
    optimizer: Any | None,
    epoch: int,
    split: str,
) -> DensityEpochMetrics:
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "abs": 0.0, "sq": 0.0, "electron": 0.0}
    entries = 0.0
    channels = 0.0
    batches = 0
    max_asymmetry = 0.0
    breakdown: dict[str, float] = {}

    for chunk in _iterate(examples, settings.batch_size, shuffle=training, seed=settings.seed + epoch):
        batch = collate_density_batch(chunk, basis)
        tensors = batch_to_tensors(batch, torch, device)

        with torch.set_grad_enabled(training):
            predicted = forward_density(model, tensors)
            loss, terms = density_loss(predicted, tensors, settings.loss_weights)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=settings.grad_clip)
                optimizer.step()

        with torch.no_grad():
            mask = (
                tensors["ao_mask"].unsqueeze(1) * tensors["ao_mask"].unsqueeze(2)
            ).unsqueeze(1) * tensors["channel_mask"][:, :, None, None]
            difference = (predicted - tensors["density_target"]) * mask
            counts = predicted_electron_counts(predicted, tensors["overlap"])
            electron_error = ((counts - tensors["channel_electrons"]) * tensors["channel_mask"]).abs()

            totals["loss"] += float(loss)
            totals["abs"] += float(difference.abs().sum())
            totals["sq"] += float((difference**2).sum())
            totals["electron"] += float(electron_error.sum())
            entries += float(mask.sum())
            channels += float(tensors["channel_mask"].sum())
            max_asymmetry = max(
                max_asymmetry, float((predicted - predicted.transpose(-1, -2)).abs().max())
            )
            breakdown = terms
            batches += 1

    scale = max(entries, 1.0)
    return DensityEpochMetrics(
        epoch=epoch,
        split=split,
        loss=totals["loss"] / max(batches, 1),
        density_mae=totals["abs"] / scale,
        density_rmse=float(np.sqrt(totals["sq"] / scale)),
        electron_mae=totals["electron"] / max(channels, 1.0),
        max_asymmetry=max_asymmetry,
        breakdown=breakdown,
    )


def _iterate(examples: SCFExampleSet, batch_size: int, *, shuffle: bool, seed: int):
    if batch_size <= 0:
        raise DataError(f"batch_size must be positive, got {batch_size}")
    order = np.arange(len(examples))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [examples[int(index)] for index in order[start : start + batch_size]]
        if chunk:
            yield chunk


__all__ = [
    "DENSITY_CHECKPOINT_VERSION",
    "DensityEpochMetrics",
    "DensityTrainingConfig",
    "DensityTrainingResult",
    "fit_basis",
    "fit_valence",
    "load_density_checkpoint",
    "save_density_checkpoint",
    "train_density_from_dataset",
    "train_density_matrix_model",
]
