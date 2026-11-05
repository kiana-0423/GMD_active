# Ported from PHAL (hotspot_al.cp2k_ml_scf.training.evaluate_density) at 96263b6; MIT License, see NOTICE.
"""Serve and evaluate a trained AO density model (Stage 10).

Two things live here:

:class:`AODensityMatrixPredictor`
    Serves a density checkpoint through the Stage 2 predictor contract, so the
    trained model is usable by exactly the same code path as the dummy one.

:func:`compare_guesses`
    The comparison that decides whether the model is worth anything: atomic
    guess versus ML guess, on density error, electron error, SCF iterations,
    energy difference and convergence rate.

The density error metrics answer "is the prediction close?"; only the SCF
iteration count answers "did it help?".  Both are reported, and a model that
wins the first while losing the second is not an improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gmd_active_learning.adapters.dft.scf.density.predictor import SCFDensityPredictor
from gmd_active_learning.adapters.dft.scf.density.prediction import (
    DENSITY_FORMAT_DENSITY_MATRIX,
    PredictionError,
    SCFDensityPrediction,
)
from gmd_active_learning.adapters.dft.scf.schema import SCFElectronicState, SCFStructure
from gmd_active_learning.adapters.dft.scf.schema import structure_hash as compute_structure_hash
from gmd_active_learning.adapters.dft.scf.density.gnn_model import (
    AOBasis,
    DensityGNNConfig,
    DensityModelError,
    build_density_model,
    channel_electron_counts,
    forward_density,
)
from gmd_active_learning.adapters.dft.scf._torch import model_digest, require_torch
from gmd_active_learning.adapters.dft.scf.density.gnn_model import ValenceTable, load_density_checkpoint
from gmd_active_learning.adapters.dft.scf.density.predictor import expected_density_electrons


class AODensityMatrixPredictor(SCFDensityPredictor):
    """Predict a full AO density matrix from a trained checkpoint."""

    predictor_type = "density_gnn"
    density_format = DENSITY_FORMAT_DENSITY_MATRIX
    #: The backend passes CP2K's AO overlap to predictors that declare this.
    accepts_overlap = True

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config=config)
        model_path = self.config.get("model_path")
        if not model_path:
            options = self.config.get("options", {})
            model_path = options.get("model_path") if isinstance(options, Mapping) else None
        if not model_path:
            raise PredictionError(
                "the density_gnn predictor requires cp2k.ml_scf.predictor.model_path"
            )
        self.model_path = Path(str(model_path))
        self._model: Any | None = None
        self._basis: AOBasis | None = None
        self._checkpoint: Mapping[str, Any] | None = None
        self._digest: str | None = None

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return str(self._checkpoint["model_name"]) if self._checkpoint else "phal-scf-density-gnn"

    @property
    def model_version(self) -> str:  # type: ignore[override]
        return str(self._checkpoint["model_version"]) if self._checkpoint else "0"

    @property
    def model_digest(self) -> str:
        if self._digest is None:
            self._digest = model_digest(self.model_path)
        return self._digest

    @property
    def basis(self) -> AOBasis:
        if self._basis is None:
            self.load_model()
        assert self._basis is not None
        return self._basis

    def load_model(self) -> "AODensityMatrixPredictor":
        if self._loaded:
            return self
        checkpoint = load_density_checkpoint(self.model_path)
        basis = AOBasis.from_dict(checkpoint["basis"])
        model = build_density_model(DensityGNNConfig.from_dict(checkpoint["model_config"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self._model = model
        self._basis = basis
        self._checkpoint = checkpoint
        self._digest = model_digest(self.model_path)
        self._loaded = True
        return self

    def predict(
        self,
        structure: SCFStructure,
        electronic_state: SCFElectronicState,
        *,
        overlap: np.ndarray | None = None,
    ) -> SCFDensityPrediction:
        """Predict ``P`` for one structure.

        ``overlap`` is CP2K's AO overlap for this geometry.  Without it the
        normalisation falls back to ``Tr(P) = Ne``, which is only correct in an
        orthonormal basis -- the prediction records which was used so a
        consumer is never misled about it.
        """

        if not self._loaded:
            self.load_model()
        assert self._model is not None and self._basis is not None
        torch = require_torch()

        if structure.n_atoms == 0:
            raise PredictionError("cannot predict a density for a structure with no atoms")

        n_ao = self._basis.n_ao(structure.atomic_numbers)
        atom_of, slot = self._basis.ao_layout(structure.atomic_numbers)
        channels = electronic_state.spin_channels
        # Deliberate fix (finding F-23): normalise to the *valence* count the
        # density lives in (checkpoint valence table, else GTH valence), not to
        # the all-electron n_electrons = sum(Z) - charge, which CP2K rejects.
        target_electrons = self.density_electrons(structure, electronic_state)
        counts = channel_electron_counts(target_electrons, electronic_state.multiplicity, channels)

        if overlap is None:
            overlap_matrix = np.eye(n_ao, dtype=np.float32)
            normalization = "trace_p"
        else:
            overlap_matrix = np.asarray(overlap, dtype=np.float32)
            if overlap_matrix.shape != (n_ao, n_ao):
                raise PredictionError(
                    f"overlap has shape {overlap_matrix.shape}, expected ({n_ao}, {n_ao})"
                )
            normalization = "trace_ps"

        tensors = {
            "atomic_numbers": torch.as_tensor([structure.atomic_numbers], dtype=torch.long),
            "positions": torch.as_tensor(
                np.asarray(structure.positions, dtype=np.float32)[None, ...], dtype=torch.float32
            ),
            "atom_mask": torch.ones((1, structure.n_atoms), dtype=torch.float32),
            "global_features": torch.as_tensor(
                [
                    [
                        float(electronic_state.charge),
                        float(electronic_state.multiplicity - 1),
                        float(electronic_state.n_electrons) / structure.n_atoms,
                        float(channels),
                    ]
                ],
                dtype=torch.float32,
            ),
            "ao_atom_index": torch.as_tensor(atom_of, dtype=torch.long).unsqueeze(0),
            "ao_slot_index": torch.as_tensor(slot, dtype=torch.long).unsqueeze(0),
            "ao_mask": torch.ones((1, n_ao), dtype=torch.float32),
            "overlap": torch.as_tensor(overlap_matrix[None, ...], dtype=torch.float32),
            "channel_electrons": torch.as_tensor([list(counts)], dtype=torch.float32),
            "channel_mask": torch.ones((1, channels), dtype=torch.float32),
        }

        with torch.no_grad():
            values = forward_density(self._model, tensors)[0].cpu().numpy().astype(float)

        if not np.isfinite(values).all():
            # An architecture cannot guarantee finiteness; a diverged weight
            # produces NaN whatever the design, so it is caught here.
            raise PredictionError("the density model produced non-finite values")

        return SCFDensityPrediction(
            model_name=self.model_name,
            model_version=self.model_version,
            model_digest=self.model_digest,
            structure_hash=compute_structure_hash(structure, electronic_state),
            electron_count=float(target_electrons),
            spin_channels=channels,
            density_format=self.density_format,
            values=values,
            # Carried so the exchange file can include S (PHAL dropped it, so a
            # density_gnn prediction could never be exported).
            overlap=None if overlap is None else np.asarray(overlap, dtype=float),
            predictor_type=self.predictor_type,
            metadata={
                "model_path": str(self.model_path),
                "n_ao": n_ao,
                "basis_set": self._basis.basis_set,
                "normalization": normalization,
                "electron_count_basis": "valence",
            },
        )

    def density_electrons(self, structure: SCFStructure, electronic_state: SCFElectronicState) -> float:
        """Valence electrons minus charge, from the checkpoint's table or the GTH table."""

        if not self._loaded:
            self.load_model()
        table = (self._checkpoint or {}).get("valence")
        if table:
            return float(ValenceTable.from_dict(table).electrons(structure.atomic_numbers, electronic_state.charge))
        return expected_density_electrons(structure, electronic_state)


@dataclass(frozen=True, slots=True)
class DensityComparison:
    """Atomic guess versus ML guess on one set of systems."""

    n_systems: int
    atomic_density_mae: float
    ml_density_mae: float
    atomic_electron_error: float
    ml_electron_error: float
    density_improvement: float
    scf_iterations_atomic: float | None = None
    scf_iterations_ml: float | None = None
    energy_difference: float | None = None
    convergence_rate_atomic: float | None = None
    convergence_rate_ml: float | None = None
    failures: tuple[str, ...] = field(default_factory=tuple)

    @property
    def improved(self) -> bool:
        """Whether the ML guess is closer to the reference than the atomic one."""

        return self.ml_density_mae < self.atomic_density_mae

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_systems": self.n_systems,
            "atomic_density_mae": self.atomic_density_mae,
            "ml_density_mae": self.ml_density_mae,
            "density_improvement": self.density_improvement,
            "atomic_electron_error": self.atomic_electron_error,
            "ml_electron_error": self.ml_electron_error,
            "improved": self.improved,
            "scf_iterations_atomic": self.scf_iterations_atomic,
            "scf_iterations_ml": self.scf_iterations_ml,
            "energy_difference": self.energy_difference,
            "convergence_rate_atomic": self.convergence_rate_atomic,
            "convergence_rate_ml": self.convergence_rate_ml,
            "n_failures": len(self.failures),
            "failures": list(self.failures),
        }


def atomic_guess_density(
    structure: SCFStructure,
    electronic_state: SCFElectronicState,
    basis: AOBasis,
) -> np.ndarray:
    """A block-diagonal atomic-guess density, the baseline the model must beat.

    Each atom carries its share of the electrons spread evenly over its own
    orbitals and nothing off-site, which is the structure of what
    ``SCF_GUESS ATOMIC`` produces.
    """

    n_ao = basis.n_ao(structure.atomic_numbers)
    channels = electronic_state.spin_channels
    counts = channel_electron_counts(
        float(electronic_state.n_electrons), electronic_state.multiplicity, channels
    )

    from gmd_active_learning.adapters.dft.scf.density.predictor import VALENCE_ELECTRONS

    weights = np.asarray(
        [float(VALENCE_ELECTRONS.get(int(number), int(number))) for number in structure.atomic_numbers],
        dtype=float,
    )
    if weights.sum() <= 0.0:  # pragma: no cover - defensive
        weights = np.ones(structure.n_atoms, dtype=float)
    share = weights / weights.sum()

    density = np.zeros((channels, n_ao, n_ao), dtype=float)
    offset = 0
    for index, number in enumerate(structure.atomic_numbers):
        width = basis.count(int(number))
        for channel in range(channels):
            per_orbital = counts[channel] * share[index] / width
            for slot in range(width):
                density[channel, offset + slot, offset + slot] = per_orbital
        offset += width
    return density


def density_error(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Elementwise error between two density matrices of the same shape."""

    left = np.asarray(predicted, dtype=float)
    right = np.asarray(reference, dtype=float)
    if left.shape != right.shape:
        raise DensityModelError(f"density shape mismatch: {left.shape} vs {right.shape}")
    difference = np.abs(left - right)
    return {
        "mae": float(difference.mean()),
        "rmse": float(np.sqrt((difference**2).mean())),
        "max": float(difference.max()),
    }


def electron_error(density: np.ndarray, overlap: np.ndarray | None, reference_count: float) -> float:
    """``|Tr(P S) - Ne|``, or ``|Tr(P) - Ne|`` without an overlap."""

    matrices = np.asarray(density, dtype=float)
    if overlap is None:
        counted = float(np.trace(matrices, axis1=1, axis2=2).sum())
    else:
        counted = float(sum(np.sum(channel * np.asarray(overlap, dtype=float)) for channel in matrices))
    return abs(counted - float(reference_count))


def compare_guesses(
    predictor: AODensityMatrixPredictor,
    examples: Sequence[Any],
    *,
    structures: Mapping[str, tuple[SCFStructure, SCFElectronicState]],
    scf_iterations: Mapping[str, tuple[int, int]] | None = None,
    energies: Mapping[str, tuple[float, float]] | None = None,
    convergence: Mapping[str, tuple[bool, bool]] | None = None,
) -> DensityComparison:
    """Compare the atomic guess and the ML guess against recorded densities.

    ``scf_iterations``, ``energies`` and ``convergence`` carry ``(atomic, ml)``
    pairs measured by actually running CP2K.  They are optional because the
    density metrics can be computed offline; the report says which of them were
    supplied rather than inventing the ones that were not.
    """

    basis = predictor.basis
    atomic_errors: list[float] = []
    ml_errors: list[float] = []
    atomic_electron: list[float] = []
    ml_electron: list[float] = []
    failures: list[str] = []

    for example in examples:
        pair = structures.get(example.sample_id)
        if pair is None:
            failures.append(f"{example.sample_id}: no structure supplied")
            continue
        structure, state = pair
        reference = np.asarray(example.density_target, dtype=float)
        overlap = example.overlap

        try:
            prediction = predictor.predict(structure, state, overlap=overlap)
        except (PredictionError, DensityModelError) as exc:
            failures.append(f"{example.sample_id}: {exc}")
            continue

        predicted = np.asarray(prediction.values, dtype=float)
        if predicted.shape != reference.shape:
            failures.append(
                f"{example.sample_id}: predicted shape {predicted.shape} != reference {reference.shape}"
            )
            continue

        guess = atomic_guess_density(structure, state, basis)
        atomic_errors.append(density_error(guess, reference)["mae"])
        ml_errors.append(density_error(predicted, reference)["mae"])
        atomic_electron.append(electron_error(guess, overlap, example.electron_count))
        ml_electron.append(electron_error(predicted, overlap, example.electron_count))

    if not ml_errors:
        raise DensityModelError(
            "no example could be compared; " + ("; ".join(failures) if failures else "the set was empty")
        )

    atomic_mae = float(np.mean(atomic_errors))
    ml_mae = float(np.mean(ml_errors))

    return DensityComparison(
        n_systems=len(ml_errors),
        atomic_density_mae=atomic_mae,
        ml_density_mae=ml_mae,
        atomic_electron_error=float(np.mean(atomic_electron)),
        ml_electron_error=float(np.mean(ml_electron)),
        density_improvement=atomic_mae - ml_mae,
        scf_iterations_atomic=_mean_first(scf_iterations),
        scf_iterations_ml=_mean_second(scf_iterations),
        energy_difference=_max_absolute_difference(energies),
        convergence_rate_atomic=_rate_first(convergence),
        convergence_rate_ml=_rate_second(convergence),
        failures=tuple(failures),
    )


def _mean_first(pairs: Mapping[str, tuple[int, int]] | None) -> float | None:
    return None if not pairs else float(np.mean([float(value[0]) for value in pairs.values()]))


def _mean_second(pairs: Mapping[str, tuple[int, int]] | None) -> float | None:
    return None if not pairs else float(np.mean([float(value[1]) for value in pairs.values()]))


def _max_absolute_difference(pairs: Mapping[str, tuple[float, float]] | None) -> float | None:
    if not pairs:
        return None
    return float(max(abs(float(second) - float(first)) for first, second in pairs.values()))


def _rate_first(pairs: Mapping[str, tuple[bool, bool]] | None) -> float | None:
    return None if not pairs else float(np.mean([1.0 if value[0] else 0.0 for value in pairs.values()]))


def _rate_second(pairs: Mapping[str, tuple[bool, bool]] | None) -> float | None:
    return None if not pairs else float(np.mean([1.0 if value[1] else 0.0 for value in pairs.values()]))


__all__ = [
    "AODensityMatrixPredictor",
    "DensityComparison",
    "atomic_guess_density",
    "compare_guesses",
    "density_error",
    "electron_error",
]
