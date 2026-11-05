# Ported from PHAL (hotspot_al.cp2k_ml_scf.training.evaluate) at 96263b6; MIT License, see NOTICE.
"""Evaluation metrics for ML-SCF density models.

Two questions are answered separately, because a model can do well at one and
badly at the other:

*Is the predicted density close to the reference?*  Measured by
:func:`evaluate_density_predictions` (density error, electron number error).

*Does it actually save SCF iterations?*  Measured by :func:`compare_scf_runs`,
which needs paired CP2K runs -- the same systems with ``SCF_GUESS ATOMIC`` and
with ``SCF_GUESS ML``.  Density error is a proxy; iteration count is the thing
the project is actually buying.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from gmd_active_learning.adapters.dft.scf.density.predictor import SCFDensityPredictor
from gmd_active_learning.adapters.dft.scf.schema import SCFElectronicState, SCFStructure
from tools.scf.training.dataset_loader import SCFExample, SCFExampleSet


@dataclass(frozen=True, slots=True)
class DensityMetrics:
    """How closely a predicted density matches its reference."""

    n_examples: int
    density_mae: float
    density_rmse: float
    density_max_error: float
    electron_mae: float
    electron_max_error: float
    n_failed: int = 0
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_examples": self.n_examples,
            "density_mae": self.density_mae,
            "density_rmse": self.density_rmse,
            "density_max_error": self.density_max_error,
            "electron_mae": self.electron_mae,
            "electron_max_error": self.electron_max_error,
            "n_failed": self.n_failed,
            "failures": list(self.failures),
        }


@dataclass(frozen=True, slots=True)
class SCFRun:
    """One recorded CP2K calculation, whatever produced its initial guess."""

    label: str
    system_id: str
    scf_iterations: int
    converged: bool
    wall_time_seconds: float | None = None
    final_energy: float | None = None
    max_force_error: float | None = None
    fallback_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    #: Full observables, when they were recorded. These are what let a
    #: benchmark compare *what CP2K produced* rather than only how fast it got
    #: there -- a mode that is faster and wrong must fail, and it can only be
    #: caught by comparing forces and stress, not iteration counts.
    forces: np.ndarray | None = field(default=None, repr=False)
    stress: np.ndarray | None = field(default=None, repr=False)
    density_matrix_path: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SCFRun":
        return cls(
            label=str(payload.get("label", "unknown")),
            system_id=str(payload["system_id"]),
            scf_iterations=int(payload["scf_iterations"]),
            converged=bool(payload["converged"]),
            wall_time_seconds=_optional_float(payload.get("wall_time_seconds")),
            final_energy=_optional_float(payload.get("final_energy")),
            max_force_error=_optional_float(payload.get("max_force_error")),
            fallback_used=bool(payload.get("fallback_used", False)),
            metadata=dict(payload.get("metadata", {})),
            forces=_optional_array(payload.get("forces")),
            stress=_optional_array(payload.get("stress")),
            density_matrix_path=_optional_text(payload.get("density_matrix_path")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "system_id": self.system_id,
            "scf_iterations": self.scf_iterations,
            "converged": self.converged,
            "wall_time_seconds": self.wall_time_seconds,
            "final_energy": self.final_energy,
            "max_force_error": self.max_force_error,
            "fallback_used": self.fallback_used,
            "metadata": dict(self.metadata),
            "forces": None if self.forces is None else np.asarray(self.forces).tolist(),
            "stress": None if self.stress is None else np.asarray(self.stress).tolist(),
            "density_matrix_path": self.density_matrix_path,
        }


@dataclass(frozen=True, slots=True)
class GuessComparison:
    """Paired comparison of a baseline guess against a candidate guess."""

    baseline_label: str
    candidate_label: str
    n_systems: int
    baseline_iterations: float
    ml_iterations: float
    iteration_reduction: float
    speedup: float
    runtime_speedup: float | None
    energy_error: float | None
    force_error: float | None
    converged: bool
    baseline_success_rate: float
    candidate_success_rate: float
    fallback_rate: float
    per_system: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_label": self.baseline_label,
            "candidate_label": self.candidate_label,
            "n_systems": self.n_systems,
            "baseline_iterations": self.baseline_iterations,
            "ml_iterations": self.ml_iterations,
            "iteration_reduction": self.iteration_reduction,
            "speedup": self.speedup,
            "runtime_speedup": self.runtime_speedup,
            "energy_error": self.energy_error,
            "force_error": self.force_error,
            "converged": self.converged,
            "baseline_success_rate": self.baseline_success_rate,
            "candidate_success_rate": self.candidate_success_rate,
            "fallback_rate": self.fallback_rate,
            "per_system": [dict(entry) for entry in self.per_system],
        }


class EvaluationError(RuntimeError):
    """Raised when an evaluation cannot be carried out."""


def evaluate_density_predictions(
    predictor: SCFDensityPredictor,
    examples: SCFExampleSet | Sequence[SCFExample],
    *,
    structures: Mapping[str, tuple[SCFStructure, SCFElectronicState]] | None = None,
) -> DensityMetrics:
    """Compare a predictor's output against each example's reference density.

    ``structures`` supplies the geometry for each example; when omitted, the
    examples must already carry predictions (see :func:`compare_densities`).
    """

    if structures is None:
        raise EvaluationError(
            "evaluate_density_predictions needs a structures mapping; "
            "features alone cannot be turned back into coordinates"
        )

    absolute: list[np.ndarray] = []
    electron_errors: list[float] = []
    failures: list[str] = []

    for example in examples:
        pair = structures.get(example.sample_id)
        if pair is None:
            failures.append(f"{example.sample_id}: no structure supplied")
            continue
        structure, state = pair
        prediction = predictor.predict(structure, state)
        validation = predictor.validate_prediction(prediction, state, structure=structure)
        if not validation.valid:
            failures.append(f"{example.sample_id}: {validation.reasons[0]}")
            continue
        predicted = np.asarray(prediction.values, dtype=float)
        reference = np.asarray(example.density_target, dtype=float)
        if predicted.shape != reference.shape:
            failures.append(
                f"{example.sample_id}: predicted shape {predicted.shape} != reference {reference.shape}"
            )
            continue
        absolute.append(np.abs(predicted - reference).ravel())
        electron_errors.append(abs(float(predicted.sum()) - float(reference.sum())))

    if not absolute:
        return DensityMetrics(
            n_examples=0,
            density_mae=float("nan"),
            density_rmse=float("nan"),
            density_max_error=float("nan"),
            electron_mae=float("nan"),
            electron_max_error=float("nan"),
            n_failed=len(failures),
            failures=tuple(failures),
        )

    stacked = np.concatenate(absolute)
    return DensityMetrics(
        n_examples=len(electron_errors),
        density_mae=float(stacked.mean()),
        density_rmse=float(np.sqrt((stacked**2).mean())),
        density_max_error=float(stacked.max()),
        electron_mae=float(np.mean(electron_errors)),
        electron_max_error=float(np.max(electron_errors)),
        n_failed=len(failures),
        failures=tuple(failures),
    )


def compare_densities(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Elementwise density error between two arrays of the same shape."""

    left = np.asarray(predicted, dtype=float)
    right = np.asarray(reference, dtype=float)
    if left.shape != right.shape:
        raise EvaluationError(f"shape mismatch: {left.shape} vs {right.shape}")
    difference = np.abs(left - right)
    return {
        "density_mae": float(difference.mean()),
        "density_rmse": float(np.sqrt((difference**2).mean())),
        "density_max_error": float(difference.max()),
        "electron_error": float(abs(left.sum() - right.sum())),
    }


def compare_scf_runs(
    baseline: Iterable[SCFRun],
    candidate: Iterable[SCFRun],
    *,
    baseline_label: str = "atomic",
    candidate_label: str = "ml",
    require_both_converged: bool = True,
) -> GuessComparison:
    """Compare two sets of runs system by system.

    Only systems present in both sets are compared; a system the candidate
    failed to converge is counted against its success rate but excluded from
    the iteration statistics, since averaging in a run that hit ``MAX_SCF``
    would flatter or punish the candidate arbitrarily.
    """

    baseline_by_id = {run.system_id: run for run in baseline}
    candidate_by_id = {run.system_id: run for run in candidate}
    shared = sorted(set(baseline_by_id) & set(candidate_by_id))
    if not shared:
        raise EvaluationError("baseline and candidate runs share no system ids")

    per_system: list[dict[str, Any]] = []
    baseline_iterations: list[float] = []
    candidate_iterations: list[float] = []
    baseline_times: list[float] = []
    candidate_times: list[float] = []
    energy_differences: list[float] = []
    force_differences: list[float] = []
    fallbacks = 0

    for system_id in shared:
        before = baseline_by_id[system_id]
        after = candidate_by_id[system_id]
        fallbacks += 1 if after.fallback_used else 0

        entry: dict[str, Any] = {
            "system_id": system_id,
            "baseline_iterations": before.scf_iterations,
            "candidate_iterations": after.scf_iterations,
            "baseline_converged": before.converged,
            "candidate_converged": after.converged,
            "fallback_used": after.fallback_used,
        }

        usable = (before.converged and after.converged) or not require_both_converged
        if usable:
            baseline_iterations.append(float(before.scf_iterations))
            candidate_iterations.append(float(after.scf_iterations))
            if before.wall_time_seconds is not None and after.wall_time_seconds is not None:
                baseline_times.append(float(before.wall_time_seconds))
                candidate_times.append(float(after.wall_time_seconds))
            if before.final_energy is not None and after.final_energy is not None:
                difference = abs(float(after.final_energy) - float(before.final_energy))
                energy_differences.append(difference)
                entry["energy_difference"] = difference
            if after.max_force_error is not None:
                force_differences.append(float(after.max_force_error))
                entry["force_error"] = float(after.max_force_error)
        entry["compared"] = usable
        per_system.append(entry)

    if not baseline_iterations:
        raise EvaluationError(
            "no system converged in both the baseline and the candidate; nothing can be compared"
        )

    baseline_mean = float(np.mean(baseline_iterations))
    candidate_mean = float(np.mean(candidate_iterations))
    runtime_speedup = (
        float(np.sum(baseline_times) / np.sum(candidate_times))
        if baseline_times and float(np.sum(candidate_times)) > 0.0
        else None
    )

    return GuessComparison(
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        n_systems=len(baseline_iterations),
        baseline_iterations=baseline_mean,
        ml_iterations=candidate_mean,
        iteration_reduction=baseline_mean - candidate_mean,
        speedup=(baseline_mean / candidate_mean) if candidate_mean > 0 else math.inf,
        runtime_speedup=runtime_speedup,
        energy_error=float(np.max(energy_differences)) if energy_differences else None,
        force_error=float(np.max(force_differences)) if force_differences else None,
        converged=all(candidate_by_id[system_id].converged for system_id in shared),
        baseline_success_rate=_success_rate(baseline_by_id, shared),
        candidate_success_rate=_success_rate(candidate_by_id, shared),
        fallback_rate=fallbacks / len(shared),
        per_system=tuple(per_system),
    )


def iteration_reduction(baseline: Sequence[int], candidate: Sequence[int]) -> dict[str, float]:
    """Summarise the iteration saving between two paired iteration counts."""

    if len(baseline) != len(candidate):
        raise EvaluationError(f"paired lengths differ: {len(baseline)} vs {len(candidate)}")
    if not baseline:
        raise EvaluationError("cannot summarise an empty comparison")
    before = np.asarray(baseline, dtype=float)
    after = np.asarray(candidate, dtype=float)
    return {
        "baseline_mean": float(before.mean()),
        "candidate_mean": float(after.mean()),
        "mean_reduction": float((before - after).mean()),
        "fraction_improved": float(np.mean(after < before)),
        "fraction_worse": float(np.mean(after > before)),
        "speedup": float(before.mean() / after.mean()) if after.mean() > 0 else math.inf,
    }


def _success_rate(runs: Mapping[str, SCFRun], system_ids: Sequence[str]) -> float:
    if not system_ids:
        return 0.0
    return sum(1 for system_id in system_ids if runs[system_id].converged) / len(system_ids)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_array(value: Any) -> np.ndarray | None:
    return None if value is None else np.asarray(value, dtype=float)


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "DensityMetrics",
    "EvaluationError",
    "GuessComparison",
    "SCFRun",
    "compare_densities",
    "compare_scf_runs",
    "evaluate_density_predictions",
    "iteration_reduction",
]
