# Ported from PHAL (hotspot_al.cp2k_ml_scf.benchmark) at 96263b6; MIT License, see NOTICE.
"""End-to-end validation: compare native CP2K against the accelerated modes.

Four modes are compared on the same systems:

1. ``native``   -- ``SCF_GUESS ATOMIC`` + Pulay mixing
2. ``restart``  -- ``SCF_GUESS RESTART`` + Pulay mixing
3. ``ml_guess`` -- ``SCF_GUESS ML`` + Pulay mixing
4. ``ml_full``  -- ``SCF_GUESS ML`` + neural mixing

The harness does not run CP2K.  It consumes recorded runs -- from a benchmark
script, a scheduler, or ``scf_history.json`` files -- so the comparison is
reproducible and can be re-derived from artifacts long after the calculations
finished.

The metric that decides whether acceleration is legitimate is ``energy_error``,
not ``speedup``.  A mode that converges faster to a different energy has not
accelerated anything; it has broken the calculation, and the report says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.scf.benchmark.evaluate import (
    EvaluationError,
    GuessComparison,
    SCFRun,
    compare_scf_runs,
)

BENCHMARK_VERSION = "phal-scf-benchmark-v1"

from gmd_active_learning.adapters.dft.scf.modes import (  # noqa: F401 - re-exported
    BENCHMARK_MODES,
    MODE_DESCRIPTIONS,
    MODE_ML_FULL,
    MODE_ML_GUESS,
    MODE_NATIVE,
    MODE_RESTART,
)

#: Energy agreement required before a mode is called equivalent, in Hartree.
DEFAULT_ENERGY_TOLERANCE = 1.0e-6
#: Force agreement, in Hartree/Bohr as CP2K prints them.
DEFAULT_FORCE_TOLERANCE = 1.0e-5
#: Stress agreement, in the unit the runs recorded (GPa for PHAL's parser).
DEFAULT_STRESS_TOLERANCE = 1.0e-3

BENCHMARK_RESULTS_NAME = "benchmark_results.json"


class BenchmarkError(RuntimeError):
    """Raised when a benchmark cannot be assembled from the recorded runs."""


@dataclass(frozen=True, slots=True)
class AcceptanceCriteria:
    """What a mode must satisfy before its speedup counts for anything.

    An accelerated mode is accepted only if it converged *and* reproduced the
    baseline's energy and forces.  Fast but incorrect must fail: a mode that
    halves the iteration count while shifting the energy has not accelerated
    the calculation, it has broken it.
    """

    energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE
    force_tolerance: float = DEFAULT_FORCE_TOLERANCE
    stress_tolerance: float = DEFAULT_STRESS_TOLERANCE
    require_converged: bool = True
    #: Fail a mode that had to fall back on more than this share of systems.
    max_fallback_rate: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModeResult:
    """One mode's row in ``benchmark_results.json``."""

    mode: str
    iterations: float | None
    runtime: float | None
    energy_error: float | None
    force_error: float | None
    stress_error: float | None
    density_error: float | None
    speedup: float | None
    converged: bool
    fallback_rate: float
    accepted: bool
    rejection_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "iterations": self.iterations,
            "runtime": self.runtime,
            "energy_error": self.energy_error,
            "force_error": self.force_error,
            "stress_error": self.stress_error,
            "density_error": self.density_error,
            "speedup": self.speedup,
            "converged": self.converged,
            "fallback_rate": self.fallback_rate,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True, slots=True)
class ModeSummary:
    """Aggregate statistics for one mode, independent of any comparison."""

    mode: str
    description: str
    n_runs: int
    n_converged: int
    mean_iterations: float | None
    mean_wall_time: float | None
    failure_rate: float
    fallback_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "description": self.description,
            "n_runs": self.n_runs,
            "n_converged": self.n_converged,
            "mean_iterations": self.mean_iterations,
            "mean_wall_time": self.mean_wall_time,
            "failure_rate": self.failure_rate,
            "fallback_rate": self.fallback_rate,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """The full four-way comparison."""

    modes: tuple[ModeSummary, ...]
    comparisons: Mapping[str, GuessComparison]
    energy_tolerance: float
    headline: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    results: tuple[ModeResult, ...] = ()
    criteria: AcceptanceCriteria | None = None

    @property
    def accepted_modes(self) -> tuple[str, ...]:
        return tuple(result.mode for result in self.results if result.accepted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_version": BENCHMARK_VERSION,
            **dict(self.headline),
            "energy_tolerance": self.energy_tolerance,
            "acceptance_criteria": None if self.criteria is None else self.criteria.to_dict(),
            "results": [result.to_dict() for result in self.results],
            "accepted_modes": list(self.accepted_modes),
            "modes": [summary.to_dict() for summary in self.modes],
            "comparisons": {name: value.to_dict() for name, value in self.comparisons.items()},
            "warnings": list(self.warnings),
        }


def load_runs(path: str | Path) -> list[SCFRun]:
    """Read recorded runs from a JSON list or ``{"runs": [...]}`` object."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{source}: invalid JSON ({exc})") from exc
    except OSError as exc:
        raise BenchmarkError(f"could not read {source}: {exc}") from exc

    if isinstance(payload, Mapping):
        payload = payload.get("runs", [])
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise BenchmarkError(f"{source}: expected a list of runs")

    runs: list[SCFRun] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, Mapping):
            raise BenchmarkError(f"{source}: run {index} is not an object")
        try:
            runs.append(SCFRun.from_dict(entry))
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkError(f"{source}: run {index} is malformed ({exc})") from exc
    if not runs:
        raise BenchmarkError(f"{source}: no runs to compare")
    return runs


def run_benchmark(
    runs: Iterable[SCFRun],
    *,
    baseline_mode: str = MODE_NATIVE,
    energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    criteria: AcceptanceCriteria | None = None,
) -> BenchmarkReport:
    """Build the four-way report from a flat list of recorded runs."""

    resolved_criteria = criteria or AcceptanceCriteria(energy_tolerance=energy_tolerance)
    by_mode: dict[str, list[SCFRun]] = {}
    for run in runs:
        by_mode.setdefault(run.label, []).append(run)

    if baseline_mode not in by_mode:
        available = ", ".join(sorted(by_mode)) or "none"
        raise BenchmarkError(
            f"no runs labelled {baseline_mode!r} to use as the baseline; found: {available}"
        )

    summaries = tuple(
        summarize_mode(mode, by_mode[mode])
        for mode in BENCHMARK_MODES
        if mode in by_mode
    )
    summaries += tuple(
        summarize_mode(mode, by_mode[mode]) for mode in sorted(set(by_mode) - set(BENCHMARK_MODES))
    )

    comparisons: dict[str, GuessComparison] = {}
    warnings: list[str] = []
    baseline_runs = by_mode[baseline_mode]

    for mode in BENCHMARK_MODES:
        if mode == baseline_mode or mode not in by_mode:
            continue
        try:
            comparison = compare_scf_runs(
                baseline_runs,
                by_mode[mode],
                baseline_label=baseline_mode,
                candidate_label=mode,
            )
        except EvaluationError as exc:
            warnings.append(f"{mode}: {exc}")
            continue
        comparisons[mode] = comparison
        if comparison.energy_error is not None and comparison.energy_error > energy_tolerance:
            warnings.append(
                f"{mode}: converged energy differs from {baseline_mode} by "
                f"{comparison.energy_error:.3e} Ha, above the {energy_tolerance:.1e} Ha tolerance; "
                "this is a correctness problem, not a speedup"
            )
        if comparison.candidate_success_rate < comparison.baseline_success_rate:
            warnings.append(
                f"{mode}: convergence success rate dropped from "
                f"{comparison.baseline_success_rate:.2f} to {comparison.candidate_success_rate:.2f}"
            )

    for mode in BENCHMARK_MODES:
        if mode not in by_mode:
            warnings.append(f"no runs recorded for mode {mode!r}; it was not compared")

    results = tuple(
        _mode_result(mode, by_mode[mode], baseline_runs, comparisons.get(mode), resolved_criteria)
        for mode in _ordered_modes(by_mode)
    )

    return BenchmarkReport(
        modes=summaries,
        comparisons=comparisons,
        energy_tolerance=energy_tolerance,
        headline=_headline(comparisons, energy_tolerance),
        warnings=tuple(warnings),
        results=results,
        criteria=resolved_criteria,
    )


def _ordered_modes(by_mode: Mapping[str, Sequence[SCFRun]]) -> list[str]:
    known = [mode for mode in BENCHMARK_MODES if mode in by_mode]
    return known + sorted(set(by_mode) - set(BENCHMARK_MODES))


def _mode_result(
    mode: str,
    runs: Sequence[SCFRun],
    baseline_runs: Sequence[SCFRun],
    comparison: GuessComparison | None,
    criteria: AcceptanceCriteria,
) -> ModeResult:
    """Score one mode against the baseline and apply the acceptance criteria."""

    summary = summarize_mode(mode, runs)
    baseline = summarize_mode("baseline", baseline_runs)

    energy_error = _paired_max(baseline_runs, runs, lambda run: run.final_energy)
    force_error = _paired_array_max(baseline_runs, runs, lambda run: run.forces)
    stress_error = _paired_array_max(baseline_runs, runs, lambda run: run.stress)
    density_error = _paired_density_max(baseline_runs, runs)

    speedup: float | None = None
    if comparison is not None:
        speedup = comparison.speedup
    elif summary.mean_iterations and baseline.mean_iterations:
        speedup = baseline.mean_iterations / summary.mean_iterations

    reasons: list[str] = []
    if criteria.require_converged and summary.n_converged < summary.n_runs:
        failed = summary.n_runs - summary.n_converged
        reasons.append(f"{failed} of {summary.n_runs} run(s) did not converge")
    if energy_error is not None and energy_error > criteria.energy_tolerance:
        reasons.append(
            f"energy differs from the baseline by {energy_error:.3e}, above {criteria.energy_tolerance:.1e}"
        )
    if force_error is not None and force_error > criteria.force_tolerance:
        reasons.append(
            f"forces differ by {force_error:.3e}, above {criteria.force_tolerance:.1e}"
        )
    if stress_error is not None and stress_error > criteria.stress_tolerance:
        reasons.append(
            f"stress differs by {stress_error:.3e}, above {criteria.stress_tolerance:.1e}"
        )
    if summary.fallback_rate > criteria.max_fallback_rate:
        reasons.append(
            f"fallback rate {summary.fallback_rate:.2f} exceeds {criteria.max_fallback_rate:.2f}"
        )

    return ModeResult(
        mode=mode,
        iterations=summary.mean_iterations,
        runtime=summary.mean_wall_time,
        energy_error=energy_error,
        force_error=force_error,
        stress_error=stress_error,
        density_error=density_error,
        speedup=speedup,
        converged=summary.n_converged == summary.n_runs,
        fallback_rate=summary.fallback_rate,
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
    )


def _paired(baseline: Sequence[SCFRun], candidate: Sequence[SCFRun]) -> list[tuple[SCFRun, SCFRun]]:
    """Pair runs by system id, keeping only systems both modes ran."""

    baseline_by_id = {run.system_id: run for run in baseline}
    return [
        (baseline_by_id[run.system_id], run) for run in candidate if run.system_id in baseline_by_id
    ]


def _paired_max(baseline, candidate, accessor) -> float | None:
    values = [
        abs(float(accessor(after)) - float(accessor(before)))
        for before, after in _paired(baseline, candidate)
        if accessor(before) is not None and accessor(after) is not None
    ]
    return max(values) if values else None


def _paired_array_max(baseline, candidate, accessor) -> float | None:
    import numpy as np

    worst: float | None = None
    for before, after in _paired(baseline, candidate):
        left, right = accessor(before), accessor(after)
        if left is None or right is None:
            continue
        first = np.asarray(left, dtype=float)
        second = np.asarray(right, dtype=float)
        if first.shape != second.shape:
            # A shape mismatch is a comparison that cannot be made, not a
            # difference of zero; skipping it silently would hide it.
            raise BenchmarkError(
                f"{after.system_id}: cannot compare arrays of shape {first.shape} and {second.shape}"
            )
        difference = float(np.abs(first - second).max())
        worst = difference if worst is None else max(worst, difference)
    return worst


def _paired_density_max(baseline, candidate) -> float | None:
    import numpy as np

    from gmd_active_learning.adapters.dft.scf.recording.density import load_matrix_file

    worst: float | None = None
    for before, after in _paired(baseline, candidate):
        if not before.density_matrix_path or not after.density_matrix_path:
            continue
        try:
            first = load_matrix_file(before.density_matrix_path)
            second = load_matrix_file(after.density_matrix_path)
        except (OSError, ValueError, RuntimeError) as exc:
            raise BenchmarkError(f"{after.system_id}: could not read a recorded density ({exc})") from exc
        if first.shape != second.shape:
            raise BenchmarkError(
                f"{after.system_id}: density shapes differ, {first.shape} vs {second.shape}"
            )
        difference = float(np.abs(first - second).max())
        worst = difference if worst is None else max(worst, difference)
    return worst


def write_benchmark_results(report: BenchmarkReport, output_dir: str | Path) -> Path:
    """Write ``benchmark_results.json`` into ``output_dir``."""

    from gmd_active_learning.utils.atomic_io import write_json_atomic

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return write_json_atomic(directory / BENCHMARK_RESULTS_NAME, report.to_dict())


def summarize_mode(mode: str, runs: Sequence[SCFRun]) -> ModeSummary:
    """Aggregate one mode's runs without reference to any other mode."""

    if not runs:
        raise BenchmarkError(f"mode {mode!r} has no runs")
    converged = [run for run in runs if run.converged]
    times = [run.wall_time_seconds for run in converged if run.wall_time_seconds is not None]
    return ModeSummary(
        mode=mode,
        description=MODE_DESCRIPTIONS.get(mode, "unlabelled mode"),
        n_runs=len(runs),
        n_converged=len(converged),
        mean_iterations=(sum(run.scf_iterations for run in converged) / len(converged)) if converged else None,
        mean_wall_time=(sum(times) / len(times)) if times else None,
        failure_rate=1.0 - len(converged) / len(runs),
        fallback_rate=sum(1 for run in runs if run.fallback_used) / len(runs),
    )


def write_report(report: BenchmarkReport, path: str | Path) -> Path:
    """Write the benchmark report as JSON."""

    from gmd_active_learning.utils.atomic_io import write_json_atomic

    return write_json_atomic(path, report.to_dict())


def runs_from_scf_histories(
    scf_root: str | Path,
    *,
    label: str,
    wall_times: Mapping[str, float] | None = None,
) -> list[SCFRun]:
    """Build runs from recorded ``<task-id>/scf_history.json`` files.

    This is the path that needs no extra instrumentation: the SCF recorder
    already writes these for every CP2K task, native or accelerated.
    """

    from gmd_active_learning.adapters.dft.scf.dataset import discover_scf_runs
    from gmd_active_learning.adapters.dft.scf.schema import SCFTrajectory, SchemaError

    runs: list[SCFRun] = []
    for history_path in discover_scf_runs(scf_root):
        system_id = history_path.parent.name
        try:
            payload = json.loads(history_path.read_text(encoding="utf-8"))
            trajectory = SCFTrajectory.from_dict(payload)
        except (json.JSONDecodeError, SchemaError, OSError) as exc:
            raise BenchmarkError(f"{history_path}: could not read SCF history ({exc})") from exc
        runs.append(
            SCFRun(
                label=label,
                system_id=system_id,
                scf_iterations=trajectory.n_iterations,
                converged=trajectory.converged,
                wall_time_seconds=None if wall_times is None else wall_times.get(system_id),
                final_energy=trajectory.final_energy,
                metadata={
                    "initial_guess": trajectory.initial_guess,
                    "mixing_method": trajectory.mixing_method,
                    "eps_scf": trajectory.eps_scf,
                },
            )
        )
    if not runs:
        raise BenchmarkError(f"no scf_history.json files found under {scf_root}")
    return runs


def _headline(comparisons: Mapping[str, GuessComparison], energy_tolerance: float) -> dict[str, Any]:
    """The flat summary block promised by the Stage 7 contract."""

    preferred = comparisons.get(MODE_ML_FULL) or comparisons.get(MODE_ML_GUESS)
    if preferred is None:
        return {
            "baseline_iterations": None,
            "ml_iterations": None,
            "speedup": None,
            "energy_error": None,
            "force_error": None,
            "converged": False,
        }
    return {
        "baseline_iterations": preferred.baseline_iterations,
        "ml_iterations": preferred.ml_iterations,
        "speedup": preferred.speedup,
        "energy_error": preferred.energy_error,
        "force_error": preferred.force_error,
        # "converged" means the accelerated mode converged everywhere *and*
        # landed on the same energy; a fast wrong answer is not convergence.
        "converged": bool(
            preferred.converged
            and (preferred.energy_error is None or preferred.energy_error <= energy_tolerance)
        ),
        "headline_mode": preferred.candidate_label,
    }


__all__ = [
    "BENCHMARK_MODES",
    "BENCHMARK_RESULTS_NAME",
    "BENCHMARK_VERSION",
    "DEFAULT_ENERGY_TOLERANCE",
    "DEFAULT_FORCE_TOLERANCE",
    "DEFAULT_STRESS_TOLERANCE",
    "MODE_DESCRIPTIONS",
    "MODE_ML_FULL",
    "MODE_ML_GUESS",
    "MODE_NATIVE",
    "MODE_RESTART",
    "AcceptanceCriteria",
    "BenchmarkError",
    "BenchmarkReport",
    "ModeResult",
    "ModeSummary",
    "load_runs",
    "run_benchmark",
    "runs_from_scf_histories",
    "summarize_mode",
    "write_benchmark_results",
    "write_report",
]
