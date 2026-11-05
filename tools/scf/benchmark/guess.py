# Ported from PHAL (hotspot_al.cp2k_ml_scf.guess_benchmark) at 96263b6; MIT License, see NOTICE.
"""Benchmark what an initial density guess is worth.

The comparison is one system, two runs, one difference:

* **A** -- CP2K's atomic superposition guess (``SCF_GUESS ATOMIC``)
* **B** -- an ML-predicted density (``SCF_GUESS ML``)

and, when a controller is wired into the SCF loop, **C** -- ML guess plus
adaptive control.

Speed alone is not a result. A guess that halves the iteration count and moves
the total energy by a millihartree has not accelerated anything, it has changed
the answer -- so energy *and* forces are compared against the atomic-guess run
of the same system, which is the reference precisely because it is the one
nobody doubts. Forces are the stricter test: they are a derivative of the
converged state, and a density that is subtly wrong shows up there first.

Wall time is reported next to iteration count because they can disagree. An ML
guess costs a model evaluation, and on a small system that can be a
non-negligible fraction of an SCF that was already short. Iterations saved is
the claim; seconds saved is whether it mattered.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gmd_active_learning.adapters.dft.scf.execution import ExecutionRequest, JobState, SchedulerBackend

REPORT_NAME = "benchmark_ml_guess.json"
BENCHMARK_FORMAT = "phal-ml-guess-benchmark-v1"

GUESS_ATOMIC = "ATOMIC"
GUESS_ML = "ML"

#: Environment variables CP2K's in-process guess reads. ENABLED is a master
#: switch: without it the hooked build honours SCF_GUESS ML by falling back to
#: ATOMIC, which produces a run that looks successful and measures nothing.
GUESS_MODEL_ENV = "PHAL_ML_SCF_GUESS_MODEL"
ML_ENABLED_ENV = "PHAL_ML_SCF_ENABLED"
#: Mixing weight against the atomic guess, read by ml_scf_guess_alpha().
GUESS_ALPHA_ENV = "PHAL_ML_SCF_GUESS_ALPHA"
#: The adaptive SCF controller, and its counterfactual override.
CONTROLLER_ENV = "PHAL_ML_SCF_CONTROLLER"
CONTROLLER_FORCE_ENV = "PHAL_ML_SCF_CONTROLLER_FORCE"

_CTRL_SUMMARY = re.compile(
    r"ML-CTRL\|\s*summary:\s*actions=(\d+)\s+accepted=(\d+)\s+rejected=(\d+)\s+fell_back=([TF])"
)
#: Only the action lines. The rejection line also starts "iteration N:", so it
#: is excluded by name rather than by hoping the shapes differ.
_CTRL_ACTION = re.compile(
    r"ML-CTRL\|\s*iteration\s+(\d+):\s*(reset_diis|switch_mixing|increase_stability|keep)\b"
)

_STEPS = re.compile(r"SCF run converged in\s+(\d+)\s+steps")
_ENERGY = re.compile(r"ENERGY\|.*?energy.*?(?::|\])\s*([-+0-9.EDed]+)")
#: v2026.2 prints "FORCES| Atom x y z |f|"; older releases printed a bare
#: "ATOMIC FORCES in [a.u.]" table. Both are read, because a benchmark that
#: silently reports no forces looks exactly like one that found them equal.
_FORCES_BAR = re.compile(
    r"^\s*FORCES\|\s+(\d+)\s+"
    r"([-+]?[\d.]+(?:[EeDd][-+]?\d+)?)\s+"
    r"([-+]?[\d.]+(?:[EeDd][-+]?\d+)?)\s+"
    r"([-+]?[\d.]+(?:[EeDd][-+]?\d+)?)",
    re.MULTILINE,
)
_FORCE_ROW = re.compile(
    r"^\s*\d+\s+\d+\s+[A-Z][a-z]?\s+"
    r"([-+]?\d+\.\d+(?:[EeDd][-+]?\d+)?)\s+"
    r"([-+]?\d+\.\d+(?:[EeDd][-+]?\d+)?)\s+"
    r"([-+]?\d+\.\d+(?:[EeDd][-+]?\d+)?)\s*$",
    re.MULTILINE,
)
#: The last column of CP2K's own timing row is max wall time for the run.
_CP2K_TIME = re.compile(r"^\s*CP2K\s+1\s+1\.0\s+\S+\s+\S+\s+\S+\s+(\S+)\s*$", re.MULTILINE)
#: The three machine-readable lines ml_scf_log_decision writes on every ML run.
_ML_SOURCE = re.compile(r"ML-SCF\|\s*ml_guess_source:\s*(\S+)")
_ML_DIGEST = re.compile(r"ML-SCF\|\s*model_digest:\s*(\S+)")
_ML_SUCCESS = re.compile(r"ML-SCF\|\s*prediction_success:\s*([TF])")
_ML_REASON = re.compile(r"ML-SCF\|\s*fallback used:\s*(.+)")

GUESS_SOURCE_ARTIFACT = "artifact"
GUESS_SOURCE_TORCHSCRIPT = "torchscript"
GUESS_SOURCE_FALLBACK = "fallback"


class GuessBenchmarkError(RuntimeError):
    """Raised when a benchmark cannot be run or read."""


@dataclass(frozen=True, slots=True)
class GuessRun:
    """One CP2K run in the benchmark."""

    system: str
    initial_guess: str
    converged: bool
    scf_steps: int | None
    #: CP2K's own reported wall time where available, else the harness's.
    runtime_s: float
    energy: float | None
    forces: np.ndarray | None = None
    ml_guess_accepted: bool = False
    controller: Mapping[str, Any] | None = None
    failure: str | None = None
    runtime_source: str = "harness"
    #: Which route produced the guess: artifact, torchscript, or fallback.
    #: ``None`` when the run printed no ML-SCF lines at all, i.e. it was not an
    #: ML run -- which is different from an ML run that fell back.
    guess_source: str | None = None
    guess_model_digest: str | None = None
    guess_fallback_reason: str | None = None
    #: The descent itself, from CP2K's own SCF lines. Kept because "how many
    #: steps" hides whether a guess started badly or descended badly, and those
    #: call for different fixes.
    residuals: tuple[float, ...] = ()
    energies: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "initial_guess": self.initial_guess,
            "converged": self.converged,
            "scf_steps": self.scf_steps,
            "runtime": self.runtime_s,
            "runtime_source": self.runtime_source,
            "energy": self.energy,
            "n_forces": 0 if self.forces is None else int(self.forces.shape[0]),
            "ml_guess_accepted": self.ml_guess_accepted,
            "ml_guess_source": self.guess_source,
            "ml_guess_model_digest": self.guess_model_digest,
            "ml_guess_fallback_reason": self.guess_fallback_reason,
            "residuals": list(self.residuals),
            "energies": list(self.energies),
            "controller": dict(self.controller) if self.controller else None,
            "failure": self.failure,
        }


def parse_run(text: str, *, system: str, initial_guess: str, runtime_s: float) -> GuessRun:
    """Read one CP2K output into a benchmark record.

    A failed run is a record with ``converged=False``, not an exception: the
    convergence *rate* across systems is one of the three metrics, and it is
    only measurable if failures are data.
    """

    steps = _STEPS.search(text)
    energy = _ENERGY.search(text)
    forces = parse_forces(text)

    failure = None
    if steps is None:
        from tools.scf.difficulty import parse_abort

        failure = parse_abort(text) or (
            "SCF run NOT converged" if "SCF run NOT converged" in text else "no convergence line"
        )

    source = _ML_SOURCE.search(text)
    digest = _ML_DIGEST.search(text)
    success = _ML_SUCCESS.search(text)
    reason = _ML_REASON.search(text)

    from gmd_active_learning.adapters.dft.scf.scf_history import parse_scf_history

    history = parse_scf_history(text)
    residuals = tuple(
        float(item.residual) for item in history.iterations if item.residual is not None
    )
    energies = tuple(float(item.energy) for item in history.iterations if item.energy is not None)

    controller = None
    summary = _CTRL_SUMMARY.search(text)
    if summary:
        controller = {
            "n_actions": int(summary.group(1)),
            "n_accepted": int(summary.group(2)),
            "n_rejected": int(summary.group(3)),
            "fell_back": summary.group(4) == "T",
            "actions": [
                {"iteration": int(iteration), "action": action}
                for iteration, action in _CTRL_ACTION.findall(text)
            ],
        }

    reported = parse_cp2k_time(text)
    return GuessRun(
        system=system,
        initial_guess=initial_guess,
        converged=steps is not None,
        scf_steps=int(steps.group(1)) if steps else None,
        runtime_s=float(reported if reported is not None else runtime_s),
        runtime_source="cp2k" if reported is not None else "harness",
        energy=float(energy.group(1).replace("D", "E")) if energy else None,
        forces=forces,
        # Accepted means CP2K said so, not that the run looked normal.
        ml_guess_accepted=bool(success and success.group(1) == "T"),
        guess_source=source.group(1) if source else None,
        guess_model_digest=digest.group(1) if digest else None,
        guess_fallback_reason=reason.group(1).strip() if reason else None,
        residuals=residuals,
        energies=energies,
        controller=controller,
        failure=failure,
    )


def parse_forces(text: str) -> np.ndarray | None:
    """Pull the atomic force table out of a CP2K output.

    Takes the last table in the file: for ``ENERGY_FORCE`` there is one, but a
    geometry step would print several and only the final one belongs to the
    converged state.
    """

    index = text.rfind("FORCES| Atom")
    if index >= 0:
        rows = [row[1:] for row in _FORCES_BAR.findall(text[index:])]
        if rows:
            return np.asarray(
                [[float(value.replace("D", "E")) for value in row] for row in rows], dtype=float
            )

    index = text.rfind("ATOMIC FORCES in [a.u.]")
    if index < 0:
        return None
    rows = _FORCE_ROW.findall(text[index : index + 200_000])
    if not rows:
        return None
    return np.asarray(
        [[float(value.replace("D", "E")) for value in row] for row in rows], dtype=float
    )


def parse_cp2k_time(text: str) -> float | None:
    """CP2K's own reported wall time for the run.

    Preferred over the time the harness measured: benchmark jobs run
    concurrently, so harness wall time includes waiting for other jobs on the
    same cores. This number is the process's own.
    """

    match = _CP2K_TIME.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


# -- running ------------------------------------------------------------


def run_benchmark(
    inputs: Mapping[str, Mapping[str, Path]],
    *,
    binary: str,
    scheduler: SchedulerBackend,
    work_dir: str | Path,
    environment: Mapping[str, str] | None = None,
    guess_model: str | Path | None = None,
    timeout: float = 7200.0,
    poll_interval: float = 5.0,
) -> list[GuessRun]:
    """Run every ``(system, guess)`` input and collect the results.

    ``inputs`` is ``{system: {guess: input_path}}``. Every job is submitted
    before any is polled, so a scheduler runs them concurrently -- but note that
    on a shared machine the concurrency itself perturbs wall time, which is why
    the report carries iteration counts as the primary speed metric.
    """

    # Absolute: each job runs in its own directory, so a relative output path
    # would be resolved against that and CP2K would abort on open.
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    base_env = dict(environment or {})
    if guess_model is not None:
        base_env[GUESS_MODEL_ENV] = str(guess_model)
        base_env[ML_ENABLED_ENV] = "1"

    submitted: list[tuple[str, str, Path, float, Any]] = []
    for system, per_guess in inputs.items():
        for guess, input_path in per_guess.items():
            run_dir = root / f"{system}-{guess.lower()}"
            run_dir.mkdir(parents=True, exist_ok=True)
            output_path = run_dir / "cp2k.out"
            if output_path.exists():
                output_path.unlink()

            request = ExecutionRequest(
                command=(binary, "-i", str(Path(input_path).resolve()), "-o", str(output_path)),
                work_dir=run_dir,
                # The atomic arm must not see the ML switches at all: it is the
                # control, and a control that could take the ML path is not one.
                environment=base_env if guess == GUESS_ML else {
                    key: value
                    for key, value in base_env.items()
                    if key not in (GUESS_MODEL_ENV, ML_ENABLED_ENV)
                },
                metadata={"system": system, "guess": guess},
            )
            try:
                job = scheduler.submit(request)
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                submitted.append((system, guess, output_path, time.monotonic(), exc))
                continue
            submitted.append((system, guess, output_path, time.monotonic(), job))

    deadline = time.monotonic() + timeout
    runs: list[GuessRun] = []
    for system, guess, output_path, started, job in submitted:
        if isinstance(job, Exception):
            runs.append(
                GuessRun(system, guess, False, None, 0.0, None, failure=f"submission failed: {job}")
            )
            continue

        state = _wait(scheduler, job, deadline=deadline, poll_interval=poll_interval)
        runtime = time.monotonic() - started
        if not output_path.is_file():
            runs.append(
                GuessRun(system, guess, False, None, runtime, None, failure=f"no output ({state.value})")
            )
            continue
        text = output_path.read_text(encoding="utf-8", errors="replace")
        runs.append(parse_run(text, system=system, initial_guess=guess, runtime_s=runtime))
    return runs


def _wait(scheduler: SchedulerBackend, job: Any, *, deadline: float, poll_interval: float) -> JobState:
    while True:
        state = scheduler.poll(job)
        if state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.PREPARED):
            return state
        if time.monotonic() >= deadline:
            return JobState.FAILED
        time.sleep(poll_interval)


@dataclass(frozen=True, slots=True)
class GuessArm:
    """One configuration to benchmark: a guess, a model, and a mixing weight."""

    label: str
    guess: str = GUESS_ATOMIC
    model: str | Path | None = None
    #: P = (1 - alpha) P_atomic + alpha P_ML. 1.0 is the pure prediction.
    alpha: float = 1.0
    #: Run under the adaptive SCF controller. Orthogonal to the guess: the
    #: controller decides strategy and the guess decides where to start, and
    #: the point of separating them is to attribute any difference to one.
    controller: bool = False
    #: "<iteration>:<action>", forcing one action for counterfactual labelling.
    force: str | None = None

    def environment(self) -> dict[str, str]:
        environment: dict[str, str] = {}
        if self.guess == GUESS_ML and self.model is not None:
            environment.update(
                {
                    GUESS_MODEL_ENV: str(self.model),
                    ML_ENABLED_ENV: "1",
                    GUESS_ALPHA_ENV: f"{self.alpha:.6f}",
                }
            )
        if self.controller:
            environment[CONTROLLER_ENV] = "1"
            if self.force:
                environment[CONTROLLER_FORCE_ENV] = self.force
        return environment


def run_arms(
    inputs: Mapping[str, Mapping[str, Path]],
    arms: Sequence[GuessArm],
    *,
    binary: str,
    scheduler: SchedulerBackend,
    work_dir: str | Path,
    timeout: float = 7200.0,
    poll_interval: float = 5.0,
    max_parallel: int = 4,
) -> dict[str, list[GuessRun]]:
    """Run every ``(system, arm)`` pair; return the runs keyed by arm label.

    ``inputs`` maps a system to its rendered input per *guess*, so several arms
    can share one file: the only thing that differs between two ML arms is the
    environment, and re-rendering identical inputs would invite them to drift.

    Chunked, because a full sweep is dozens of concurrent CP2K processes and a
    machine that runs out of memory produces truncated output that reads exactly
    like a system which failed to converge.
    """

    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[GuessArm, str, Path, Path]] = []
    for system, per_guess in inputs.items():
        for arm in arms:
            if arm.guess not in per_guess:
                continue
            run_dir = root / f"{system}-{arm.label}"
            run_dir.mkdir(parents=True, exist_ok=True)
            jobs.append((arm, system, per_guess[arm.guess], run_dir))

    results: dict[str, list[GuessRun]] = {arm.label: [] for arm in arms}
    for start in range(0, len(jobs), max(1, max_parallel)):
        chunk = jobs[start : start + max(1, max_parallel)]
        submitted = []
        for arm, system, input_path, run_dir in chunk:
            output_path = run_dir / "cp2k.out"
            if output_path.exists():
                output_path.unlink()
            request = ExecutionRequest(
                command=(binary, "-i", str(Path(input_path).resolve()), "-o", str(output_path)),
                work_dir=run_dir,
                environment=arm.environment(),
                metadata={"system": system, "arm": arm.label},
            )
            started = time.monotonic()
            try:
                job = scheduler.submit(request)
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                job = exc
            submitted.append((arm, system, output_path, started, job))

        deadline = time.monotonic() + timeout
        for arm, system, output_path, started, job in submitted:
            if isinstance(job, Exception):
                results[arm.label].append(
                    GuessRun(system, arm.guess, False, None, 0.0, None, failure=str(job))
                )
                continue
            state = _wait(scheduler, job, deadline=deadline, poll_interval=poll_interval)
            runtime = time.monotonic() - started
            if not output_path.is_file():
                results[arm.label].append(
                    GuessRun(
                        system, arm.guess, False, None, runtime, None,
                        failure=f"no output ({state.value})",
                    )
                )
                continue
            text = output_path.read_text(encoding="utf-8", errors="replace")
            results[arm.label].append(
                parse_run(text, system=system, initial_guess=arm.guess, runtime_s=runtime)
            )
    return results


# -- reporting ----------------------------------------------------------


def compare(runs: Sequence[GuessRun], *, reference_guess: str = GUESS_ATOMIC) -> dict[str, Any]:
    """Build ``benchmark_ml_guess.json``.

    Errors are against the reference *run of the same system*, so a system that
    the reference itself failed on contributes to the convergence-rate metric
    and to nothing else -- there is no baseline to be accurate against.
    """

    by_system: dict[str, dict[str, GuessRun]] = {}
    for run in runs:
        by_system.setdefault(run.system, {})[run.initial_guess] = run

    entries: list[dict[str, Any]] = []
    for system in sorted(by_system):
        reference = by_system[system].get(reference_guess)
        for guess, run in sorted(by_system[system].items()):
            entry = run.to_dict()
            entry["energy_error"] = _energy_error(reference, run)
            entry["force_error"] = _force_error(reference, run)
            entry["steps_vs_reference"] = (
                None
                if reference is None or run.scf_steps is None or reference.scf_steps is None
                else run.scf_steps - reference.scf_steps
            )
            entry["speedup"] = (
                None
                if reference is None
                or not run.converged
                or not reference.converged
                or not run.scf_steps
                else reference.scf_steps / run.scf_steps
            )
            entries.append(entry)

    # An ML arm that silently fell back to ATOMIC produces two identical runs,
    # which reads as "the model matched the baseline" when it means "the model
    # never ran". That is a benchmark failure, not a footnote: a fallback row
    # is evidence about nothing, and averaging it into a speedup manufactures a
    # result out of the baseline.
    ml_runs = [entry for entry in entries if entry["initial_guess"] == GUESS_ML]
    ineffective = [entry for entry in ml_runs if not entry["ml_guess_accepted"]]
    warnings: list[str] = []
    for entry in ineffective:
        warnings.append(
            f"{entry['system']}: ML was requested but the guess came from "
            f"{entry['ml_guess_source'] or 'no ML path at all'}"
            + (f" ({entry['ml_guess_fallback_reason']})" if entry["ml_guess_fallback_reason"] else "")
        )
    valid = bool(ml_runs) and not ineffective

    return {
        "benchmark_format": BENCHMARK_FORMAT,
        "reference_guess": reference_guess,
        "n_systems": len(by_system),
        # False when any ML run fell back. A caller must treat this as a failed
        # benchmark rather than a benchmark with a caveat.
        "valid": valid,
        "n_ml_runs": len(ml_runs),
        "n_fallback_runs": len(ineffective),
        "guess_sources": _source_counts(entries),
        "warnings": warnings,
        "summary": _summary(entries, reference_guess),
        "runs": entries,
        "note": (
            "energy_error and force_error are against the reference guess for the same "
            "system, in hartree and hartree/bohr. A guess that saves iterations but moves "
            "the converged energy has changed the answer, not accelerated it."
        ),
    }


def _source_counts(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """How many runs came from each route, ML runs only."""

    counts: dict[str, int] = {}
    for entry in entries:
        if entry["initial_guess"] != GUESS_ML:
            continue
        key = str(entry.get("ml_guess_source") or "none")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _energy_error(reference: GuessRun | None, run: GuessRun) -> float | None:
    if reference is None or reference.energy is None or run.energy is None:
        return None
    if not (reference.converged and run.converged):
        return None
    return abs(float(run.energy) - float(reference.energy))


def _force_error(reference: GuessRun | None, run: GuessRun) -> float | None:
    if reference is None or reference.forces is None or run.forces is None:
        return None
    if not (reference.converged and run.converged):
        return None
    if reference.forces.shape != run.forces.shape:
        return None
    return float(np.abs(run.forces - reference.forces).max())


def _summary(entries: Sequence[Mapping[str, Any]], reference_guess: str) -> dict[str, Any]:
    by_guess: dict[str, list[Mapping[str, Any]]] = {}
    for entry in entries:
        by_guess.setdefault(str(entry["initial_guess"]), []).append(entry)

    summary: dict[str, Any] = {}
    for guess, items in sorted(by_guess.items()):
        converged = [item for item in items if item["converged"]]
        steps = [float(item["scf_steps"]) for item in converged if item["scf_steps"]]
        runtime = [float(item["runtime"]) for item in converged]
        energy_errors = [
            float(item["energy_error"]) for item in items if item.get("energy_error") is not None
        ]
        force_errors = [
            float(item["force_error"]) for item in items if item.get("force_error") is not None
        ]
        speedups = [float(item["speedup"]) for item in items if item.get("speedup") is not None]

        summary[guess] = {
            "n_runs": len(items),
            "n_converged": len(converged),
            "convergence_rate": len(converged) / len(items) if items else 0.0,
            "mean_scf_steps": float(np.mean(steps)) if steps else None,
            "total_scf_steps": int(sum(steps)) if steps else 0,
            "mean_runtime_s": float(np.mean(runtime)) if runtime else None,
            "max_energy_error": max(energy_errors) if energy_errors else None,
            "max_force_error": max(force_errors) if force_errors else None,
            "mean_speedup": float(np.mean(speedups)) if speedups else None,
        }

    reference = summary.get(reference_guess, {})
    for guess, detail in summary.items():
        if guess == reference_guess:
            continue
        base = reference.get("total_scf_steps") or 0
        detail["total_steps_vs_reference"] = (
            detail["total_scf_steps"] - base if base else None
        )
    return summary


def write_report(report: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
