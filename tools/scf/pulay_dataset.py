# Ported from PHAL (hotspot_al.cp2k_ml_scf.pulay_dataset) at 96263b6; MIT License, see NOTICE.
"""Generate and validate a CP2K Pulay-matrix dataset for ML-SCF training.

One sample is one converged SCF run, kept whole::

    runtime/cp2k/pulay_dataset/
        dataset.json                  manifest: format, systems, counts
        dataset_report.json           statistics (see :func:`dataset_report`)
        <sample_id>/
            sample.json               structure, settings, convergence, checks
            cp2k.out                  the run it all came from
            trajectory/
                overlap.npy
                metadata.json
                iter_0000/
                    density.npy
                    pulay_matrix.npy      CP2K's own B, when it mixed here
                    metadata.json         iteration, residual, energy, alpha
                ...
                final/

The point of the layout is that a sample is self-describing and independently
checkable. Nothing is aggregated into a single array at generation time,
because that is where the AO-reconstruction mistake became invisible: once
matrices are stacked, a wrong metric looks exactly like a right one.

Generation runs through the existing :class:`SchedulerBackend`, so the same
code path drives ``local``, ``slurm`` and ``pbs``.

**This module does not train anything.** It produces data and refuses data it
cannot vouch for.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from gmd_active_learning.adapters.dft.scf.execution import ExecutionRequest, JobState, SchedulerBackend
from gmd_active_learning.adapters.dft.scf.pulay_matrix import (
    DEFAULT_CONDITION_LIMIT,
    METRIC_GSPACE_KERKER,
    REPRESENTATION_CP2K_PULAY,
    PulayRecordError,
    b_matrix_diagnostics,
    read_records,
    verify_records,
)
from tools.scf.difficulty import assess_dataset, difficulty_summary
from gmd_active_learning.adapters.dft.scf.recording.trajectory import (
    PULAY_MATRIX_NAME,
    TRAJECTORY_DIR_NAME,
    SCFTrajectoryError,
    load_trajectory,
    record_scf_trajectory,
)

DATASET_DIR_NAME = "pulay_dataset"
DEFAULT_DATASET_ROOT = Path("runtime/cp2k/pulay_dataset")
DATASET_FORMAT = "phal-pulay-dataset-v1"
MANIFEST_NAME = "dataset.json"
REPORT_NAME = "dataset_report.json"
SAMPLE_NAME = "sample.json"
OUTPUT_NAME = "cp2k.out"

#: Environment variable ``ml_pulay_recorder.F`` reads.
PULAY_RECORD_ENV = "PHAL_ML_SCF_PULAY_RECORD_DIR"


class PulayDatasetError(RuntimeError):
    """Raised when a dataset cannot be generated or is unusable."""


@dataclass(frozen=True, slots=True)
class PulayTask:
    """One CP2K calculation to run for the dataset."""

    task_id: str
    input_path: Path
    #: Free-form description of what this structure is, copied into sample.json
    #: so a sample can be traced back without re-reading the CP2K input.
    structure: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SampleValidation:
    """Why a sample was kept or rejected."""

    sample_id: str
    accepted: bool
    reasons: tuple[str, ...] = ()
    n_frames: int = 0
    n_pulay: int = 0
    n_ao: int = 0
    scf_steps: int | None = None
    max_pulay_deviation: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "n_frames": self.n_frames,
            "n_pulay_matrices": self.n_pulay,
            "n_ao": self.n_ao,
            "scf_steps": self.scf_steps,
            "max_pulay_deviation": self.max_pulay_deviation,
        }


def load_task_list(path: str | Path) -> list[PulayTask]:
    """Read a task list.

    ``[{"task_id": ..., "input": "path/to.inp", "structure": {...}}, ...]``
    """

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PulayDatasetError(f"{source}: cannot be read as a task list ({exc})") from exc
    if not isinstance(payload, list):
        raise PulayDatasetError(f"{source}: task list must be a JSON array")

    tasks: list[PulayTask] = []
    seen: set[str] = set()
    for index, entry in enumerate(payload):
        if not isinstance(entry, Mapping) or "input" not in entry:
            raise PulayDatasetError(f"{source}: entry {index} has no 'input'")
        task_id = str(entry.get("task_id") or f"task-{index:04d}")
        if task_id in seen:
            raise PulayDatasetError(f"{source}: duplicate task_id {task_id!r}")
        seen.add(task_id)
        tasks.append(
            PulayTask(
                task_id=task_id,
                input_path=(source.parent / str(entry["input"])).resolve()
                if not Path(str(entry["input"])).is_absolute()
                else Path(str(entry["input"])),
                structure=dict(entry.get("structure", {})),
                resources=dict(entry.get("resources", {})),
            )
        )
    return tasks


# -- generation ---------------------------------------------------------


def generate_dataset(
    tasks: Sequence[PulayTask],
    *,
    root: str | Path = DEFAULT_DATASET_ROOT,
    binary: str,
    scheduler: SchedulerBackend,
    config: Mapping[str, Any],
    poll_interval: float = 5.0,
    timeout: float = 3600.0,
    sleep: Callable[[float], None] = time.sleep,
) -> list[SampleValidation]:
    """Run every task, then collect and validate what came back.

    All tasks are submitted before any is polled, so a scheduler runs them
    concurrently. A task that fails or times out becomes a rejected sample with
    a reason rather than an exception: one bad geometry in a hundred must not
    discard the other ninety-nine.
    """

    # Absolute, because each job runs with its own sample directory as the
    # working directory: a relative output path would be resolved against that
    # and CP2K would abort on open, before it printed anything to say why.
    dataset_root = Path(root).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    submitted: list[tuple[PulayTask, Path, Any]] = []
    for task in tasks:
        sample_dir = dataset_root / task.task_id
        record_dir = sample_dir / "pulay_records"
        record_dir.mkdir(parents=True, exist_ok=True)
        output_path = sample_dir / OUTPUT_NAME
        if output_path.exists():
            # CP2K appends; a stale file would be parsed together with this run.
            output_path.unlink()

        request = ExecutionRequest(
            command=(binary, "-i", str(task.input_path), "-o", str(output_path)),
            work_dir=sample_dir,
            environment={PULAY_RECORD_ENV: str(record_dir)},
            resources=dict(task.resources),
            metadata={"task_id": task.task_id},
        )
        try:
            job = scheduler.submit(request)
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            submitted.append((task, sample_dir, exc))
            continue
        submitted.append((task, sample_dir, job))

    results: list[SampleValidation] = []
    deadline = time.monotonic() + timeout
    for task, sample_dir, job in submitted:
        if isinstance(job, Exception):
            results.append(
                SampleValidation(task.task_id, False, (f"submission failed: {job}",))
            )
            continue

        state = _wait(scheduler, job, deadline=deadline, poll_interval=poll_interval, sleep=sleep)
        if state not in (JobState.COMPLETED, JobState.PREPARED):
            results.append(
                SampleValidation(task.task_id, False, (f"job did not complete (state {state.value})",))
            )
            continue

        results.append(collect_sample(task, sample_dir, config=config))

    write_manifest(dataset_root, results)
    return results


def _wait(
    scheduler: SchedulerBackend,
    job: Any,
    *,
    deadline: float,
    poll_interval: float,
    sleep: Callable[[float], None],
) -> JobState:
    while True:
        state = scheduler.poll(job)
        if state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.PREPARED):
            return state
        if time.monotonic() >= deadline:
            return JobState.FAILED
        sleep(poll_interval)


def collect_sample(
    task: PulayTask,
    sample_dir: str | Path,
    *,
    config: Mapping[str, Any],
) -> SampleValidation:
    """Turn one finished run into a validated sample directory."""

    directory = Path(sample_dir)
    output_path = directory / OUTPUT_NAME
    if not output_path.is_file():
        return SampleValidation(task.task_id, False, ("CP2K produced no output",))

    text = output_path.read_text(encoding="utf-8", errors="replace")
    converged = "SCF run converged" in text
    steps = _scf_steps(text)

    record_dir = directory / "pulay_records"
    try:
        artifacts = record_scf_trajectory(
            text,
            output_root=directory.parent,
            task_id=directory.name,
            config=config,
            scf_converged=converged,
            strict=True,
            pulay_record_dir=record_dir if record_dir.is_dir() else None,
        )
    except (SCFTrajectoryError, PulayRecordError) as exc:
        return SampleValidation(task.task_id, False, (f"trajectory: {exc}",), scf_steps=steps)

    if artifacts is None:
        return SampleValidation(
            task.task_id, False, ("trajectory recording is disabled in the config",), scf_steps=steps
        )

    validation = validate_sample(artifacts.directory, converged=converged, scf_steps=steps)
    (directory / SAMPLE_NAME).write_text(
        json.dumps(
            {
                "sample_id": task.task_id,
                "structure": dict(task.structure),
                "input_path": str(task.input_path),
                "scf_converged": converged,
                "scf_steps": steps,
                "trajectory": str(artifacts.directory),
                "n_ao": artifacts.n_ao,
                "spin_channels": artifacts.spin_channels,
                "n_electrons": artifacts.n_electrons,
                "representation": REPRESENTATION_CP2K_PULAY,
                "metric": METRIC_GSPACE_KERKER,
                "validation": validation.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return validation


def _scf_steps(text: str) -> int | None:
    import re

    match = re.search(r"SCF run converged in\s+(\d+)\s+steps", text)
    return int(match.group(1)) if match else None


# -- validation ---------------------------------------------------------


def validate_sample(
    trajectory_dir: str | Path,
    *,
    converged: bool,
    scf_steps: int | None = None,
) -> SampleValidation:
    """Accept a sample only if every part of it is present and self-consistent.

    Each rejection names one thing, so a report says *why* a run was dropped
    rather than how many were.
    """

    directory = Path(trajectory_dir)
    sample_id = directory.parent.name
    reasons: list[str] = []

    if not converged:
        # An unconverged run has no reliable final density, so its hindsight
        # target is meaningless -- which is the whole label.
        reasons.append("SCF did not converge")

    try:
        densities, overlap, metadata = load_trajectory(directory)
    except SCFTrajectoryError as exc:
        return SampleValidation(sample_id, False, (f"trajectory unreadable: {exc}",), scf_steps=scf_steps)

    n_frames = len(densities)
    n_ao = int(densities[0].shape[-1]) if densities else 0

    if overlap is None or overlap.size == 0:
        reasons.append("missing overlap matrix")
    elif not np.isfinite(overlap).all():
        reasons.append("overlap contains non-finite values")

    for index, density in enumerate(densities):
        if not np.isfinite(density).all():
            reasons.append(f"frame {index}: density contains non-finite values")
            break
        if float(np.abs(density - np.swapaxes(density, 1, 2)).max()) > 1e-8:
            reasons.append(f"frame {index}: density is not Hermitian")
            break

    representation = metadata.get("pulay_representation")
    metric = metadata.get("pulay_metric")
    n_pulay = int(metadata.get("n_pulay_matrices") or 0)

    if n_pulay == 0:
        reasons.append("no CP2K Pulay matrices were recorded")
    else:
        if representation != REPRESENTATION_CP2K_PULAY:
            reasons.append(
                f"wrong representation {representation!r}; expected {REPRESENTATION_CP2K_PULAY!r}"
            )
        if metric != METRIC_GSPACE_KERKER:
            reasons.append(f"wrong metric {metric!r}; expected {METRIC_GSPACE_KERKER!r}")

    verification = metadata.get("pulay_verification") or {}
    deviation = float(verification.get("max_deviation", float("nan")))
    failed = verification.get("failed_iterations") or []
    if failed:
        reasons.append(
            f"Pulay matrices at iterations {failed} do not reproduce CP2K's own coefficients"
        )

    # A trajectory with no mixing history teaches nothing about mixing.
    if n_frames < 4:
        reasons.append(f"trajectory too short to learn from ({n_frames} frames)")

    return SampleValidation(
        sample_id=sample_id,
        accepted=not reasons,
        reasons=tuple(reasons),
        n_frames=n_frames,
        n_pulay=n_pulay,
        n_ao=n_ao,
        scf_steps=scf_steps,
        max_pulay_deviation=deviation,
    )


def validate_dataset(root: str | Path) -> list[SampleValidation]:
    """Re-validate every sample already on disk."""

    dataset_root = Path(root)
    results: list[SampleValidation] = []
    for sample_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        trajectory = sample_dir / TRAJECTORY_DIR_NAME
        # A directory that was never a sample -- rendered inputs, scratch --
        # is not a rejected sample. Counting it as one would understate the
        # acceptance rate with something that was never a run.
        if not any(
            (sample_dir / name).exists()
            for name in (SAMPLE_NAME, OUTPUT_NAME, TRAJECTORY_DIR_NAME)
        ):
            continue
        if not trajectory.is_dir():
            results.append(SampleValidation(sample_dir.name, False, ("no trajectory directory",)))
            continue
        sample_file = sample_dir / SAMPLE_NAME
        converged, steps = True, None
        if sample_file.is_file():
            try:
                payload = json.loads(sample_file.read_text(encoding="utf-8"))
                converged = bool(payload.get("scf_converged", True))
                steps = payload.get("scf_steps")
            except json.JSONDecodeError:
                pass
        results.append(validate_sample(trajectory, converged=converged, scf_steps=steps))
    return results


def write_manifest(root: str | Path, results: Sequence[SampleValidation]) -> Path:
    dataset_root = Path(root)
    accepted = [result for result in results if result.accepted]
    path = dataset_root / MANIFEST_NAME
    path.write_text(
        json.dumps(
            {
                "dataset_format": DATASET_FORMAT,
                "representation": REPRESENTATION_CP2K_PULAY,
                "metric": METRIC_GSPACE_KERKER,
                "n_samples": len(results),
                "n_accepted": len(accepted),
                "n_rejected": len(results) - len(accepted),
                "samples": [result.to_dict() for result in results],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


# -- statistics ---------------------------------------------------------


def dataset_report(root: str | Path, results: Sequence[SampleValidation] | None = None) -> dict[str, Any]:
    """Summarise a dataset: size, shapes, element distribution, convergence.

    The element distribution is over the recorded Pulay matrices themselves.
    It is worth reporting because ``B`` spans many orders of magnitude across
    an SCF, and a model that normalises its input needs to know how wide that
    range is before anyone is surprised by it.
    """

    dataset_root = Path(root)
    checks = list(results) if results is not None else validate_dataset(dataset_root)
    accepted = [check for check in checks if check.accepted]

    rejection_counts: dict[str, int] = {}
    for check in checks:
        for reason in check.reasons:
            # Collapse per-frame detail so the histogram stays readable.
            key = reason.split(":")[0].strip()
            rejection_counts[key] = rejection_counts.get(key, 0) + 1

    depths: list[int] = []
    elements: list[float] = []
    condition_numbers: list[float] = []
    diagnostics: list[Mapping[str, Any]] = []
    for check in accepted:
        trajectory = dataset_root / check.sample_id / TRAJECTORY_DIR_NAME
        for matrix_path in sorted(trajectory.glob(f"*/{PULAY_MATRIX_NAME}")):
            matrix = np.load(matrix_path)
            depths.append(int(matrix.shape[0]))
            elements.extend(np.abs(matrix[np.abs(matrix) > 0.0]).tolist())
            detail = _frame_diagnostics(matrix_path.parent, matrix)
            diagnostics.append(detail)
            condition = detail.get("condition_number")
            if condition is not None:
                condition_numbers.append(float(condition))

    magnitudes = np.log10(np.asarray(elements)) if elements else np.asarray([])
    scf_steps = [check.scf_steps for check in accepted if check.scf_steps is not None]

    return {
        "chemistry": _chemistry(dataset_root, accepted),
        "conditioning": _conditioning(diagnostics),
        # Over every sample, not only the accepted ones: the runs that failed
        # are the hard class, and they are what robustness work is aimed at.
        "difficulty": difficulty_summary(assess_dataset(dataset_root)),
        "dataset_format": DATASET_FORMAT,
        "representation": REPRESENTATION_CP2K_PULAY,
        "metric": METRIC_GSPACE_KERKER,
        "systems": {
            "n_total": len(checks),
            "n_accepted": len(accepted),
            "n_rejected": len(checks) - len(accepted),
            "acceptance_rate": (len(accepted) / len(checks)) if checks else 0.0,
        },
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "iterations": {
            "n_trajectory_frames": int(sum(check.n_frames for check in accepted)),
            "n_pulay_matrices": int(sum(check.n_pulay for check in accepted)),
            "frames_per_system": _summary([check.n_frames for check in accepted]),
        },
        "matrix_dimensions": {
            "history_depth": _summary(depths),
            "n_ao": _summary([check.n_ao for check in accepted]),
        },
        "element_distribution": {
            "n_elements": int(magnitudes.size),
            "log10_abs": _summary(magnitudes.tolist()),
            "note": (
                "log10 of |B_ij| over non-zero elements. B spans many decades across an "
                "SCF; a model that normalises by max|B| removes the scale but still sees "
                "this spread within each matrix."
            ),
            "condition_number_log10": _summary(
                [float(np.log10(value)) for value in condition_numbers if np.isfinite(value) and value > 0]
            ),
        },
        "convergence": {
            "scf_steps": _summary(scf_steps),
            "n_converged": len(scf_steps),
        },
        "verification": {
            "max_pulay_deviation": max(
                (check.max_pulay_deviation for check in accepted if np.isfinite(check.max_pulay_deviation)),
                default=float("nan"),
            ),
            "note": (
                "Largest disagreement between B^-1 1/(1^T B^-1 1) and the coefficients "
                "CP2K derived from the same B. Non-zero would mean the exported matrix is "
                "not the one CP2K solved."
            ),
        },
    }


def _frame_diagnostics(frame_dir: Path, matrix: np.ndarray) -> dict[str, Any]:
    """Per-frame ``B`` diagnostics, preferring what was recorded at run time.

    Falling back to recomputation keeps datasets generated before the
    diagnostics existed readable, rather than silently reporting nothing for
    them.
    """

    metadata_path = frame_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            stored = json.loads(metadata_path.read_text(encoding="utf-8")).get("pulay_diagnostics")
        except json.JSONDecodeError:
            stored = None
        if isinstance(stored, Mapping) and stored.get("finite"):
            return dict(stored)
    try:
        return b_matrix_diagnostics(matrix)
    except PulayRecordError:
        return {}


def _conditioning(diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How hard the recorded solves are.

    ``B`` is a Gram matrix of residuals that become collinear as the SCF
    converges, so it goes near-singular exactly where a mixer matters most.
    These numbers say how much of the dataset sits in that regime -- which is a
    modelling constraint, not a data defect.
    """

    if not diagnostics:
        return {"n_matrices": 0}

    conditions = [float(entry["condition_number"]) for entry in diagnostics if "condition_number" in entry]
    finite = [value for value in conditions if np.isfinite(value) and value > 0.0]
    deficient = [entry for entry in diagnostics if entry.get("rank_deficient")]
    needs_shift = [entry for entry in diagnostics if entry.get("regularization_required")]

    return {
        "n_matrices": len(diagnostics),
        "condition_number_log10": _summary([float(np.log10(value)) for value in finite]),
        "n_singular": len(conditions) - len(finite),
        "rank_deficit": _summary(
            [float(entry["depth"] - entry["rank"]) for entry in diagnostics if "rank" in entry]
        ),
        "n_rank_deficient": len(deficient),
        "n_indefinite": sum(1 for entry in diagnostics if entry.get("positive_definite") is False),
        "eigenvalue_min_log10": _summary(
            [
                float(np.log10(abs(entry["eigenvalue_min"])))
                for entry in diagnostics
                if entry.get("eigenvalue_min")
            ]
        ),
        "eigenvalue_max_log10": _summary(
            [
                float(np.log10(abs(entry["eigenvalue_max"])))
                for entry in diagnostics
                if entry.get("eigenvalue_max")
            ]
        ),
        "coefficient_l1": _summary(
            [
                float(entry["coefficient_l1"])
                for entry in diagnostics
                if np.isfinite(entry.get("coefficient_l1", np.inf))
            ]
        ),
        "regularization": {
            "n_required": len(needs_shift),
            "fraction_required": len(needs_shift) / len(diagnostics),
            "condition_limit": float(
                diagnostics[0].get("condition_limit", DEFAULT_CONDITION_LIMIT)
            ),
            "relative_shift": _summary(
                [
                    float(entry["regularization_relative"])
                    for entry in needs_shift
                    if np.isfinite(entry.get("regularization_relative", np.inf))
                ]
            ),
            "note": (
                "Diagonal shift s, relative to the largest eigenvalue, that would bring "
                "cond(B + sI) to the limit. Reported only; nothing shifts the matrix CP2K "
                "solved. A high fraction means a model consuming B needs its own "
                "regularisation to be numerically stable."
            ),
        },
        "coefficient_l1_note": (
            "||c||_1 for the native solve. sum(c) = 1, so values well above 1 mean the "
            "extrapolation relies on cancellation between large opposite-signed terms."
        ),
    }


def _chemistry(root: Path, accepted: Sequence[SampleValidation]) -> dict[str, Any]:
    """Element and geometry diversity, from what each sample recorded.

    A dataset can be large and still be one molecule at ninety geometries. This
    is the section that tells them apart, so it counts distinct elements and
    families rather than only samples.
    """

    elements: dict[str, int] = {}
    families: dict[str, int] = {}
    formulas: dict[str, int] = {}
    n_atoms: list[float] = []
    volumes: list[float] = []
    min_distances: list[float] = []
    n_periodic = 0
    described = 0

    for check in accepted:
        payload = _sample_payload(root / check.sample_id)
        structure = payload.get("structure") if isinstance(payload, Mapping) else None
        if not isinstance(structure, Mapping) or not structure:
            continue
        described += 1
        for element in structure.get("elements", ()) or ():
            elements[str(element)] = elements.get(str(element), 0) + 1
        family = structure.get("group") or structure.get("family")
        if family:
            families[str(family)] = families.get(str(family), 0) + 1
        formula = structure.get("formula")
        if formula:
            formulas[str(formula)] = formulas.get(str(formula), 0) + 1
        if structure.get("n_atoms") is not None:
            n_atoms.append(float(structure["n_atoms"]))
        if structure.get("cell_volume_A3"):
            volumes.append(float(structure["cell_volume_A3"]))
        if structure.get("min_distance_A"):
            min_distances.append(float(structure["min_distance_A"]))
        if structure.get("periodic"):
            n_periodic += 1

    return {
        "n_described": described,
        "n_undescribed": len(accepted) - described,
        "elements": dict(sorted(elements.items())),
        "n_distinct_elements": len(elements),
        "families": dict(sorted(families.items())),
        "n_distinct_formulas": len(formulas),
        "formulas": dict(sorted(formulas.items())),
        "n_periodic": n_periodic,
        "n_molecular": described - n_periodic,
        "geometry": {
            "n_atoms": _summary(n_atoms),
            "cell_volume_A3": _summary(volumes),
            "min_distance_A": _summary(min_distances),
        },
        "note": (
            "Diversity is what the first trained model lacked: nine geometries of two "
            "molecules share one basis, one element pair and one n_ao, so nothing in the "
            "training set distinguished a transferable rule from a memorised one."
        ),
    }


def _sample_payload(sample_dir: Path) -> Mapping[str, Any]:
    path = sample_dir / SAMPLE_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
    }


def write_dataset_report(root: str | Path, results: Sequence[SampleValidation] | None = None) -> Path:
    dataset_root = Path(root)
    path = dataset_root / REPORT_NAME
    path.write_text(
        json.dumps(dataset_report(dataset_root, results), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
