# Ported from PHAL (hotspot_al.cp2k_ml_scf.validation) at 96263b6; MIT License, see NOTICE.
"""Validation for SCF samples and whole SCF datasets.

Validation is split into two severities.  An *error* means the record cannot be
trained on and must be rejected; a *warning* means the record is usable but
scientifically suspect (a non-converged trajectory, a residual history that
never decreases).  Keeping the distinction lets the dataset absorb the messy
tail of a real active-learning campaign without silently training on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from gmd_active_learning.adapters.dft.scf.schema import (
    SCFSample,
    SCFTrajectory,
    SchemaError,
)
from gmd_active_learning.utils.electronic_state import multiplicity_is_consistent

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One problem found in a sample, located by field path."""

    severity: str
    field_path: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.field_path}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {"severity": self.severity, "field": self.field_path, "message": self.message}


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Aggregated issues for one sample or one dataset split."""

    subject: str
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == WARNING)

    @property
    def ok(self) -> bool:
        """Whether the subject is usable, i.e. carries no errors."""

        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "ok": self.ok,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def merged_with(self, other: "ValidationReport") -> "ValidationReport":
        return ValidationReport(subject=self.subject, issues=self.issues + other.issues)


def validate_payload(payload: Any, *, subject: str = "sample") -> ValidationReport:
    """Validate a raw JSON payload, reporting decode failures as errors."""

    try:
        sample = SCFSample.from_dict(payload)
    except SchemaError as exc:
        return ValidationReport(subject=subject, issues=(ValidationIssue(ERROR, "sample", str(exc)),))
    except (TypeError, ValueError) as exc:
        return ValidationReport(
            subject=subject,
            issues=(ValidationIssue(ERROR, "sample", f"could not decode sample: {exc}"),),
        )
    return validate_sample(sample, subject=subject)


def validate_sample(
    sample: SCFSample,
    *,
    subject: str | None = None,
    check_density: bool = False,
) -> ValidationReport:
    """Validate a decoded sample's physics and internal consistency.

    ``check_density`` additionally opens and checks the recorded density
    matrix.  It is off by default because it costs a file read per sample, and
    because a sample without a density is perfectly valid for trajectory-level
    work.  Training on ``target="density_matrix"`` turns it on.
    """

    issues: list[ValidationIssue] = []
    issues.extend(_structure_issues(sample))
    issues.extend(_electronic_state_issues(sample))
    issues.extend(validate_trajectory(sample.trajectory))
    issues.extend(_artifact_issues(sample))
    if check_density:
        issues.extend(validate_density_artifacts(sample))
    return ValidationReport(subject=subject or sample.sample_id, issues=tuple(issues))


def validate_density_artifacts(
    sample: SCFSample,
    *,
    electron_tolerance: float = 1.0e-3,
    hermitian_tolerance: float = 1.0e-6,
) -> tuple[ValidationIssue, ...]:
    """Open the recorded density and check it against the sample it belongs to.

    Every failure here is an *error*: a density that is the wrong shape, is not
    Hermitian, or does not hold the right number of electrons cannot be trained
    on, and using it would teach a model to produce initial guesses CP2K will
    reject.
    """

    import numpy as np

    artifacts = sample.artifacts
    path_text = artifacts.density_matrix_path
    if not path_text:
        return (
            ValidationIssue(
                ERROR,
                "artifacts.density_matrix_path",
                "no density matrix is recorded for this sample",
            ),
        )

    source = Path(path_text)
    if not source.is_file():
        return (
            ValidationIssue(ERROR, "artifacts.density_matrix_path", f"density file is missing: {source}"),
        )

    from gmd_active_learning.adapters.dft.scf.recording.density import load_matrix_file

    try:
        density = load_matrix_file(source)
    except (OSError, ValueError) as exc:
        return (
            ValidationIssue(ERROR, "artifacts.density_matrix_path", f"could not read {source}: {exc}"),
        )

    issues: list[ValidationIssue] = []

    if density.ndim != 3 or density.shape[1] != density.shape[2]:
        return (
            ValidationIssue(
                ERROR,
                "artifacts.density_matrix_path",
                f"density must be (spin, nao, nao), got shape {density.shape}",
            ),
        )

    if not np.isfinite(density).all():
        # Reported first and alone: every later comparison against a NaN is
        # false, so continuing would produce misleading secondary errors.
        return (
            ValidationIssue(ERROR, "artifacts.density_matrix_path", "density contains non-finite values"),
        )

    expected_channels = sample.electronic_state.spin_channels
    if density.shape[0] != expected_channels:
        issues.append(
            ValidationIssue(
                ERROR,
                "artifacts.density_matrix_path",
                f"density has {density.shape[0]} spin channel(s), the electronic state declares "
                f"{expected_channels}",
            )
        )

    if artifacts.n_ao is not None and artifacts.n_ao != density.shape[1]:
        issues.append(
            ValidationIssue(
                ERROR,
                "artifacts.n_ao",
                f"recorded n_ao={artifacts.n_ao} disagrees with the matrix dimension {density.shape[1]}",
            )
        )

    asymmetry = float(np.abs(density - np.swapaxes(density, 1, 2)).max())
    if asymmetry > hermitian_tolerance:
        issues.append(
            ValidationIssue(
                ERROR,
                "artifacts.density_matrix_path",
                f"density matrix is not Hermitian (max asymmetry {asymmetry:g})",
            )
        )

    overlap = None
    if artifacts.overlap_path:
        overlap_source = Path(artifacts.overlap_path)
        if not overlap_source.is_file():
            issues.append(
                ValidationIssue(ERROR, "artifacts.overlap_path", f"overlap file is missing: {overlap_source}")
            )
        else:
            try:
                overlap = load_matrix_file(overlap_source)
            except (OSError, ValueError) as exc:
                issues.append(
                    ValidationIssue(ERROR, "artifacts.overlap_path", f"could not read {overlap_source}: {exc}")
                )
            else:
                if overlap.shape != density.shape[1:]:
                    issues.append(
                        ValidationIssue(
                            ERROR,
                            "artifacts.overlap_path",
                            f"overlap shape {overlap.shape} does not match density {density.shape[1:]}",
                        )
                    )
                    overlap = None
                elif not np.isfinite(overlap).all():
                    issues.append(
                        ValidationIssue(ERROR, "artifacts.overlap_path", "overlap contains non-finite values")
                    )
                    overlap = None

    if overlap is None:
        # Deliberate deviation from PHAL (findings F-21/SCF-008): PHAL fell back
        # to Tr(P), which contradicts docs/scf/density_representation.md section 4
        # ("an unverifiable count is refused, not approximated"); Tr(P) runs
        # 20-50% low in a non-orthogonal AO basis.
        issues.append(
            ValidationIssue(
                ERROR,
                "artifacts.overlap_path",
                "no overlap matrix recorded, so the electron count Tr(P S) cannot be verified; "
                "refused rather than approximated with Tr(P)",
            )
        )
        return issues
    counted = float(sum(np.sum(channel * overlap) for channel in density))
    method = "Tr(PS)"

    # The recorded density holds *valence* electrons under a pseudopotential,
    # which is not the all-electron count in the electronic state. The recorded
    # count is therefore the reference, and the state's count is only used when
    # nothing else is available.
    reference = artifacts.density_electron_count
    if reference is None:
        issues.append(
            ValidationIssue(
                WARNING,
                "artifacts.density_electron_count",
                "no recorded electron count; density normalisation cannot be verified",
            )
        )
    else:
        tolerance = electron_tolerance * max(1.0, abs(float(reference)))
        if abs(counted - float(reference)) > tolerance:
            issues.append(
                ValidationIssue(
                    ERROR,
                    "artifacts.density_electron_count",
                    f"{method} = {counted:.6f} but the recorded electron count is {float(reference):.6f}",
                )
            )

    return tuple(issues)


def validate_trajectory(trajectory: SCFTrajectory) -> tuple[ValidationIssue, ...]:
    """Validate an SCF trajectory in isolation."""

    issues: list[ValidationIssue] = []
    if trajectory.n_iterations == 0:
        issues.append(ValidationIssue(ERROR, "trajectory.iterations", "trajectory has no iterations"))
        return tuple(issues)

    steps = [iteration.step for iteration in trajectory.iterations]
    if steps != sorted(steps):
        issues.append(ValidationIssue(ERROR, "trajectory.iterations", f"iteration steps are not ordered: {steps}"))
    if len(set(steps)) != len(steps):
        issues.append(ValidationIssue(ERROR, "trajectory.iterations", "iteration steps contain duplicates"))
    if any(step < 0 for step in steps):
        issues.append(ValidationIssue(ERROR, "trajectory.iterations", "iteration steps must be non-negative"))

    for index, iteration in enumerate(trajectory.iterations):
        path = f"trajectory.iterations[{index}]"
        if iteration.residual is not None:
            if not math.isfinite(iteration.residual):
                issues.append(ValidationIssue(ERROR, f"{path}.residual", "residual is not finite"))
            elif iteration.residual < 0.0:
                issues.append(
                    ValidationIssue(ERROR, f"{path}.residual", f"residual is negative ({iteration.residual!r})")
                )
        if iteration.energy is not None and not math.isfinite(iteration.energy):
            issues.append(ValidationIssue(ERROR, f"{path}.energy", "energy is not finite"))

    if trajectory.eps_scf is not None and trajectory.eps_scf <= 0.0:
        issues.append(ValidationIssue(ERROR, "trajectory.eps_scf", f"EPS_SCF must be positive, got {trajectory.eps_scf!r}"))
    if trajectory.max_scf is not None:
        if trajectory.max_scf <= 0:
            issues.append(ValidationIssue(ERROR, "trajectory.max_scf", f"MAX_SCF must be positive, got {trajectory.max_scf!r}"))
        elif trajectory.n_iterations > trajectory.max_scf:
            issues.append(
                ValidationIssue(
                    ERROR,
                    "trajectory.iterations",
                    f"{trajectory.n_iterations} iterations exceed MAX_SCF={trajectory.max_scf}",
                )
            )

    if not trajectory.converged:
        issues.append(ValidationIssue(WARNING, "trajectory.converged", "SCF did not converge"))
    elif trajectory.eps_scf is not None:
        final = trajectory.residual_history[-1]
        if final is not None and final > trajectory.eps_scf:
            issues.append(
                ValidationIssue(
                    WARNING,
                    "trajectory.converged",
                    f"marked converged but final residual {final:g} exceeds EPS_SCF {trajectory.eps_scf:g}",
                )
            )

    residuals = [value for value in trajectory.residual_history if value is not None]
    if len(residuals) >= 2 and residuals[-1] > residuals[0]:
        issues.append(
            ValidationIssue(
                WARNING,
                "trajectory.residual_history",
                f"residual grew from {residuals[0]:g} to {residuals[-1]:g}",
            )
        )
    if trajectory.initial_guess is None:
        issues.append(ValidationIssue(WARNING, "trajectory.initial_guess", "initial guess type is unknown"))
    return tuple(issues)


def validate_many(payloads: Iterable[tuple[str, Any]]) -> tuple[ValidationReport, ...]:
    """Validate ``(subject, payload)`` pairs, preserving input order."""

    return tuple(validate_payload(payload, subject=subject) for subject, payload in payloads)


def summarize(reports: Iterable[ValidationReport]) -> dict[str, Any]:
    """Aggregate reports into a JSON-safe summary."""

    materialized = list(reports)
    failed = [report for report in materialized if not report.ok]
    return {
        "n_subjects": len(materialized),
        "n_ok": len(materialized) - len(failed),
        "n_failed": len(failed),
        "n_errors": sum(len(report.errors) for report in materialized),
        "n_warnings": sum(len(report.warnings) for report in materialized),
        "failed": [report.to_dict() for report in failed],
    }


def _structure_issues(sample: SCFSample) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    structure = sample.structure
    if structure.n_atoms == 0:
        issues.append(ValidationIssue(ERROR, "structure.atomic_numbers", "structure has no atoms"))
    if any(number <= 0 for number in structure.atomic_numbers):
        issues.append(ValidationIssue(ERROR, "structure.atomic_numbers", "atomic numbers must be positive"))
    if any(number > 118 for number in structure.atomic_numbers):
        issues.append(ValidationIssue(ERROR, "structure.atomic_numbers", "atomic numbers must be <= 118"))
    volume = _cell_volume(structure.cell)
    if any(structure.pbc) and abs(volume) < 1e-8:
        issues.append(ValidationIssue(ERROR, "structure.cell", "periodic structure has a degenerate cell"))
    return issues


def _electronic_state_issues(sample: SCFSample) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    state = sample.electronic_state
    expected_electrons = sum(sample.structure.atomic_numbers) - state.charge
    if state.n_electrons != expected_electrons:
        issues.append(
            ValidationIssue(
                ERROR,
                "electronic_state.n_electrons",
                f"n_electrons={state.n_electrons} disagrees with sum(Z) - charge = {expected_electrons}",
            )
        )
    if not multiplicity_is_consistent(state.n_electrons, state.multiplicity):
        issues.append(
            ValidationIssue(
                ERROR,
                "electronic_state.multiplicity",
                f"multiplicity {state.multiplicity} is impossible for {state.n_electrons} electrons",
            )
        )
    expected_channels = 2 if state.spin_polarized else 1
    if state.spin_channels != expected_channels:
        issues.append(
            ValidationIssue(
                ERROR,
                "electronic_state.spin_channels",
                f"spin_channels={state.spin_channels} disagrees with spin_polarized={state.spin_polarized}",
            )
        )
    if state.multiplicity > 1 and not state.spin_polarized:
        issues.append(
            ValidationIssue(
                ERROR,
                "electronic_state.spin_polarized",
                f"multiplicity {state.multiplicity} requires a spin-polarized treatment",
            )
        )
    return issues


def _artifact_issues(sample: SCFSample) -> list[ValidationIssue]:
    # Density files are optional in Stage 1; only their self-consistency is checked.
    artifacts = sample.artifacts
    if artifacts.is_empty:
        return []
    if artifacts.density_format is None:
        return [
            ValidationIssue(
                WARNING,
                "artifacts.density_format",
                "an electronic-structure artifact is declared without a density_format",
            )
        ]
    return []


def _cell_volume(cell: tuple[tuple[float, float, float], ...]) -> float:
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = cell
    return (
        ax * (by * cz - bz * cy)
        - ay * (bx * cz - bz * cx)
        + az * (bx * cy - by * cx)
    )


__all__ = [
    "ERROR",
    "WARNING",
    "ValidationIssue",
    "ValidationReport",
    "summarize",
    "validate_density_artifacts",
    "validate_many",
    "validate_payload",
    "validate_sample",
    "validate_trajectory",
]
