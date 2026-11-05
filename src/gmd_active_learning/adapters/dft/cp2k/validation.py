# Ported from PHAL (hotspot_al.cp2k.cp2k_result:CP2KResult.validation_errors) at 96263b6; MIT License, see NOTICE.
"""Scientific validity of a CP2K label, kept separate from execution status.

Baseline (findings F-27, kept): the output exists, an SCF convergence marker is present with no
non-convergence/ABORT marker, and energy, forces and stress are finite. A process that merely exits 0
is not a valid label. GMD additions: the number of force rows equals the region's atom count (PHAL
checked it only when writing the dataset and crashed out of ``poll_job``), the energy must be present
(F-12) and CP2K must have terminated normally (``PROGRAM ENDED``, i.e. the output is not truncated).
"""

from __future__ import annotations

import numpy as np

from gmd_active_learning.adapters.dft.cp2k.config import FailureClass, ValidationSettings
from gmd_active_learning.adapters.dft.cp2k.parse import CP2KOutput
from gmd_active_learning.contracts.labels import ValidationStatus


def validation_reasons(output: CP2KOutput | None, *, n_atoms: int, settings: ValidationSettings) -> list[str]:
    """Every reason why ``output`` is not a valid label (empty list = valid)."""

    if output is None:
        return ["CP2K output file was not produced"]
    reasons: list[str] = []
    if output.aborted:
        reasons.append(f"CP2K aborted: {output.abort_message or 'ABORT marker in output'}")
    if output.scf_not_converged_marker:
        reasons.append("SCF did not converge (CP2K reported 'SCF run NOT converged')")
    elif settings.require_scf_convergence and not output.scf_success_marker:
        reasons.append("no SCF convergence marker in the output")
    if output.energy is None:
        if settings.require_energy:
            reasons.append("no total energy (ENERGY| Total FORCE_EVAL) in the output")
    elif not np.isfinite(output.energy):
        reasons.append("energy is not finite")
    if output.forces is None:
        reasons.append("no force block in the output")
    else:
        if output.forces.shape != (n_atoms, 3):
            reasons.append(f"force block has {output.forces.shape[0]} rows for a region of {n_atoms} atoms")
        if not np.all(np.isfinite(output.forces)):
            reasons.append("forces contain non-finite values")
    if output.stress is not None and not np.all(np.isfinite(output.stress)):
        reasons.append("stress contains non-finite values")
    if output.scf_residual is not None and not np.isfinite(output.scf_residual):
        reasons.append("SCF residual is not finite")
    if settings.require_normal_termination and not output.normal_termination:
        reasons.append("CP2K did not terminate normally (no 'PROGRAM ENDED'; output truncated or killed)")
    return reasons


def validation_status(reasons: list[str]) -> ValidationStatus:
    return ValidationStatus.VALID if not reasons else ValidationStatus.INVALID


def output_is_complete(text: str) -> bool:
    """PHAL ``CP2KBackend.output_is_complete``: success marker present and no failure marker."""

    if any(marker in text for marker in ("SCF run NOT converged", "SCF not converged", "ABORT")):
        return False
    return any(marker in text for marker in ("SCF run converged", "SCF converged"))


def classify_failure(
    output: CP2KOutput | None,
    *,
    stage: str,
    exit_code: int | None,
    cancelled: bool = False,
    walltime_exceeded: bool = False,
    reasons: list[str] | None = None,
) -> FailureClass:
    """Map one failed attempt to the failure class that selects its retry ladder."""

    if cancelled:
        return FailureClass.CANCELLED
    if walltime_exceeded:
        return FailureClass.WALLTIME
    if output is None:
        return FailureClass.EXECUTION_FAILED
    if output.scf_not_converged_marker:
        return FailureClass.SCF_NOT_CONVERGED
    if stage == "hopt" and output.geo_opt_max_iter_reached:
        return FailureClass.GEO_OPT_NOT_CONVERGED
    if output.aborted:
        # A CP2K abort other than SCF non-convergence (bad input, missing basis) is deterministic.
        return FailureClass.INVALID_RESULT
    if not output.normal_termination or (exit_code not in (None, 0)):
        return FailureClass.EXECUTION_FAILED
    if not output.scf_success_marker:
        return FailureClass.SCF_NOT_CONVERGED
    return FailureClass.INVALID_RESULT


__all__ = ["classify_failure", "output_is_complete", "validation_reasons", "validation_status"]
