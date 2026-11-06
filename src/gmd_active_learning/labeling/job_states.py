# Ported from PHAL (hotspot_al.backends.base.JobState) at 96263b6; MIT License, see NOTICE.
"""PHAL ``JobState`` vocabulary <-> GMD ``ExecutionStatus`` (ledger BE-001).

GMD replaced PHAL's job states with :class:`~gmd_active_learning.contracts.ExecutionStatus`. Its
persisted values differ in two places. ``pending`` became ``queued``. ``timeout`` became ``failed``
with ``raw_state="TIMEOUT"`` and ``metadata["walltime_exceeded"] = True``, because a walltime kill is a
failure that needs its own retry policy, not a separate terminal state. The other seven values are
spelled identically. These helpers translate both ways without loss, e.g. to read or report PHAL-era
state.
"""

from __future__ import annotations

from typing import Any

from gmd_active_learning.contracts import ExecutionStatus, SchedulerJob

#: every PHAL JobState value (hotspot_al.backends.base.JobState)
PHAL_JOB_STATES = ("prepared", "submitted", "pending", "running", "completed", "failed", "cancelled", "timeout", "unknown")

_RENAMED = {"pending": ExecutionStatus.QUEUED}


def from_phal_job_state(value: str) -> tuple[ExecutionStatus, str | None, dict[str, Any]]:
    """``(state, raw_state, metadata)`` for a PHAL JobState value; raises ``ValueError`` for others."""

    value = str(value).lower()
    if value not in PHAL_JOB_STATES:
        raise ValueError(f"not a PHAL job state: {value!r} (expected one of {', '.join(PHAL_JOB_STATES)})")
    if value == "timeout":
        return ExecutionStatus.FAILED, "TIMEOUT", {"walltime_exceeded": True}
    return (_RENAMED[value] if value in _RENAMED else ExecutionStatus(value)), None, {}


def phal_job_state(job: SchedulerJob) -> str:
    """PHAL's JobState value for a GMD scheduler job (walltime failures report ``timeout``)."""

    if job.state is ExecutionStatus.FAILED and (job.raw_state == "TIMEOUT" or bool((job.metadata or {}).get("walltime_exceeded"))):
        return "timeout"
    if job.state is ExecutionStatus.QUEUED:
        return "pending"
    return job.state.value


__all__ = ["PHAL_JOB_STATES", "from_phal_job_state", "phal_job_state"]
