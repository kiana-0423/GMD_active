"""BE-001: PHAL's JobState values map losslessly onto GMD's ExecutionStatus representation."""

from __future__ import annotations

from pathlib import Path

import pytest

from gmd_active_learning.adapters.schedulers.base import ExecutionRequest
from gmd_active_learning.contracts import ExecutionStatus, SchedulerJob
from gmd_active_learning.labeling.job_states import PHAL_JOB_STATES, from_phal_job_state, phal_job_state

REF = Path(__file__).resolve().parents[2] / "ref" / "lammps_monitor"


def _job(state: ExecutionStatus, raw_state: str | None = None, metadata: dict | None = None) -> SchedulerJob:
    return SchedulerJob(job_id="1", backend="local", state=state, work_dir="w", raw_state=raw_state, metadata=dict(metadata or {}))


@pytest.mark.parametrize("value", PHAL_JOB_STATES)
def test_every_phal_job_state_round_trips(value: str) -> None:
    state, raw_state, metadata = from_phal_job_state(value)
    assert phal_job_state(_job(state, raw_state, metadata)) == value


def test_only_pending_and_timeout_are_spelled_differently() -> None:
    renamed = {value: from_phal_job_state(value)[0].value for value in PHAL_JOB_STATES if from_phal_job_state(value)[0].value != value}
    assert renamed == {"pending": "queued", "timeout": "failed"}
    assert from_phal_job_state("timeout") == (ExecutionStatus.FAILED, "TIMEOUT", {"walltime_exceeded": True})
    assert {status.value for status in ExecutionStatus} == (set(PHAL_JOB_STATES) - {"pending", "timeout"}) | {"queued"}


def test_walltime_failures_from_any_scheduler_report_timeout() -> None:
    assert phal_job_state(_job(ExecutionStatus.FAILED, raw_state="TIMEOUT")) == "timeout"  # local scheduler
    assert phal_job_state(_job(ExecutionStatus.FAILED, raw_state="F", metadata={"walltime_exceeded": True})) == "timeout"  # PBS/Slurm
    assert phal_job_state(_job(ExecutionStatus.FAILED, raw_state="FAILED")) == "failed"
    with pytest.raises(ValueError, match="not a PHAL job state"):
        from_phal_job_state("queued")


def test_execution_request_from_command_stringifies_like_phal(tmp_path: Path) -> None:
    request = ExecutionRequest.from_command(["cp2k.psmp", "-i", Path("job.inp"), 3], work_dir=str(tmp_path), environment={"A": "1"})
    assert request.command == ("cp2k.psmp", "-i", "job.inp", "3") and request.work_dir == tmp_path
    assert dict(request.environment) == {"A": "1"} and dict(request.metadata) == {}


@pytest.mark.skipif(not (REF / "src" / "hotspot_al").is_dir(), reason="ref/lammps_monitor not present")
def test_phal_job_states_are_phal_enum_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(REF / "src"))
    from hotspot_al.backends.base import JobState

    assert tuple(state.value for state in JobState) == PHAL_JOB_STATES
