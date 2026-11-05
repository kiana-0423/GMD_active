"""Local scheduler: real subprocesses, exit-code capture, cancellation, walltime, restart safety."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from gmd_active_learning.adapters.schedulers.base import scheduler_job_from_dict, scheduler_job_to_dict
from gmd_active_learning.adapters.schedulers.config import SchedulerConfig, SchedulerResources
from gmd_active_learning.adapters.schedulers.local import JOB_RECORD, LocalScheduler
from gmd_active_learning.contracts.labels import ExecutionStatus


def _wait(scheduler: LocalScheduler, job, timeout: float = 20.0) -> ExecutionStatus:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = scheduler.poll(job)
        if state.terminal:
            return state
        time.sleep(0.05)
    raise AssertionError(f"job {job.job_id} did not finish: {job.state}")


def _scheduler(**local: object) -> LocalScheduler:
    return LocalScheduler(SchedulerConfig.model_validate({"local": {"cancel_grace_seconds": 2.0, **local}}))


@pytest.mark.parametrize("code, expected", [(0, ExecutionStatus.COMPLETED), (3, ExecutionStatus.FAILED)])
def test_exit_code_is_recorded(tmp_path: Path, code: int, expected: ExecutionStatus) -> None:
    scheduler = _scheduler()
    job = scheduler.submit(tmp_path / "w", [sys.executable, "-c", f"print('out'); raise SystemExit({code})"])
    assert job.state is ExecutionStatus.SUBMITTED
    assert _wait(scheduler, job) is expected
    assert job.exit_code == code
    assert (tmp_path / "w" / "job.stdout").read_text() == "out\n"


def test_environment_is_exported_to_the_job(tmp_path: Path) -> None:
    scheduler = _scheduler()
    job = scheduler.submit(tmp_path / "w", ["sh", "-c", 'printf "%s" "$GMD_TEST_VALUE"'], environment={"GMD_TEST_VALUE": "a b"})
    assert _wait(scheduler, job) is ExecutionStatus.COMPLETED
    assert (tmp_path / "w" / "job.stdout").read_text() == "a b"


def test_cancel_stops_the_process(tmp_path: Path) -> None:
    scheduler = _scheduler()
    job = scheduler.submit(tmp_path / "w", ["sleep", "60"])
    time.sleep(0.2)
    assert scheduler.poll(job) is ExecutionStatus.RUNNING
    started = time.monotonic()
    assert scheduler.cancel(job) is ExecutionStatus.CANCELLED
    assert time.monotonic() - started < 10
    assert scheduler.poll(job) is ExecutionStatus.CANCELLED
    pid = int(job.job_id)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_cancel_escalates_to_sigkill_when_sigterm_is_ignored(tmp_path: Path) -> None:
    scheduler = _scheduler(cancel_grace_seconds=0.5)
    code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)"
    job = scheduler.submit(tmp_path / "w", [sys.executable, "-c", code])
    deadline = time.monotonic() + 10
    stdout = tmp_path / "w" / "job.stdout"
    while time.monotonic() < deadline and not (stdout.exists() and "ready" in stdout.read_text()):
        time.sleep(0.05)
    assert scheduler.cancel(job) is ExecutionStatus.CANCELLED


def test_walltime_is_enforced(tmp_path: Path) -> None:
    scheduler = _scheduler()
    job = scheduler.submit(tmp_path / "w", ["sleep", "60"], SchedulerResources(walltime=1))
    state = _wait(scheduler, job, timeout=15)
    assert state is ExecutionStatus.FAILED
    assert job.raw_state == "TIMEOUT"


def test_walltime_enforcement_can_be_disabled(tmp_path: Path) -> None:
    scheduler = _scheduler(enforce_walltime=False)
    job = scheduler.submit(tmp_path / "w", ["sleep", "1.5"], SchedulerResources(walltime=1))
    assert _wait(scheduler, job) is ExecutionStatus.COMPLETED


def test_poll_survives_a_controller_restart(tmp_path: Path) -> None:
    first = _scheduler()
    job = first.submit(tmp_path / "w", ["sh", "-c", "sleep 1; exit 5"])
    persisted = scheduler_job_to_dict(job)
    del first  # the controller "restarts": no Popen handle survives

    second = _scheduler()
    restored = scheduler_job_from_dict(persisted)
    assert second.poll(restored) is ExecutionStatus.RUNNING
    assert _wait(second, restored) is ExecutionStatus.FAILED
    assert restored.exit_code == 5


def test_a_restarted_controller_can_cancel(tmp_path: Path) -> None:
    job = _scheduler().submit(tmp_path / "w", ["sleep", "60"])
    restored = scheduler_job_from_dict(scheduler_job_to_dict(job))
    second = _scheduler()
    time.sleep(0.2)
    assert second.cancel(restored) is ExecutionStatus.CANCELLED
    with pytest.raises(ProcessLookupError):
        os.kill(int(job.job_id), 0)


def test_vanished_process_without_exit_code_is_failed_not_running(tmp_path: Path) -> None:
    scheduler = _scheduler()
    job = scheduler.submit(tmp_path / "w", ["sh", "-c", "true"])
    _wait(scheduler, job)
    (tmp_path / "w" / ".gmd_exit_code").unlink()  # simulate SIGKILL before the wrapper could record it
    restored = scheduler_job_from_dict(scheduler_job_to_dict(job))
    assert _scheduler().poll(restored) is ExecutionStatus.FAILED
    assert restored.raw_state == "VANISHED"


def test_missing_job_record_is_unknown_and_reconcile_keeps_it_unknown(tmp_path: Path) -> None:
    scheduler = _scheduler()
    job = scheduler.submit(tmp_path / "w", ["sleep", "0.2"])
    _wait(scheduler, job)
    (tmp_path / "w" / ".gmd_exit_code").unlink()
    (tmp_path / "w" / JOB_RECORD).unlink()
    restored = scheduler_job_from_dict(scheduler_job_to_dict(job))
    assert _scheduler().reconcile(restored) is ExecutionStatus.UNKNOWN


def test_local_scheduler_is_restart_safe_capability() -> None:
    assert "restart_safe" in LocalScheduler.capabilities
    assert _scheduler().check_runtime().available
