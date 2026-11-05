# Ported from PHAL (hotspot_al.backends.schedulers:LocalSchedulerBackend) at 96263b6; MIT License, see NOTICE.
"""Local subprocess scheduler that survives a controller restart.

PHAL kept the ``Popen`` handle in memory, so after a restart ``poll`` returned a stale state forever,
and ``cancel`` sent SIGTERM without waiting (findings F-26, ledger SCH-001). Here the command runs in
a detached session through a shell wrapper that writes its exit code to ``.gmd_exit_code``; the pid,
process-group id and start time are persisted in ``.gmd_local_job.json``. Any later process (or a
new scheduler instance) can therefore poll, enforce the walltime and cancel the job.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gmd_active_learning.adapters.registry import BackendRole, RuntimeStatus
from gmd_active_learning.adapters.schedulers.base import (
    EXIT_CODE_FILE,
    SchedulerBackend,
    ensure_fresh_work_dir,
    read_exit_code,
    render_job_body,
)
from gmd_active_learning.adapters.schedulers.config import SchedulerConfig, SchedulerResources
from gmd_active_learning.contracts.labels import ExecutionStatus, SchedulerJob
from gmd_active_learning.core.exceptions import SchedulerError
from gmd_active_learning.utils.atomic_io import write_json_atomic

LOGGER = logging.getLogger(__name__)

SCRIPT_NAME = "gmd_local_job.sh"
JOB_RECORD = ".gmd_local_job.json"
CANCEL_MARKER = ".gmd_cancelled"
WALLTIME_MARKER = ".gmd_walltime_exceeded"


class LocalScheduler(SchedulerBackend):
    """Run jobs as detached local processes."""

    name = "local"
    role = BackendRole.SCHEDULER
    capabilities = frozenset({"submit", "poll", "cancel", "reconcile", "exit_code", "walltime", "restart_safe"})

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        super().__init__(config)
        self._processes: dict[str, subprocess.Popen[bytes]] = {}

    def check_runtime(self) -> RuntimeStatus:
        shell = self.config.local.shell
        available = Path(shell).exists() or bool(_which(shell))
        return RuntimeStatus(self.name, self.role, available, f"shell {shell}" if available else f"shell not found: {shell}")

    def submit(
        self,
        work_dir: str | Path,
        command: Sequence[str],
        resources: SchedulerResources | Mapping[str, Any] | None = None,
        *,
        environment: Mapping[str, str] | None = None,
        job_name: str | None = None,
    ) -> SchedulerJob:
        work_dir = Path(work_dir).resolve()
        ensure_fresh_work_dir(work_dir)
        resolved = self.resolve_resources(resources)
        ignored = [key for key in ("partition", "queue", "account", "memory") if getattr(resolved, key) is not None]
        if (resolved.nodes or 1) > 1:
            ignored.append("nodes")
        if resolved.extra_directives:
            ignored.append("extra_directives")
        if ignored:
            LOGGER.warning("local scheduler ignores resource fields: %s", ", ".join(ignored))
        script = work_dir / SCRIPT_NAME
        script.write_text(
            "\n".join(["#!" + self.config.local.shell, "set -u", *render_job_body(work_dir, command, environment), ""]),
            encoding="utf-8",
        )
        script.chmod(0o750)
        try:
            process = subprocess.Popen(
                [self.config.local.shell, str(script)],
                cwd=work_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            raise SchedulerError(f"could not start local job in {work_dir}: {exc}") from exc
        job_id = str(process.pid)
        self._processes[job_id] = process
        record = {
            "pid": process.pid,
            "pgid": process.pid,  # start_new_session makes the wrapper its own group leader
            "script": str(script),
            "started_at": time.time(),
            "walltime_seconds": resolved.walltime_seconds if self.config.local.enforce_walltime else None,
            "job_name": job_name,
        }
        write_json_atomic(work_dir / JOB_RECORD, record)
        LOGGER.info("local job %s started in %s", job_id, work_dir)
        return self._new_job(job_id, work_dir, script=str(script), walltime_seconds=record["walltime_seconds"])

    def poll(self, job: SchedulerJob) -> ExecutionStatus:
        work_dir = Path(job.work_dir)
        record = _read_record(work_dir)
        exit_code = read_exit_code(work_dir)
        process = self._processes.get(job.job_id)
        if process is not None:
            process.poll()  # reap a finished child so it does not linger as a zombie
        if (work_dir / CANCEL_MARKER).exists() or job.metadata.get("cancel_requested"):
            if exit_code is not None or not self._alive(job, record):
                return self._set(job, ExecutionStatus.CANCELLED, raw_state="CANCELLED", exit_code=exit_code)
        if (work_dir / WALLTIME_MARKER).exists():
            if exit_code is not None or not self._alive(job, record):
                return self._set(job, ExecutionStatus.FAILED, raw_state="TIMEOUT", exit_code=exit_code)
        if exit_code is not None:
            state = ExecutionStatus.COMPLETED if exit_code == 0 else ExecutionStatus.FAILED
            return self._set(job, state, raw_state="EXITED", exit_code=exit_code)
        if record is None:
            job.metadata["poll_error"] = f"{JOB_RECORD} missing in {work_dir}"
            return self._set(job, ExecutionStatus.UNKNOWN, raw_state="NO_RECORD")
        if self._alive(job, record):
            limit = record.get("walltime_seconds")
            if limit and time.time() - float(record["started_at"]) > float(limit):
                LOGGER.warning("local job %s exceeded its walltime of %s s; terminating", job.job_id, limit)
                (work_dir / WALLTIME_MARKER).write_text(f"{limit}\n", encoding="utf-8")
                self._terminate(job, record)
                return self._set(job, ExecutionStatus.FAILED, raw_state="TIMEOUT", exit_code=read_exit_code(work_dir))
            return self._set(job, ExecutionStatus.RUNNING, raw_state="RUNNING")
        # The wrapper writes the exit code before it exits; re-read once to close that window.
        exit_code = read_exit_code(work_dir)
        if exit_code is not None:
            state = ExecutionStatus.COMPLETED if exit_code == 0 else ExecutionStatus.FAILED
            return self._set(job, state, raw_state="EXITED", exit_code=exit_code)
        returncode = process.returncode if process is not None else None
        job.metadata["poll_error"] = "process ended without recording an exit code (killed or host restarted)"
        return self._set(job, ExecutionStatus.FAILED, raw_state="VANISHED", exit_code=returncode)

    def cancel(self, job: SchedulerJob) -> ExecutionStatus:
        work_dir = Path(job.work_dir)
        state = self.poll(job)
        if state.terminal:
            return state
        job.metadata["cancel_requested"] = True
        (work_dir / CANCEL_MARKER).write_text("cancelled\n", encoding="utf-8")
        record = _read_record(work_dir)
        if record is not None:
            self._terminate(job, record)
        if self._alive(job, record):
            raise SchedulerError(f"local job {job.job_id} did not stop after SIGTERM/SIGKILL")
        return self._set(job, ExecutionStatus.CANCELLED, raw_state="CANCELLED", exit_code=read_exit_code(work_dir))

    # -- process helpers ---------------------------------------------------------------------------
    def _alive(self, job: SchedulerJob, record: Mapping[str, Any] | None) -> bool:
        process = self._processes.get(job.job_id)
        if process is not None:
            return process.poll() is None
        if record is None:
            return False
        return _pid_runs_script(int(record["pid"]), str(record.get("script", "")))

    def _terminate(self, job: SchedulerJob, record: Mapping[str, Any]) -> None:
        pgid = int(record.get("pgid", record["pid"]))
        _signal_group(pgid, signal.SIGTERM)
        deadline = time.monotonic() + float(self.config.local.cancel_grace_seconds)
        while time.monotonic() < deadline and self._alive(job, record):
            time.sleep(0.05)
        if self._alive(job, record):
            _signal_group(pgid, signal.SIGKILL)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and self._alive(job, record):
                time.sleep(0.05)
        _signal_group(pgid, signal.SIGKILL, quiet=True)  # stragglers that ignored SIGTERM in the group


def _read_record(work_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((work_dir / JOB_RECORD).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _signal_group(pgid: int, sig: signal.Signals, *, quiet: bool = False) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        if not quiet:
            raise SchedulerError(f"not allowed to signal process group {pgid}: {exc}") from exc


def _pid_runs_script(pid: int, script: str) -> bool:
    """Whether ``pid`` is alive and still runs our wrapper (guards against pid reuse)."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False  # a different user's process: the pid was reused
    try:
        result = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], text=True, capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return True  # cannot verify; the pid exists
    command_line = result.stdout.strip()
    if not command_line or "<defunct>" in command_line:
        return False
    return not script or script in command_line


def _which(program: str) -> str | None:
    from shutil import which

    return which(program)


__all__ = ["CANCEL_MARKER", "EXIT_CODE_FILE", "JOB_RECORD", "LocalScheduler", "SCRIPT_NAME", "WALLTIME_MARKER"]
