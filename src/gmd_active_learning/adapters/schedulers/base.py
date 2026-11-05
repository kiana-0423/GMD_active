# Ported from PHAL (hotspot_al.backends.schedulers, hotspot_al.backends.base:SchedulerBackend) at 96263b6;
# MIT License, see NOTICE.
"""Common scheduler protocol, job artifacts and command execution.

Every scheduler implements ``submit(work_dir, command, resources) -> SchedulerJob``,
``poll(job) -> ExecutionStatus`` (with ``job.exit_code`` filled when known), ``cancel(job)`` and
``reconcile(job)``. Jobs run through a small shell wrapper that records the command's exit code in
``<work_dir>/.gmd_exit_code`` (written atomically), so a definitive outcome survives a controller
restart and can be recovered when the scheduler itself has forgotten the job.

``ExecutionStatus.UNKNOWN`` never means "failed": callers must not resubmit on UNKNOWN; they call
:meth:`SchedulerBackend.reconcile`, which inspects the scheduler and the work directory.
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from gmd_active_learning.adapters.registry import BackendRole, RuntimeStatus
from gmd_active_learning.adapters.schedulers.config import SchedulerConfig, SchedulerResources
from gmd_active_learning.contracts.labels import ExecutionStatus, SchedulerJob
from gmd_active_learning.core.exceptions import SchedulerError

LOGGER = logging.getLogger(__name__)

EXIT_CODE_FILE = ".gmd_exit_code"
JOB_STDOUT = "job.stdout"
JOB_STDERR = "job.stderr"
SCHEDULER_STDOUT = "scheduler.out"
SCHEDULER_STDERR = "scheduler.err"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_JOB_NAME_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")

class SubmissionRejected(SchedulerError):
    """The scheduler definitely did not accept the job (non-zero submit status, missing command).

    A plain :class:`SchedulerError` from ``submit`` (e.g. a timeout) leaves the outcome unknown: the job
    may exist, so callers must reconcile rather than resubmit.
    """


TERMINAL_STATES = frozenset({ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED})


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """Scheduler-neutral request: what to run, where, with which extra environment."""

    command: tuple[str, ...]
    work_dir: Path
    environment: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_command(
        cls,
        command: Sequence[str | Path],
        *,
        work_dir: str | Path,
        environment: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionRequest:
        return cls(
            command=tuple(str(part) for part in command),
            work_dir=Path(work_dir),
            environment=dict(environment or {}),
            metadata=dict(metadata or {}),
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_scheduler_command(
    argv: Sequence[str],
    *,
    timeout: float,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one scheduler CLI call with a timeout; launch problems become :class:`SchedulerError`.

    A non-zero exit status is returned to the caller, which decides whether it is an error.
    """

    argv = [str(part) for part in argv]
    try:
        return subprocess.run(argv, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise SchedulerError(f"scheduler command not found: {argv[0]!r}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SchedulerError(f"scheduler command {shlex.join(argv)!r} timed out after {timeout:g} s") from exc
    except OSError as exc:
        raise SchedulerError(f"scheduler command {shlex.join(argv)!r} could not be executed: {exc}") from exc


def read_exit_code(work_dir: str | Path) -> int | None:
    """Exit code recorded by the job wrapper, or None when the command has not finished."""

    path = Path(work_dir) / EXIT_CODE_FILE
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    try:
        return int(text)
    except ValueError:
        LOGGER.warning("ignoring malformed exit-code file %s: %r", path, text)
        return None


def sanitize_job_name(name: str | None, *, default: str = "gmd_job", max_length: int = 64) -> str:
    cleaned = _JOB_NAME_SAFE.sub("_", (name or default)).strip("_") or default
    if not cleaned[0].isalpha():
        cleaned = f"j{cleaned}"
    return cleaned[:max_length]


def render_job_body(
    work_dir: Path,
    command: Sequence[str],
    environment: Mapping[str, str] | None = None,
) -> list[str]:
    """Shell lines shared by every scheduler: run ``command`` and record its exit code atomically."""

    if not command:
        raise SchedulerError("cannot submit an empty command")
    lines = [f"cd {shlex.quote(str(work_dir))}"]
    for key, value in sorted((environment or {}).items()):
        if _ENV_NAME.match(key) is None:
            raise SchedulerError(f"invalid environment variable name {key!r}")
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.extend(
        [
            f"{shlex.join(list(command))} > {JOB_STDOUT} 2> {JOB_STDERR}",
            "rc=$?",
            f'printf "%s\\n" "$rc" > {EXIT_CODE_FILE}.tmp && mv -f {EXIT_CODE_FILE}.tmp {EXIT_CODE_FILE}',
            'exit "$rc"',
        ]
    )
    return lines


def ensure_fresh_work_dir(work_dir: Path) -> None:
    """Refuse to run a second job in a directory that already holds a job outcome."""

    work_dir.mkdir(parents=True, exist_ok=True)
    if (work_dir / EXIT_CODE_FILE).exists():
        raise SchedulerError(
            f"{work_dir} already holds a finished job ({EXIT_CODE_FILE}); use a new directory per attempt"
        )


class SchedulerBackend(ABC):
    """Execution contract for local, Slurm, PBS or plugin schedulers."""

    name: ClassVar[str]
    role: ClassVar[BackendRole] = BackendRole.SCHEDULER
    capabilities: ClassVar[frozenset[str]] = frozenset({"submit", "poll", "cancel", "reconcile", "exit_code"})

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> SchedulerBackend:
        """Build from the complete application configuration (reads the ``scheduler`` section)."""

        section = config.get("scheduler", {}) if isinstance(config, Mapping) else {}
        return cls(SchedulerConfig.model_validate(section or {}))

    @property
    def timeout(self) -> float:
        return float(self.config.command_timeout_seconds)

    def resolve_resources(self, resources: SchedulerResources | Mapping[str, Any] | None) -> SchedulerResources:
        if resources is None:
            return self.config.default_resources
        if not isinstance(resources, SchedulerResources):
            resources = SchedulerResources.model_validate(dict(resources))
        return self.config.default_resources.merged(resources)

    @abstractmethod
    def submit(
        self,
        work_dir: str | Path,
        command: Sequence[str],
        resources: SchedulerResources | Mapping[str, Any] | None = None,
        *,
        environment: Mapping[str, str] | None = None,
        job_name: str | None = None,
    ) -> SchedulerJob:
        """Start ``command`` in ``work_dir`` and return a handle in state SUBMITTED."""

    def submit_request(
        self,
        request: ExecutionRequest,
        resources: SchedulerResources | Mapping[str, Any] | None = None,
        *,
        job_name: str | None = None,
    ) -> SchedulerJob:
        job = self.submit(
            request.work_dir,
            request.command,
            resources,
            environment=request.environment,
            job_name=job_name,
        )
        job.metadata.update({key: value for key, value in request.metadata.items() if key not in job.metadata})
        return job

    @abstractmethod
    def poll(self, job: SchedulerJob) -> ExecutionStatus:
        """Refresh ``job.state`` (and ``job.exit_code``/``job.raw_state``) and return the state."""

    @abstractmethod
    def cancel(self, job: SchedulerJob) -> ExecutionStatus:
        """Cancel the job; raises :class:`SchedulerError` when cancellation cannot be confirmed."""

    @abstractmethod
    def check_runtime(self) -> RuntimeStatus:
        """Whether the scheduler commands are available."""

    def reconcile(self, job: SchedulerJob) -> ExecutionStatus:
        """Return a definitive state when one can be established, else UNKNOWN.

        Consults the scheduler first; if it cannot tell, the exit code recorded by the job wrapper in
        the work directory decides. Never resubmits anything.
        """

        state = self.poll(job)
        if state is not ExecutionStatus.UNKNOWN:
            return state
        exit_code = read_exit_code(job.work_dir)
        if exit_code is None:
            job.metadata["reconcile"] = "scheduler state unknown and no exit-code file in the work directory"
            return ExecutionStatus.UNKNOWN
        resolved = self._terminal_state(job, exit_code)
        job.metadata["reconcile"] = f"resolved from {EXIT_CODE_FILE} (exit code {exit_code})"
        self._set(job, resolved, raw_state=job.raw_state or "RECONCILED", exit_code=exit_code)
        return resolved

    # -- helpers -----------------------------------------------------------------------------------
    @staticmethod
    def _terminal_state(job: SchedulerJob, exit_code: int | None) -> ExecutionStatus:
        if job.metadata.get("cancel_requested"):
            return ExecutionStatus.CANCELLED
        if exit_code is None:
            return ExecutionStatus.FAILED
        return ExecutionStatus.COMPLETED if exit_code == 0 else ExecutionStatus.FAILED

    @staticmethod
    def _set(
        job: SchedulerJob,
        state: ExecutionStatus,
        *,
        raw_state: str | None = None,
        exit_code: int | None = None,
    ) -> ExecutionStatus:
        job.state = state
        if raw_state is not None:
            job.raw_state = raw_state
        if exit_code is not None:
            job.exit_code = exit_code
        return state

    def _new_job(self, job_id: str, work_dir: Path, **metadata: Any) -> SchedulerJob:
        return SchedulerJob(
            job_id=job_id,
            backend=self.name,
            state=ExecutionStatus.SUBMITTED,
            work_dir=str(work_dir),
            submitted_at=utc_now(),
            metadata=dict(metadata),
        )


def scheduler_job_to_dict(job: SchedulerJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "backend": job.backend,
        "state": job.state.value,
        "work_dir": job.work_dir,
        "submitted_at": job.submitted_at,
        "exit_code": job.exit_code,
        "raw_state": job.raw_state,
        "metadata": dict(job.metadata),
    }


def scheduler_job_from_dict(data: Mapping[str, Any]) -> SchedulerJob:
    return SchedulerJob(
        job_id=str(data["job_id"]),
        backend=str(data["backend"]),
        state=ExecutionStatus(data.get("state", ExecutionStatus.UNKNOWN.value)),
        work_dir=str(data["work_dir"]),
        submitted_at=data.get("submitted_at"),
        exit_code=data.get("exit_code"),
        raw_state=data.get("raw_state"),
        metadata=dict(data.get("metadata", {})),
    )


__all__ = [
    "EXIT_CODE_FILE",
    "SubmissionRejected",
    "JOB_STDERR",
    "JOB_STDOUT",
    "SCHEDULER_STDERR",
    "SCHEDULER_STDOUT",
    "TERMINAL_STATES",
    "ExecutionRequest",
    "SchedulerBackend",
    "ensure_fresh_work_dir",
    "read_exit_code",
    "render_job_body",
    "run_scheduler_command",
    "sanitize_job_name",
    "scheduler_job_from_dict",
    "scheduler_job_to_dict",
    "utc_now",
]
