# Ported from PHAL (hotspot_al.backends.base, hotspot_al.backends.schedulers.LocalSchedulerBackend)
# at 96263b6; MIT License, see NOTICE.
"""Minimal scheduler-neutral execution types used by the SCF tooling.

PHAL's SCF tooling (Pulay dataset generation, guess/controller benchmarks,
strategy grid) submits CP2K runs through a ``SchedulerBackend`` with
``submit``/``poll``/``cancel``. This module keeps exactly that interface so the
tooling can run, with one real implementation (:class:`LocalSchedulerBackend`,
a local subprocess). Batch schedulers are resolved by ``module:Class`` from the
GMD_active scheduler adapters (owned by the labeling area) via
:func:`resolve_scheduler`; nothing here pretends a cluster is available.

Status values map one-to-one onto
:class:`gmd_active_learning.contracts.ExecutionStatus` (see :func:`to_execution_status`).
"""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from gmd_active_learning.core.exceptions import CapabilityUnavailableError, ConfigurationError

LOGGER = logging.getLogger(__name__)


class BackendRole(str, Enum):
    DFT = "dft"
    SCHEDULER = "scheduler"


class JobState(str, Enum):
    """Portable state vocabulary used by scheduler adapters."""

    PREPARED = "prepared"
    SUBMITTED = "submitted"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    """Result of a backend runtime health check."""

    backend: str
    role: BackendRole
    available: bool
    detail: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """Scheduler-neutral request to execute one external command."""

    command: tuple[str, ...]
    work_dir: Path
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_command(
        cls,
        command: Sequence[str | Path],
        *,
        work_dir: str | Path,
        stdout_path: str | Path | None = None,
        stderr_path: str | Path | None = None,
        environment: Mapping[str, str] | None = None,
        resources: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ExecutionRequest":
        return cls(
            command=tuple(str(part) for part in command),
            work_dir=Path(work_dir),
            stdout_path=None if stdout_path is None else Path(stdout_path),
            stderr_path=None if stderr_path is None else Path(stderr_path),
            environment=dict(environment or {}),
            resources=dict(resources or {}),
            metadata=dict(metadata or {}),
        )


@dataclass(slots=True)
class BackendJob:
    """Opaque job handle shared between the tooling and scheduler backends."""

    backend: str
    state: JobState
    external_id: str | None = None
    native_handle: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SchedulerBackend(ABC):
    """The submit/poll/cancel interface the SCF tooling needs."""

    backend_name: str = "scheduler"
    role = BackendRole.SCHEDULER

    @abstractmethod
    def submit(self, request: ExecutionRequest) -> BackendJob:
        raise NotImplementedError

    @abstractmethod
    def poll(self, job: BackendJob) -> JobState:
        raise NotImplementedError

    def cancel(self, job: BackendJob) -> None:  # pragma: no cover - optional
        job.state = JobState.CANCELLED


class LocalSchedulerBackend(SchedulerBackend):
    """Run each request as a local subprocess (real execution, no batch system)."""

    backend_name = "local"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LocalSchedulerBackend":
        return cls(config)

    def check_runtime(self) -> RuntimeStatus:
        return RuntimeStatus(self.backend_name, self.role, True, "local subprocess runtime available")

    def submit(self, request: ExecutionRequest) -> BackendJob:
        request.work_dir.mkdir(parents=True, exist_ok=True)
        stdout_handle = None if request.stdout_path is None else _open_output(request.work_dir, request.stdout_path)
        stderr_handle = None if request.stderr_path is None else _open_output(request.work_dir, request.stderr_path)
        environment = {**os.environ, **request.environment}
        try:
            process = subprocess.Popen(
                list(request.command),
                cwd=request.work_dir,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                env=environment,
            )
        except OSError as exc:
            LOGGER.error("could not start %s: %s", request.command[:1], exc)
            return BackendJob(
                backend=self.backend_name,
                state=JobState.FAILED,
                metadata={**dict(request.metadata), "error": str(exc)},
            )
        finally:
            if stdout_handle is not None:
                stdout_handle.close()
            if stderr_handle is not None:
                stderr_handle.close()
        return BackendJob(
            backend=self.backend_name,
            state=JobState.SUBMITTED,
            external_id=str(process.pid),
            native_handle=process,
            metadata=dict(request.metadata),
        )

    def poll(self, job: BackendJob) -> JobState:
        process = job.native_handle
        if not isinstance(process, subprocess.Popen):
            return job.state
        returncode = process.poll()
        if returncode is None:
            job.state = JobState.RUNNING
        else:
            job.state = JobState.COMPLETED if returncode == 0 else JobState.FAILED
            job.metadata["returncode"] = returncode
        return job.state

    def cancel(self, job: BackendJob) -> None:
        process = job.native_handle
        if isinstance(process, subprocess.Popen) and process.poll() is None:
            process.terminate()
        job.state = JobState.CANCELLED


class GmdSchedulerBridge(SchedulerBackend):
    """Drive a ``gmd_active_learning.adapters.schedulers`` backend (local/Slurm/PBS).

    Those take ``submit(work_dir, command, ...)`` and return
    :class:`~gmd_active_learning.contracts.SchedulerJob` /
    :class:`~gmd_active_learning.contracts.ExecutionStatus`; the SCF tooling
    speaks PHAL's ``submit(request) -> BackendJob`` / ``poll -> JobState``.
    stdout/stderr redirection, which the GMD request does not carry, is done
    by a ``/bin/sh -c 'exec ... > out 2> err'`` wrapper.
    """

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.backend_name = f"gmd:{getattr(backend, 'backend_name', type(backend).__name__)}"

    def submit(self, request: ExecutionRequest) -> BackendJob:
        import shlex

        command = list(request.command)
        if request.stdout_path is not None or request.stderr_path is not None:
            redirect = ""
            if request.stdout_path is not None:
                redirect += f" > {shlex.quote(str(request.stdout_path))}"
            if request.stderr_path is not None:
                redirect += f" 2> {shlex.quote(str(request.stderr_path))}"
            command = ["/bin/sh", "-c", f"exec {shlex.join(command)}{redirect}"]
        job = self.backend.submit(
            request.work_dir,
            command,
            dict(request.resources) or None,
            environment=dict(request.environment),
            job_name=str(request.metadata.get("label") or request.metadata.get("system") or "") or None,
        )
        return BackendJob(
            backend=self.backend_name,
            state=JobState.SUBMITTED,
            external_id=job.job_id,
            native_handle=job,
            metadata=dict(request.metadata),
        )

    def poll(self, job: BackendJob) -> JobState:
        status = self.backend.poll(job.native_handle)
        job.state = from_execution_status(status)
        exit_code = getattr(job.native_handle, "exit_code", None)
        if exit_code is not None:
            job.metadata["returncode"] = exit_code
        return job.state

    def cancel(self, job: BackendJob) -> None:
        self.backend.cancel(job.native_handle)
        job.state = JobState.CANCELLED


_GMD_SCHEDULERS = {
    "slurm": "gmd_active_learning.adapters.schedulers.slurm:SlurmScheduler",
    "pbs": "gmd_active_learning.adapters.schedulers.pbs:PBSScheduler",
    "gmd-local": "gmd_active_learning.adapters.schedulers.local:LocalScheduler",
}


def resolve_scheduler(name_or_path: str, config: Mapping[str, Any] | None = None) -> SchedulerBackend:
    """``local``, ``slurm``, ``pbs``, ``gmd-local`` or a ``module:Class`` path.

    ``slurm``/``pbs``/``gmd-local`` are the GMD_active scheduler adapters,
    driven through :class:`GmdSchedulerBridge` and configured from the
    ``scheduler`` section of ``config``. An unknown name raises instead of
    silently running locally.
    """

    if name_or_path == "local":
        return LocalSchedulerBackend(config)
    path = _GMD_SCHEDULERS.get(name_or_path, name_or_path)
    if ":" not in path:
        raise ConfigurationError(
            f"unknown scheduler {name_or_path!r}; use local, slurm, pbs, gmd-local or a 'module:Class' path"
        )
    module_name, _, class_name = path.partition(":")
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise CapabilityUnavailableError(f"scheduler {name_or_path!r} is not importable: {exc}") from exc
    factory = getattr(cls, "from_config", None)
    backend = factory(dict(config or {})) if callable(factory) else cls(dict(config or {}))
    if isinstance(backend, SchedulerBackend):
        return backend
    if hasattr(backend, "submit_request"):  # a gmd_active_learning.adapters.schedulers backend
        return GmdSchedulerBridge(backend)
    raise ConfigurationError(f"{path} is neither an SCF SchedulerBackend nor a GMD scheduler adapter")


def to_execution_status(state: JobState) -> Any:
    """Map a :class:`JobState` onto the contracts' ``ExecutionStatus``."""

    from gmd_active_learning.contracts.labels import ExecutionStatus

    mapping = {
        JobState.PREPARED: ExecutionStatus.PREPARED,
        JobState.SUBMITTED: ExecutionStatus.SUBMITTED,
        JobState.PENDING: ExecutionStatus.QUEUED,
        JobState.RUNNING: ExecutionStatus.RUNNING,
        JobState.COMPLETED: ExecutionStatus.COMPLETED,
        JobState.FAILED: ExecutionStatus.FAILED,
        JobState.CANCELLED: ExecutionStatus.CANCELLED,
        JobState.TIMEOUT: ExecutionStatus.FAILED,
        JobState.UNKNOWN: ExecutionStatus.UNKNOWN,
    }
    return mapping[state]


def from_execution_status(status: Any) -> JobState:
    """Map the contracts' ``ExecutionStatus`` back onto :class:`JobState`."""

    value = getattr(status, "value", status)
    return {
        "prepared": JobState.PREPARED,
        "submitted": JobState.SUBMITTED,
        "queued": JobState.PENDING,
        "running": JobState.RUNNING,
        "completed": JobState.COMPLETED,
        "failed": JobState.FAILED,
        "cancelled": JobState.CANCELLED,
    }.get(str(value), JobState.UNKNOWN)


def _open_output(work_dir: Path, path: Path):
    resolved = path if path.is_absolute() else work_dir / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved.open("w", encoding="utf-8")


__all__ = [
    "BackendJob",
    "BackendRole",
    "ExecutionRequest",
    "GmdSchedulerBridge",
    "JobState",
    "LocalSchedulerBackend",
    "RuntimeStatus",
    "SchedulerBackend",
    "from_execution_status",
    "resolve_scheduler",
    "to_execution_status",
]
