# Ported from PHAL (hotspot_al.backends.schedulers:_BatchSchedulerBackend) at 96263b6; MIT License, see NOTICE.
"""Shared batch-scheduler logic: script rendering, submission, cancellation, job-id parsing.

Differences from PHAL (findings F-26, ledger SCH-002): resources are typed and rendered per scheduler
by the subclass; every CLI call has a timeout; a failing or silent submit command raises
:class:`SchedulerError` with its stderr; the job script records the command's exit code in the work
directory so :meth:`reconcile` can decide once the scheduler has forgotten the job.
"""

from __future__ import annotations

import logging
import shutil
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from gmd_active_learning.adapters.registry import RuntimeStatus
from gmd_active_learning.adapters.schedulers.base import (
    SCHEDULER_STDERR,
    SCHEDULER_STDOUT,
    SchedulerBackend,
    ensure_fresh_work_dir,
    render_job_body,
    run_scheduler_command,
    SubmissionRejected,
    sanitize_job_name,
)
from gmd_active_learning.adapters.schedulers.config import SchedulerResources
from gmd_active_learning.contracts.labels import ExecutionStatus, SchedulerJob
from gmd_active_learning.core.exceptions import SchedulerError

LOGGER = logging.getLogger(__name__)


class BatchScheduler(SchedulerBackend):
    """Base for queueing systems driven by submit/query/cancel command-line tools."""

    directive_prefix: ClassVar[str]
    foreign_prefixes: ClassVar[tuple[str, ...]]
    script_name: ClassVar[str]

    @property
    @abstractmethod
    def submit_command(self) -> str: ...

    @property
    @abstractmethod
    def cancel_command(self) -> str: ...

    @abstractmethod
    def resource_directives(self, resources: SchedulerResources, *, job_name: str, work_dir: Path) -> list[str]:
        """Structured resources rendered as directive lines for this scheduler."""

    @abstractmethod
    def parse_job_id(self, stdout: str) -> str | None: ...

    def check_runtime(self) -> RuntimeStatus:
        resolved = shutil.which(self.submit_command)
        return RuntimeStatus(
            self.name,
            self.role,
            resolved is not None,
            resolved or f"scheduler command not found: {self.submit_command}",
        )

    def render_script(
        self,
        work_dir: Path,
        command: Sequence[str],
        resources: SchedulerResources,
        *,
        environment: Mapping[str, str] | None = None,
        job_name: str | None = None,
    ) -> str:
        name = sanitize_job_name(job_name or resources.job_name)
        lines = [f"#!{self.config.shell}"]
        lines.extend(self.resource_directives(resources, job_name=name, work_dir=work_dir))
        lines.extend(self._extra_directives(resources.extra_directives))
        lines.append("set -u")
        lines.extend(render_job_body(work_dir, command, environment))
        lines.append("")
        return "\n".join(lines)

    def _extra_directives(self, directives: Sequence[str]) -> list[str]:
        rendered: list[str] = []
        for raw in directives:
            line = str(raw).strip()
            if not line:
                continue
            for prefix in self.foreign_prefixes:
                if line.startswith(prefix):
                    raise SchedulerError(
                        f"directive {line!r} belongs to another scheduler; {self.name} needs {self.directive_prefix}"
                    )
            rendered.append(line if line.startswith(self.directive_prefix) else f"{self.directive_prefix} {line}")
        return rendered

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
        script = work_dir / self.script_name
        script.write_text(
            self.render_script(work_dir, command, resolved, environment=environment, job_name=job_name),
            encoding="utf-8",
        )
        if shutil.which(self.submit_command) is None:
            raise SubmissionRejected(f"scheduler command not found: {self.submit_command!r}")
        result = run_scheduler_command([self.submit_command, script.name], timeout=self.timeout, cwd=work_dir)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip() or "no output"
            raise SubmissionRejected(f"{self.submit_command} failed with exit status {result.returncode}: {detail}")
        job_id = self.parse_job_id(result.stdout)
        if not job_id:
            detail = result.stderr.strip() or "no output"
            raise SchedulerError(f"{self.submit_command} did not report a job id ({detail})")
        LOGGER.info("%s job %s submitted from %s", self.name, job_id, work_dir)
        return self._new_job(job_id, work_dir, script=str(script))

    def cancel(self, job: SchedulerJob) -> ExecutionStatus:
        state = self.poll(job)
        if state.terminal:
            return state
        job.metadata["cancel_requested"] = True
        result = run_scheduler_command([self.cancel_command, job.job_id], timeout=self.timeout)
        if result.returncode != 0:
            state = self.poll(job)
            if state.terminal:
                return state
            detail = (result.stderr or result.stdout).strip() or "no output"
            raise SchedulerError(f"{self.cancel_command} {job.job_id} failed ({result.returncode}): {detail}")
        return self._set(job, ExecutionStatus.CANCELLED, raw_state="CANCEL_REQUESTED")

    def _query(self, argv: Sequence[str], job: SchedulerJob) -> str | None:
        """Run a status query; failures are recorded on the job and yield None (state unknown)."""

        try:
            result = run_scheduler_command(argv, timeout=self.timeout)
        except SchedulerError as exc:
            job.metadata["poll_error"] = str(exc)
            LOGGER.warning("%s", exc)
            return None
        if result.returncode != 0:
            job.metadata["poll_error"] = f"{argv[0]} exit {result.returncode}: {result.stderr.strip()}"
            return None
        return result.stdout

    @staticmethod
    def _log_paths(work_dir: Path) -> tuple[str, str]:
        return str(work_dir / SCHEDULER_STDOUT), str(work_dir / SCHEDULER_STDERR)


__all__ = ["BatchScheduler"]
