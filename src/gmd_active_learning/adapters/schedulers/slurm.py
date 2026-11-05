# Ported from PHAL (hotspot_al.backends.schedulers:SlurmSchedulerBackend, _slurm_state) at 96263b6;
# MIT License, see NOTICE.
"""Slurm scheduler: ``sbatch`` / ``squeue`` with ``sacct`` fallback / ``scancel``."""

from __future__ import annotations

import re
from pathlib import Path

from gmd_active_learning.adapters.registry import BackendRole
from gmd_active_learning.adapters.schedulers.base import read_exit_code
from gmd_active_learning.adapters.schedulers.batch import BatchScheduler
from gmd_active_learning.adapters.schedulers.config import SchedulerResources, format_hms
from gmd_active_learning.contracts.labels import ExecutionStatus, SchedulerJob

# Explicit mapping of every documented Slurm job state (squeue %T / sacct State). TIMEOUT and
# DEADLINE are FAILED with the raw state kept, so callers can classify a walltime failure.
SLURM_STATES: dict[str, ExecutionStatus] = {
    "PENDING": ExecutionStatus.QUEUED,
    "CONFIGURING": ExecutionStatus.QUEUED,
    "REQUEUED": ExecutionStatus.QUEUED,
    "REQUEUE_FED": ExecutionStatus.QUEUED,
    "REQUEUE_HOLD": ExecutionStatus.QUEUED,
    "RESV_DEL_HOLD": ExecutionStatus.QUEUED,
    "RUNNING": ExecutionStatus.RUNNING,
    "COMPLETING": ExecutionStatus.RUNNING,
    "SIGNALING": ExecutionStatus.RUNNING,
    "STAGE_OUT": ExecutionStatus.RUNNING,
    "RESIZING": ExecutionStatus.RUNNING,
    "SUSPENDED": ExecutionStatus.RUNNING,
    "STOPPED": ExecutionStatus.RUNNING,
    "COMPLETED": ExecutionStatus.COMPLETED,
    "CANCELLED": ExecutionStatus.CANCELLED,
    "FAILED": ExecutionStatus.FAILED,
    "OUT_OF_MEMORY": ExecutionStatus.FAILED,
    "NODE_FAIL": ExecutionStatus.FAILED,
    "BOOT_FAIL": ExecutionStatus.FAILED,
    "PREEMPTED": ExecutionStatus.FAILED,
    "SPECIAL_EXIT": ExecutionStatus.FAILED,
    "TIMEOUT": ExecutionStatus.FAILED,
    "DEADLINE": ExecutionStatus.FAILED,
    "REVOKED": ExecutionStatus.UNKNOWN,  # federation: the job runs on a sibling cluster
}
WALLTIME_STATES = frozenset({"TIMEOUT", "DEADLINE"})
_JOB_ID = re.compile(r"(\d+(?:_\d+)?)(?:;\S+)?\s*$")


def normalize_slurm_state(raw: str | None) -> str:
    if not raw:
        return ""
    return raw.strip().split()[0].rstrip("+").upper()


def slurm_state(raw: str | None) -> ExecutionStatus:
    return SLURM_STATES.get(normalize_slurm_state(raw), ExecutionStatus.UNKNOWN)


def parse_slurm_exit_code(raw: str | None) -> int | None:
    """``"rc:signal"`` from sacct -> rc, or 128 + signal when killed by a signal."""

    if not raw or ":" not in raw:
        return None
    code, _, sig = raw.strip().partition(":")
    try:
        code_value, sig_value = int(code), int(sig)
    except ValueError:
        return None
    return 128 + sig_value if code_value == 0 and sig_value else code_value


class SlurmScheduler(BatchScheduler):
    name = "slurm"
    role = BackendRole.SCHEDULER
    directive_prefix = "#SBATCH"
    foreign_prefixes = ("#PBS",)
    script_name = "gmd_job.sbatch"

    @property
    def submit_command(self) -> str:
        return self.config.slurm.sbatch

    @property
    def cancel_command(self) -> str:
        return self.config.slurm.scancel

    def resource_directives(self, resources: SchedulerResources, *, job_name: str, work_dir: Path) -> list[str]:
        stdout, stderr = self._log_paths(work_dir)
        lines = [
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --chdir={work_dir}",
            f"#SBATCH --output={stdout}",
            f"#SBATCH --error={stderr}",
        ]
        if resources.nodes is not None:
            lines.append(f"#SBATCH --nodes={resources.nodes}")
        if resources.ntasks is not None:
            lines.append(f"#SBATCH --ntasks={resources.ntasks}")
        if resources.ntasks_per_node is not None:
            lines.append(f"#SBATCH --ntasks-per-node={resources.ntasks_per_node}")
        if resources.cpus_per_task is not None:
            lines.append(f"#SBATCH --cpus-per-task={resources.cpus_per_task}")
        if resources.walltime_seconds is not None:
            lines.append(f"#SBATCH --time={format_hms(resources.walltime_seconds)}")
        memory = resources.memory_parts()
        if memory is not None:
            lines.append(f"#SBATCH --mem={memory[0]}{memory[1]}")
        partition = resources.partition or resources.queue
        if partition:
            lines.append(f"#SBATCH --partition={partition}")
        if resources.account:
            lines.append(f"#SBATCH --account={resources.account}")
        return lines

    def parse_job_id(self, stdout: str) -> str | None:
        for line in reversed(stdout.strip().splitlines()):
            match = _JOB_ID.search(line.strip())
            if match:
                return match.group(1)
        return None

    def poll(self, job: SchedulerJob) -> ExecutionStatus:
        output = self._query([self.config.slurm.squeue, "-h", "-j", job.job_id, "-o", "%T"], job)
        raw = normalize_slurm_state(output.splitlines()[0]) if output and output.strip() else ""
        exit_code: int | None = None
        if not raw and self.config.slurm.sacct_fallback:
            output = self._query(
                [self.config.slurm.sacct, "-n", "-X", "-P", "-j", job.job_id, "--format=State,ExitCode"], job
            )
            first = next((line for line in (output or "").splitlines() if line.strip()), "")
            if first:
                state_field, _, exit_field = first.partition("|")
                raw = normalize_slurm_state(state_field)
                exit_code = parse_slurm_exit_code(exit_field)
        if not raw:
            return self._set(job, ExecutionStatus.UNKNOWN, raw_state="NOT_FOUND")
        state = slurm_state(raw)
        if state.terminal:
            exit_code = exit_code if exit_code is not None else read_exit_code(job.work_dir)
            if state is not ExecutionStatus.CANCELLED and job.metadata.get("cancel_requested"):
                state = ExecutionStatus.CANCELLED
            if raw in WALLTIME_STATES:
                job.metadata["walltime_exceeded"] = True
        return self._set(job, state, raw_state=raw, exit_code=exit_code)


__all__ = ["SLURM_STATES", "SlurmScheduler", "parse_slurm_exit_code", "slurm_state"]
