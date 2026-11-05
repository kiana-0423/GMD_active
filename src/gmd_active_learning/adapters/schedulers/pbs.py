# Ported from PHAL (hotspot_al.backends.schedulers:PBSSchedulerBackend, _pbs_state) at 96263b6;
# MIT License, see NOTICE.
"""PBS Professional / Torque scheduler: ``qsub`` / ``qstat -f`` with ``qstat -x -f`` history / ``qdel``.

Fixes relative to PHAL (F-26): Torque's ``C`` (completed) state is no longer COMPLETED regardless of
the exit status; resources are rendered as PBS ``-l`` requests instead of reusing Slurm directive
text; ``qstat -f`` continuation lines are joined; ``history_query_enabled`` is configurable through
``scheduler.pbs``.
"""

from __future__ import annotations

import math
from pathlib import Path

from gmd_active_learning.adapters.registry import BackendRole
from gmd_active_learning.adapters.schedulers.base import read_exit_code
from gmd_active_learning.adapters.schedulers.batch import BatchScheduler
from gmd_active_learning.adapters.schedulers.config import SchedulerResources, format_hms
from gmd_active_learning.contracts.labels import ExecutionStatus, SchedulerJob

PBS_ACTIVE_STATES: dict[str, ExecutionStatus] = {
    "Q": ExecutionStatus.QUEUED,
    "H": ExecutionStatus.QUEUED,
    "W": ExecutionStatus.QUEUED,
    "T": ExecutionStatus.QUEUED,  # being moved to a new location
    "M": ExecutionStatus.QUEUED,  # moved to another server
    "S": ExecutionStatus.RUNNING,  # suspended while running
    "U": ExecutionStatus.RUNNING,  # suspended (workstation busy)
    "R": ExecutionStatus.RUNNING,
    "E": ExecutionStatus.RUNNING,  # exiting after having run
    "B": ExecutionStatus.RUNNING,  # array job has begun
}
PBS_FINISHED_STATES = frozenset({"F", "C", "X"})
# Walltime kills: PBS Pro JOB_EXEC_KILL_WALLTIME (-29) and Torque's -11.
PBS_WALLTIME_EXIT_CODES = frozenset({-29, -11})


def parse_qstat_full(output: str) -> dict[str, str]:
    """``key = value`` pairs of ``qstat -f`` (keys lower-cased, continuation lines joined)."""

    fields: dict[str, str] = {}
    last_key: str | None = None
    for line in output.splitlines():
        if not line.strip() or line.startswith("Job Id:"):
            continue
        if " = " in line:
            key, value = line.split(" = ", 1)
            last_key = key.strip().lower()
            fields[last_key] = value.strip()
        elif last_key is not None and line[:1] in {"\t", " "}:
            fields[last_key] += line.strip()
    return fields


def pbs_state(fields: dict[str, str]) -> tuple[ExecutionStatus, str, int | None]:
    """(state, raw job_state, exit status) from parsed ``qstat -f`` fields."""

    raw = fields.get("job_state", "").strip().upper()
    if raw in PBS_ACTIVE_STATES:
        return PBS_ACTIVE_STATES[raw], raw, None
    if raw in PBS_FINISHED_STATES:
        exit_text = fields.get("exit_status")
        if exit_text is None:
            # PHAL parity: PBS Pro "F" without an exit status is a failure (e.g. deleted before running).
            # Torque "C" / subjob "X" without one cannot be judged here: reconcile with the work dir.
            return (ExecutionStatus.FAILED if raw == "F" else ExecutionStatus.UNKNOWN), raw, None
        try:
            exit_status = int(exit_text)
        except ValueError:
            return ExecutionStatus.UNKNOWN, raw, None
        return (ExecutionStatus.COMPLETED if exit_status == 0 else ExecutionStatus.FAILED), raw, exit_status
    return ExecutionStatus.UNKNOWN, raw, None


class PBSScheduler(BatchScheduler):
    name = "pbs"
    role = BackendRole.SCHEDULER
    directive_prefix = "#PBS"
    foreign_prefixes = ("#SBATCH",)
    script_name = "gmd_job.pbs"

    @property
    def submit_command(self) -> str:
        return self.config.pbs.qsub

    @property
    def cancel_command(self) -> str:
        return self.config.pbs.qdel

    def resource_directives(self, resources: SchedulerResources, *, job_name: str, work_dir: Path) -> list[str]:
        stdout, stderr = self._log_paths(work_dir)
        lines = [f"#PBS -N {job_name[:15] if self.config.pbs.flavor == 'torque' else job_name}"]
        lines += [f"#PBS -o {stdout}", f"#PBS -e {stderr}"]
        nodes = resources.nodes or 1
        if resources.ntasks_per_node is not None:
            per_node = resources.ntasks_per_node
        elif resources.ntasks is not None:
            per_node = math.ceil(resources.ntasks / nodes)
        else:
            per_node = None
        threads = resources.cpus_per_task or 1
        memory = resources.memory_parts()
        memory_text = None if memory is None else f"{memory[0]}{(memory[1] or 'b').lower()}{'b' if memory[1] else ''}"
        if self.config.pbs.flavor == "pbspro":
            chunk = [f"select={nodes}"]
            if per_node is not None or resources.cpus_per_task is not None:
                ranks = per_node or 1
                chunk += [f"ncpus={ranks * threads}", f"mpiprocs={ranks}"]
                if resources.cpus_per_task is not None:
                    chunk.append(f"ompthreads={threads}")
            if memory_text is not None:
                chunk.append(f"mem={memory_text}")
            if len(chunk) > 1 or resources.nodes is not None:
                lines.append(f"#PBS -l {':'.join(chunk)}")
        else:
            if resources.nodes is not None or per_node is not None:
                lines.append(f"#PBS -l nodes={nodes}:ppn={(per_node or 1) * threads}")
            if memory_text is not None:
                lines.append(f"#PBS -l mem={memory_text}")
        if resources.walltime_seconds is not None:
            lines.append(f"#PBS -l walltime={format_hms(resources.walltime_seconds)}")
        queue = resources.queue or resources.partition
        if queue:
            lines.append(f"#PBS -q {queue}")
        if resources.account:
            lines.append(f"#PBS -A {resources.account}")
        return lines

    def parse_job_id(self, stdout: str) -> str | None:
        tokens = stdout.strip().split()
        return tokens[-1] if tokens else None

    def poll(self, job: SchedulerJob) -> ExecutionStatus:
        qstat = self.config.pbs.qstat
        output = self._query([qstat, "-f", job.job_id], job) or ""
        if not output.strip() and self.config.pbs.history_query_enabled:
            # Finished jobs leave the live queue; ask the server history before declaring UNKNOWN.
            output = self._query([qstat, "-x", "-f", job.job_id], job) or ""
        if not output.strip():
            return self._set(job, ExecutionStatus.UNKNOWN, raw_state="NOT_FOUND")
        state, raw, exit_status = pbs_state(parse_qstat_full(output))
        if state.terminal:
            if exit_status is None:
                exit_status = read_exit_code(job.work_dir)
            if exit_status in PBS_WALLTIME_EXIT_CODES:
                job.metadata["walltime_exceeded"] = True
            if job.metadata.get("cancel_requested"):
                state = ExecutionStatus.CANCELLED
        return self._set(job, state, raw_state=raw or "NO_STATE", exit_code=exit_status)


__all__ = ["PBSScheduler", "parse_qstat_full", "pbs_state"]
