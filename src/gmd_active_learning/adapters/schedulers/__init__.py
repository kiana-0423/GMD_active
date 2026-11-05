"""Scheduler adapters (local, Slurm, PBS) behind one protocol.

``submit(work_dir, command, resources) -> SchedulerJob``, ``poll(job) -> ExecutionStatus``,
``cancel(job)``, ``reconcile(job)``. Construct them through
``gmd_active_learning.adapters.registry.create_backend(config, "scheduler")``.
"""

from gmd_active_learning.adapters.schedulers.base import ExecutionRequest, SchedulerBackend, read_exit_code
from gmd_active_learning.adapters.schedulers.config import SchedulerConfig, SchedulerResources

__all__ = ["ExecutionRequest", "SchedulerBackend", "SchedulerConfig", "SchedulerResources", "read_exit_code"]
