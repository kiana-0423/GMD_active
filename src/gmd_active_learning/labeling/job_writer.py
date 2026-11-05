"""Write (but do not submit) a batch script for a prepared labeling directory.

Rendering goes through the scheduler adapters, so Slurm and PBS scripts carry their own typed resource
directives (PHAL reused Slurm directive text for PBS, findings F-26) and record the exit code in
``.gmd_exit_code`` like submitted jobs. Submission itself is the job of
:class:`gmd_active_learning.labeling.service.LabelingService`.
"""

from __future__ import annotations

import logging
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gmd_active_learning.adapters.schedulers.base import render_job_body
from gmd_active_learning.adapters.schedulers.config import SchedulerConfig, SchedulerResources
from gmd_active_learning.core.exceptions import ConfigurationError

LOGGER = logging.getLogger(__name__)

SCRIPT_NAMES = {"slurm": "submit.slurm", "pbs": "submit.pbs", "local": "submit.sh"}


def write_job_script(
    job_dir: str | Path,
    scheduler: str,
    command: str | Sequence[str],
    *,
    resources: SchedulerResources | Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
    job_name: str = "gmd_al",
    scheduler_config: SchedulerConfig | None = None,
) -> Path:
    """Render ``command`` for ``scheduler`` (slurm, pbs or local) into ``job_dir``; returns the path."""

    from gmd_active_learning.adapters.schedulers.pbs import PBSScheduler
    from gmd_active_learning.adapters.schedulers.slurm import SlurmScheduler

    name = str(scheduler).strip().lower()
    if name not in SCRIPT_NAMES:
        raise ConfigurationError(f"unknown scheduler {scheduler!r}; expected one of {sorted(SCRIPT_NAMES)}")
    job_dir = Path(job_dir).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    argv = shlex.split(command) if isinstance(command, str) else [str(part) for part in command]
    config = scheduler_config or SchedulerConfig()
    if resources is not None and not isinstance(resources, SchedulerResources):
        resources = SchedulerResources.model_validate(dict(resources))
    resolved = config.default_resources.merged(resources)  # type: ignore[arg-type]
    if name == "local":
        content = "\n".join([f"#!{config.shell}", "set -u", *render_job_body(job_dir, argv, environment), ""])
    else:
        backend = SlurmScheduler(config) if name == "slurm" else PBSScheduler(config)
        content = backend.render_script(job_dir, argv, resolved, environment=environment, job_name=job_name)
    path = job_dir / SCRIPT_NAMES[name]
    path.write_text(content, encoding="utf-8")
    LOGGER.debug("wrote %s job script %s", name, path)
    return path


__all__ = ["SCRIPT_NAMES", "write_job_script"]
