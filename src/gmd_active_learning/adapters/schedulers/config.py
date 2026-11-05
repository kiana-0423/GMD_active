"""Typed configuration for scheduler adapters (``scheduler`` section) and job resources.

Resources are structured fields, rendered per scheduler; raw directive text is only accepted through
``extra_directives`` and must carry the right prefix for the target scheduler (PHAL reused
``cp2k.slurm_directives`` for PBS jobs, findings F-26).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_WALLTIME = re.compile(r"^(?:(?:(?P<d>\d+)-)?(?P<h>\d+):)?(?P<m>\d+):(?P<s>\d+)$")
_MEMORY = re.compile(r"^(?P<value>\d+)(?P<unit>[KMGT]?)(?:B|b)?$", re.IGNORECASE)


def walltime_seconds(value: str | int | float | None) -> int | None:
    """Parse ``[[D-]HH:]MM:SS`` or plain seconds into integer seconds (None stays None)."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("walltime must be a duration, not a boolean")
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError("walltime must be positive")
        return int(round(value))
    text = str(value).strip()
    if text.isdigit():
        return walltime_seconds(int(text))
    match = _WALLTIME.match(text)
    if match is None:
        raise ValueError(f"walltime {value!r} is not '[[D-]HH:]MM:SS' or seconds")
    days = int(match.group("d") or 0)
    hours = int(match.group("h") or 0)
    seconds = ((days * 24 + hours) * 60 + int(match.group("m"))) * 60 + int(match.group("s"))
    if seconds <= 0:
        raise ValueError("walltime must be positive")
    return seconds


def format_hms(seconds: int) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class SchedulerResources(BaseModel):
    """Scheduler-neutral resource request for one job.

    ``ntasks`` is the total number of MPI ranks, ``cpus_per_task`` the threads per rank. ``memory``
    is per node (``4G``, ``500M``). ``partition`` (Slurm) and ``queue`` (PBS) are accepted by both
    renderers. ``extra_directives`` are appended verbatim after the structured ones; a line may omit
    the ``#SBATCH``/``#PBS`` prefix but must not carry the other scheduler's prefix.
    """

    model_config = ConfigDict(extra="forbid")

    nodes: int | None = Field(default=None, ge=1)
    ntasks: int | None = Field(default=None, ge=1)
    ntasks_per_node: int | None = Field(default=None, ge=1)
    cpus_per_task: int | None = Field(default=None, ge=1)
    walltime: str | int | None = None
    memory: str | None = None
    partition: str | None = None
    queue: str | None = None
    account: str | None = None
    job_name: str | None = None
    extra_directives: list[str] = Field(default_factory=list)

    @field_validator("walltime")
    @classmethod
    def _check_walltime(cls, value: str | int | None) -> str | int | None:
        walltime_seconds(value)
        return value

    @field_validator("memory")
    @classmethod
    def _check_memory(cls, value: str | None) -> str | None:
        if value is not None and _MEMORY.match(value.strip()) is None:
            raise ValueError(f"memory {value!r} must look like '4G', '500M' or '1024'")
        return value

    @property
    def walltime_seconds(self) -> int | None:
        return walltime_seconds(self.walltime)

    def memory_parts(self) -> tuple[int, str] | None:
        """``(value, unit)`` with unit in '', K, M, G, T (upper case)."""

        if self.memory is None:
            return None
        match = _MEMORY.match(self.memory.strip())
        assert match is not None  # validated
        return int(match.group("value")), match.group("unit").upper()

    def merged(self, override: SchedulerResources | None) -> SchedulerResources:
        """Fields set in ``override`` win; extra directives are concatenated."""

        if override is None:
            return self
        data = self.model_dump()
        for key, value in override.model_dump(exclude_unset=True).items():
            if key == "extra_directives":
                data[key] = [*data[key], *value]
            else:
                data[key] = value
        return SchedulerResources.model_validate(data)


class LocalSchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shell: str = "/bin/sh"
    cancel_grace_seconds: float = Field(default=10.0, ge=0.0)
    enforce_walltime: bool = True


class SlurmSchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sbatch: str = "sbatch"
    squeue: str = "squeue"
    sacct: str = "sacct"
    scancel: str = "scancel"
    sacct_fallback: bool = True


class PBSSchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qsub: str = "qsub"
    qstat: str = "qstat"
    qdel: str = "qdel"
    history_query_enabled: bool = True
    flavor: Literal["pbspro", "torque"] = "pbspro"


class SchedulerConfig(BaseModel):
    """``scheduler`` section: per-engine settings. The engine itself is chosen by ``backends.scheduler``."""

    model_config = ConfigDict(extra="forbid")

    command_timeout_seconds: float = Field(default=60.0, gt=0.0)
    shell: str = "/bin/bash"
    local: LocalSchedulerConfig = Field(default_factory=LocalSchedulerConfig)
    slurm: SlurmSchedulerConfig = Field(default_factory=SlurmSchedulerConfig)
    pbs: PBSSchedulerConfig = Field(default_factory=PBSSchedulerConfig)
    default_resources: SchedulerResources = Field(default_factory=SchedulerResources)

    @model_validator(mode="after")
    def _check_shell(self) -> SchedulerConfig:
        if not self.shell.strip():
            raise ValueError("scheduler.shell must not be empty")
        return self


__all__ = [
    "LocalSchedulerConfig",
    "PBSSchedulerConfig",
    "SchedulerConfig",
    "SchedulerResources",
    "SlurmSchedulerConfig",
    "format_hms",
    "walltime_seconds",
]
