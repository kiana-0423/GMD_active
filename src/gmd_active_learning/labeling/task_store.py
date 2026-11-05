"""Atomic ``task.json`` persistence for :class:`~gmd_active_learning.contracts.LabelTask`.

PHAL's ``task.json`` held metadata only (no job id, no status) and the job table lived in memory, so
nothing could be resumed after a restart (findings F-24, ledger LAB-007/008). Here every state change
is written atomically, together with the scheduler job, attempts and failure reasons.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from gmd_active_learning.adapters.schedulers.base import scheduler_job_from_dict, scheduler_job_to_dict
from gmd_active_learning.contracts.labels import ExecutionStatus, LabelTask, ValidationStatus
from gmd_active_learning.core.exceptions import LabelingError
from gmd_active_learning.utils.atomic_io import write_json_atomic

TASK_FILE = "task.json"
LOCK_FILE = ".task.lock"
SCHEMA = "gmd_active_learning.label_task/1"


def task_to_dict(task: LabelTask) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "task_id": task.task_id,
        "region_id": task.region_id,
        "engine": task.engine,
        "settings_hash": task.settings_hash,
        "work_dir": task.work_dir,
        "attempt": task.attempt,
        "execution": task.execution.value,
        "validation": task.validation.value,
        "job": None if task.job is None else scheduler_job_to_dict(task.job),
        "failure_reasons": list(task.failure_reasons),
        "metadata": task.metadata,
    }


def task_from_dict(data: dict[str, Any]) -> LabelTask:
    if data.get("schema") != SCHEMA:
        raise LabelingError(f"unsupported task schema {data.get('schema')!r}")
    return LabelTask(
        task_id=data["task_id"],
        region_id=data["region_id"],
        engine=data["engine"],
        settings_hash=data["settings_hash"],
        work_dir=data["work_dir"],
        attempt=int(data.get("attempt", 0)),
        execution=ExecutionStatus(data["execution"]),
        validation=ValidationStatus(data["validation"]),
        job=None if data.get("job") is None else scheduler_job_from_dict(data["job"]),
        failure_reasons=list(data.get("failure_reasons", [])),
        metadata=dict(data.get("metadata", {})),
    )


def save_task(task: LabelTask) -> Path:
    return write_json_atomic(Path(task.work_dir) / TASK_FILE, task_to_dict(task))


def load_task(work_dir: str | Path) -> LabelTask:
    path = Path(work_dir) / TASK_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LabelingError(f"no labeling task at {path}") from exc
    except json.JSONDecodeError as exc:
        raise LabelingError(f"corrupt task file {path}: {exc}") from exc
    return task_from_dict(data)


@contextmanager
def task_lock(work_dir: str | Path) -> Iterator[None]:
    """Exclusive advisory lock so two controllers never act on the same task at once."""

    path = Path(work_dir)
    path.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path / LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


__all__ = ["LOCK_FILE", "SCHEMA", "TASK_FILE", "load_task", "save_task", "task_from_dict", "task_lock", "task_to_dict"]
