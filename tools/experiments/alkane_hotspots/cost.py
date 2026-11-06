"""Measured cost of labeling tasks and budget decisions.

Costs are measured, never modelled: the wall time of every attempt comes from the labeling service's
attempt directory (the local scheduler's ``.gmd_local_job.json`` start time and the modification
time of the ``.gmd_exit_code`` file the job wrapper writes when CP2K exits). If those files are
absent (another scheduler), the attempt record's ``submitted_at``/``finished_at`` timestamps are used
and the source is recorded. ``cpu_core_s = wall_s x allocated cores`` (OMP threads x MPI ranks).

Budget decisions use the mean measured cost of finished labels of the same kind (full labels: same
chain length) and, before any measurement exists, the per-job walltime x cores upper bound.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.experiments.alkane_hotspots.config import BudgetsConfig

LOGGER = logging.getLogger(__name__)

EXIT_CODE_FILE = ".gmd_exit_code"
LOCAL_JOB_FILE = ".gmd_local_job.json"


def _iso_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return None


def attempt_wall_seconds(task_work_dir: str | Path, attempt: dict[str, Any]) -> tuple[float | None, str]:
    """Measured wall seconds of one attempt and the source of the measurement."""

    attempt_dir = Path(task_work_dir) / attempt["dir"]
    record_path = attempt_dir / LOCAL_JOB_FILE
    exit_path = attempt_dir / EXIT_CODE_FILE
    if record_path.exists() and exit_path.exists():
        try:
            started = float(json.loads(record_path.read_text(encoding="utf-8"))["started_at"])
            return max(0.0, exit_path.stat().st_mtime - started), "local_job_record"
        except (KeyError, ValueError, json.JSONDecodeError):
            LOGGER.warning("unreadable %s; falling back to attempt timestamps", record_path)
    start, end = _iso_seconds(attempt.get("submitted_at")), _iso_seconds(attempt.get("finished_at"))
    if start is not None and end is not None:
        return max(0.0, end - start), "attempt_timestamps"
    return None, "unavailable"


def measured_cost(task_work_dir: str | Path, attempts: list[dict[str, Any]], cores: int) -> dict[str, Any]:
    """``{wall_s, cpu_core_s, attempts, retries_wall_s, cores, timing_sources, stages}`` for one task."""

    total = 0.0
    retries = 0.0
    sources: list[str] = []
    stages: dict[str, float] = {}
    complete = True
    for attempt in attempts:
        wall, source = attempt_wall_seconds(task_work_dir, attempt)
        sources.append(source)
        if wall is None:
            complete = False
            continue
        total += wall
        stages[attempt["stage"]] = stages.get(attempt["stage"], 0.0) + wall
        if attempt.get("kind") == "retry":
            retries += wall
    return {
        "wall_s": total,
        "cpu_core_s": total * cores,
        "attempts": len(attempts),
        "retries_wall_s": retries,
        "cores": cores,
        "timing_sources": sorted(set(sources)),
        "stage_wall_s": stages,
        "complete": complete,
    }


@dataclass
class BudgetDecision:
    allowed: bool
    reason: str
    estimate_core_s: float
    estimate_source: str


class BudgetLedger:
    """Counts and measured core-seconds from the label records, plus in-flight estimates."""

    def __init__(self, records: list[dict[str, Any]], budgets: BudgetsConfig, cores: int) -> None:
        self.budgets = budgets
        self.cores = cores
        self.records = records
        self.inflight: dict[str, tuple[str, float]] = {}

    def launched(self, kind: str) -> int:
        counted = sum(1 for r in self.records if r["kind"] == kind and r.get("task_id") and r["status"] != "budget_skipped")
        return counted + sum(1 for k, _ in self.inflight.values() if k == kind)

    def spent_core_s(self) -> float:
        return float(sum((r.get("cost") or {}).get("cpu_core_s", 0.0) for r in self.records if r["status"] != "budget_skipped"))

    def inflight_core_s(self) -> float:
        return float(sum(estimate for _, estimate in self.inflight.values()))

    def measured(self, kind: str, chain_length: int | None = None) -> list[float]:
        values = []
        for record in self.records:
            cost = record.get("cost") or {}
            if record["kind"] != kind or record["status"] not in {"valid", "invalid"} or not cost.get("complete", True):
                continue
            if chain_length is not None and record.get("chain_length") != chain_length:
                continue
            values.append(float(cost.get("cpu_core_s", 0.0)))
        return values

    def estimate(self, kind: str, chain_length: int | None = None) -> tuple[float, str]:
        same = self.measured(kind, chain_length if kind == "full" else None)
        if same:
            return sum(same) / len(same), f"measured mean of {len(same)} {kind} label(s)" + (f" C{chain_length}" if kind == "full" else "")
        bound = float(self.budgets.per_job_walltime_s * self.cores)
        return bound, "upper bound per_job_walltime_s x cores (no measurement yet)"

    def decide(self, kind: str, n_atoms: int, chain_length: int | None = None) -> BudgetDecision:
        estimate, source = self.estimate(kind, chain_length)
        limit_atoms = self.budgets.max_atoms_full if kind == "full" else self.budgets.max_atoms_per_fragment
        if n_atoms > limit_atoms:
            return BudgetDecision(False, f"{n_atoms} atoms > budgets.max_atoms_{'full' if kind == 'full' else 'per_fragment'}={limit_atoms}", estimate, source)
        max_count = self.budgets.max_full_labels if kind == "full" else self.budgets.max_fragment_labels
        if self.launched(kind) >= max_count:
            return BudgetDecision(False, f"budgets.max_{kind}_labels={max_count} reached", estimate, source)
        limit = self.budgets.max_total_cpu_core_hours * 3600.0
        projected = self.spent_core_s() + self.inflight_core_s() + estimate
        if projected > limit:
            return BudgetDecision(
                False,
                f"projected {projected / 3600.0:.2f} core-h (spent {self.spent_core_s() / 3600.0:.2f}, in flight "
                f"{self.inflight_core_s() / 3600.0:.2f}, this job {estimate / 3600.0:.2f}) > budgets.max_total_cpu_core_hours={self.budgets.max_total_cpu_core_hours}",
                estimate,
                source,
            )
        return BudgetDecision(True, "within budget", estimate, source)


__all__ = ["BudgetDecision", "BudgetLedger", "attempt_wall_seconds", "measured_cost"]
