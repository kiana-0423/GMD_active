"""Measured attempt timing and budget decisions."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ._helpers import ROOT  # noqa: F401
from tools.experiments.alkane_hotspots.config import BudgetsConfig
from tools.experiments.alkane_hotspots.cost import BudgetLedger, attempt_wall_seconds, measured_cost


def _attempt(task: Path, name: str, started: float, ended: float, kind: str = "initial", stage: str = "sp") -> dict:
    directory = task / "attempts" / name
    directory.mkdir(parents=True)
    (directory / ".gmd_local_job.json").write_text(json.dumps({"started_at": started}))
    exit_file = directory / ".gmd_exit_code"
    exit_file.write_text("0\n")
    os.utime(exit_file, (ended, ended))
    return {"dir": f"attempts/{name}", "kind": kind, "stage": stage}


def test_wall_time_comes_from_the_job_record_and_exit_file(tmp_path: Path) -> None:
    attempts = [_attempt(tmp_path, "01-hopt", 1000.0, 1100.0, stage="hopt"), _attempt(tmp_path, "02-sp", 1100.0, 1160.0, kind="stage"),
                _attempt(tmp_path, "03-sp", 1200.0, 1290.0, kind="retry")]
    cost = measured_cost(tmp_path, attempts, cores=5)
    assert cost["wall_s"] == pytest.approx(250.0) and cost["cpu_core_s"] == pytest.approx(1250.0)
    assert cost["retries_wall_s"] == pytest.approx(90.0) and cost["attempts"] == 3
    assert cost["stage_wall_s"] == {"hopt": pytest.approx(100.0), "sp": pytest.approx(150.0)}
    assert cost["timing_sources"] == ["local_job_record"] and cost["complete"]


def test_fallback_to_attempt_timestamps(tmp_path: Path) -> None:
    attempt = {"dir": "attempts/01-sp", "submitted_at": "2026-11-07T10:00:00+00:00", "finished_at": "2026-11-07T10:02:30+00:00"}
    assert attempt_wall_seconds(tmp_path, attempt) == (150.0, "attempt_timestamps")
    assert attempt_wall_seconds(tmp_path, {"dir": "x"}) == (None, "unavailable")


def _record(kind: str, core_s: float, status: str = "valid", n: int = 16) -> dict:
    return {"kind": kind, "status": status, "task_id": "t", "chain_length": n, "cost": {"cpu_core_s": core_s, "complete": True}}


def test_budget_uses_measured_mean_and_refuses_over_budget() -> None:
    budgets = BudgetsConfig(max_full_labels=3, max_fragment_labels=10, max_total_cpu_core_hours=2.0, per_job_walltime_s=3600,
                            max_atoms_per_fragment=40, max_atoms_full=100)
    ledger = BudgetLedger([], budgets, cores=2)
    estimate, source = ledger.estimate("full", 16)
    assert estimate == 7200.0 and "upper bound" in source
    assert ledger.decide("full", 50, 16).allowed
    ledger.records = [_record("full", 3000.0), _record("full", 5000.0)]
    estimate, source = ledger.estimate("full", 16)
    assert estimate == 4000.0 and "measured mean of 2" in source
    decision = ledger.decide("full", 50, 16)  # spent 8000 + 4000 > 7200
    assert not decision.allowed and "max_total_cpu_core_hours" in decision.reason
    assert not ledger.decide("fragment", 41).allowed and "max_atoms_per_fragment" in ledger.decide("fragment", 41).reason


def test_count_budget_and_inflight_accounting() -> None:
    budgets = BudgetsConfig(max_full_labels=2, max_fragment_labels=1, max_total_cpu_core_hours=100.0, per_job_walltime_s=60,
                            max_atoms_per_fragment=40, max_atoms_full=100)
    ledger = BudgetLedger([_record("full", 10.0), _record("full", 0.0, status="budget_skipped")], budgets, cores=1)
    assert ledger.launched("full") == 1 and ledger.decide("full", 10, 16).allowed
    ledger.inflight["x"] = ("full", 10.0)
    decision = ledger.decide("full", 10, 16)
    assert not decision.allowed and "max_full_labels=2" in decision.reason
    assert ledger.inflight_core_s() == 10.0 and ledger.spent_core_s() == 10.0
