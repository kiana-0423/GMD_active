"""Compute-allowance enforcement: ledger from disk, scopes, room checks, cancellation on poll (offline)."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from .conftest import CONFIGS, copy_profile

from tools.experiments.alkane_repair.budget import (
    BudgetGuard,
    BudgetRefused,
    ComputeLedger,
    GuardedService,
    load_authorization,
    scan_attempt,
)
from tools.experiments.alkane_repair.config import load_benchmark
from tools.experiments.alkane_repair.generate import generate, load_records, write_structures


def _attempt(task_dir: Path, name: str, *, started: float, ended: float | None = None, pid: int | None = None) -> Path:
    attempt = task_dir / "attempts" / name
    attempt.mkdir(parents=True)
    (attempt / ".gmd_local_job.json").write_text(json.dumps({"pid": pid if pid is not None else 999999, "started_at": started}))
    if ended is not None:
        exit_file = attempt / ".gmd_exit_code"
        exit_file.write_text("0\n")
        os.utime(exit_file, (ended, ended))
    else:
        for item in attempt.iterdir():
            os.utime(item, (started, started))
    return attempt


def _task(root: Path, name: str, parent_id: str, mode: str = "full") -> Path:
    task = root / name
    task.mkdir(parents=True)
    (task / "region.json").write_text(json.dumps({"mode": mode, "run_id": parent_id, "metadata": {"parent_id": parent_id}}))
    return task


@pytest.fixture()
def bench(tmp_path: Path):
    loaded = load_benchmark(copy_profile(tmp_path))
    (tmp_path / "repair_smoke_authorization.yaml").write_text((CONFIGS / "repair_smoke_authorization.yaml").read_text())
    write_structures(loaded, generate(loaded))
    return loaded


def test_scan_attempt_states(tmp_path: Path) -> None:
    now = time.time()
    finished = _attempt(tmp_path / "t", "01", started=now - 100, ended=now - 40)
    assert scan_attempt(finished, 5, now) == (pytest.approx(now - 100), pytest.approx(now - 40), False)
    running = _attempt(tmp_path / "t", "02", started=now - 30, pid=os.getpid())
    started, ended, alive = scan_attempt(running, 5, now)
    assert ended is None and alive
    dead = _attempt(tmp_path / "t", "03", started=now - 50)  # killed: no exit code, process gone
    started, ended, alive = scan_attempt(dead, 5, now)
    assert not alive and ended == pytest.approx(now - 50)
    assert scan_attempt(tmp_path / "t" / "missing", 5, now) is None


def test_ledger_classifies_every_attempt_and_survives_restart(bench) -> None:
    auth = load_authorization(bench)
    records = load_records(bench.root)
    ledger = ComputeLedger(bench, auth)
    q0_base = sorted(ledger.q0_bases())[0]
    id_train = next(r for r in records if r["partition"] == "id_train" and r["base_id"] == q0_base)
    dev = next(r for r in records if r["partition"] == "development")
    now = time.time()
    _attempt(_task(bench.root / "cp2k_tasks", "a", id_train["parent_id"]), "01", started=now - 1000, ended=now - 900)            # 100 s
    retry_task = _task(bench.root / "cp2k_tasks", "b", dev["parent_id"], mode="cluster")
    _attempt(retry_task, "01", started=now - 500, ended=now - 450)                                                               # 50 s failed
    _attempt(retry_task, "02", started=now - 400, ended=now - 380)                                                               # 20 s retry
    _attempt(_task(bench.sealed_root / "cp2k_tasks", "c", "x"), "01", started=now - 300, ended=now - 290)                        # 10 s
    run_task = bench.root / "repair_runs" / "D__case__s1" / "run" / "cp2k_tasks" / "d"
    run_task.mkdir(parents=True)
    _attempt(run_task, "01", started=now - 60, pid=os.getpid())                                                                   # running
    consumption = ledger.consumption(now)
    cores = auth.cores_per_job
    assert consumption["stages_core_s"]["initial_model"] == pytest.approx(100 * cores) and consumption["q0_core_s"] == pytest.approx(100 * cores)
    assert consumption["stages_core_s"]["fidelity_audit"] == pytest.approx(70 * cores)  # failed attempt and retry both count
    assert consumption["stages_core_s"]["final_test"] == pytest.approx(10 * cores)
    assert consumption["runs_core_s"]["D__case__s1"] == pytest.approx(60 * cores, rel=0.05) and len(consumption["running"]) == 1
    again = ComputeLedger(bench, auth).consumption(now)  # a restart recomputes the same expenditure from disk
    assert again["total_core_s"] == pytest.approx(consumption["total_core_s"])
    names = dict(ledger.limits({"stage": "initial_model", "q": 0}))
    assert set(names) == {"total", "stage:initial_model", "initial_model_q0"}
    assert dict(ledger.limits({"stage": "repair_runs", "run": "r"}))["run:r"] == pytest.approx(auth.per_run_core_h * 3600)
    assert dict(ledger.limits({"stage": "unclassified"}))["stage:unclassified"] == 0.0


def test_room_and_live_enforcement(bench) -> None:
    auth = load_authorization(bench)
    ledger = ComputeLedger(bench, auth)
    guard = BudgetGuard(ledger, lambda kind: 100.0, bench.root / "execution" / "journal.jsonl")
    assert guard.room({"stage": "repair_val"}, "full")["allowed"]
    with pytest.raises(BudgetRefused, match="no room"):
        BudgetGuard(ledger, lambda kind: auth.stages["repair_val"] * 3600, guard.journal_path).require_room({"stage": "repair_val"}, "full", label="x")
    # a running job that has consumed its run allowance is cancelled on the next poll
    run_task = bench.root / "repair_runs" / "C__case__s1" / "run" / "cp2k_tasks" / "t1"
    run_task.mkdir(parents=True)
    elapsed = auth.per_run_core_h * 3600 / auth.cores_per_job
    _attempt(run_task, "01", started=time.time() - elapsed, pid=os.getpid())
    cancelled = []

    class Inner:
        settings_hash = "s"

        def poll(self, task: object) -> SimpleNamespace:
            return SimpleNamespace(task_id="t1", work_dir=str(run_task))

        def cancel(self, task_id: str) -> SimpleNamespace:
            cancelled.append(task_id)
            return SimpleNamespace(task_id="t1", work_dir=str(run_task), state="cancelled")

        def work_dir_for(self, task_id: str) -> Path:
            return run_task

    service = GuardedService(Inner(), guard, kind_of=lambda work: "fragment")
    result = service.poll("t1")
    assert cancelled == ["t1"] and result.state == "cancelled" and service.settings_hash == "s"
    journal = [json.loads(line) for line in guard.journal_path.read_text().splitlines()]
    assert journal[-1]["event"] == "cancelled_for_budget" and "run:C__case__s1" in journal[-1]["violation"]["exceeded"]
    assert (guard.journal_path.parent / "ledger_snapshot.json").is_file()
    with pytest.raises(BudgetRefused):
        service.submit("t1")  # the run has no room left for another job


def test_authorization_must_match_protocol_and_launcher(bench, tmp_path: Path) -> None:
    path = tmp_path / "repair_smoke_authorization.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["authorization"]["stages"]["repair_val"] = 99.0
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(BudgetRefused, match="exceeds the protocol budget"):
        load_authorization(bench)
    raw["authorization"]["stages"]["repair_val"] = 0.5
    raw["authorization"]["enforcement"]["attempt_wall_cap_s"] = 1234.0
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(BudgetRefused, match="launcher's timeout"):
        load_authorization(bench)
    path.unlink()
    with pytest.raises(BudgetRefused, match="no execution authorization"):
        load_authorization(bench)


def test_initial_model_labels_need_an_explicit_qualification_stage(bench) -> None:
    from gmd_active_learning.core.exceptions import ConfigurationError

    from tools.experiments.alkane_repair.references import stage_structures

    with pytest.raises(ConfigurationError, match="explicit qualification stage"):
        stage_structures(bench, "initial_model")


def test_launcher_timeout_kills_a_process_that_ignores_sigterm() -> None:
    """The launcher form used for CP2K ends even a process that ignores SIGTERM (SIGKILL after the grace)."""

    started = time.monotonic()
    result = subprocess.run(["/usr/bin/timeout", "-k", "1", "1", "/bin/sh", "-c", 'trap "" TERM; while :; do :; done'], capture_output=True, timeout=60)
    assert result.returncode in (137, -9) and time.monotonic() - started < 10  # SIGKILL (shell: 137; Python: -9)


def test_attempt_cap_follows_the_launcher(bench) -> None:
    from tools.experiments.alkane_repair.budget import attempt_cap_core_s, timeout_of_launcher

    auth = load_authorization(bench)
    wall, grace = timeout_of_launcher(bench.alpha.config.cp2k.launcher)
    assert attempt_cap_core_s(auth) == pytest.approx((wall + grace) * auth.cores_per_job)
    assert timeout_of_launcher(["/bin/sh", "-c", "x"]) is None
