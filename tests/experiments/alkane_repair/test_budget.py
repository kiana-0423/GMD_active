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


# ---------------------------------------------------------------------------------------------- amendments


def _amend(bench, *entries) -> Path:
    path = bench.path.parent / f"{bench.config.name}_authorization_amendments.yaml"
    path.write_text(yaml.safe_dump({"amendments": list(entries)}))
    return path


def _entry(**overrides):
    entry = {"id": "A1", "at": "2026-11-07T00:00:00Z", "granted_by": "user", "statement": "raise q0", "rationale": "measured costs",
             "changes": {"initial_model_q0_core_h": {"from": 1.0, "to": 0.8}}, "unchanged": {"total_core_h": 4.0, "stages.initial_model": 1.0}}
    entry.update(overrides)
    return entry


def test_amendments_chain_and_keep_the_invariants(bench) -> None:
    from tools.experiments.alkane_repair.budget import authorization_path

    original = authorization_path(bench).read_bytes()
    _amend(bench, _entry(), _entry(id="A2", changes={"initial_model_q0_core_h": {"from": 0.8, "to": 0.9}}, unchanged={}))
    auth = load_authorization(bench)
    assert auth.initial_model_q0_core_h == 0.9 and [a["id"] for a in auth.applied_amendments] == ["A1", "A2"]
    assert authorization_path(bench).read_bytes() == original  # the original record is never rewritten
    assert dict(ComputeLedger(bench, auth).limits({"stage": "initial_model", "q": 0}))["initial_model_q0"] == pytest.approx(0.9 * 3600)
    for bad, match in ((_entry(changes={"initial_model_q0_core_h": {"from": 2.0, "to": 0.8}}), "does not match the current allowance"),
                       (_entry(changes={"initial_model_q0_core_h": {"from": 1.0, "to": 1.5}}), "exceeds the initial-model allowance"),
                       (_entry(changes={"stages.repair_runs": {"from": 1.0, "to": 2.0}}), "exceeds the protocol budget|more than the total"),
                       (_entry(changes={"cores_per_job": {"from": 2, "to": 4}}), "not an amendable allowance"),
                       (_entry(unchanged={"total_core_h": 5.0}), "declared unchanged"),
                       (_entry(unchanged={"initial_model_q0_core_h": 1.0}), "declared unchanged")):
        _amend(bench, bad)
        with pytest.raises(BudgetRefused, match=match):
            load_authorization(bench)
    _amend(bench, _entry(), _entry())
    with pytest.raises(BudgetRefused, match="appears twice"):
        load_authorization(bench)


def test_executor_journals_amendments_and_refuses_tampering(bench) -> None:
    from gmd_active_learning.contracts.ids import file_sha256

    from tools.experiments.alkane_repair.budget import authorization_path
    from tools.experiments.alkane_repair.execute import ExecutionStop, Journal, verify_authorization_record

    journal = Journal(bench)
    journal.state["start"] = {"authorization": {"sha256": file_sha256(authorization_path(bench))}}
    _amend(bench, _entry())
    verify_authorization_record(bench, journal, load_authorization(bench))
    verify_authorization_record(bench, journal, load_authorization(bench))  # journalled once
    events = [json.loads(line) for line in journal.journal_path.read_text().splitlines()]
    assert [e["event"] for e in events] == ["authorization_amendment"] and events[0]["amendment"]["id"] == "A1"
    assert events[0]["effective"]["initial_model_q0_core_h"] == 0.8
    _amend(bench, _entry(rationale="edited afterwards"))
    with pytest.raises(ExecutionStop, match="changed after it was recorded"):
        verify_authorization_record(bench, journal, load_authorization(bench))
    _amend(bench)
    with pytest.raises(ExecutionStop, match="missing from the amendments file"):
        verify_authorization_record(bench, journal, load_authorization(bench))
    _amend(bench, _entry())
    path = authorization_path(bench)
    path.write_text(path.read_text().replace("total_core_h: 4.0", "total_core_h: 4.0 # edited"))
    with pytest.raises(ExecutionStop, match="original authorization file changed"):
        verify_authorization_record(bench, journal, load_authorization(bench))


def test_submit_of_an_existing_job_is_reconciled_not_refused(bench, tmp_path: Path) -> None:
    from tools.experiments.alkane_repair.budget import would_launch

    auth = load_authorization(bench)
    work = bench.root / "cp2k_tasks" / "task"
    assert would_launch(work)
    work.mkdir(parents=True)
    (work / "task.json").write_text(json.dumps({"execution": "prepared", "metadata": {"attempts": []}}))
    assert would_launch(work)
    (work / "task.json").write_text(json.dumps({"execution": "running", "metadata": {"attempts": [{"n": 1}]}}))
    assert not would_launch(work)
    submitted = []

    class Inner:
        def work_dir_for(self, task_id: str) -> Path:
            return work

        def submit(self, task: str) -> str:
            submitted.append(task)
            return task

    full = BudgetGuard(ComputeLedger(bench, auth), lambda kind: auth.total_core_h * 3600 * 2, bench.root / "execution" / "journal.jsonl")
    service = GuardedService(Inner(), full, kind_of=lambda w: "full")
    assert service.submit("t") == "t" and submitted == ["t"]  # no room for a new job, but this one already exists
    (work / "task.json").write_text(json.dumps({"execution": "prepared", "metadata": {"attempts": []}}))
    with pytest.raises(BudgetRefused):
        service.submit("t")


def test_gate_stops_an_unaffordable_stage_with_a_total_preserving_proposal(bench) -> None:
    from tools.experiments.alkane_repair.execute import ExecutionStop, Journal, gate

    auth = load_authorization(bench)
    records = load_records(bench.root)
    n_dev = len({r["parent_id"] for r in records if r["partition"] == "development"})
    cost_h = 1.5 * auth.stages["development"] / n_dev  # measured full-label cost that makes development unaffordable
    now = time.time()
    wall = cost_h * 3600 / auth.cores_per_job
    for i, record in enumerate([r for r in records if r["partition"] in ("id_train", "id_val")][:3]):
        _attempt(_task(bench.root / "cp2k_tasks", f"q{i}", record["parent_id"]), "01", started=now - 10000 + i * 1000, ended=now - 10000 + i * 1000 + wall)
    journal = Journal(bench)
    journal.state["stages"]["qualification_q0"] = {"status": "done", "accepted": True}
    with pytest.raises(ExecutionStop, match="projected not to fit"):
        gate(bench, journal, "development", "development")
    state = journal.stage("development")
    assert state["status"] == "unaffordable" and not journal.done("development")
    proposal = state["proposal"]
    if proposal["feasible"]:
        assert proposal["sum_after"] == pytest.approx(sum(auth.stages.values())) and proposal["proposed_stages"]["development"] > auth.stages["development"]
        assert all(t["to"] == "development" or t["to"] in proposal["deficits"] for t in proposal["transfers"])
    assert load_authorization(bench).stages == auth.stages  # nothing is moved automatically
    assert (bench.root / "execution" / "projections" / "development.json").is_file()
