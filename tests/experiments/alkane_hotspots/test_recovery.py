"""Recovery demonstration logic with fakes and real local processes (no CP2K), plus one real-CP2K
cancellation test (marked ``cp2k``)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ._helpers import CONFIGS
from gmd_active_learning.adapters.schedulers.local import LocalScheduler
from gmd_active_learning.contracts import ExecutionStatus
from gmd_active_learning.retraining.registry import GateReport, ModelRegistry
from gmd_active_learning.workflow.store import WorkflowStore
from tools.experiments.alkane_hotspots.recovery import (
    process_group_alive,
    read_tasks,
    strict_gate_audit,
    submitted_tasks,
    verify_resume,
)


def _snapshot(job_ids: dict[str, list[tuple[str, str]]]) -> dict:
    return {"tasks": [{"task_id": task, "state": "submitted", "cycle": 0, "attempt": 1, "job_id": attempts[0][1]} for task, attempts in job_ids.items()],
            "task_dirs": sorted(job_ids),
            "attempts": {task: [{"index": i + 1, "stage": "sp", "kind": kind, "job_id": job, "submitted_at": "t", "pgid": None, "state": "running"}
                                for i, (kind, job) in enumerate(attempts)] for task, attempts in job_ids.items()}}


def test_resume_that_adopts_the_job_passes() -> None:
    before = _snapshot({"t1": [("initial", "100")]})
    after = _snapshot({"t1": [("initial", "100")], "t2": [("initial", "200")]})
    result = verify_resume(before, after, samples=[{"task_id": "t1"}, {"task_id": "t2"}], label_files=["t1.json", "t2.json"], dataset_versions=["a", "b"])
    assert result["passed"] and result["adopted"][0]["same_job_after_resume"] and result["new_task_dirs_after_resume"] == ["t2"]


@pytest.mark.parametrize(
    ("after_jobs", "samples", "message"),
    [
        ({"t1": [("initial", "100"), ("initial", "101")]}, [{"task_id": "t1"}], "not adopted"),  # resubmitted
        ({"t1": [("initial", "999")]}, [{"task_id": "t1"}], "not adopted"),  # a different job
        ({"t1": [("initial", "100")]}, [{"task_id": "t1"}, {"task_id": "t1"}], "more than one sample"),
        ({}, [], "not adopted"),  # the task directory vanished
    ],
)
def test_resume_problems_are_detected(after_jobs, samples, message) -> None:  # noqa: ANN001
    before = _snapshot({"t1": [("initial", "100")]})
    result = verify_resume(before, _snapshot(after_jobs), samples=samples, label_files=[], dataset_versions=[])
    assert not result["passed"] and any(message in problem for problem in result["problems"])


def test_retry_after_adoption_is_reported_not_hidden() -> None:
    before = _snapshot({"t1": [("initial", "100")]})
    after = _snapshot({"t1": [("initial", "100"), ("retry", "150")]})
    result = verify_resume(before, after, samples=[], label_files=[], dataset_versions=[])
    assert result["passed"] and result["adopted"][0]["later_attempts"][0]["kind"] == "retry"


def test_read_tasks_and_submitted_filter(tmp_path: Path) -> None:
    store = WorkflowStore(tmp_path / "workflow.sqlite")
    store.ensure_run("run", config_hash="h", initial_model_id=None, stage="sampling")
    store.upsert_task("a", candidate_id="c", run_id="run", cycle=0, state="submitting", work_dir="/w/a")
    store.upsert_task("b", candidate_id="c", run_id="run", cycle=0, state="submitted", work_dir="/w/b", job={"job_id": "77"})
    rows = read_tasks(tmp_path / "workflow.sqlite", "run")
    assert [row["task_id"] for row in rows] == ["a", "b"]
    assert [row["task_id"] for row in submitted_tasks(rows)] == ["b"]
    assert read_tasks(tmp_path / "missing.sqlite", "run") == []


# ---------------------------------------------------------------------------------------------- gates + rollback


def _model_files(tmp_path: Path, name: str) -> tuple[Path, Path]:
    checkpoint, export = tmp_path / f"{name}.pt", tmp_path / f"{name}_export.pt"
    checkpoint.write_bytes(name.encode() * 10)
    export.write_bytes((name + "x").encode() * 10)
    return checkpoint, export


def _registry(tmp_path: Path) -> tuple[WorkflowStore, ModelRegistry, str, str]:
    store = WorkflowStore(tmp_path / "workflow.sqlite")
    registry = ModelRegistry(tmp_path / "models", store)
    store.ensure_run("run", config_hash="h", initial_model_id=None, stage="sampling")
    c0, e0 = _model_files(tmp_path, "m0")
    m0 = registry.register(checkpoint=c0, export=e0, dataset_version="d0", parent_model_id=None, species=[1, 6], cutoff=4.0, metrics={"force_rmse": 0.4},
                           train_mode="fresh", backend="gmd_sgt")
    registry.promote("run", m0.model_id, gate_report=None, reason="initial", force_initial=True)
    c1, e1 = _model_files(tmp_path, "m1")
    m1 = registry.register(checkpoint=c1, export=e1, dataset_version="d1", parent_model_id=m0.model_id, species=[1, 6], cutoff=4.0,
                           metrics={"force_rmse": 0.3, "n_validation": 4.0}, train_mode="finetune", backend="gmd_sgt",
                           metadata={"parent_metrics_on_this_dataset": {"force_rmse": 0.4}})
    registry.promote("run", m1.model_id, gate_report=GateReport(True, {}), reason="cycle 0 gates passed")
    return store, registry, m0.model_id, m1.model_id


def test_strict_gate_rejects_on_recorded_metrics_and_rolls_back(tmp_path: Path) -> None:
    store, registry, m0, m1 = _registry(tmp_path)
    export = store.model(m1)["metadata"]["source_export"]
    history = [{"kind": "validate_export", "export": export, "result": {"max_force_deviation": 1e-6}},
               {"kind": "engine_smoke", "export": export, "result": {"passed": True}}]
    result = strict_gate_audit(store, registry, "run", history, {"max_force_rmse": 0.05, "max_relative_regression": 0.0})
    assert result["outcome"] == "rolled_back" and result["failed_checks"] == ["force_rmse"]
    assert result["strict_report"]["checks"]["force_rmse"]["value"] == 0.3  # the model's recorded validation metric
    assert store.run("run").active_model_id == m0 and registry.state(m1) == "rejected" and registry.state(m0) == "promoted"
    assert store.promotions("run")[-1]["action"] == "rollback"


def test_strict_gate_pass_or_no_promotion_changes_nothing(tmp_path: Path) -> None:
    store, registry, _m0, m1 = _registry(tmp_path)
    export = store.model(m1)["metadata"]["source_export"]
    history = [{"kind": "validate_export", "export": export, "result": {"max_force_deviation": 1e-6}},
               {"kind": "engine_smoke", "export": export, "result": {"passed": True}}]
    result = strict_gate_audit(store, registry, "run", history, {"max_force_rmse": 0.5, "max_relative_regression": 0.0})
    assert result["outcome"] == "strict_gates_passed" and store.run("run").active_model_id == m1
    missing_evidence = strict_gate_audit(store, registry, "run", [], {"max_force_rmse": 0.5, "max_relative_regression": 0.0})
    assert missing_evidence["outcome"] == "rolled_back" and "export_matches_checkpoint" in missing_evidence["failed_checks"]
    fresh = WorkflowStore(tmp_path / "other.sqlite")
    fresh.ensure_run("run", config_hash="h", initial_model_id=None, stage="sampling")
    assert strict_gate_audit(fresh, ModelRegistry(tmp_path / "m2", fresh), "run", [], {})["outcome"] == "no_promotion"


# ---------------------------------------------------------------------------------------------- real processes, no CP2K


def test_process_group_helper_sees_live_processes_only() -> None:
    process = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        assert any(p["pid"] == process.pid for p in process_group_alive(process.pid))
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    assert process_group_alive(process.pid) == []


def test_local_cancel_from_a_fresh_scheduler_terminates_the_job_group(tmp_path: Path) -> None:
    """A restarted controller has no Popen handle: cancel must still stop the wrapper and its children."""

    submitter = LocalScheduler()
    job = submitter.submit(tmp_path / "job", [sys.executable, "-c", "import subprocess; subprocess.run(['sleep', '60'])"])
    record = json.loads((tmp_path / "job" / ".gmd_local_job.json").read_text())
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not any("sleep" in p["command"] for p in process_group_alive(record["pgid"])):
        time.sleep(0.1)
    assert any("sleep" in p["command"] for p in process_group_alive(record["pgid"]))  # grandchild running
    fresh = LocalScheduler()
    assert fresh.cancel(job) is ExecutionStatus.CANCELLED
    assert process_group_alive(record["pgid"]) == []
    assert (tmp_path / "job" / ".gmd_cancelled").is_file()
    assert LocalScheduler().poll(job) is ExecutionStatus.CANCELLED  # stays cancelled for any later poll


# ---------------------------------------------------------------------------------------------- real CP2K


@pytest.mark.cp2k
@pytest.mark.integration
def test_real_cp2k_job_is_cancelled_by_a_new_service_instance(tmp_path: Path) -> None:
    from tools.experiments.alkane_hotspots.closed_loop import load_closed_loop
    from tools.experiments.alkane_hotspots.recovery import cancellation_check

    if shutil.which("cp2k.psmp") is None:
        pytest.skip("cp2k.psmp not on PATH")
    cl = load_closed_loop(CONFIGS / "closed_loop_smoke.yaml")
    if not (cl.alpha.artifacts_root / "parents.jsonl").is_file():
        pytest.skip(f"smoke parents not generated in {cl.alpha.artifacts_root} (run `generate` with smoke.yaml)")
    result = cancellation_check(cl, tmp_path, wait_s=120.0)
    assert result["passed"], result["checks"]
    assert result["attempt"]["failure_class"] == "cancelled" and "never retried" in result["attempt"]["retry_decision"]
