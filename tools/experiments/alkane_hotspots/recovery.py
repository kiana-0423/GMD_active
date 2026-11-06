"""Bounded, real recovery demonstration on the closed loop's recovery run.

1. **Interrupt.** The recovery run (``closed-loop-run --run-kind recovery``) is started in a subprocess
   and killed with SIGKILL as soon as the controller has recorded a submitted CP2K job. The CP2K job
   keeps running: the local scheduler starts it in its own session so that a restarted controller can
   adopt it (by design).
2. **Resume.** The run is resumed in this process. The controller's polling reconciles the persisted
   job (pid + process-group record) instead of submitting again. Checked: every task submitted before
   the kill keeps its single initial attempt and job id; no second job, label, sample or dataset
   version exists for it; CP2K task directories are not duplicated.
3. **Gate rejection on real metrics + rollback.** The promoted model of the recovery run is audited
   with ``recovery.strict_promotion`` (the same ``evaluate_gates`` on the model's recorded real
   validation metrics, export check and GMD smoke test). A failing model is rolled back with the
   registry (it becomes ``rejected``), and GMD restarts from the run's last checkpoint with the
   restored model; the energy jump at the boundary is recorded as a model change.
4. **Cancellation.** A real CP2K job is submitted through ``LabelingService``; once its CP2K process
   runs, a *new* service instance (as after a controller restart, with no in-memory process handle)
   cancels it. Checked: no live process remains in the job's process group, the task records
   ``cancelled`` without a retry, and the workflow bridge does not report it as a completed calculation.

Each outcome is recorded as it happened (``recovery/recovery_report.json``); nothing is assumed.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from gmd_active_learning.contracts import FrameAction, stable_id
from gmd_active_learning.core.exceptions import ConfigurationError, WorkflowError
from gmd_active_learning.utils.atomic_io import write_json_atomic

from .closed_loop import (
    REPO_ROOT,
    LoadedClosedLoop,
    assemble,
    load_closed_loop,
    model_change_jumps,
    read_segment_frames,
    require_exclusive_cp2k,
    run_closed_loop,
)
from .config import LoadedConfig

LOGGER = logging.getLogger(__name__)


# ============================================================================================ store snapshots


def read_tasks(db_path: str | Path, run_id: str, *, retries: int = 20, delay_s: float = 0.25) -> list[dict[str, Any]]:
    """Task rows of a (possibly live) workflow store, read-only."""

    path = Path(db_path)
    if not path.is_file():
        return []
    last: Exception | None = None
    for _ in range(retries):
        try:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
            try:
                connection.row_factory = sqlite3.Row
                rows = connection.execute("SELECT * FROM tasks WHERE run_id=? ORDER BY task_id", (run_id,)).fetchall()
            finally:
                connection.close()
            out = []
            for row in rows:
                data = dict(row)
                data["job"] = None if data["job"] is None else json.loads(data["job"])
                out.append(data)
            return out
        except sqlite3.OperationalError as exc:  # locked or schema not yet created by the writer
            last = exc
            time.sleep(delay_s)
    raise WorkflowError(f"cannot read {path}: {last}")


def submitted_tasks(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rows whose job was recorded by the controller (state ``submitted`` with a job id)."""

    return [dict(row) for row in rows if row.get("state") == "submitted" and (row.get("job") or {}).get("job_id")]


def process_group_alive(pgid: int) -> list[dict[str, Any]]:
    """Live (non-zombie) processes of a process group."""

    listing = subprocess.run(["ps", "-axo", "pid=,pgid=,stat=,command="], capture_output=True, text=True, check=True).stdout
    alive = []
    for line in listing.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, group, stat, command = int(parts[0]), int(parts[1]), parts[2], parts[3]
        if group == int(pgid) and not stat.startswith("Z"):
            alive.append({"pid": pid, "stat": stat, "command": command[:160]})
    return alive


def attempt_job_records(task_dir: Path) -> list[dict[str, Any]]:
    task = json.loads((task_dir / "task.json").read_text())
    attempts = (task.get("metadata") or {}).get("attempts") or []
    out = []
    for attempt in attempts:
        record_path = task_dir / attempt["dir"] / ".gmd_local_job.json"
        record = json.loads(record_path.read_text()) if record_path.is_file() else {}
        out.append({"index": attempt.get("index"), "stage": attempt.get("stage"), "kind": attempt.get("kind"), "job_id": attempt.get("job_id"),
                    "submitted_at": attempt.get("submitted_at"), "pgid": record.get("pgid"), "state": attempt.get("state")})
    return out


def snapshot(work_dir: Path, run_id: str) -> dict[str, Any]:
    rows = read_tasks(work_dir / "workflow.sqlite", run_id)
    tasks_root = work_dir / "cp2k_tasks"
    task_dirs = sorted(p.name for p in tasks_root.iterdir() if (p / "task.json").is_file()) if tasks_root.is_dir() else []
    attempts = {name: attempt_job_records(tasks_root / name) for name in task_dirs}
    return {"tasks": [{k: row[k] for k in ("task_id", "state", "cycle", "attempt")} | {"job_id": (row.get("job") or {}).get("job_id")}
                      for row in rows], "task_dirs": task_dirs, "attempts": attempts}


def verify_resume(before: Mapping[str, Any], after: Mapping[str, Any], *, samples: Sequence[Mapping[str, Any]],
                  label_files: Sequence[str], dataset_versions: Sequence[str]) -> dict[str, Any]:
    """No duplicate jobs/labels/samples; every pre-kill job adopted (same single initial attempt and job id)."""

    adopted, problems = [], []
    for task in before["tasks"]:
        if task["job_id"] is None:
            continue
        after_attempts = after["attempts"].get(task["task_id"], [])
        before_attempts = before["attempts"].get(task["task_id"], [])
        initial = [a for a in after_attempts if a["kind"] == "initial"]
        same_job = bool(initial) and initial[0]["job_id"] == task["job_id"] and len(initial) == 1
        extra = [a for a in after_attempts[len(before_attempts):]]
        adopted.append({"task_id": task["task_id"], "job_id": task["job_id"], "same_job_after_resume": same_job, "n_initial_attempts": len(initial),
                        "later_attempts": [{k: a[k] for k in ("index", "stage", "kind", "job_id")} for a in extra]})
        if not same_job:
            problems.append(f"task {task['task_id']}: initial job not adopted ({initial})")
    task_ids = [t["task_id"] for t in after["tasks"]]
    if len(task_ids) != len(set(task_ids)):
        problems.append("duplicate task ids")
    missing_dirs = sorted(set(before["task_dirs"]) - set(after["task_dirs"]))
    if missing_dirs:
        problems.append(f"task directories disappeared: {missing_dirs}")
    sample_tasks = [s["task_id"] for s in samples]
    if len(sample_tasks) != len(set(sample_tasks)):
        problems.append("more than one sample for a task")
    label_ids = [Path(name).stem for name in label_files]
    if len(label_ids) != len(set(label_ids)):
        problems.append("duplicate label files")
    if len(dataset_versions) != len(set(dataset_versions)):
        problems.append("duplicate dataset versions")
    all_jobs = [a["job_id"] for attempts in after["attempts"].values() for a in attempts if a["job_id"]]
    if len(all_jobs) != len(set(all_jobs)):
        problems.append("a job id appears twice")
    return {"passed": not problems, "problems": problems, "adopted": adopted, "n_tasks": len(task_ids), "n_samples": len(sample_tasks),
            "n_label_files": len(label_ids), "n_dataset_versions": len(dataset_versions),
            "new_task_dirs_after_resume": sorted(set(after["task_dirs"]) - set(before["task_dirs"]))}


# ============================================================================================ 1. interrupt


def interrupt_after_submission(cl: LoadedClosedLoop, *, python: str = sys.executable, poll_s: float = 2.0) -> dict[str, Any]:
    """Run the recovery run in a subprocess; SIGKILL it once a CP2K job is recorded as submitted."""

    rec = cl.config.recovery
    assert rec is not None
    work = cl.run_dir("recovery")
    out = work.parent
    out.mkdir(parents=True, exist_ok=True)
    existing = read_tasks(work / "workflow.sqlite", rec.run_id)
    if existing:
        raise WorkflowError(f"the recovery run {rec.run_id} already has tasks ({len(existing)}); its interruption was done before. "
                            f"Delete {out} to repeat the demonstration")
    command = [python, "-W", "ignore", "-m", "tools.experiments.alkane_hotspots", "closed-loop-run", "--config", str(cl.alpha.path),
               "--closed-loop", str(cl.path), "--run-kind", "recovery"]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    log = (out / "interrupted_run.log").open("w")
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        while True:
            if process.poll() is not None:
                log.close()
                tail = (out / "interrupted_run.log").read_text()[-3000:]
                raise WorkflowError(f"the recovery run exited (code {process.returncode}) before any CP2K submission; nothing to interrupt.\n{tail}")
            rows = submitted_tasks(read_tasks(work / "workflow.sqlite", rec.run_id))
            if rows:
                break
            if time.monotonic() - started > rec.kill_timeout_s:
                raise WorkflowError(f"no CP2K submission within recovery.kill_timeout_s={rec.kill_timeout_s}")
            time.sleep(poll_s)
        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=60)
    finally:
        if process.poll() is None:
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=60)
        if not log.closed:
            log.close()
    killed_at = time.monotonic() - started
    state = snapshot(work, rec.run_id)
    jobs_alive = {}
    for task_id, attempts in state["attempts"].items():
        for attempt in attempts:
            if attempt["pgid"]:
                jobs_alive[f"{task_id}:{attempt['index']}"] = bool(process_group_alive(int(attempt["pgid"])))
    result = {"controller_pid": process.pid, "signal": "SIGKILL", "returncode": process.returncode, "killed_after_s": killed_at,
              "snapshot_after_kill": state, "cp2k_jobs_alive_after_kill": jobs_alive,
              "note": "local CP2K jobs run in their own session and survive the controller by design, so a restart can adopt them"}
    write_json_atomic(out / "interruption.json", result)
    return result


# ============================================================================================ 3. strict gate audit + rollback


def strict_gate_audit(store: Any, registry: Any, run_id: str, history: Sequence[Mapping[str, Any]], strict: Mapping[str, Any]) -> dict[str, Any]:
    """Audit the run's last promoted model with the strict gates; roll back when it fails."""

    from gmd_active_learning.retraining.registry import PromotionGateConfig, evaluate_gates

    config = PromotionGateConfig.model_validate(dict(strict))
    promotions = store.promotions(run_id)
    promoted = [p for p in promotions if p["action"] == "promote" and p["previous_model_id"]]
    rejected = [p for p in promotions if p["action"] == "reject"]
    if not promoted:
        return {"outcome": "no_promotion", "detail": "the controller's own gates did not promote a model in the recovery run",
                "controller_rejections": rejected, "rolled_back": False}
    model_id = promoted[-1]["model_id"]
    record = store.model(model_id)
    source_export = (record.get("metadata") or {}).get("source_export")
    export_check = next((h["result"] for h in history if h["kind"] == "validate_export" and h["export"] == source_export), None)
    smoke = next((h["result"] for h in history if h["kind"] == "engine_smoke" and h["export"] == source_export), None)
    parent_metrics = (record.get("metadata") or {}).get("parent_metrics_on_this_dataset")
    report = evaluate_gates(config, metrics=record["metrics"], parent_metrics=parent_metrics, export_check=export_check, engine_smoke=smoke)
    failed = sorted(name for name, check in report.checks.items() if not check["passed"])
    result: dict[str, Any] = {"model_id": model_id, "metrics": record["metrics"], "parent_metrics": parent_metrics,
                              "strict_gates": config.model_dump(), "strict_report": report.to_dict(), "failed_checks": failed}
    if report.passed:
        result.update(outcome="strict_gates_passed", rolled_back=False, detail="the promoted model also passes the strict gates; nothing rolled back")
        return result
    if store.run(run_id).active_model_id != model_id:
        result.update(outcome="not_active", rolled_back=False, detail="the audited model is no longer active")
        return result
    restored = registry.rollback(run_id, reason=f"strict gate audit failed {failed} on real validation metrics")
    result.update(outcome="rolled_back", rolled_back=True, rejected_model_id=model_id, restored_model_id=restored.model_id,
                  rejected_state=registry.state(model_id), restored_state=registry.state(restored.model_id))
    return result


# ============================================================================================ 4. GMD restart with the restored model


def restart_with_active_model(asm: Any, *, n_steps: int, reason: str) -> dict[str, Any]:
    """One GMD segment from the run's last checkpoint with the run's active model, recorded in the store.

    The previous model is the one that ran the last finished segment; when it differs, GMD restarts with
    ``model_change allow`` and the energy jump at the boundary is recorded."""

    from gmd_active_learning.workflow.protocols import SegmentRequest

    controller = asm.controller
    store, registry = controller.store, controller.registry
    run = store.run(asm.run_id)
    if run.last_checkpoint is None:
        raise WorkflowError("the recovery run has no MD checkpoint to restart from")
    model = registry.get(run.active_model_id)
    segments = store.segments(asm.run_id)
    finished = [seg for seg in segments if seg["status"] in ("completed", "stopped")]
    if not finished:
        raise WorkflowError("the recovery run has no finished MD segment")
    previous_model_id = finished[-1]["model_id"]
    previous = None if previous_model_id == model.model_id else registry.get(previous_model_id)
    index = len(segments)
    request = SegmentRequest(asm.run_id, index, asm.work_dir / "md" / f"segment_{index:05d}", model, int(n_steps), Path(run.last_checkpoint), previous,
                             {"reason": reason})
    store.upsert_segment(asm.run_id, index, cycle=run.cycle, model_id=model.model_id, status="running", work_dir=str(request.work_dir),
                         start_step=run.last_step)

    def on_frame(frame: Any) -> FrameAction:
        asm.guard.check(frame)
        return FrameAction.CONTINUE

    result = controller.md.completed_segment(request) or controller.md.run_segment(request, on_frame)
    store.upsert_segment(asm.run_id, index, status=result.status, end_step=result.end_step,
                         checkpoint=None if result.checkpoint is None else str(result.checkpoint), detail=result.detail)
    if result.status == "failed":
        raise WorkflowError(f"GMD restart ({reason}) failed: {result.detail}")
    store.update_run(asm.run_id, last_checkpoint=str(result.checkpoint), last_step=result.end_step)
    backend = controller.trainer.inner

    def reevaluate(model_id: str, frame: Any) -> float:
        artifact = registry.get(model_id)
        return backend.predict(artifact.export_path, frame.positions, frame.numbers, frame.cell, frame.pbc, species=artifact.species).energy

    rows = [dict(row, run_id=asm.run_id) for row in store.segments(asm.run_id)]
    jumps = model_change_jumps(rows[-2:], read_segment_frames, reevaluate)
    status_path = request.work_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    return {"reason": reason, "segment_index": index, "status": result.status, "start_step": run.last_step, "end_step": result.end_step,
            "model_id": model.model_id, "previous_model_id": previous_model_id, "gmd_status": {k: status.get(k) for k in ("model_changed_from", "start_step", "step", "state")},
            "model_change": jumps}


# ============================================================================================ 5. cancellation


def cancellation_check(cl: LoadedClosedLoop, out_dir: Path, *, wait_s: float, settle_s: float = 3.0,
                       service_factory: Callable[[Path], Any] | None = None) -> dict[str, Any]:
    """Submit a real CP2K job, cancel it from a fresh service instance, and record what happened."""

    from gmd_active_learning.labeling.service import LabelingService
    from gmd_active_learning.workflow.assembly import ServiceLabeler

    from .families import load_parents, load_positions
    from .labels import cp2k_raw_config, parent_region

    task_root = out_dir / "cancellation_tasks"

    def factory(root: Path) -> Any:
        if service_factory is not None:
            return service_factory(root)
        raw = cp2k_raw_config(cl.alpha)
        raw["cp2k"] = {**raw["cp2k"], "task_dir": str(root)}
        return LabelingService.from_config(raw, base_dir=root)

    parents = load_parents(cl.alpha.artifacts_root)
    parent = next(p for p in parents if p["split"] != "final_eval")
    region = parent_region(parent, load_positions(cl.alpha.artifacts_root, parent))
    data = region.to_dict()
    data["region_id"] = stable_id("cancellation-check", region.region_id, time.time_ns())  # a fresh task every invocation
    from gmd_active_learning.contracts import ExtractedRegion

    region = ExtractedRegion.from_dict(data)
    submitter = factory(task_root)
    task = submitter.submit(submitter.prepare(region).task_id)
    work = Path(task.work_dir)
    attempt_dir = work / task.metadata["attempts"][-1]["dir"]
    record = json.loads((attempt_dir / ".gmd_local_job.json").read_text())
    pgid = int(record["pgid"])
    deadline = time.monotonic() + wait_s
    running: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        running = [p for p in process_group_alive(pgid) if "cp2k" in p["command"]]
        if running:
            break
        time.sleep(0.2)
    if not running:
        raise WorkflowError(f"the CP2K process of task {task.task_id} did not start within {wait_s} s; cancellation not exercised")
    time.sleep(settle_s)  # let CP2K get into its SCF so the cancellation interrupts real work
    before = process_group_alive(pgid)
    if not any("cp2k" in p["command"] for p in before):
        raise WorkflowError(f"the CP2K process of task {task.task_id} ended within {settle_s} s; cancellation not exercised")
    canceller = factory(task_root)  # a new instance: no in-memory process handle, as after a controller restart
    started = time.monotonic()
    cancelled = canceller.cancel(task.task_id)
    latency = time.monotonic() - started
    after = process_group_alive(pgid)
    polled = canceller.poll(task.task_id)
    reloaded = canceller.load(task.task_id)
    attempts = reloaded.metadata.get("attempts") or []
    view = ServiceLabeler(canceller).poll(reloaded)
    checks = {
        "no_live_process_in_job_group": not after,
        "task_execution_cancelled": reloaded.execution.value == "cancelled",
        "task_finished": reloaded.metadata.get("stage") == "done",
        "single_attempt_no_retry": len(attempts) == 1,
        "failure_class_cancelled": bool(attempts) and attempts[-1].get("failure_class") == "cancelled",
        "poll_after_cancel_no_resubmission": len(polled.metadata.get("attempts") or []) == 1,
        "workflow_bridge_reports_cancelled": view.execution.value == "cancelled",
        "cancel_marker_written": (attempt_dir / ".gmd_cancelled").is_file(),
    }
    output = attempt_dir / attempts[-1]["output"] if attempts else None
    return {"task_id": task.task_id, "job_id": task.job.job_id if task.job else None, "pgid": pgid, "processes_before_cancel": before,
            "cp2k_output_bytes_at_cancel": None if output is None or not output.is_file() else output.stat().st_size, "settle_s": settle_s,
            "processes_after_cancel": after, "cancel_latency_s": latency, "execution_after_cancel": cancelled.execution.value,
            "attempt": {k: attempts[-1].get(k) for k in ("state", "raw_state", "exit_code", "failure_class", "retry_decision", "reasons")} if attempts else None,
            "failure_reasons": reloaded.failure_reasons, "checks": checks, "passed": all(checks.values())}


# ============================================================================================ driver


def recovery_demo(cl: LoadedClosedLoop, *, skip_interrupt: bool = False) -> dict[str, Any]:
    from .manifest import utc_now

    rec = cl.config.recovery
    if rec is None:
        raise ConfigurationError(f"{cl.path} has no recovery section")
    require_exclusive_cp2k(cl, [cl.out_dir])
    work = cl.run_dir("recovery")
    out = work.parent
    report: dict[str, Any] = {"created_at": utc_now(), "config": str(cl.path), "run_id": rec.run_id, "smoke_only": cl.alpha.config.cp2k.smoke_only}
    if skip_interrupt:
        interruption = json.loads((out / "interruption.json").read_text())
    else:
        interruption = interrupt_after_submission(cl)
    report["interruption"] = interruption

    run_report = run_closed_loop(cl, kind="recovery")
    asm = assemble(cl, "recovery")
    store = asm.controller.store
    after = snapshot(work, rec.run_id)
    labels_dir = work / "labels"
    report["resume"] = {"final_stage": run_report["final_stage"], "summary": run_report["summary"],
                        "verification": verify_resume(interruption["snapshot_after_kill"], after, samples=store.samples(rec.run_id),
                                                      label_files=sorted(p.name for p in labels_dir.glob("*.json")) if labels_dir.is_dir() else [],
                                                      dataset_versions=[d for d in asm.controller.datasets.inner.versions()])}
    # The controller ends after its last cycle's promotion; run the promoted model for one segment (as the next
    # cycle would), so that the audited model has driven the MD before it is rolled back.
    segments = store.segments(rec.run_id)
    finished = [seg for seg in segments if seg["status"] in ("completed", "stopped")]
    active = store.run(rec.run_id).active_model_id
    if finished and finished[-1]["model_id"] != active:
        report["segment_with_promoted_model"] = restart_with_active_model(asm, n_steps=rec.restart_steps, reason="promoted model drives the MD")
    audit = strict_gate_audit(store, asm.controller.registry, rec.run_id, asm.trainer.history(), rec.strict_promotion)
    report["strict_gate_audit"] = audit
    if audit.get("rolled_back"):
        report["restart_after_rollback"] = restart_with_active_model(asm, n_steps=rec.restart_steps, reason="restart after rollback")
    else:
        report["restart_after_rollback"] = {"executed": False, "reason": audit["outcome"]}
    if rec.cancellation_check:
        require_exclusive_cp2k(cl, [cl.out_dir])
        report["cancellation"] = cancellation_check(cl, out, wait_s=rec.cancellation_wait_s)
    else:
        report["cancellation"] = {"executed": False, "reason": "recovery.cancellation_check is false"}
    from gmd_active_learning.workflow.controller import summarize

    report["final_summary"] = summarize(store, rec.run_id)
    report["image_check"] = asm.guard.flush()
    write_json_atomic(out / "recovery_report.json", report)
    return report


def _arguments(parser: Any) -> None:
    parser.add_argument("--closed-loop", required=True, help="configs/demos/alkane_hotspots/closed_loop_{smoke,pilot}.yaml")
    parser.add_argument("--skip-interrupt", action="store_true", help="reuse recovery/interruption.json (the run was interrupted before)")


def cmd_recovery_demo(args: Any, loaded: LoadedConfig) -> int:
    cl = load_closed_loop(args.closed_loop, alpha=loaded)
    report = recovery_demo(cl, skip_interrupt=args.skip_interrupt)
    short = {"interruption": {k: report["interruption"][k] for k in ("killed_after_s", "cp2k_jobs_alive_after_kill")},
             "resume": {"final_stage": report["resume"]["final_stage"], "verification_passed": report["resume"]["verification"]["passed"],
                        "problems": report["resume"]["verification"]["problems"]},
             "strict_gate_audit": {k: report["strict_gate_audit"].get(k) for k in ("outcome", "failed_checks", "restored_model_id")},
             "restart_after_rollback": {**{k: report["restart_after_rollback"].get(k) for k in ("status", "start_step", "end_step", "executed")},
                                        "jump_ev": [j.get("jump_ev") for j in report["restart_after_rollback"].get("model_change", [])]},
             "cancellation": {k: report["cancellation"].get(k) for k in ("passed", "checks", "cancel_latency_s", "executed")}}
    print(json.dumps(short, indent=1, default=str))
    ok = report["resume"]["verification"]["passed"] and report["cancellation"].get("passed", True)
    return 0 if ok else 1


def register_cli(register: Callable[..., None]) -> None:
    register("recovery-demo", "Interrupt/resume, strict-gate rejection + rollback + GMD restart, CP2K cancellation (real, bounded)",
             _arguments, cmd_recovery_demo)


__all__ = [
    "cancellation_check",
    "interrupt_after_submission",
    "process_group_alive",
    "read_tasks",
    "recovery_demo",
    "restart_with_active_model",
    "snapshot",
    "strict_gate_audit",
    "submitted_tasks",
    "verify_resume",
]
