# Ported from PHAL (hotspot_al.cp2k.cp2k_task_submitter:CP2KTaskSubmitter) at 96263b6; MIT License, see NOTICE.
"""Labeling service: ``prepare(region) -> LabelTask``, ``submit``, ``poll``, ``collect -> LabelResult``.

Every operation reloads ``task.json`` under an exclusive lock and writes it back atomically, so the
service is restartable and idempotent:

- ``prepare`` derives ``task_id`` from the region content and the settings hash; preparing the same
  region twice returns the existing task (never a second directory or job);
- ``submit`` is a no-op once a job exists; an interrupted submission (intent marker without job id)
  turns the task UNKNOWN instead of resubmitting;
- ``poll`` never resubmits on UNKNOWN; it asks the scheduler to reconcile and otherwise waits;
- capped regions run the cap-only GEO_OPT first and the single point on the relaxed caps (F-17);
- failures follow the bounded per-class retry ladder, and an ML-SCF failure is resubmitted natively
  (F-25). Every attempt has its own directory ``attempts/NN-<stage>/``; nothing is overwritten.

Differences from PHAL's ``CP2KTaskSubmitter`` (F-05, F-24): extraction/capping/embedding happen
upstream (the service labels whatever region it is given, in any extraction mode); labels are
collected by ``poll``/``collect``; dataset writing belongs to the datasets area.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gmd_active_learning.adapters.dft.cp2k.backend import STAGE_HOPT, STAGE_SINGLE_POINT, CP2KBackend
from gmd_active_learning.adapters.dft.cp2k.config import CP2KConfig, FailureClass, RetryStep
from gmd_active_learning.adapters.dft.cp2k.input import cap_indices
from gmd_active_learning.adapters.dft.cp2k.parse import CP2KOutput, parse_cp2k_output_file, read_last_xyz_frame
from gmd_active_learning.adapters.dft.cp2k.validation import classify_failure
from gmd_active_learning.adapters.registry import BackendRegistry, BackendRole, create_typed_backend
from gmd_active_learning.adapters.schedulers.base import (
    SchedulerBackend,
    SubmissionRejected,
    read_exit_code,
    sanitize_job_name,
)
from gmd_active_learning.adapters.schedulers.config import SchedulerResources
from gmd_active_learning.contracts.ids import content_hash, file_sha256, stable_id
from gmd_active_learning.contracts.labels import ExecutionStatus, LabelResult, LabelTask, SchedulerJob, ValidationStatus
from gmd_active_learning.contracts.regions import ExtractedRegion
from gmd_active_learning.core.exceptions import LabelingError, SchedulerError
from gmd_active_learning.labeling.retry_policy import KIND_INITIAL, KIND_RETRY, KIND_STAGE, RetryPolicy
from gmd_active_learning.labeling.task_store import load_task, save_task, task_lock
from gmd_active_learning.utils.atomic_io import write_json_atomic

LOGGER = logging.getLogger(__name__)

REGION_FILE = "region.json"
LABELED_REGION_FILE = "region_labeled.json"
RESULT_FILE = "result.json"
INTENT_FILE = ".submit_intent"
STAGE_DONE = "done"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def region_fingerprint(region: ExtractedRegion) -> str:
    """Hash of everything in a region that changes the calculation."""

    return content_hash(
        {
            "numbers": region.numbers,
            "positions": np.round(region.positions, 10),
            "cell": None if region.cell is None else np.round(region.cell, 10),
            "pbc": list(region.pbc),
            "roles": [role.value for role in region.roles],
            "charge": int(region.charge),
            "multiplicity": region.multiplicity,
            "h_capping_skipped": bool(region.metadata.get("h_capping_skipped", False)),
        },
        length=16,
    )


class LabelingService:
    """CP2K labeling with persistent, idempotent task state."""

    def __init__(
        self,
        backend: CP2KBackend,
        scheduler: SchedulerBackend,
        *,
        task_root: str | Path,
        resources: SchedulerResources | None = None,
    ) -> None:
        self.backend = backend
        self.scheduler = scheduler
        self.task_root = Path(task_root).resolve()
        self.resources = resources if resources is not None else backend.config.resources
        self.policy = RetryPolicy(backend.config.retry)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        base_dir: str | Path = ".",
        registry: BackendRegistry | None = None,
    ) -> LabelingService:
        """Build the DFT backend and scheduler from ``backends.dft`` / ``backends.scheduler``."""

        backend = create_typed_backend(config, BackendRole.DFT, CP2KBackend, registry=registry)
        scheduler = create_typed_backend(config, BackendRole.SCHEDULER, SchedulerBackend, registry=registry)
        cp2k = backend.config if isinstance(backend.config, CP2KConfig) else CP2KConfig()
        root = Path(cp2k.task_dir)
        if not root.is_absolute():
            root = Path(base_dir) / root
        return cls(backend, scheduler, task_root=root)

    # -- identity ----------------------------------------------------------------------------------
    @property
    def settings_hash(self) -> str:
        return self.backend.settings_hash()

    def task_id_for(self, region: ExtractedRegion) -> str:
        return stable_id("label", region.region_id, region_fingerprint(region), self.settings_hash)

    def work_dir_for(self, task_id: str) -> Path:
        return self.task_root / task_id

    def load(self, task_id: str) -> LabelTask:
        return load_task(self.work_dir_for(task_id))

    # -- prepare -----------------------------------------------------------------------------------
    def prepare(self, region: ExtractedRegion) -> LabelTask:
        """Create (or return the existing) task for ``region``; input problems raise here."""

        task_id = self.task_id_for(region)
        work_dir = self.work_dir_for(task_id)
        with task_lock(work_dir):
            if (work_dir / "task.json").exists():
                return load_task(work_dir)
            first_stage = STAGE_HOPT if self.backend.needs_h_opt(region) else STAGE_SINGLE_POINT
            # Build the first input now so unsupported combinations fail before a task exists.
            probe = self.backend.build_input(region, stage=first_stage, project_name=first_stage)
            write_json_atomic(work_dir / REGION_FILE, region.to_dict())
            task = LabelTask(
                task_id=task_id,
                region_id=region.region_id,
                engine=self.backend.name,
                settings_hash=self.settings_hash,
                work_dir=str(work_dir),
                metadata={
                    "stage": first_stage,
                    "stages": [STAGE_HOPT, STAGE_SINGLE_POINT] if first_stage == STAGE_HOPT else [STAGE_SINGLE_POINT],
                    "attempts": [],
                    "current_step": None,
                    "native": not self.backend.ml_scf_enabled,
                    "ml_fallback_used": False,
                    "unknown_polls": 0,
                    "n_atoms": region.n_atoms,
                    "n_caps": len(cap_indices(region)),
                    "region_file": REGION_FILE,
                    "labeled_region_file": None,
                    "result_file": None,
                    "electronic_state": probe.electronic_state.as_metadata(),
                    "periodic": probe.geometry.periodic,
                    "variant": self.backend.variant_id(),
                    "created_at": _now(),
                },
            )
            save_task(task)
            LOGGER.info("prepared labeling task %s for region %s (%s)", task_id, region.region_id, first_stage)
            return task

    # -- submit ------------------------------------------------------------------------------------
    def submit(self, task: LabelTask | str) -> LabelTask:
        """Submit the first job; a no-op if the task already has (or had) a job."""

        task_id = task if isinstance(task, str) else task.task_id
        work_dir = self.work_dir_for(task_id)
        with task_lock(work_dir):
            current = load_task(work_dir)
            if current.execution is not ExecutionStatus.PREPARED or current.metadata.get("attempts"):
                return current
            return self._launch(current, stage=current.metadata["stage"], kind=KIND_INITIAL, step=None)

    def _region_for_stage(self, task: LabelTask, stage: str) -> ExtractedRegion:
        name = task.metadata.get("labeled_region_file") if stage == STAGE_SINGLE_POINT else None
        path = Path(task.work_dir) / (name or REGION_FILE)
        return ExtractedRegion.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _backend_for(self, task: LabelTask) -> CP2KBackend:
        return self.backend.native_backend() if task.metadata.get("native", True) else self.backend

    def _launch(self, task: LabelTask, *, stage: str, kind: str, step: RetryStep | None, trigger: str | None = None) -> LabelTask:
        attempts: list[dict[str, Any]] = task.metadata.setdefault("attempts", [])
        index = len(attempts) + 1
        attempt_dir = Path(task.work_dir) / "attempts" / f"{index:02d}-{stage}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        intent = attempt_dir / INTENT_FILE
        try:
            descriptor = os.open(intent, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            task.execution = ExecutionStatus.UNKNOWN
            task.failure_reasons.append(f"attempt {index}: an earlier submission was interrupted; reconcile before retrying")
            save_task(task)
            return task
        os.write(descriptor, _now().encode())
        os.close(descriptor)

        backend = self._backend_for(task)
        region = self._region_for_stage(task, stage)
        cp2k_input = backend.build_input(region, stage=stage, project_name=stage, step=step)
        input_path = attempt_dir / f"{stage}.inp"
        output_path = attempt_dir / f"{stage}.out"
        input_path.write_text(backend.decorate_input(cp2k_input, attempt_dir, region=region), encoding="utf-8")
        write_json_atomic(attempt_dir / "geometry.json", {"offset": cp2k_input.geometry.offset.tolist(), "periodic": cp2k_input.geometry.periodic})
        record: dict[str, Any] = {
            "index": index,
            "stage": stage,
            "kind": kind,
            "trigger_class": trigger,
            "step": None if step is None else step.model_dump(exclude_none=True),
            "native": bool(task.metadata.get("native", True)),
            "dir": str(attempt_dir.relative_to(task.work_dir)),
            "input": input_path.name,
            "input_sha256": file_sha256(input_path),
            "output": output_path.name,
            "job_id": None,
            "submitted_at": None,
            "state": ExecutionStatus.PREPARED.value,
            "exit_code": None,
            "raw_state": None,
            "failure_class": None,
            "reasons": [],
        }
        attempts.append(record)
        task.metadata["stage"] = stage
        task.metadata["current_step"] = record["step"]
        request = backend.execution_request(input_path, output_file=output_path, resources=self.resources)
        try:
            job = self.scheduler.submit_request(request, self.resources, job_name=sanitize_job_name(f"gmd_{stage}_{task.task_id[-8:]}"))
        except SubmissionRejected as exc:
            intent.unlink(missing_ok=True)
            record["state"] = "submit_rejected"
            record["reasons"] = [str(exc)]
            task.failure_reasons.append(f"attempt {index}: submission rejected: {exc}")
            if kind == KIND_INITIAL:
                attempts.pop()  # nothing reached the scheduler; a later submit() may try again
                task.execution = ExecutionStatus.PREPARED
                save_task(task)
                raise
            self._finalize(task, ExecutionStatus.FAILED, [f"retry submission rejected: {exc}"])
            raise
        except SchedulerError as exc:
            record["state"] = ExecutionStatus.UNKNOWN.value
            record["reasons"] = [f"submission outcome unknown: {exc}"]
            task.execution = ExecutionStatus.UNKNOWN
            task.failure_reasons.append(f"attempt {index}: submission outcome unknown ({exc}); reconcile before retrying")
            save_task(task)
            LOGGER.error("task %s: submission outcome unknown: %s", task.task_id, exc)
            return task
        record.update({"job_id": job.job_id, "submitted_at": job.submitted_at, "state": job.state.value})
        task.job = job
        task.attempt = index
        task.execution = job.state
        task.metadata["unknown_polls"] = 0
        save_task(task)
        LOGGER.info("task %s attempt %d (%s, %s) submitted as %s job %s", task.task_id, index, stage, kind, job.backend, job.job_id)
        return task

    # -- poll --------------------------------------------------------------------------------------
    def poll(self, task: LabelTask | str) -> LabelTask:
        """Refresh the job; on completion validate, advance to the next stage or retry."""

        task_id = task if isinstance(task, str) else task.task_id
        work_dir = self.work_dir_for(task_id)
        with task_lock(work_dir):
            current = load_task(work_dir)
            if current.metadata.get("stage") == STAGE_DONE or current.job is None:
                return current
            if current.job.state.terminal and current.metadata["attempts"][-1].get("finished_at"):
                return current
            state = self.scheduler.poll(current.job)
            if state is ExecutionStatus.UNKNOWN:
                state = self.scheduler.reconcile(current.job)
            return self._apply_state(current, state)

    def reconcile(self, task: LabelTask | str) -> LabelTask:
        """Explicit reconciliation: scheduler + work-dir artifacts; never resubmits by itself."""

        task_id = task if isinstance(task, str) else task.task_id
        work_dir = self.work_dir_for(task_id)
        with task_lock(work_dir):
            current = load_task(work_dir)
            if current.metadata.get("stage") == STAGE_DONE:
                return current
            if current.job is not None:
                return self._apply_state(current, self.scheduler.reconcile(current.job))
            attempts = current.metadata.get("attempts", [])
            if not attempts:
                return current
            attempt_dir = Path(current.work_dir) / attempts[-1]["dir"]
            exit_code = read_exit_code(attempt_dir)
            if exit_code is None:
                return current
            # The job ran although its id was never recorded (interrupted submission).
            current.job = SchedulerJob(
                job_id="unrecorded",
                backend=self.scheduler.name,
                state=ExecutionStatus.COMPLETED if exit_code == 0 else ExecutionStatus.FAILED,
                work_dir=str(attempt_dir),
                exit_code=exit_code,
                raw_state="RECONCILED",
            )
            return self._apply_state(current, current.job.state)

    def resolve_unknown(self, task: LabelTask | str, *, reason: str) -> LabelTask:
        """Workflow decision after reconciliation proved the job is gone: treat it as failed.

        The failure then follows the normal retry ladder (``execution_failed``).
        """

        task_id = task if isinstance(task, str) else task.task_id
        work_dir = self.work_dir_for(task_id)
        with task_lock(work_dir):
            current = load_task(work_dir)
            if current.execution is not ExecutionStatus.UNKNOWN:
                return current
            if current.job is None:
                current.job = SchedulerJob(job_id="unrecorded", backend=self.scheduler.name, state=ExecutionStatus.FAILED, work_dir=current.work_dir)
            current.job.state = ExecutionStatus.FAILED
            current.job.metadata["resolved_unknown"] = reason
            return self._apply_state(current, ExecutionStatus.FAILED)

    def cancel(self, task: LabelTask | str) -> LabelTask:
        task_id = task if isinstance(task, str) else task.task_id
        work_dir = self.work_dir_for(task_id)
        with task_lock(work_dir):
            current = load_task(work_dir)
            if current.metadata.get("stage") == STAGE_DONE:
                return current
            if current.job is None:
                self._finalize(current, ExecutionStatus.CANCELLED, ["cancelled before submission"])
                return current
            return self._apply_state(current, self.scheduler.cancel(current.job))

    # -- state machine -----------------------------------------------------------------------------
    def _apply_state(self, task: LabelTask, state: ExecutionStatus) -> LabelTask:
        attempt = task.metadata["attempts"][-1]
        assert task.job is not None
        attempt.update({"state": state.value, "exit_code": task.job.exit_code, "raw_state": task.job.raw_state})
        if state is ExecutionStatus.UNKNOWN:
            polls = int(task.metadata.get("unknown_polls", 0)) + 1
            task.metadata["unknown_polls"] = polls
            task.execution = ExecutionStatus.UNKNOWN
            if polls >= self.backend.config.max_unknown_polls_warning:
                LOGGER.warning("task %s: scheduler state unknown for %d polls; reconcile needed", task.task_id, polls)
            save_task(task)
            return task
        task.metadata["unknown_polls"] = 0
        if not state.terminal:
            task.execution = state
            save_task(task)
            return task
        attempt["finished_at"] = _now()
        return self._finish_attempt(task, state)

    def _finish_attempt(self, task: LabelTask, state: ExecutionStatus) -> LabelTask:
        attempt = task.metadata["attempts"][-1]
        stage = attempt["stage"]
        attempt_dir = Path(task.work_dir) / attempt["dir"]
        output_path = attempt_dir / attempt["output"]
        output = parse_cp2k_output_file(output_path) if output_path.exists() else None
        job = task.job
        assert job is not None
        cancelled = state is ExecutionStatus.CANCELLED
        walltime = job.raw_state in {"TIMEOUT", "DEADLINE"} or bool(job.metadata.get("walltime_exceeded"))
        if output is not None:
            attempt["output_sha256"] = file_sha256(output_path)
            attempt["provenance"] = output.provenance()
        if stage == STAGE_HOPT:
            reasons = self._hopt_reasons(task, output, attempt_dir) if state is ExecutionStatus.COMPLETED else []
            if state is ExecutionStatus.COMPLETED and not reasons:
                attempt["outcome"] = "relaxed"
                save_task(task)
                return self._launch(
                    task,
                    stage=STAGE_SINGLE_POINT,
                    kind=KIND_STAGE,
                    step=None if task.metadata.get("current_step") is None else RetryStep.model_validate(task.metadata["current_step"]),
                )
        else:
            reasons = self.backend.validate(output, n_atoms=int(task.metadata["n_atoms"]))
            if state is ExecutionStatus.COMPLETED and not reasons:
                attempt["outcome"] = "valid"
                self._write_result(task, output, ValidationStatus.VALID, [])
                task.execution = ExecutionStatus.COMPLETED
                task.validation = ValidationStatus.VALID
                task.metadata["stage"] = STAGE_DONE
                save_task(task)
                LOGGER.info("task %s: valid label after %d attempt(s)", task.task_id, task.attempt)
                return task
        if not reasons:
            reasons = [f"job ended {state.value} (exit code {job.exit_code}, scheduler state {job.raw_state})"]
        elif state is not ExecutionStatus.COMPLETED:
            reasons = [f"job ended {state.value} (exit code {job.exit_code})", *reasons]
        if job.metadata.get("resolved_unknown"):
            failure = FailureClass.EXECUTION_FAILED
            reasons = [f"job declared lost by the workflow: {job.metadata['resolved_unknown']}", *reasons]
        else:
            failure = classify_failure(
                output,
                stage=stage,
                exit_code=job.exit_code,
                cancelled=cancelled,
                walltime_exceeded=walltime,
                reasons=reasons,
            )
        attempt.update({"outcome": "failed", "failure_class": failure.value, "reasons": reasons})
        task.failure_reasons.append(f"attempt {attempt['index']} ({stage}): {failure.value}: {'; '.join(reasons)}")
        ml_active = not task.metadata.get("native", True)
        current_step = None if task.metadata.get("current_step") is None else RetryStep.model_validate(task.metadata["current_step"])
        decision = self.policy.decide(failure, task.metadata["attempts"], ml_active=ml_active, current_step=current_step)
        attempt["retry_decision"] = decision.reason
        LOGGER.warning("task %s attempt %d failed (%s): %s", task.task_id, attempt["index"], failure.value, decision.reason)
        if decision.retry:
            if decision.native_fallback:
                task.metadata["native"] = True
                task.metadata["ml_fallback_used"] = True
            trigger = FailureClass.ML_SCF_FAILED.value if decision.native_fallback else failure.value
            save_task(task)
            return self._launch(task, stage=stage, kind=KIND_RETRY, step=decision.step, trigger=trigger)
        final_state = ExecutionStatus.CANCELLED if cancelled else (ExecutionStatus.COMPLETED if state is ExecutionStatus.COMPLETED else ExecutionStatus.FAILED)
        self._write_result(task, output if stage == STAGE_SINGLE_POINT else None, ValidationStatus.INVALID, reasons)
        self._finalize(task, final_state, [decision.reason])
        return task

    def _finalize(self, task: LabelTask, execution: ExecutionStatus, reasons: list[str]) -> None:
        task.execution = execution
        task.validation = ValidationStatus.INVALID
        task.failure_reasons.extend(reason for reason in reasons if reason)
        task.metadata["stage"] = STAGE_DONE
        save_task(task)

    def _hopt_reasons(self, task: LabelTask, output: CP2KOutput | None, attempt_dir: Path) -> list[str]:
        settings = self.backend.config.h_only_opt
        if output is None:
            return ["GEO_OPT output was not produced"]
        reasons: list[str] = []
        if output.aborted:
            reasons.append(f"CP2K aborted: {output.abort_message}")
        if output.scf_not_converged_marker:
            reasons.append("SCF did not converge during the cap relaxation")
        if not output.normal_termination:
            reasons.append("GEO_OPT did not terminate normally")
        if settings.require_converged and not output.geo_opt_completed:
            reasons.append("cap-only GEO_OPT did not converge within max_iter")
        trajectory = attempt_dir / f"{STAGE_HOPT}-pos-1.xyz"
        if reasons:
            return reasons
        if not trajectory.exists():
            return [f"GEO_OPT trajectory {trajectory.name} missing"]
        region = self._region_for_stage(task, STAGE_HOPT)
        symbols, positions = read_last_xyz_frame(trajectory)
        if len(symbols) != region.n_atoms:
            return [f"GEO_OPT trajectory has {len(symbols)} atoms, region has {region.n_atoms}"]
        offset = np.asarray(json.loads((attempt_dir / "geometry.json").read_text())["offset"], dtype=float)
        relaxed = positions - offset
        caps = set(cap_indices(region))
        fixed = [index for index in range(region.n_atoms) if index not in caps]
        drift = float(np.max(np.abs(relaxed[fixed] - region.positions[fixed]))) if fixed else 0.0
        if drift > settings.fixed_atom_tolerance:
            return [f"fixed atoms moved by {drift:.2e} A during the cap-only GEO_OPT"]
        positions_out = region.positions.copy()
        positions_out[sorted(caps)] = relaxed[sorted(caps)]
        metadata = dict(region.metadata)
        metadata["h_caps_relaxed"] = {
            "task_id": task.task_id,
            "max_cap_displacement": float(np.max(np.linalg.norm(positions_out - region.positions, axis=1))),
        }
        relaxed_region = replace(region, positions=positions_out, metadata=metadata)
        write_json_atomic(Path(task.work_dir) / LABELED_REGION_FILE, relaxed_region.to_dict())
        task.metadata["labeled_region_file"] = LABELED_REGION_FILE
        return []

    def _write_result(self, task: LabelTask, output: CP2KOutput | None, validation: ValidationStatus, reasons: list[str]) -> None:
        attempt = task.metadata["attempts"][-1]
        region = self._region_for_stage(task, STAGE_SINGLE_POINT)
        payload = {
            "task_id": task.task_id,
            "region_id": task.region_id,
            "validation": validation.value,
            "validation_reasons": reasons,
            "converged": bool(output is not None and output.scf_converged),
            "energy": None if output is None else output.energy,
            "forces": None if output is None or output.forces is None else output.forces.tolist(),
            "stress": None if output is None or output.stress is None else output.stress.tolist(),
            "scf_steps": None if output is None else output.scf_steps,
            "engine": task.engine,
            "engine_version": None if output is None else output.version,
            "output_file": str(Path(attempt["dir"]) / attempt["output"]),
            "output_sha256": attempt.get("output_sha256"),
            "positions": region.positions.tolist(),
            "caps_relaxed": task.metadata.get("labeled_region_file") is not None,
            "settings_hash": task.settings_hash,
            "provenance": None if output is None else output.provenance(),
            "electronic_state": task.metadata.get("electronic_state"),
            "attempts": len(task.metadata["attempts"]),
            "ml_fallback_used": bool(task.metadata.get("ml_fallback_used")),
        }
        write_json_atomic(Path(task.work_dir) / RESULT_FILE, payload)
        task.metadata["result_file"] = RESULT_FILE

    # -- collect -----------------------------------------------------------------------------------
    def collect(self, task: LabelTask | str) -> LabelResult:
        """The final :class:`LabelResult` (VALID or INVALID with reasons); raises if still running."""

        task_id = task if isinstance(task, str) else task.task_id
        current = self.load(task_id)
        if current.metadata.get("stage") != STAGE_DONE:
            raise LabelingError(f"task {task_id} has not finished (execution {current.execution.value})")
        result_file = current.metadata.get("result_file")
        if not result_file:
            return LabelResult(
                task_id=current.task_id,
                region_id=current.region_id,
                forces=np.zeros((0, 3)),
                energy=None,
                converged=False,
                validation=ValidationStatus.INVALID,
                validation_reasons=list(current.failure_reasons),
                engine=current.engine,
                metadata={"execution": current.execution.value},
            )
        data = json.loads((Path(current.work_dir) / result_file).read_text(encoding="utf-8"))
        forces = np.zeros((0, 3)) if data["forces"] is None else np.asarray(data["forces"], dtype=float)
        validation = ValidationStatus(data["validation"])
        reasons = list(data["validation_reasons"])
        if validation is ValidationStatus.INVALID:
            reasons = reasons + [reason for reason in current.failure_reasons if reason not in reasons]
        return LabelResult(
            task_id=current.task_id,
            region_id=current.region_id,
            forces=forces,
            energy=data["energy"],
            converged=bool(data["converged"]),
            validation=validation,
            validation_reasons=reasons,
            stress=None if data["stress"] is None else np.asarray(data["stress"], dtype=float),
            scf_steps=data["scf_steps"],
            engine=data["engine"],
            engine_version=data["engine_version"],
            output_sha256=data["output_sha256"],
            metadata={
                "execution": current.execution.value,
                "positions": data["positions"],
                "caps_relaxed": data["caps_relaxed"],
                "settings_hash": data["settings_hash"],
                "provenance": data["provenance"],
                "electronic_state": data["electronic_state"],
                "attempts": data["attempts"],
                "ml_fallback_used": data["ml_fallback_used"],
                "output_file": data["output_file"],
                "work_dir": current.work_dir,
            },
        )


__all__ = ["LabelingService", "region_fingerprint"]
