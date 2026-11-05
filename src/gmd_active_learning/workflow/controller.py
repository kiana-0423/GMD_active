"""Persistent closed-loop active-learning controller.

    SAMPLING (segmented MD + online monitoring + event capture)
    -> EXTRACTION -> SELECTION -> SUBMISSION -> POLLING (+ reconciliation, validation, retries)
    -> INGESTION -> TRAINING_DECISION
    -> PUBLICATION -> TRAINING -> EVALUATION (incl. export validation, engine smoke test)
    -> PROMOTION (gates; reject keeps the active model) -> SAMPLING of the next cycle

Every stage is idempotent and the current stage is persisted before work starts, so a crashed
controller resumes where it stopped. External effects follow a write-ahead pattern: the intent is
stored, the action performed, the result stored; on restart an intent without a result is
reconciled against the backend (``Labeler.find_submission``, ``MDEngine.completed_segment``)
instead of being repeated. Deterministic IDs and uniqueness constraints in the store prevent
duplicate events, candidates, jobs, samples, dataset versions and models.
"""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
from ase.io import write as ase_write
from pydantic import BaseModel, ConfigDict, Field

from gmd_active_learning.active_learning.candidates import Candidate, CandidateConfig, CandidateSelector
from gmd_active_learning.contracts import (
    DatasetVersion,
    EventRecord,
    ExecutionStatus,
    ExtractedRegion,
    Frame,
    FrameAction,
    LabelTask,
    ModelArtifact,
    ValidationStatus,
    content_hash,
    stable_id,
)
from gmd_active_learning.core.exceptions import WorkflowError
from gmd_active_learning.retraining.registry import ModelRegistry, PromotionGateConfig, evaluate_gates
from gmd_active_learning.retraining.triggers import RetrainTriggerConfig, evaluate_trigger
from gmd_active_learning.utils.atomic_io import read_json_or_default, write_json_atomic
from gmd_active_learning.utils.logging import get_logger
from gmd_active_learning.workflow.protocols import (
    DatasetStore,
    FrameEnricher,
    FrameMonitor,
    Labeler,
    MDEngine,
    RegionExtractor,
    SegmentRequest,
    SegmentResult,
    Trainer,
)
from gmd_active_learning.workflow.store import WorkflowStore, utc_now

LOGGER = get_logger(__name__)


class Stage(str, Enum):
    SAMPLING = "sampling"
    EXTRACTION = "extraction"
    SELECTION = "selection"
    SUBMISSION = "submission"
    POLLING = "polling"
    INGESTION = "ingestion"
    TRAINING_DECISION = "training_decision"
    PUBLICATION = "publication"
    TRAINING = "training"
    EVALUATION = "evaluation"
    PROMOTION = "promotion"
    WAITING = "waiting"  # labels outstanding past the poll budget; resume later
    DONE = "done"
    STOPPED = "stopped"


ACTIVE_TASK_STATES = {"prepared", "submitting", "submitted", "unknown", "collected"}


class WorkflowConfig(BaseModel):
    """``workflow`` config section."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1)
    max_cycles: int = Field(1, ge=1)
    segment_steps: int = Field(1000, ge=1)
    segments_per_cycle: int = Field(1, ge=1)
    max_pending_tasks: int = Field(50, ge=1, description="bounded labeling queue")
    backpressure: str = Field("pause_sampling", pattern="^(pause_sampling|stop)$")
    stop_md_on_monitor_stop: bool = True
    poll_interval_s: float = Field(5.0, ge=0.0)
    max_poll_wait_s: float = Field(3600.0, ge=0.0)
    rollback_on_segment_failure: bool = True
    save_event_frames: bool = True
    candidates: CandidateConfig = Field(default_factory=CandidateConfig)
    retraining: RetrainTriggerConfig = Field(default_factory=RetrainTriggerConfig)
    promotion: PromotionGateConfig = Field(default_factory=PromotionGateConfig)


class ActiveLearningController:
    def __init__(
        self,
        config: WorkflowConfig,
        *,
        work_dir: str | Path,
        md: MDEngine,
        monitor: FrameMonitor,
        extractor: RegionExtractor,
        labeler: Labeler,
        datasets: DatasetStore,
        trainer: Trainer,
        enricher: FrameEnricher | None = None,
        config_hash: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.store = WorkflowStore(self.work_dir / "workflow.sqlite")
        self.registry = ModelRegistry(self.work_dir / "models", self.store)
        self.md, self.monitor, self.extractor = md, monitor, extractor
        self.labeler, self.datasets, self.trainer, self.enricher = labeler, datasets, trainer, enricher
        self.selector = CandidateSelector(config.candidates)
        self.run_id = config.run_id
        self.config_hash = config_hash or content_hash(config.model_dump(mode="json"))
        self._sleep, self._clock = sleep, clock
        self._stop_requested = False
        self._pending_events: dict[str, EventRecord] = {}

    # ================================================================== public API
    def initialise(self, initial_model: ModelArtifact) -> None:
        """Create the run with an already registered initial model (promoted as the initial model)."""

        run = self.store.ensure_run(self.run_id, config_hash=self.config_hash, initial_model_id=None, stage=Stage.SAMPLING.value)
        if run.active_model_id is None:
            self.registry.promote(self.run_id, initial_model.model_id, gate_report=None, reason="initial model", force_initial=True)

    def request_stop(self) -> None:
        """Stop at the next stage boundary (and ask MD to stop at its next control boundary).

        Persisted, so a request made from another process (``gmd-al stop``) is honoured too."""

        self._stop_requested = True
        self.store.put_blob(f"{self.run_id}:stop_requested", True)

    def _stop_pending(self) -> bool:
        return self._stop_requested or bool(self.store.get_blob(f"{self.run_id}:stop_requested", False))

    def run(self, *, max_stage_steps: int = 10_000, resume_stopped: bool = True) -> str:
        """Advance the state machine until DONE/STOPPED/WAITING; returns the final stage."""

        run = self.store.ensure_run(self.run_id, config_hash=self.config_hash, initial_model_id=None, stage=Stage.SAMPLING.value)
        if run.active_model_id is None:
            raise WorkflowError("call initialise(initial_model) before run()")
        self._restore_component_state()
        if run.stage == Stage.WAITING.value:
            self._transition(Stage.POLLING, detail={"resumed": True})
        elif run.stage == Stage.STOPPED.value and resume_stopped:
            last = self.store.transitions(self.run_id)[-1]["detail"].get("resume_stage", Stage.SAMPLING.value)
            self._transition(Stage(last), detail={"resumed_after_stop": True})
        for _ in range(max_stage_steps):
            stage = Stage(self.store.run(self.run_id).stage)
            if stage in {Stage.DONE, Stage.STOPPED, Stage.WAITING}:
                return stage.value
            if self._stop_pending():
                self.store.put_blob(f"{self.run_id}:stop_requested", False)
                self._stop_requested = False
                self._transition(Stage.STOPPED, detail={"reason": "stop requested", "resume_stage": stage.value})
                return Stage.STOPPED.value
            getattr(self, f"_stage_{stage.value}")()
        raise WorkflowError("state machine did not settle; possible livelock")

    # ================================================================== helpers
    @property
    def _run(self):  # noqa: ANN202 - RunRecord
        return self.store.run(self.run_id)

    def _transition(self, stage: Stage, *, cycle: int | None = None, detail: dict[str, Any] | None = None, **fields: Any) -> None:
        self.store.transition(self.run_id, stage.value, cycle=cycle, detail=detail, **fields)

    def _restore_component_state(self) -> None:
        state = self.store.get_blob(f"{self.run_id}:monitor")
        if state is not None:
            self.monitor.load_state(state)

    def _save_component_state(self) -> None:
        self.store.put_blob(f"{self.run_id}:monitor", self.monitor.to_state())

    def _active_model(self) -> ModelArtifact:
        model_id = self._run.active_model_id
        if model_id is None:
            raise WorkflowError("no active model")
        return self.registry.get(model_id)

    def _pending_task_count(self) -> int:
        return len(self.store.tasks(self.run_id, states=ACTIVE_TASK_STATES))

    def _region_path(self, region_id: str) -> Path:
        return self.work_dir / "regions" / f"{region_id}.json"

    def _load_region(self, region_id: str) -> ExtractedRegion:
        return ExtractedRegion.from_dict(read_json_or_default(self._region_path(region_id), None) or {})

    # ================================================================== SAMPLING
    def _stage_sampling(self) -> None:
        run = self._run
        if self._pending_task_count() >= self.config.max_pending_tasks:
            if self.config.backpressure == "stop":
                self._transition(Stage.STOPPED, detail={"reason": "labeling queue full"})
            else:
                LOGGER.info("labeling queue full (%d); pausing sampling", self._pending_task_count())
                self._transition(Stage.POLLING, detail={"reason": "backpressure"})
            return
        segments = self.store.segments(self.run_id)
        in_cycle = [seg for seg in segments if seg["cycle"] == run.cycle and seg["status"] in {"completed", "stopped"}]
        if len(in_cycle) >= self.config.segments_per_cycle or (in_cycle and in_cycle[-1]["status"] == "stopped"):
            self._transition(Stage.EXTRACTION)
            return
        index = len([seg for seg in segments if seg["status"] in {"completed", "stopped"}])
        model = self._active_model()
        previous_segment = next((seg for seg in reversed(segments) if seg["status"] in {"completed", "stopped"}), None)
        previous_model = None
        if previous_segment is not None and previous_segment["model_id"] != model.model_id:
            previous_model = self.registry.get(previous_segment["model_id"])
        request = SegmentRequest(
            run_id=self.run_id,
            segment_index=index,
            work_dir=self.work_dir / "md" / f"segment_{index:05d}",
            model=model,
            n_steps=self.config.segment_steps,
            restart_from=None if run.last_checkpoint is None else Path(run.last_checkpoint),
            previous_model=previous_model,
        )
        self.store.upsert_segment(self.run_id, index, cycle=run.cycle, model_id=model.model_id, status="running", work_dir=str(request.work_dir), start_step=run.last_step)

        result = self.md.completed_segment(request)
        if result is not None:  # finished before a controller crash: replay its frames
            LOGGER.info("segment %d already finished; replaying its frames", index)
            for frame in self.md.read_frames(result, request):
                self._on_frame(frame, model)
        else:
            result = self.md.run_segment(request, lambda frame: self._on_frame(frame, model))
        self._finish_segment(index, request, result, model)

    def _on_frame(self, frame: Frame, model: ModelArtifact) -> FrameAction:
        if self.enricher is not None:
            frame = self.enricher.enrich(frame, model)
        result = self.monitor.process(frame)
        self._record_events(self.monitor.drain_events())
        if self._stop_pending():
            return FrameAction.STOP
        if result.action is FrameAction.STOP and not self.config.stop_md_on_monitor_stop:
            return FrameAction.SAVE
        return result.action

    def _record_events(self, events: list[EventRecord]) -> None:
        cycle = self._run.cycle
        for event in events:
            path = None
            if self.config.save_event_frames:
                event_dir = self.work_dir / "events" / event.event_id
                event_dir.mkdir(parents=True, exist_ok=True)
                ase_write(event_dir / "frames.extxyz", [frame.to_atoms() for frame in event.frames], format="extxyz")
                write_json_atomic(event_dir / "event.json", event.summary())
                path = str(event_dir)
            # Keep the full record for extraction (frames are needed); kept in memory and on disk.
            self._pending_events[event.event_id] = event
            self.store.record_event(self.run_id, cycle, event.summary(), path=path)


    def _finish_segment(self, index: int, request: SegmentRequest, result: SegmentResult, model: ModelArtifact) -> None:
        if result.status == "failed":
            self.store.upsert_segment(self.run_id, index, status="failed", detail=result.detail)
            promoted_this_run = [p for p in self.store.promotions(self.run_id) if p["model_id"] == model.model_id and p["action"] == "promote" and p["previous_model_id"]]
            if self.config.rollback_on_segment_failure and promoted_this_run:
                self.registry.rollback(self.run_id, reason=f"segment {index} failed with newly promoted model: {result.detail}")
                return  # SAMPLING again, now with the previous model
            self._transition(Stage.STOPPED, detail={"reason": "md segment failed", "segment": index, **result.detail})
            return
        if result.status == "stopped":
            self._record_events(self.monitor.close("md_stopped"))
        self._save_component_state()
        self.store.upsert_segment(
            self.run_id, index, status=result.status, end_step=result.end_step,
            checkpoint=None if result.checkpoint is None else str(result.checkpoint), detail=result.detail,
        )
        self.store.update_run(self.run_id, last_checkpoint=None if result.checkpoint is None else str(result.checkpoint), last_step=result.end_step)

    # ================================================================== EXTRACTION / SELECTION
    def _event_record(self, row: dict[str, Any]) -> EventRecord:
        event = self._pending_events.get(row["event_id"])
        if event is not None:
            return event
        if row["path"] is None:
            raise WorkflowError(f"event {row['event_id']} frames were not saved; cannot extract after restart")
        from ase.io import read

        summary = row["summary"]
        frames = [Frame.from_atoms(atoms, run_id=self.run_id) for atoms in read(Path(row["path"]) / "frames.extxyz", index=":")]
        n_pre = len(summary["pre_frame_steps"])
        return EventRecord(
            event_id=summary["event_id"], run_id=self.run_id, trigger_step=summary["trigger_step"], trigger_steps=summary["trigger_steps"],
            pre_frames=frames[:n_pre], trigger_frame=frames[n_pre], post_frames=frames[n_pre + 1 :],
            post_frames_requested=summary["post_frames_requested"], complete=summary["complete"],
            hotspot_atom_ids=summary["hotspot_atom_ids"], trigger_reasons=summary["trigger_reasons"], max_score=summary["max_score"],
            truncation_reason=summary["truncation_reason"], model_version=summary["model_version"], metadata=summary["metadata"],
        )

    def _stage_extraction(self) -> None:
        cycle = self._run.cycle
        for row in self.store.events(self.run_id, state="captured"):
            event = self._event_record(row)
            try:
                regions = self.extractor.extract(event)
            except Exception as exc:  # noqa: BLE001 - recorded, extraction of other events continues
                LOGGER.warning("extraction failed for %s: %s", event.event_id, exc)
                self.store.set_event_state(event.event_id, "extraction_failed")
                continue
            for region, score in regions:
                write_json_atomic(self._region_path(region.region_id), region.to_dict())
                candidate = self.selector.make_candidate(region, score=score)
                self.store.record_candidate(
                    self.run_id, cycle, candidate_id=region.region_id, event_id=event.event_id, score=score,
                    region_path=str(self._region_path(region.region_id)), fingerprint=candidate.fingerprint.tolist(),
                )
            self.store.set_event_state(event.event_id, "extracted")
        self._transition(Stage.SELECTION)

    def _stage_selection(self) -> None:
        order = "insertion" if self.config.candidates.tie_break == "insertion" else "score"
        pending = self.store.candidates(self.run_id, state="pending", order=order)
        labeled_states = {"selected", "labeled"}
        history = [np.asarray(row["fingerprint"]) for row in self.store.candidates(self.run_id) if row["state"] in labeled_states]
        total = len([row for row in self.store.candidates(self.run_id) if row["state"] in labeled_states])
        candidates = [
            Candidate(region=self._load_region(row["candidate_id"]), score=row["score"], fingerprint=np.asarray(row["fingerprint"]), event_id=row["event_id"])
            for row in pending
        ]
        decision = self.selector.select(candidates, labeled_fingerprints=history, already_labeled_total=total)
        for candidate in decision.selected:
            self.store.set_candidate_state(candidate.candidate_id, "selected")
        for candidate_id, reason in decision.rejected.items():
            self.store.set_candidate_state(candidate_id, "rejected", reason)
        cycle = self._run.cycle
        reasons: dict[str, int] = {}
        for reason in decision.rejected.values():
            reasons[reason] = reasons.get(reason, 0) + 1
        self.store.record_round(
            self.run_id, cycle, n_events=len([row for row in self.store.events(self.run_id) if row["cycle"] == cycle]),
            n_candidates=len(candidates), n_selected=len(decision.selected), rejected_reasons=reasons,
        )
        self._transition(Stage.SUBMISSION, detail={"selected": len(decision.selected), "rejected": len(decision.rejected)})

    # ================================================================== SUBMISSION / POLLING / INGESTION
    def _task_from_row(self, row: dict[str, Any]) -> LabelTask:
        from gmd_active_learning.contracts import SchedulerJob

        job = None if row["job"] is None else SchedulerJob(**{**row["job"], "state": ExecutionStatus(row["job"]["state"])})
        return LabelTask(
            task_id=row["task_id"], region_id=row["candidate_id"], engine=(row["job"] or {}).get("backend", "dft"), settings_hash="",
            work_dir=row["work_dir"], attempt=row["attempt"], execution=ExecutionStatus(row["execution"] or "prepared"),
            validation=ValidationStatus(row["validation"] or "pending"), job=job, failure_reasons=row["failure_reasons"],
        )

    def _save_task(self, task: LabelTask, state: str, *, candidate_id: str | None = None) -> None:
        job = None
        if task.job is not None:
            job = {"job_id": task.job.job_id, "backend": task.job.backend, "state": task.job.state.value, "work_dir": task.job.work_dir,
                   "submitted_at": task.job.submitted_at, "exit_code": task.job.exit_code, "raw_state": task.job.raw_state, "metadata": task.job.metadata}
        fields: dict[str, Any] = dict(state=state, attempt=task.attempt, execution=task.execution.value, validation=task.validation.value,
                                      job=job, work_dir=task.work_dir, failure_reasons=task.failure_reasons)
        if candidate_id is not None:
            fields.update(candidate_id=candidate_id, run_id=self.run_id, cycle=self._run.cycle)
        self.store.upsert_task(task.task_id, **fields)

    def _stage_submission(self) -> None:
        for row in self.store.candidates(self.run_id, state="selected"):
            if self._pending_task_count() >= self.config.max_pending_tasks:
                LOGGER.info("labeling queue full; remaining selections wait")
                break
            region = self._load_region(row["candidate_id"])
            task = self.labeler.prepare(region)
            existing = self.store.task(task.task_id)
            if existing is not None and existing["state"] != "submitting":
                continue  # already submitted or finished: never duplicate
            # write-ahead: record the intent, then submit, then record the job
            self._save_task(task, "submitting", candidate_id=row["candidate_id"])
            try:
                found = self.labeler.find_submission(task)
                task = found if found is not None else self.labeler.submit(task)
            except Exception as exc:  # noqa: BLE001 - PHAL OnlineEventScheduler: a raising submitter fails the task, the queue continues
                LOGGER.warning("submission of %s failed: %s", task.task_id, exc)
                task.failure_reasons = [*task.failure_reasons, f"submission failed: {type(exc).__name__}: {exc}"]
                self._save_task(task, "failed")
                self.store.set_candidate_state(row["candidate_id"], "failed", task.failure_reasons[-1][-500:])
                continue
            self._save_task(task, "submitted")
            self.store.set_candidate_state(row["candidate_id"], "labeling")
        self._transition(Stage.POLLING)

    def _stage_polling(self) -> None:
        deadline = self._clock() + self.config.max_poll_wait_s
        while True:
            # recover submissions interrupted by a crash
            for row in self.store.tasks(self.run_id, states={"submitting"}):
                task = self._task_from_row(row)
                found = self.labeler.find_submission(task)
                task = found if found is not None else self.labeler.submit(task)
                self._save_task(task, "submitted")
            for row in self.store.tasks(self.run_id, states={"submitted", "unknown"}):
                self._advance_task(self._task_from_row(row))
            if not self.store.tasks(self.run_id, states={"submitted", "unknown", "submitting"}):
                self._transition(Stage.INGESTION)
                return
            if self._clock() >= deadline or self._stop_pending():
                self._transition(Stage.WAITING, detail={"pending": self._pending_task_count()})
                return
            self._sleep(self.config.poll_interval_s)

    def _advance_task(self, task: LabelTask) -> None:
        task = self.labeler.poll(task)
        if task.execution is ExecutionStatus.UNKNOWN:
            task = self.labeler.reconcile(task)
            if task.execution is ExecutionStatus.UNKNOWN:
                self._save_task(task, "unknown")  # never resubmit on an unknown state
                return
        if not task.execution.terminal:
            self._save_task(task, "submitted")
            return
        result = None
        if task.execution is ExecutionStatus.COMPLETED:
            result = self.labeler.collect(task)
            if result.validation is ValidationStatus.VALID:
                task.validation = ValidationStatus.VALID
                self._write_label(task, result)
                self._save_task(task, "collected")
                return
            task.validation = ValidationStatus.INVALID
            task.failure_reasons = [*task.failure_reasons, *result.validation_reasons]
        else:
            task.failure_reasons = [*task.failure_reasons, f"execution {task.execution.value}"]
        retry = self.labeler.retry(task, result)
        if retry is not None:
            retry = self.labeler.submit(retry)
            self._save_task(retry, "submitted")
            return
        final_state = "invalid" if task.execution is ExecutionStatus.COMPLETED else "failed"
        self._save_task(task, final_state)
        self.store.set_candidate_state(task.region_id, final_state, "; ".join(task.failure_reasons)[-500:])

    def _label_path(self, task_id: str) -> Path:
        return self.work_dir / "labels" / f"{task_id}.json"

    def _write_label(self, task: LabelTask, result: Any) -> None:
        write_json_atomic(
            self._label_path(task.task_id),
            {
                "task_id": result.task_id, "region_id": result.region_id, "forces": np.asarray(result.forces).tolist(), "energy": result.energy,
                "converged": result.converged, "validation": result.validation.value, "validation_reasons": result.validation_reasons,
                "stress": None if result.stress is None else np.asarray(result.stress).tolist(), "scf_steps": result.scf_steps,
                "engine": result.engine, "engine_version": result.engine_version, "output_sha256": result.output_sha256, "metadata": result.metadata,
            },
        )

    def _read_label(self, task_id: str) -> Any:
        from gmd_active_learning.contracts import LabelResult

        data = read_json_or_default(self._label_path(task_id), None)
        if data is None:
            raise WorkflowError(f"collected label for {task_id} is missing")
        data["validation"] = ValidationStatus(data["validation"])
        return LabelResult(**data)

    def _stage_ingestion(self) -> None:
        cycle = self._run.cycle
        for row in self.store.tasks(self.run_id, states={"collected"}):
            region = self._load_region(row["candidate_id"])
            label = self._read_label(row["task_id"])
            if label.forces.shape != (region.n_atoms, 3):
                self._save_task(self._task_from_row(row), "invalid")
                self.store.set_candidate_state(row["candidate_id"], "invalid", "label atom count does not match region")
                continue
            provenance = {"task_id": row["task_id"], "run_id": self.run_id, "cycle": cycle, "model_version": region.model_version,
                          "event_id": region.event_id, "parent_step": region.parent_step}
            sample_id, path = self.datasets.add_sample(region, label, provenance=provenance)
            group = region.event_id or f"{region.run_id}:{region.parent_step}"
            self.store.record_sample(sample_id=sample_id, task_id=row["task_id"], candidate_id=row["candidate_id"], run_id=self.run_id, cycle=cycle, path=str(path), split_group=group)
            self.store.upsert_task(row["task_id"], state="ingested")
            self.store.set_candidate_state(row["candidate_id"], "labeled")
        self._transition(Stage.TRAINING_DECISION)

    # ================================================================== TRAINING
    def _stage_training_decision(self) -> None:
        new_samples = len(self.store.samples(self.run_id, unpublished_only=True))
        decision = evaluate_trigger(
            self.config.retraining, new_samples=new_samples, pending_tasks=self._pending_task_count(),
            last_trained_at=self.store.get_blob(f"{self.run_id}:last_trained_at"),
            manual=bool(self.store.get_blob(f"{self.run_id}:manual_retrain", False)),
        )
        LOGGER.info("training decision: %s (%d new samples)", decision.reason, new_samples)
        if decision.due:
            self._transition(Stage.PUBLICATION, detail={"trigger": decision.reason})
        else:
            self._next_cycle(reason=f"no training: {decision.reason}")

    def request_retraining(self) -> None:
        self.store.put_blob(f"{self.run_id}:manual_retrain", True)

    def _stage_publication(self) -> None:
        run = self._run
        parent = None if run.active_dataset_version is None else self.datasets.get(run.active_dataset_version)
        samples = self.store.samples(self.run_id)
        unpublished = [row for row in samples if row["dataset_version"] is None]
        if not unpublished:
            raise WorkflowError("publication with no new samples")
        version = self.datasets.publish(
            [Path(row["path"]) for row in unpublished], parent=parent,
            split_groups={row["sample_id"]: row["split_group"] for row in samples},
        )
        self.store.record_dataset(self.run_id, run.cycle, version.to_dict())
        self.store.mark_samples_published([row["sample_id"] for row in unpublished], version.version)
        self.store.put_blob(f"{self.run_id}:candidate_dataset", version.version)
        self._transition(Stage.TRAINING, detail={"dataset": version.version, "n_structures": version.n_structures})

    def _candidate_dataset(self) -> DatasetVersion:
        version = self.store.get_blob(f"{self.run_id}:candidate_dataset")
        if version is None:
            raise WorkflowError("no published dataset for training")
        return self.datasets.get(version)

    def _stage_training(self) -> None:
        dataset = self._candidate_dataset()
        parent = self._active_model()
        work_dir = self.work_dir / "training" / stable_id("train", dataset.version, parent.model_id)
        output = self.trainer.train(dataset, parent=parent, work_dir=work_dir)
        if not output.checkpoint.is_file():
            raise WorkflowError(f"trainer reported {output.checkpoint}, which does not exist")
        self.store.put_blob(f"{self.run_id}:training_output", {
            "checkpoint": str(output.checkpoint), "species": output.species, "cutoff": output.cutoff,
            "mode": output.mode, "summary": output.summary, "work_dir": str(work_dir),
        })
        self.store.put_blob(f"{self.run_id}:last_trained_at", utc_now())
        self.store.put_blob(f"{self.run_id}:manual_retrain", False)
        self._transition(Stage.EVALUATION)

    def _stage_evaluation(self) -> None:
        dataset = self._candidate_dataset()
        output = self.store.get_blob(f"{self.run_id}:training_output")
        checkpoint, work_dir = Path(output["checkpoint"]), Path(output["work_dir"])
        parent = self._active_model()
        metrics = self.trainer.evaluate(checkpoint, dataset)
        parent_metrics = self.trainer.evaluate(Path(parent.checkpoint_path), dataset)
        export = self.trainer.export(checkpoint, work_dir / "export")
        export_check = self.trainer.validate_export(checkpoint, export, dataset)
        smoke = self.trainer.engine_smoke_test(export, work_dir / "engine_smoke") if self.config.promotion.require_engine_smoke_test else None
        artifact = self.registry.register(
            checkpoint=checkpoint, export=export, dataset_version=dataset.version, parent_model_id=parent.model_id,
            species=output["species"], cutoff=output["cutoff"], metrics=metrics, train_mode=output["mode"], backend="gmd_sgt",
            metadata={"training_summary": output["summary"], "parent_metrics_on_this_dataset": parent_metrics},
            extra_files={f"committee_{k + 1}": path for k, path in enumerate(output["summary"].get("committee_checkpoints") or [])},
        )
        report = evaluate_gates(self.config.promotion, metrics=metrics, parent_metrics=parent_metrics, export_check=export_check, engine_smoke=smoke)
        self.store.put_blob(f"{self.run_id}:gate_report", {"model_id": artifact.model_id, **report.to_dict(), "export_check": export_check, "engine_smoke": smoke})
        self._transition(Stage.PROMOTION, detail={"model_id": artifact.model_id, "passed": report.passed})

    def _stage_promotion(self) -> None:
        report = self.store.get_blob(f"{self.run_id}:gate_report")
        model_id = report["model_id"]
        dataset = self._candidate_dataset()
        state = self.registry.state(model_id)
        if state == "registered":
            from gmd_active_learning.retraining.registry import GateReport

            gate = GateReport(report["passed"], report["checks"])
            if gate.passed:
                self.registry.promote(self.run_id, model_id, gate_report=gate, reason=f"cycle {self._run.cycle} gates passed")
                self.store.update_run(self.run_id, active_dataset_version=dataset.version)
            else:
                failed = sorted(name for name, check in gate.checks.items() if not check["passed"])
                self.registry.reject(self.run_id, model_id, reason=f"failed gates: {failed}")
        self._next_cycle(reason=f"model {model_id} {self.registry.state(model_id)}")

    def _next_cycle(self, *, reason: str) -> None:
        run = self._run
        if run.cycle + 1 >= self.config.max_cycles:
            self._transition(Stage.DONE, cycle=run.cycle, detail={"reason": reason})
        else:
            self._transition(Stage.SAMPLING, cycle=run.cycle + 1, detail={"reason": reason})


def summarize(store: WorkflowStore, run_id: str) -> dict[str, Any]:
    """Machine-readable run report (used by the CLI and tests)."""

    run = store.run(run_id)
    return {
        "run_id": run_id,
        "stage": run.stage,
        "cycle": run.cycle,
        "active_model_id": run.active_model_id,
        "active_dataset_version": run.active_dataset_version,
        "segments": [{k: seg[k] for k in ("segment_index", "cycle", "model_id", "status", "end_step")} for seg in store.segments(run_id)],
        "events": len(store.events(run_id)),
        "candidates": {state: len(store.candidates(run_id, state=state)) for state in ("pending", "selected", "labeling", "labeled", "rejected", "invalid", "failed")},
        "tasks": {state: len(store.tasks(run_id, states={state})) for state in ("submitting", "submitted", "unknown", "collected", "ingested", "invalid", "failed")},
        "samples": len(store.samples(run_id)),
        "models": [{"model_id": m["model_id"], "state": m["state"], "metrics": m["metrics"]} for m in store.models()],
        "promotions": [{k: p[k] for k in ("model_id", "previous_model_id", "action", "reason")} for p in store.promotions(run_id)],
        "rounds": [{k: r[k] for k in ("round_id", "n_events", "n_candidates", "n_selected", "n_rejected", "rejected_reasons")} for r in store.rounds(run_id)],
    }
