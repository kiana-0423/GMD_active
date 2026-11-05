"""Closed-loop controller tests with deterministic fake backends (crash and failure injection)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fakes.workflow_fakes import (
    FakeDatasetStore,
    FakeExtractor,
    FakeLabeler,
    FakeMD,
    FakeMonitor,
    FakeTrainer,
    SimulatedCrash,
)
from gmd_active_learning.active_learning.candidates import CandidateConfig
from gmd_active_learning.retraining.registry import ModelRegistry, PromotionGateConfig
from gmd_active_learning.retraining.triggers import RetrainTriggerConfig
from gmd_active_learning.workflow.controller import ActiveLearningController, WorkflowConfig, summarize
from gmd_active_learning.workflow.store import WorkflowStore


def _config(**overrides: Any) -> WorkflowConfig:
    data: dict[str, Any] = dict(
        run_id="run-a",
        max_cycles=2,
        segment_steps=60,
        poll_interval_s=0.0,
        max_poll_wait_s=10.0,
        candidates=CandidateConfig(max_per_cycle=4, max_per_event=2, diversity_threshold=0.0),
        retraining=RetrainTriggerConfig(min_new_samples=1),
        promotion=PromotionGateConfig(max_force_rmse=0.5, require_engine_smoke_test=True),
    )
    data.update(overrides)
    return WorkflowConfig(**data)


class Harness:
    def __init__(self, root: Path, **config_overrides: Any) -> None:
        self.root = root
        self.config = _config(**config_overrides)
        self.md = FakeMD()
        self.labeler = FakeLabeler(root / "dft")
        self.datasets = FakeDatasetStore(root / "datasets")
        self.trainer = FakeTrainer()

    def controller(self) -> ActiveLearningController:
        return ActiveLearningController(
            self.config, work_dir=self.root / "wf", md=self.md, monitor=FakeMonitor(), extractor=FakeExtractor(),
            labeler=self.labeler, datasets=self.datasets, trainer=self.trainer, sleep=lambda _s: None,
        )

    def initial_model(self, controller: ActiveLearningController):  # noqa: ANN201
        ckpt = self.root / "m0.json"
        ckpt.write_text(json.dumps({"error_scale": 0.3}))
        export = self.root / "m0_export.json"
        export.write_text(ckpt.read_text())
        artifact = controller.registry.register_external(checkpoint=ckpt, export=export, species=[18], cutoff=6.0, backend="gmd_sgt")
        controller.initialise(artifact)
        return artifact


def test_full_loop_labels_trains_promotes_and_continues_with_new_model(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    controller = harness.controller()
    m0 = harness.initial_model(controller)
    assert controller.run() == "done"
    report = summarize(controller.store, "run-a")
    assert report["events"] >= 1
    assert report["samples"] >= 1 and report["tasks"]["ingested"] == report["samples"]
    promoted = [p for p in report["promotions"] if p["action"] == "promote" and p["previous_model_id"] == m0.model_id]
    assert promoted, report["promotions"]
    new_model = promoted[0]["model_id"]
    # The next MD segment ran with the promoted model and the change was passed to the engine.
    second = [run for run in harness.md.runs if run["model"] == new_model]
    assert second and second[0]["previous_model"] == m0.model_id
    # Provenance: every sample traces to a task, a candidate region, an event and a parent frame.
    store = controller.store
    for sample in store.samples("run-a"):
        task = store.task(sample["task_id"])
        assert task is not None and task["state"] == "ingested"
        region = json.loads((tmp_path / "wf" / "regions" / f"{sample['candidate_id']}.json").read_text())
        assert region["event_id"] == sample["split_group"] and region["model_version"] == m0.model_id
    # The registry holds immutable, verified artifacts.
    ModelRegistry(tmp_path / "wf" / "models", store).verify(new_model)


def test_crash_after_submission_does_not_duplicate_jobs(tmp_path: Path) -> None:
    harness = Harness(tmp_path, max_cycles=1)
    controller = harness.controller()
    harness.initial_model(controller)
    harness.labeler.crash_after_submit = True
    with pytest.raises(SimulatedCrash):
        controller.run()
    resumed = harness.controller()  # new process, same work dir
    assert resumed.run() == "done"
    assert harness.labeler.submissions and all(count == 1 for count in harness.labeler.submissions.values())
    store = resumed.store
    assert len(store.samples("run-a")) == len(store.tasks("run-a", states={"ingested"}))


def test_crash_during_ingestion_does_not_duplicate_samples(tmp_path: Path) -> None:
    harness = Harness(tmp_path, max_cycles=1)
    controller = harness.controller()
    harness.initial_model(controller)
    harness.datasets.crash_after_write = True
    with pytest.raises(SimulatedCrash):
        controller.run()
    resumed = harness.controller()
    assert resumed.run() == "done"
    samples = resumed.store.samples("run-a")
    assert len({row["task_id"] for row in samples}) == len(samples)
    assert len(samples) == len(resumed.store.tasks("run-a", states={"ingested"}))


def test_crash_after_md_segment_replays_frames_without_rerunning_md(tmp_path: Path) -> None:
    harness = Harness(tmp_path, max_cycles=1)
    controller = harness.controller()
    harness.initial_model(controller)
    harness.md.crash_after_segment_files = True
    with pytest.raises(SimulatedCrash):
        controller.run()
    resumed = harness.controller()
    assert resumed.run() == "done"
    assert len(harness.md.runs) == 1  # MD was not run again; its frames were replayed
    assert len(resumed.store.events("run-a")) >= 1


def test_unknown_scheduler_state_is_reconciled_not_resubmitted(tmp_path: Path) -> None:
    harness = Harness(tmp_path, max_cycles=1)
    controller = harness.controller()
    harness.initial_model(controller)
    original_prepare = harness.labeler.prepare

    def prepare(region):  # noqa: ANN001, ANN202
        task = original_prepare(region)
        harness.labeler.unknown_once.add(task.task_id)
        return task

    harness.labeler.prepare = prepare  # type: ignore[method-assign]
    assert controller.run() == "done"
    assert all(count == 1 for count in harness.labeler.submissions.values())


def test_invalid_label_is_retried_then_ingested(tmp_path: Path) -> None:
    harness = Harness(tmp_path, max_cycles=1)
    controller = harness.controller()
    harness.initial_model(controller)

    original_collect = harness.labeler.collect
    harness.labeler.invalid_first_attempt = {"__all__"}

    def collect(task):  # noqa: ANN001, ANN202
        if task.attempt == 0:
            harness.labeler.invalid_first_attempt.add(task.region_id)
        return original_collect(task)

    harness.labeler.collect = collect  # type: ignore[method-assign]
    assert controller.run() == "done"
    ingested = controller.store.tasks("run-a", states={"ingested"})
    assert ingested and all(task["attempt"] == 1 for task in ingested)
    assert all("SCF not converged" in task["failure_reasons"] for task in ingested)


def test_failed_model_is_rejected_and_active_model_kept(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.trainer.bad_model = True
    controller = harness.controller()
    m0 = harness.initial_model(controller)
    assert controller.run() == "done"
    store = controller.store
    assert store.run("run-a").active_model_id == m0.model_id
    rejected = [p for p in store.promotions("run-a") if p["action"] == "reject"]
    assert rejected and "force_rmse" in rejected[0]["reason"]
    assert all(run["model"] == m0.model_id for run in harness.md.runs)


def test_segment_failure_with_new_model_rolls_back(tmp_path: Path) -> None:
    harness = Harness(tmp_path, max_cycles=3)
    controller = harness.controller()
    m0 = harness.initial_model(controller)

    original = harness.md.run_segment

    def run_segment(request, on_frame):  # noqa: ANN001, ANN202
        if request.model.model_id != m0.model_id and not harness.md.fail_segments_with_model:
            harness.md.fail_segments_with_model.add(request.model.model_id)
        return original(request, on_frame)

    harness.md.run_segment = run_segment  # type: ignore[method-assign]
    controller.run()
    actions = [p["action"] for p in controller.store.promotions("run-a")]
    assert "rollback" in actions
    rollback = [p for p in controller.store.promotions("run-a") if p["action"] == "rollback"][0]
    assert rollback["model_id"] == m0.model_id
    # After the rollback MD continued with the previous model.
    failed_model = rollback["previous_model_id"]
    idx = [i for i, run in enumerate(harness.md.runs) if run["model"] == failed_model][0]
    assert harness.md.runs[idx + 1]["model"] == m0.model_id


def test_labels_outstanding_past_poll_budget_wait_and_resume(tmp_path: Path) -> None:
    harness = Harness(tmp_path, max_cycles=1, max_poll_wait_s=0.0)
    harness.labeler.polls_to_complete = 3
    controller = harness.controller()
    harness.initial_model(controller)
    assert controller.run() == "waiting"
    assert controller.run() == "waiting"
    assert harness.controller().run() == "done"
    assert all(count == 1 for count in harness.labeler.submissions.values())


def test_backpressure_pauses_sampling(tmp_path: Path) -> None:
    harness = Harness(tmp_path, max_cycles=1, max_pending_tasks=1, max_poll_wait_s=0.0)
    harness.labeler.polls_to_complete = 1000
    controller = harness.controller()
    harness.initial_model(controller)
    assert controller.run() == "waiting"
    pending = controller.store.tasks("run-a", states={"submitted"})
    assert len(pending) == 1  # bounded queue: never more than max_pending_tasks outstanding
    assert len(controller.store.candidates("run-a", state="selected")) >= 1  # the rest wait their turn


def test_no_training_on_incomplete_labels(tmp_path: Path) -> None:
    harness = Harness(tmp_path, max_cycles=1, max_poll_wait_s=0.0)
    harness.labeler.polls_to_complete = 1000
    controller = harness.controller()
    harness.initial_model(controller)
    controller.run()
    assert harness.trainer.trained == []


def test_stop_request_from_another_process_is_persisted_and_resumable(tmp_path: Path) -> None:
    harness = Harness(tmp_path, max_cycles=1)
    controller = harness.controller()
    harness.initial_model(controller)
    other_process = harness.controller()
    other_process.request_stop()
    assert controller.run() == "stopped"
    assert harness.md.runs == [] or len(harness.md.runs) <= 1
    assert harness.controller().run() == "done"
