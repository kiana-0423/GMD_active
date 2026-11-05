"""Event -> labeling-task scheduling and per-round summaries (ledger AL-005, AL-006).

PHAL's OnlineEventScheduler was an in-memory FIFO: task_id from the event id (uuid-suffixed, so not
reproducible), status queued/submitted/failed/skipped, no persistence. Here the queue is the
persistent workflow store: task ids are deterministic per region, a raising submitter fails only
that task (PHAL: status "failed" + failed list) and the queue continues, and a controller killed in
the middle of the queue resubmits nothing that was already submitted. PHAL's "skipped" (block
cooldown) happens upstream in extraction (no region, so no task); "queued" is the store's
"selected" candidate state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
from gmd_active_learning.contracts import LabelTask
from gmd_active_learning.retraining.registry import PromotionGateConfig
from gmd_active_learning.retraining.triggers import RetrainTriggerConfig
from gmd_active_learning.workflow.controller import ActiveLearningController, WorkflowConfig, summarize


@dataclass
class QueueLabeler(FakeLabeler):
    """Records every submit call in order; can raise for chosen regions or crash at the n-th submit."""

    fail_regions: frozenset[str] = frozenset()
    crash_at_submission: int | None = None
    calls: list[str] | None = None

    def submit(self, task: LabelTask) -> LabelTask:
        self.calls = [*(self.calls or []), task.task_id]
        if task.region_id in self.fail_regions:
            raise RuntimeError("sbatch: error: invalid partition")
        task = super().submit(task)
        if self.crash_at_submission is not None and len(self.calls) == self.crash_at_submission:
            self.crash_at_submission = None
            raise SimulatedCrash("controller killed in the middle of the submission queue")
        return task


def _controller(root: Path, labeler: FakeLabeler, **overrides: Any) -> ActiveLearningController:
    config = WorkflowConfig(**{
        "run_id": "sched", "max_cycles": 1, "segment_steps": 60, "poll_interval_s": 0.0, "max_poll_wait_s": 10.0,
        "candidates": CandidateConfig(max_per_cycle=10, max_per_event=10, diversity_threshold=0.0),
        "retraining": RetrainTriggerConfig(min_new_samples=1),
        "promotion": PromotionGateConfig(max_force_rmse=0.5),
        **overrides,
    })
    md = FakeMD(anomaly_segments=(0,))
    return ActiveLearningController(config, work_dir=root / "wf", md=md, monitor=FakeMonitor(), extractor=FakeExtractor(),
                                    labeler=labeler, datasets=FakeDatasetStore(root / "datasets"), trainer=FakeTrainer(), sleep=lambda _s: None)


def _initialise(root: Path, controller: ActiveLearningController) -> None:
    ckpt = root / "m0.json"
    ckpt.write_text(json.dumps({"error_scale": 0.3}))
    export = root / "m0_export.json"
    export.write_text(ckpt.read_text())
    controller.initialise(controller.registry.register_external(checkpoint=ckpt, export=export, species=[18], cutoff=6.0, backend="gmd_sgt"))


def _run_to_selection(root: Path, labeler: FakeLabeler) -> list[str]:
    """Selected candidate ids of a fresh run, in queue order (dry pass to learn the queue)."""

    controller = _controller(root, labeler)
    _initialise(root, controller)
    controller.run()
    return [row["task_id"] for row in controller.store.tasks("sched")]


def test_replayed_queue_gives_identical_task_ids_and_transitions(tmp_path: Path) -> None:
    first = _run_to_selection(tmp_path / "a", QueueLabeler(tmp_path / "a" / "dft"))
    second = _run_to_selection(tmp_path / "b", QueueLabeler(tmp_path / "b" / "dft"))
    assert len(first) >= 3, "the fake trajectory must queue several labeling tasks"
    assert first == second  # deterministic ids (PHAL: uuid-suffixed event ids, new task dirs per rerun)


def test_raising_submitter_fails_only_that_task_and_the_queue_continues(tmp_path: Path) -> None:
    probe = QueueLabeler(tmp_path / "probe" / "dft")
    controller = _controller(tmp_path / "probe", probe)
    _initialise(tmp_path / "probe", controller)
    controller.run()
    regions = [row["candidate_id"] for row in controller.store.tasks("sched")]
    failing = regions[0]

    labeler = QueueLabeler(tmp_path / "dft", fail_regions=frozenset({failing}))
    controller = _controller(tmp_path, labeler)
    _initialise(tmp_path, controller)
    assert controller.run() == "done"
    tasks = {row["candidate_id"]: row for row in controller.store.tasks("sched")}
    assert tasks[failing]["state"] == "failed"
    assert any("submission failed: RuntimeError: sbatch: error: invalid partition" in reason for reason in tasks[failing]["failure_reasons"])
    others = [row for cid, row in tasks.items() if cid != failing]
    assert others and all(row["state"] == "ingested" for row in others)
    failed_list = [row["candidate_id"] for row in controller.store.candidates("sched", state="failed")]
    assert failed_list == [failing]


@pytest.mark.parametrize("crash_at", [1, 2, 3])
def test_restart_mid_queue_resubmits_nothing_already_submitted(tmp_path: Path, crash_at: int) -> None:
    labeler = QueueLabeler(tmp_path / "dft", crash_at_submission=crash_at)
    controller = _controller(tmp_path, labeler)
    _initialise(tmp_path, controller)
    with pytest.raises(SimulatedCrash):
        controller.run()
    submitted_before_crash = set(labeler.submissions)
    assert len(submitted_before_crash) == crash_at
    resumed = _controller(tmp_path, labeler)  # a new controller process on the same work dir
    assert resumed.run() == "done"
    assert all(count == 1 for count in labeler.submissions.values()), labeler.submissions
    assert len(labeler.submissions) > crash_at  # the rest of the queue was submitted after the restart
    assert sorted(labeler.calls or []) == sorted(set(labeler.calls or []))  # no task id submitted twice
    tasks = resumed.store.tasks("sched")
    assert all(row["state"] == "ingested" for row in tasks)


def test_round_summary_records_events_candidates_and_selected(tmp_path: Path) -> None:
    labeler = QueueLabeler(tmp_path / "dft")
    controller = _controller(tmp_path, labeler, max_cycles=2,
                             candidates=CandidateConfig(max_per_cycle=2, max_per_event=10, diversity_threshold=0.0))
    _initialise(tmp_path, controller)
    assert controller.run() == "done"
    store = controller.store
    rounds = store.rounds("sched")
    assert [r["round_id"] for r in rounds] == [0, 1]
    first = rounds[0]
    assert first["n_events"] == len([e for e in store.events("sched") if e["cycle"] == 0]) >= 1
    assert first["n_candidates"] == len(store.candidates("sched", cycle=0)) >= 3
    assert first["n_selected"] == 2  # the cycle budget
    assert first["n_rejected"] == first["n_candidates"] - 2 and first["rejected_reasons"] == {"cycle_budget": first["n_rejected"]}
    report = summarize(store, "sched")["rounds"]
    assert {key for key in report[0]} >= {"round_id", "n_events", "n_candidates", "n_selected"}  # PHAL RoundSummary names
    assert report[0]["n_selected"] == 2
