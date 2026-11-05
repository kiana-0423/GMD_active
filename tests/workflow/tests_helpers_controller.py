"""Small controller harness shared by workflow tests (fake backends only)."""

from __future__ import annotations

import json
from pathlib import Path

from fakes.workflow_fakes import FakeDatasetStore, FakeExtractor, FakeLabeler, FakeMD, FakeMonitor, FakeTrainer
from gmd_active_learning.active_learning.candidates import CandidateConfig
from gmd_active_learning.retraining.registry import PromotionGateConfig
from gmd_active_learning.retraining.triggers import RetrainTriggerConfig
from gmd_active_learning.workflow.controller import ActiveLearningController, WorkflowConfig
from gmd_active_learning.workflow.store import WorkflowStore


def run_one_cycle(root: Path, *, train: bool) -> tuple[WorkflowStore, list[str]]:
    """One cycle of the fake loop; ``train`` decides whether the trigger can fire."""

    trainer = FakeTrainer()
    config = WorkflowConfig(run_id="trig", max_cycles=1, segment_steps=60, poll_interval_s=0.0, max_poll_wait_s=10.0,
                            candidates=CandidateConfig(max_per_cycle=4, max_per_event=2, diversity_threshold=0.0),
                            retraining=RetrainTriggerConfig(min_new_samples=1 if train else 1000, interval_hours=None),
                            promotion=PromotionGateConfig(max_force_rmse=0.5))
    controller = ActiveLearningController(config, work_dir=root / "wf", md=FakeMD(), monitor=FakeMonitor(), extractor=FakeExtractor(),
                                          labeler=FakeLabeler(root / "dft"), datasets=FakeDatasetStore(root / "datasets"), trainer=trainer,
                                          sleep=lambda _s: None)
    ckpt = root / "m0.json"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text(json.dumps({"error_scale": 0.3}))
    controller.initialise(controller.registry.register_external(checkpoint=ckpt, export=ckpt, species=[18], cutoff=6.0, backend="gmd_sgt"))
    assert controller.run() == "done"
    return controller.store, trainer.trained
