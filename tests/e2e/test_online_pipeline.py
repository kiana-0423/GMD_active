"""Online end-to-end protocol, closed (ledger TST-003, AL-005, LAB-007/008 through the workflow):

segmented GMD run (fake GMD executable, real GMD adapter and frame tail) -> OnlineMonitorDriver
(with committee forces from a fake GMD_SGT evaluator) -> event.json + frames.extxyz under the event
id -> real EventRegionExtractor -> candidate selection -> LabelingService with a scripted fake CP2K
and the local scheduler -> polled to COMPLETED, validated and collected -> VersionedDatasetStore ->
training (fake trainer standing in for GMD_SGT) -> gated promotion -> the next GMD segment restarts
from the checkpoint with the promoted model.

PHAL's online e2e stopped at a "prepared" dry-run CP2K job and never polled it (LAB-008); this one
closes the loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from e2e.helpers import fake_cp2k_class, fake_gmd_executable
from fakes.workflow_fakes import FakeTrainer
from gmd_active_learning.active_learning.candidates import CandidateConfig
from gmd_active_learning.adapters.gmd.engine import GMDConfig, GMDEngine, iter_frames
from gmd_active_learning.contracts import Frame, ModelArtifact
from gmd_active_learning.datasets.config import DatasetsConfig
from gmd_active_learning.datasets.store import VersionedDatasetStore
from gmd_active_learning.events.config import EventsConfig
from gmd_active_learning.extraction.config import CappingConfig, EmbeddingConfig, ExtractionConfig
from gmd_active_learning.extraction.dispatch import EventRegionExtractor
from gmd_active_learning.hotspots.config import HotspotsConfig
from gmd_active_learning.labeling.service import LabelingService
from gmd_active_learning.monitors.config import MonitorConfig
from gmd_active_learning.monitors.online import OnlineMonitorDriver
from gmd_active_learning.retraining.registry import PromotionGateConfig
from gmd_active_learning.retraining.triggers import RetrainTriggerConfig
from gmd_active_learning.workflow.assembly import DriverFrameMonitor, ServiceLabeler
from gmd_active_learning.workflow.controller import ActiveLearningController, WorkflowConfig, summarize
from gmd_active_learning.workflow.protocols import TrainOutput

WATER = "6\n7.5 7.5 7.5\nO 1.85 1.91 1.86\nH 2.16 2.53 2.53\nH 2.47 1.16 1.91\nO 4.70 4.20 5.00\nH 4.95 4.80 4.30\nH 5.30 3.50 5.10\n"


class FakeSGTEvaluator:
    """Stands in for GMDSGTBackend.enrich: a two-member committee that disagrees where forces are large."""

    def __init__(self) -> None:
        self.calls = 0

    def enrich(self, frame: Frame, model: ModelArtifact) -> Frame:
        self.calls += 1
        spread = 0.05 * np.abs(frame.forces)
        frame.committee_forces = np.stack([frame.forces + spread, frame.forces - spread])
        return frame


class WaterTrainer(FakeTrainer):
    def train(self, dataset, *, parent, work_dir):  # noqa: ANN001, ANN201
        out = super().train(dataset, parent=parent, work_dir=work_dir)
        return TrainOutput(out.checkpoint, [1, 8], out.cutoff, out.mode, out.summary)


def test_online_loop_polls_the_label_to_completion_and_resumes_md_with_the_new_model(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    FakeCP2K = fake_cp2k_class()  # noqa: N806
    fake_cp2k = FakeCP2K(tmp_path / "fake_cp2k")
    monkeypatch.setenv("FAKE_CP2K_DIR", str(fake_cp2k.state))
    structure = tmp_path / "water.xyz"
    structure.write_text(WATER)
    initial_export = tmp_path / "m0_export.pt"
    initial_export.write_text("initial torchscript")
    from gmd_active_learning.contracts.ids import file_sha256

    md = GMDEngine(GMDConfig(executable=str(fake_gmd_executable(tmp_path / "bin")), initial_structure=str(structure), frame_interval=1,
                             reader_poll_s=0.01, env={"FAKE_GMD_BURST": "6:3:30.0", "FAKE_GMD_BURST_MODEL": file_sha256(initial_export)[:12]}))
    monitor = DriverFrameMonitor(OnlineMonitorDriver(
        MonitorConfig.model_validate({"running_stats": {"warmup_frames": 3}}),
        EventsConfig.model_validate({"pre_trigger_frames": 2, "post_trigger_frames": 2}), HotspotsConfig()))
    extractor = EventRegionExtractor(ExtractionConfig.model_validate({"mode": "cluster", "core_radius": 1.2, "buffer_radius": 1.0,
                                                                      "extract_radius": 3.6, "boundary_thickness": 0.5}),
                                     CappingConfig(), EmbeddingConfig(), state_dir=tmp_path / "run")
    service = LabelingService.from_config({
        "backends": {"dft": {"engine": "cp2k"}, "scheduler": {"engine": "local"}, "discover_entry_points": False},
        "cp2k": {"executable": str(fake_cp2k.executable), "task_dir": "labels"},
        "scheduler": {"command_timeout_seconds": 30},
    }, base_dir=tmp_path / "run")
    evaluator, trainer = FakeSGTEvaluator(), WaterTrainer()
    config = WorkflowConfig(run_id="online", max_cycles=2, segment_steps=12, poll_interval_s=0.05, max_poll_wait_s=60.0,
                            candidates=CandidateConfig(max_per_cycle=4, max_per_event=2, diversity_threshold=0.0),
                            retraining=RetrainTriggerConfig(min_new_samples=1), promotion=PromotionGateConfig(max_force_rmse=0.5))
    controller = ActiveLearningController(config, work_dir=tmp_path / "run", md=md, monitor=monitor, extractor=extractor,
                                          labeler=ServiceLabeler(service), datasets=VersionedDatasetStore(tmp_path / "run" / "datasets", DatasetsConfig()),
                                          trainer=trainer, enricher=evaluator)
    checkpoint = tmp_path / "m0.json"
    checkpoint.write_text(json.dumps({"error_scale": 0.3}))
    m0 = controller.registry.register_external(checkpoint=checkpoint, export=initial_export, species=[1, 8], cutoff=4.0, backend="gmd_sgt")
    controller.initialise(m0)

    assert controller.run() == "done"
    report = summarize(controller.store, "online")
    # Monitoring saw every streamed frame (committee attached) and captured the burst as one event.
    events = controller.store.events("online")
    assert len(events) == 1 and events[0]["summary"]["hotspot_atom_ids"] == [3]
    event_dir = Path(events[0]["path"])
    assert event_dir.name == events[0]["event_id"] and (event_dir / "event.json").is_file()
    assert json.loads((event_dir / "event.json").read_text())["event_id"] == events[0]["event_id"]
    assert len(list(iter_frames(event_dir / "frames.extxyz", run_id="online"))) == 5
    # The CP2K job was prepared, submitted, polled to completion, validated, collected and ingested.
    tasks = controller.store.tasks("online")
    assert tasks and all(row["state"] == "ingested" for row in tasks)
    for row in tasks:
        task = service.load(row["task_id"])
        assert task.execution.value == "completed" and task.validation.value == "valid"
        assert list(Path(task.work_dir).rglob("*.inp")) and list(Path(task.work_dir).rglob("*.out"))
    assert report["samples"] == len(tasks) >= 1 and report["rounds"][0]["n_selected"] == len(tasks)
    # Training, gated promotion, and the next segment restarted from the checkpoint with the new model.
    promoted = [p for p in report["promotions"] if p["action"] == "promote" and p["previous_model_id"] == m0.model_id]
    assert promoted
    segments = report["segments"]
    assert len(segments) == 2 and segments[0]["model_id"] == m0.model_id and segments[1]["model_id"] == promoted[0]["model_id"]
    second = controller.registry.get(promoted[0]["model_id"])
    frames = list(iter_frames(tmp_path / "run" / "md" / "segment_00001" / "frames.extxyz", run_id="online"))
    assert frames[0].step == segments[0]["end_step"] and all(frame.model_sha256 == second.export_sha256 for frame in frames)
    # The restart re-emits its start frame; the controller monitors every step exactly once.
    assert evaluator.calls == 13 + 12
    assert "restart_from" in (tmp_path / "run" / "md" / "segment_00001" / "run.in").read_text()
