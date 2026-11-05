from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gmd_active_learning.adapters.registry import BackendRole, create_backend
from gmd_active_learning.config import build_config
from gmd_active_learning.contracts import ExecutionStatus, Frame, LabelTask
from gmd_active_learning.core.exceptions import CapabilityUnavailableError
from gmd_active_learning.workflow.assembly import DriverFrameMonitor, ServiceLabeler, build_controller, hotspot_grouper
from gmd_active_learning.workflow.protocols import FrameMonitor, Labeler

SGT_PATH = os.environ.get("GMD_SGT_PATH", str(Path(__file__).resolve().parents[3] / "GMD_SGT"))


def _raw(tmp_path: Path) -> dict:
    gmd = tmp_path / "gmd"
    gmd.write_text("#!/bin/sh\nexit 0\n")
    gmd.chmod(0o755)
    (tmp_path / "box.xyz").write_text("3\n8 8 8\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")
    return {
        "workflow": {"run_id": "asm"},
        "md": {"executable": "gmd", "initial_structure": "box.xyz"},
        "mlip": {"source_path": SGT_PATH},
        "cp2k": {"executable": "cp2k.psmp", "task_dir": "cp2k_tasks"},
        "hotspots": {"merge_radius": 1.6},
    }


def test_hotspot_grouper_splits_separate_molecules_under_pbc() -> None:
    positions = np.array([[0.1, 4, 4], [9.9, 4, 4], [5, 5, 5], [5.9, 5, 5]])  # pair 0-1 is 0.2 A apart through x
    frame = Frame(run_id="r", step=0, atom_ids=[10, 11, 12, 13], numbers=[8, 1, 8, 1], positions=positions,
                  cell=np.eye(3) * 10.0, pbc=(True, True, True))
    groups = hotspot_grouper(1.6)(frame, [13, 10, 11, 12])
    assert sorted(sorted(group) for group in groups) == [[10, 11], [12, 13]]


class _FakeDriver:
    def __init__(self) -> None:
        self.state = {"n": 0}

    def process(self, frame: Frame):  # noqa: ANN201
        self.state["n"] += 1
        return SimpleNamespace(action="continue"), [SimpleNamespace(event_id=f"e{frame.step}")]

    def finish(self, reason: str):  # noqa: ANN201
        return SimpleNamespace(event_id="open", reason=reason)

    def to_state(self) -> dict:
        return dict(self.state)

    def load_state(self, state: dict) -> None:
        self.state = dict(state)


def test_driver_bridge_buffers_events_and_state() -> None:
    bridge = DriverFrameMonitor(_FakeDriver())
    assert isinstance(bridge, FrameMonitor)
    frame = Frame(run_id="r", step=5, atom_ids=[0], numbers=[1], positions=np.zeros((1, 3)), cell=None, pbc=(False,) * 3)
    bridge.process(frame)
    assert [event.event_id for event in bridge.drain_events()] == ["e5"]
    assert bridge.drain_events() == []
    assert [event.event_id for event in bridge.close("md_stopped")] == ["open"]
    bridge.load_state({"n": 7})
    assert bridge.to_state() == {"n": 7}


class _FakeService:
    def __init__(self, task: LabelTask) -> None:
        self.task = task

    def load(self, task_id: str) -> LabelTask:
        return self.task

    def poll(self, task_id: str) -> LabelTask:
        return self.task


def _task(stage: str, execution: ExecutionStatus, attempts: int = 1) -> LabelTask:
    return LabelTask("t", "r", "cp2k", "h", "/tmp", execution=execution, metadata={"stage": stage, "attempts": [{}] * attempts})


@pytest.mark.parametrize(
    ("stage", "execution", "expected"),
    [
        ("sp", ExecutionStatus.RUNNING, ExecutionStatus.RUNNING),
        ("hopt", ExecutionStatus.COMPLETED, ExecutionStatus.RUNNING),  # cap relaxation done; single point queued
        ("done", ExecutionStatus.COMPLETED, ExecutionStatus.COMPLETED),
        ("done", ExecutionStatus.FAILED, ExecutionStatus.FAILED),
        ("sp", ExecutionStatus.UNKNOWN, ExecutionStatus.UNKNOWN),
    ],
)
def test_service_bridge_reports_terminal_only_when_the_service_finished(stage, execution, expected) -> None:  # noqa: ANN001
    labeler = ServiceLabeler(_FakeService(_task(stage, execution)))
    assert isinstance(labeler, Labeler)
    assert labeler.poll(_task(stage, execution)).execution is expected
    assert labeler.retry(_task(stage, execution), None) is None  # the service owns the retry ladder


def test_find_submission_recognises_prior_attempts() -> None:
    assert ServiceLabeler(_FakeService(_task("sp", ExecutionStatus.SUBMITTED, attempts=1))).find_submission(_task("sp", ExecutionStatus.PREPARED)) is not None
    assert ServiceLabeler(_FakeService(_task("sp", ExecutionStatus.PREPARED, attempts=0))).find_submission(_task("sp", ExecutionStatus.PREPARED)) is None


def test_registry_builds_md_and_mlip_backends_and_controller(tmp_path: Path) -> None:
    config = build_config(_raw(tmp_path), base_dir=tmp_path)
    md = create_backend(config.raw, BackendRole.MD)
    assert md.name == "gmd" and "cooperative_stop" in md.capabilities
    try:
        controller = build_controller(config, work_dir=tmp_path / "run")
    except CapabilityUnavailableError as exc:
        pytest.skip(f"GMD_SGT unavailable: {exc}")
    assert controller.run_id == "asm"
    assert controller.trainer.name == "gmd_sgt" and controller.enricher is controller.trainer
    with pytest.raises(CapabilityUnavailableError):
        create_backend({**config.raw, "backends": {"md": {"engine": "gmd", "required_capabilities": ["sockets"]}}}, BackendRole.MD)
