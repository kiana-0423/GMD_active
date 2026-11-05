"""Online driver: frame loop, events via callback, actions, errors, timeouts, segmented resume (MON-014, EVT-002)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gmd_active_learning.contracts.events import EventRecord
from gmd_active_learning.contracts.frame import Frame
from gmd_active_learning.contracts.monitoring import FrameAction, Stage
from gmd_active_learning.monitors import POLL_TIMEOUT, OnlineMonitorDriver, StagedOODMonitor

from .frames import CountingCommittee, CountingForces, ethane_box_frames, h4_frame, staged_config

EVENTS_1_1 = {"pre_trigger_frames": 1, "post_trigger_frames": 1, "maxlen": 4}


def test_trigger_produces_one_event_via_callback() -> None:
    events: list[EventRecord] = []
    reference = CountingForces()
    driver = OnlineMonitorDriver(
        staged_config(), EVENTS_1_1, reference_forces=reference, committee_provider=CountingCommittee(), on_event=events.append
    )
    summary = driver.run([h4_frame(0, 2.0), h4_frame(1, 0.1)])
    assert len(summary.results) == 2 and summary.results[0].triggered
    assert len(events) == 1 and summary.events == events
    event = events[0]
    assert event.trigger_frame.step == 0 and [frame.step for frame in event.post_frames] == [1]
    assert event.trigger_steps == [0] and event.hotspot_atom_ids == [0] and event.complete
    assert reference.calls == 1
    assert summary.results[0].action is FrameAction.SAVE
    assert summary.results[0].metadata["hotspots"][0]["atom_ids"] == [0]


def test_progress_callback_and_progress_file(tmp_path: Path) -> None:
    progress: list[dict[str, Any]] = []
    path = tmp_path / "progress.json"
    config = staged_config(driver={"progress_file": str(path)})
    driver = OnlineMonitorDriver(config, EVENTS_1_1, reference_forces=CountingForces(), on_progress=progress.append)
    driver.run([h4_frame(0, 2.0), h4_frame(1, 0.1)])
    assert [item["processed_frames"] for item in progress] == [1, 2]
    assert progress[-1]["triggered_frames"] >= 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["processed_frames"] == 2 and payload["last_stage"] == "light"


def test_continue_after_frame_error_within_budget() -> None:
    progress: list[dict[str, Any]] = []
    config = staged_config(driver={"continue_on_error": True, "max_errors": 2})
    driver = OnlineMonitorDriver(config, EVENTS_1_1, reference_forces=CountingForces(fail_first=True), on_progress=progress.append)
    summary = driver.run([h4_frame(0, 2.0), h4_frame(1, 2.0)])
    assert len(summary.results) == 1 and summary.errors == 1
    assert progress[0]["last_stage"] == "error" and progress[0]["errors"] == 1
    assert progress[-1]["processed_frames"] == 1


def test_error_without_budget_raises_and_flushes_open_event_as_error() -> None:
    class FailOnSecond(CountingForces):
        def __call__(self, frame: Frame) -> np.ndarray:
            if frame.step == 1:
                raise RuntimeError("boom")
            return super().__call__(frame)

    events: list[EventRecord] = []
    config = staged_config(stages={"scheduled_full_interval": 1, "physics_interval": "always"})
    driver = OnlineMonitorDriver(config, {"post_trigger_frames": 3}, reference_forces=FailOnSecond(), on_event=events.append)
    with pytest.raises(RuntimeError, match="boom"):
        driver.run([h4_frame(0, 2.0), h4_frame(1, 2.0)])
    assert len(events) == 1 and not events[0].complete and events[0].truncation_reason == "error"


class ScriptedSource:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.timeouts: list[float | None] = []

    def next_frame(self, timeout: float | None = None) -> Any:
        self.timeouts.append(timeout)
        return self.script.pop(0) if self.script else None


def test_frame_source_distinguishes_timeout_from_end_of_stream() -> None:
    frames = [h4_frame(0, 0.1), h4_frame(1, 0.1)]
    source = ScriptedSource([frames[0], POLL_TIMEOUT, POLL_TIMEOUT, frames[1], None])
    config = staged_config(driver={"max_consecutive_timeouts": 3, "poll_timeout_s": 0.5})
    summary = OnlineMonitorDriver(config, EVENTS_1_1, reference_forces=CountingForces()).run(source)
    assert [result.step for result in summary.results] == [0, 1] and summary.stop_reason == "end_of_stream"
    assert set(source.timeouts) == {0.5}
    stalled = ScriptedSource([frames[0], POLL_TIMEOUT, POLL_TIMEOUT, POLL_TIMEOUT, frames[1]])
    summary = OnlineMonitorDriver(config, EVENTS_1_1, reference_forces=CountingForces()).run(stalled)
    assert [result.step for result in summary.results] == [0] and summary.stop_reason == "timeout"


def test_stop_action_ends_run_and_marks_open_event_stopped() -> None:
    frames = ethane_box_frames(8, burst={4: (3, [40.0, 0.0, 0.0])})
    frames[6].positions[1] = frames[6].positions[0] + [0.4, 0.0, 0.0]
    events: list[EventRecord] = []
    driver = OnlineMonitorDriver(
        {"running_stats": {"warmup_frames": 2}}, {"pre_trigger_frames": 2, "post_trigger_frames": 5}, on_event=events.append
    )
    summary = driver.run(frames)
    assert summary.stop_reason == "stopped" and summary.last_action is FrameAction.STOP
    assert [result.step for result in summary.results][-1] == 60
    first_trigger = next(result.step for result in summary.results if result.triggered)
    assert events[-1].complete is False and events[-1].truncation_reason == "stopped"
    assert events[0].trigger_step == first_trigger and 40 in events[-1].trigger_steps


def test_max_frames_flushes_with_reason() -> None:
    frames = ethane_box_frames(8, burst={4: (3, [40.0, 0.0, 0.0])})
    summary = OnlineMonitorDriver({"running_stats": {"warmup_frames": 2}}, {"post_trigger_frames": 5}).run(frames, max_frames=6)
    assert summary.stop_reason == "max_frames" and summary.events[-1].truncation_reason == "max_frames"


def test_segmented_run_with_persisted_state_matches_continuous_run(tmp_path: Path) -> None:
    frames = ethane_box_frames(14, burst={5: (3, [30.0, 0.0, 0.0]), 9: (12, [0.0, 25.0, 0.0])})
    monitor_cfg = {"running_stats": {"warmup_frames": 2}, "stages": {"scheduled_full_interval": 4}}
    events_cfg = {"pre_trigger_frames": 2, "post_trigger_frames": 3, "cooldown_steps": 15}
    continuous = OnlineMonitorDriver(monitor_cfg, events_cfg, reference_forces=CountingForces()).run(frames)

    probe = OnlineMonitorDriver(monitor_cfg, events_cfg, reference_forces=CountingForces())
    pending_after = []
    for frame in frames:
        probe.process(frame)
        pending_after.append(probe.buffer.has_pending)
    cut = pending_after.index(True) + 1  # cut inside an open event's post window

    segment_events: list[EventRecord] = []
    first = OnlineMonitorDriver(monitor_cfg, events_cfg, reference_forces=CountingForces(), on_event=segment_events.append)
    results = [first.process(frame)[0] for frame in frames[:cut]]
    assert first.buffer.has_pending
    state_path = tmp_path / "monitor_state.json"
    first.save_state(str(state_path))
    second = OnlineMonitorDriver(monitor_cfg, events_cfg, reference_forces=CountingForces(), on_event=segment_events.append)
    second.load_state(json.loads(state_path.read_text(encoding="utf-8")))
    results += [second.process(frame)[0] for frame in frames[cut:]]
    second.finish()

    assert [result.stage for result in results] == [result.stage for result in continuous.results]
    for expected, actual in zip(continuous.results, results):
        np.testing.assert_array_equal(actual.atom_scores, expected.atom_scores)
        assert actual.action is expected.action
    assert [event.summary() for event in segment_events] == [event.summary() for event in continuous.events]
    assert len(continuous.events) >= 2


def test_event_store_writes_each_event(tmp_path: Path) -> None:
    from gmd_active_learning.events import EventStore

    frames = ethane_box_frames(8, burst={4: (3, [40.0, 0.0, 0.0])})
    events_cfg = {"pre_trigger_frames": 1, "post_trigger_frames": 1, "store": {"enabled": True, "directory": str(tmp_path / "events")}}
    summary = OnlineMonitorDriver({"running_stats": {"warmup_frames": 2}}, events_cfg).run(frames)
    store = EventStore(tmp_path / "events")
    assert store.event_ids() == sorted(event.event_id for event in summary.events)


def test_events_and_hotspots_can_be_disabled() -> None:
    frames = ethane_box_frames(8, burst={4: (3, [40.0, 0.0, 0.0])})
    events: list[EventRecord] = []
    summary = OnlineMonitorDriver(
        {"running_stats": {"warmup_frames": 2}}, {"enabled": False}, {"enabled": False}, on_event=events.append
    ).run(frames)
    triggered = [result for result in summary.results if result.triggered]
    assert triggered and not events and not summary.events
    assert all("hotspots" not in result.metadata for result in triggered)


def test_driver_selects_global_lj_engine() -> None:
    frames = ethane_box_frames(3)
    summary = OnlineMonitorDriver({"engine": "global_lj"}).run(frames)
    assert summary.results and all(result.metadata["engine"] == "global_lj" for result in summary.results)
    assert all(result.stage is Stage.FULL for result in summary.results)


def test_injected_monitor_is_used() -> None:
    monitor = StagedOODMonitor({"enabled": False})
    summary = OnlineMonitorDriver({}, monitor=monitor).run(ethane_box_frames(2))
    assert all(result.stage is Stage.SKIPPED for result in summary.results)
