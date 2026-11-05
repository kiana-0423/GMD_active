"""Rolling event buffer (EVT-001; F-07, F-15): ports of PHAL test_rolling_buffer.py plus the fixes."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gmd_active_learning.contracts import stable_id
from gmd_active_learning.contracts.frame import Frame
from gmd_active_learning.core.exceptions import ConfigurationError, MonitorError
from gmd_active_learning.events import EventBuffer, load_events_config


def _frame(step: int, n_atoms: int = 1, run_id: str = "run") -> Frame:
    positions = np.zeros((n_atoms, 3))
    positions[:, 0] = np.arange(n_atoms) + float(step)
    return Frame(run_id, step, np.arange(n_atoms), np.ones(n_atoms, dtype=int), positions, None, (False,) * 3, forces=np.zeros((n_atoms, 3)))


def _buffer(pre: int, post: int, **extra: object) -> EventBuffer:
    return EventBuffer({"pre_trigger_frames": pre, "post_trigger_frames": post, **extra})


def _capture(buffer: EventBuffer, frame: Frame, scores: list[float], reasons: list[str] | None = None, atoms: list[int] | None = None, **kw: object):
    return buffer.capture_event(
        frame, hotspot_atom_ids=atoms or [0], atom_scores=np.asarray(scores), trigger_reasons=reasons or ["force_large"], **kw
    )


def test_push_without_pending_returns_none() -> None:
    assert _buffer(2, 1).push(_frame(0)) is None


def test_insufficient_pre_frames_are_truncated() -> None:
    buffer = _buffer(5, 0)
    buffer.push(_frame(1))
    trigger = _frame(2)
    buffer.push(trigger)
    event = _capture(buffer, trigger, [7.0])
    assert event is not None and [frame.step for frame in event.pre_frames] == [1]
    assert event.complete and event.metadata["pre_frames_requested"] == 5


def test_consecutive_triggers_with_zero_post_window_make_separate_events() -> None:
    buffer = _buffer(1, 0)
    first, second = _frame(1), _frame(2)
    buffer.push(first)
    first_event = _capture(buffer, first, [7.0])
    buffer.push(second)
    second_event = _capture(buffer, second, [8.0])
    assert first_event is not None and second_event is not None
    assert (first_event.trigger_step, second_event.trigger_step) == (1, 2)


def test_overlapping_triggers_merge_without_extending_the_window() -> None:
    buffer = _buffer(1, 3)
    before, first, second, third, fourth = (_frame(step, 4) for step in range(5))
    buffer.push(before)
    assert _capture(buffer, first, [0.0, 4.0, 3.0, 0.0], ["force_large"], [1, 2], metadata={"detector": "first"}) is None
    buffer.push(first)
    assert _capture(buffer, second, [0.0, 2.0, 8.0, 5.0], ["force_large", "committee_deviation"], [2, 3], metadata={"detector": "second"}) is None
    assert buffer.push(second) is None
    assert buffer.push(third) is None
    event = buffer.push(fourth)
    assert event is not None and event.complete
    assert event.trigger_frame.step == 1
    assert [frame.step for frame in event.pre_frames] == [0]
    assert [frame.step for frame in event.post_frames] == [2, 3, 4]
    assert event.hotspot_atom_ids == [1, 2, 3]
    assert event.metadata["atom_scores"] == [0.0, 4.0, 8.0, 5.0]
    assert event.max_score == 8.0
    assert event.trigger_reasons == ["force_large", "committee_deviation"]
    assert event.trigger_steps == [1, 2]
    assert event.metadata["latest_trigger_step"] == 2 and event.metadata["trigger_count"] == 2
    assert event.metadata["trigger_metadata"] == [{"step": 1, "detector": "first"}, {"step": 2, "detector": "second"}]


def test_many_triggers_form_one_event_then_a_new_one() -> None:
    buffer = _buffer(1, 2)
    first, second, third, fourth, fifth = (_frame(step) for step in range(1, 6))
    buffer.push(first)
    _capture(buffer, first, [1.0])
    _capture(buffer, second, [2.0], ["committee_deviation"])
    buffer.push(second)
    completed = buffer.push(third)
    assert completed is not None and completed.trigger_step == 1
    assert [frame.step for frame in completed.post_frames] == [2, 3]
    assert completed.metadata["atom_scores"] == [2.0]
    _capture(buffer, third, [3.0])  # PHAL order: the trigger frame was already pushed
    buffer.push(fourth)
    second_event = buffer.push(fifth)
    assert second_event is not None and second_event.trigger_step == 3
    assert [frame.step for frame in second_event.pre_frames] == [2]
    assert [frame.step for frame in second_event.post_frames] == [4, 5]
    assert buffer.counters["events"] == 2


def test_merge_rejects_a_different_atom_set_and_keeps_the_pending_event() -> None:
    buffer = _buffer(0, 1)
    first = _frame(1)
    buffer.push(first)
    _capture(buffer, first, [1.0])
    with pytest.raises(MonitorError):
        _capture(buffer, _frame(2, n_atoms=2), [1.0, 2.0])
    event = buffer.flush()
    assert event is not None and event.trigger_step == 1


def test_flush_marks_an_unfinished_event_incomplete_f07() -> None:
    buffer = _buffer(1, 3)
    trigger = _frame(10)
    assert _capture(buffer, trigger, [7.0]) is None
    buffer.push(trigger)
    buffer.push(_frame(11))
    event = buffer.flush("stopped")
    assert event is not None and event.trigger_step == 10
    assert [frame.step for frame in event.post_frames] == [11]
    assert event.post_frames_requested == 3
    assert event.complete is False and event.truncation_reason == "stopped"
    assert buffer.flush() is None


def test_maxlen_bounds_history() -> None:
    buffer = _buffer(3, 0, maxlen=2)
    for step in range(4):
        buffer.push(_frame(step))
    trigger = _frame(4)
    buffer.push(trigger)
    event = _capture(buffer, trigger, [7.0])
    assert event is not None and [frame.step for frame in event.pre_frames] == [3]


def test_finished_events_are_not_retained_beyond_recent_summaries_f07() -> None:
    buffer = _buffer(2, 0, retain_recent=3)
    for step in range(50):
        frame = _frame(step)
        _capture(buffer, frame, [7.0])
        buffer.push(frame)
    assert buffer.counters["events"] == 50
    assert len(buffer.recent) == 3 and [item["trigger_step"] for item in buffer.recent] == [47, 48, 49]
    assert len(buffer.history_steps) == 3  # pre_trigger_frames + 1
    assert not hasattr(buffer, "events")


def test_event_ids_are_stable_ids_of_run_and_first_trigger_step_f15() -> None:
    def run() -> list[str]:
        buffer = _buffer(1, 1)
        ids = []
        for step in range(6):
            frame = _frame(step, run_id="run-42")
            if step in {1, 2, 4}:
                _capture(buffer, frame, [7.0])
            event = buffer.push(frame)
            if event is not None:
                ids.append(event.event_id)
        return ids

    first, second = run(), run()
    assert first == second == [stable_id("evt", "run-42", 1), stable_id("evt", "run-42", 4)]


def test_cooldown_suppresses_new_events() -> None:
    buffer = _buffer(0, 1, cooldown_steps=20)
    events = []
    for step in range(0, 80, 10):
        frame = _frame(step)
        _capture(buffer, frame, [7.0])
        event = buffer.push(frame)
        if event is not None:
            events.append(event)
    # event 0 ends at step 10 -> cooldown until 30; event at 30 ends at 40 -> until 60; then 60.
    assert [event.trigger_step for event in events] == [0, 30, 60]
    assert buffer.counters["suppressed_by_cooldown"] == 2  # steps 20 and 50


def test_disabled_events_capture_nothing() -> None:
    buffer = EventBuffer({"enabled": False})
    frame = _frame(1)
    assert _capture(buffer, frame, [7.0]) is None and buffer.push(frame) is None
    assert buffer.history_steps == [] and buffer.flush() is None


def test_state_round_trip_resumes_an_open_event() -> None:
    config = {"pre_trigger_frames": 2, "post_trigger_frames": 3, "cooldown_steps": 5}
    buffer = EventBuffer(config)
    for step in range(3):
        buffer.push(_frame(step, 3))
    trigger = _frame(3, 3)
    _capture(buffer, trigger, [1.0, 9.0, 2.0], atoms=[1])
    buffer.push(trigger)
    buffer.push(_frame(4, 3))
    state = json.loads(json.dumps(buffer.to_state()))
    restored = EventBuffer.from_state(state, config)
    assert restored.pending_event_id == buffer.pending_event_id
    events = [restored.push(_frame(step, 3)) for step in (5, 6)]
    event = events[-1]
    assert event is not None and event.complete
    assert [frame.step for frame in event.frames] == [1, 2, 3, 4, 5, 6]
    np.testing.assert_array_equal(event.trigger_frame.positions, trigger.positions)
    assert restored.to_state()["cooldown_until"] == 11


def test_store_requires_directory() -> None:
    with pytest.raises(ConfigurationError):
        load_events_config({"store": {"enabled": True}})
