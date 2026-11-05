"""Event persistence (EVT-002) and EventRecord state/pickle round trips (CORE-003)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from ase.io import read

from gmd_active_learning.contracts.events import EventRecord
from gmd_active_learning.contracts.frame import Frame
from gmd_active_learning.events import EventBuffer, EventStore, event_from_state, event_to_state, read_event, write_event


def _frame(step: int) -> Frame:
    rng = np.random.default_rng(step)
    return Frame(
        "run-7",
        step,
        np.array([11, 5, 42]),
        np.array([6, 1, 8]),
        rng.uniform(0.0, 9.0, (3, 3)),
        np.diag([9.0, 9.5, 10.0]),
        (True, True, False),
        forces=rng.normal(size=(3, 3)),
        committee_forces=rng.normal(size=(2, 3, 3)),
        time_fs=0.5 * step,
        model_version="model-v3",
        metadata={"source": "test"},
    )


def _event(post: int = 2) -> EventRecord:
    buffer = EventBuffer({"pre_trigger_frames": 2, "post_trigger_frames": post})
    for step in range(3):
        buffer.push(_frame(step))
    trigger = _frame(3)
    buffer.capture_event(trigger, hotspot_atom_ids=[42, 5], atom_scores=np.array([1.0, 7.5, 9.0]), trigger_reasons=["force_burst"])
    buffer.push(trigger)
    event = None
    for step in (4, 5):
        event = buffer.push(_frame(step)) or event
    return event or buffer.flush("end_of_stream")  # type: ignore[return-value]


def test_written_event_round_trips_bit_exactly(tmp_path: Path) -> None:
    event = _event()
    event_dir = write_event(event, tmp_path)
    assert event_dir.name == event.event_id
    payload = json.loads((event_dir / "event.json").read_text(encoding="utf-8"))
    assert payload["frame_steps"] == [1, 2, 3, 4, 5] and payload["n_frames"] == len(event.pre_frames) + 1 + len(event.post_frames)
    assert payload["hotspot_atom_ids"] == [5, 42] and payload["complete"] is True
    restored = read_event(event_dir)
    assert restored.summary() == event.summary()
    for original, copy in zip(event.frames, restored.frames):
        np.testing.assert_array_equal(copy.positions, original.positions)
        np.testing.assert_array_equal(copy.forces, original.forces)
        np.testing.assert_array_equal(copy.committee_forces, original.committee_forces)
        np.testing.assert_array_equal(copy.atom_ids, original.atom_ids)
        assert copy.pbc == original.pbc
    extxyz = read(event_dir / "frames.extxyz", index=":")
    assert len(extxyz) == 5 and extxyz[2].info["step"] == 3
    assert list(extxyz[0].arrays["atom_id"]) == [11, 5, 42]


def test_rewriting_an_event_is_idempotent(tmp_path: Path) -> None:
    event = _event()
    store = EventStore(tmp_path, write_extxyz=False)
    store(event)
    first = (tmp_path / event.event_id / "event.json").read_bytes()
    store(event)
    assert (tmp_path / event.event_id / "event.json").read_bytes() == first
    assert store.event_ids() == [event.event_id]
    assert not (tmp_path / event.event_id / "frames.extxyz").exists()
    assert store.read(event.event_id).summary() == event.summary()
    assert not [path for path in (tmp_path / event.event_id).iterdir() if path.name.startswith(".")]


def test_incomplete_event_keeps_its_truncation_marker(tmp_path: Path) -> None:
    event = _event(post=5)
    assert not event.complete
    restored = read_event(write_event(event, tmp_path))
    assert restored.complete is False and restored.truncation_reason == "end_of_stream"
    assert restored.post_frames_requested == 5


def test_event_state_and_pickle_round_trip() -> None:
    event = _event()
    state = json.loads(json.dumps(event_to_state(event)))
    restored = event_from_state(state)
    assert restored.summary() == event.summary()
    assert restored.trigger_frame.metadata == {"source": "test"}
    np.testing.assert_array_equal(restored.trigger_frame.committee_forces, event.trigger_frame.committee_forces)
    pickled = pickle.loads(pickle.dumps(event))
    assert pickled.summary() == event.summary()
    np.testing.assert_array_equal(pickled.post_frames[-1].positions, event.post_frames[-1].positions)
