"""EVT-002: PHAL-compatible event.json view, and the event files written for the golden events.

The three golden scenarios are replayed through GMD's staged monitor and online driver with PHAL's
scoring conventions. For every event, ``phal_event_payload`` must give PHAL's ``event.json``
(``tests/fixtures/phal_reference/*/events.json``) with identical keys and values. The exception is
``event_id``, which is GMD's stable id (F-15). Floats are compared to fixture precision. The written
event directory holds ``len(pre) + 1 + len(post)`` frames, and positions and forces round-trip
bit-exactly through ``frames.npz``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from gmd_active_learning.events import phal_event_payload, read_event, write_event
from gmd_active_learning.events.phal_view import phal_ood_metadata
from monitors.phal_replay import SCENARIOS, load_scenario
from monitors.test_phal_parity import ATOL, RTOL, _replay

#: keys PHAL wrote (online_monitor.write_event) that the fixture capture script recorded too
PHAL_EVENT_KEYS = {"event_id", "step", "time", "hotspot_atoms", "trigger_reason", "ood_scores", "backend", "model_version", "metadata"}
#: the capture script's own additions (frame steps, checked against the written frames instead)
CAPTURE_ONLY_KEYS = {"pre_frame_steps", "post_frame_steps"}


def _assert_close(actual: object, expected: object, path: str) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and sorted(actual) == sorted(expected), f"{path}: {sorted(actual)} != {sorted(expected)}"
        for key, value in expected.items():
            _assert_close(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_close(left, right, f"{path}[{index}]")
    elif isinstance(expected, float) and not isinstance(expected, bool):
        assert actual == pytest.approx(expected, rel=RTOL, abs=ATOL), path
    else:
        assert actual == expected and type(actual) is type(expected), f"{path}: {actual!r} != {expected!r}"


@pytest.fixture(scope="module", params=SCENARIOS)
def replayed(request: pytest.FixtureRequest):  # noqa: ANN201
    scenario = load_scenario(request.param)
    _evaluations, events, _driver = _replay(scenario, negate_lj=True)
    for event in events:  # the replayed frames carry no model version; PHAL's was the first model file
        event.model_version = scenario.config["allegro"]["model_paths"][0]
    return scenario, events


def test_phal_view_matches_phal_event_json(replayed) -> None:  # noqa: ANN001
    scenario, events = replayed
    assert len(events) == len(scenario.events)
    lazy = scenario.config["ood_score"]["lj_lazy_threshold"]
    for event, reference in zip(events, scenario.events):
        assert set(reference) - CAPTURE_ONLY_KEYS == PHAL_EVENT_KEYS
        payload = json.loads(json.dumps(phal_event_payload(event, backend=reference["backend"], lj_lazy_threshold=lazy)))
        assert set(payload) == PHAL_EVENT_KEYS
        expected = {key: value for key, value in reference.items() if key in PHAL_EVENT_KEYS}
        expected["event_id"] = payload["event_id"] = None  # GMD ids are stable ids (F-15)
        expected["metadata"] = {**expected["metadata"], "event_id": None}
        payload["metadata"]["event_id"] = None
        _assert_close(payload, expected, f"{scenario.name}:{reference['event_id']}")


def test_written_event_frames_match_phal_counts_and_round_trip_bit_exactly(replayed, tmp_path: Path) -> None:  # noqa: ANN001
    scenario, events = replayed
    for event, reference in zip(events, scenario.events):
        directory = write_event(event, tmp_path)
        expected_count = len(reference["pre_frame_steps"]) + 1 + len(reference["post_frame_steps"])
        frames = read(directory / "frames.extxyz", index=":")
        assert len(frames) == expected_count == len(event.frames)
        assert [atoms.info["step"] for atoms in frames] == [*reference["pre_frame_steps"], reference["step"], *reference["post_frame_steps"]]
        restored = read_event(directory)
        for original, copy in zip(event.frames, restored.frames, strict=True):
            assert np.array_equal(original.positions, copy.positions) and original.positions.tobytes() == copy.positions.tobytes()
            assert original.forces.tobytes() == copy.forces.tobytes()


def test_ood_mapping_rules() -> None:
    base = {"label_threshold": 6.0, "physics_threshold": 5.0, "screen_threshold": 4.0}
    assert phal_ood_metadata(base, lj_lazy_threshold=3.0) == {**base, "lj_fit_count": 0, "lj_lazy_threshold": 3.0}
    full = {**base, "scheduled_full": True, "screen_max_score": 1.0, "lj_fit_count": 2}
    assert "screen_max_score" not in phal_ood_metadata(full, lj_lazy_threshold=3.0)  # PHAL's score_full dropped it


def test_view_needs_driver_trigger_metadata() -> None:
    scenario = load_scenario("pt111_co_slab")
    _evaluations, events, _driver = _replay(scenario, negate_lj=True)
    event = events[0]
    event.metadata = {**event.metadata, "trigger_metadata": [{"step": event.trigger_step}]}
    with pytest.raises(KeyError):
        phal_event_payload(event, backend=None, lj_lazy_threshold=3.0)
