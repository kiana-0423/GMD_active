"""MonitorResult carries per-metric scores (CORE-002) and events their requested context (EVT-002)."""

from __future__ import annotations

import numpy as np
from ase.io import read

from gmd_active_learning.contracts import Frame, MetricStatus
from gmd_active_learning.monitors.online import OnlineMonitorDriver

from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "phal_reference" / "pt111_co_slab" / "input_frames.extxyz"


def test_metric_scores_and_requested_context_are_populated() -> None:
    driver = OnlineMonitorDriver({"stages": {"light_interval": 1}}, {"pre_trigger_frames": 2, "post_trigger_frames": 2})
    events = []
    results = []
    for atoms in read(FIXTURE, index=":"):
        frame = Frame.from_atoms(atoms, run_id="slab")
        result, completed = driver.process(frame)
        results.append(result)
        events.extend(completed)
    tail = driver.finish("end_of_stream")
    if tail is not None:
        events.append(tail)
    scored = [r for r in results if r.metric_scores]
    assert scored, "staged results must expose per-metric scores"
    for result in scored:
        for name, values in result.metric_scores.items():
            assert values.shape == result.atom_scores.shape
            if result.metrics[name].status is not MetricStatus.MEASURED:
                assert not np.any(np.isfinite(values) & (values != 0)), name
    assert events and all(event.pre_frames_requested == 2 and event.post_frames_requested == 2 for event in events)
    assert all(event.summary()["pre_frames_requested"] == 2 for event in events)
