"""Staged light -> physics -> full monitor (MON-012, MON-013), feature switches, actions and state."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from gmd_active_learning.contracts.frame import Frame
from gmd_active_learning.contracts.monitoring import FrameAction, MetricStatus, Stage
from gmd_active_learning.core.exceptions import MonitorError
from gmd_active_learning.monitors import (
    LocalLJEstimator,
    StagedOODMonitor,
    build_frame_monitor,
    interval_due,
    load_monitor_config,
)

from .frames import CountingCommittee, CountingForces, ethane_box_frames, h4_frame, staged_config


class CountingLJ:
    def __init__(self, cutoff: float = 4.0) -> None:
        self.calls = 0
        self.estimator = LocalLJEstimator(cutoff=cutoff)

    def __call__(self, pairs: Any, forces: np.ndarray, mask: np.ndarray | None) -> Any:
        self.calls += 1
        return self.estimator(pairs, forces, mask)


def _run(monitor: StagedOODMonitor, frames: list[Frame]) -> list[Any]:
    return [monitor.evaluate(frame) for frame in frames]


# -- interval rules (PHAL _interval_due) ---------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "index", "triggered", "expected"),
    [
        ("triggered", 3, True, True),
        ("triggered", 3, False, False),
        ("never", 0, True, False),
        ("off", 0, True, False),
        ("always", 7, False, True),
        (3, 6, False, True),
        (3, 7, True, False),
        (0, 0, True, False),
    ],
)
def test_interval_due(spec: Any, index: int, triggered: bool, expected: bool) -> None:
    assert interval_due(spec, index, triggered=triggered) is expected


def test_invalid_interval_string_is_a_config_error_not_triggered() -> None:
    from gmd_active_learning.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        load_monitor_config({"stages": {"physics_interval": "sometimes"}})
    assert load_monitor_config({"stages": {"full_interval": "4"}}).stages.full_interval == 4


# -- cascade (ports of PHAL test_online_monitor.py) ----------------------------------------------


def test_scheduled_full_frames_and_reference_call_count() -> None:
    reference, committee = CountingForces(), CountingCommittee()
    monitor = StagedOODMonitor(staged_config(), reference_forces=reference, committee_provider=committee)
    results = _run(monitor, [h4_frame(0, 2.0), h4_frame(1, 2.0), h4_frame(2, 2.0)])
    assert [result.stage for result in results] == [Stage.FULL, Stage.LIGHT, Stage.FULL]
    assert reference.calls == 2 and committee.calls == 2  # PHAL: 2 x (1 MLIP + 2 committee models)
    assert results[0].triggered and results[0].hotspot_atom_ids == [0]


def test_cascade_from_light_to_physics_to_full() -> None:
    config = staged_config(stages={"scheduled_full_interval": 1})
    config["metrics"]["force"]["weight"] = 1.0
    reference, committee = CountingForces(), CountingCommittee()
    monitor = StagedOODMonitor(config, reference_forces=reference, committee_provider=committee)
    result = monitor.evaluate(h4_frame(0, 2.0))
    assert result.stage is Stage.FULL
    assert result.metadata["screen_max_score"] >= 1.0 and result.metadata["physics_max_score"] >= 1.0
    assert reference.calls == 1 and committee.calls == 1


def test_light_trigger_failing_physics_stays_physics_and_untriggered() -> None:
    config = staged_config(stages={"scheduled_full_interval": 1, "physics_threshold": 50.0, "label_threshold": 50.0})
    config["metrics"]["force"]["weight"] = 1.0
    committee = CountingCommittee()
    monitor = StagedOODMonitor(config, reference_forces=CountingForces(), committee_provider=committee)
    result = monitor.evaluate(h4_frame(0, 2.0))
    assert result.stage is Stage.PHYSICS and not result.triggered
    assert result.metrics["committee"].status is MetricStatus.NOT_EVALUATED
    assert committee.calls == 0


def test_integer_physics_interval_escalates_only_on_multiples() -> None:
    config = staged_config(stages={"scheduled_full_interval": 1, "physics_interval": 2, "full_interval": "never"})
    config["metrics"]["force"]["weight"] = 1.0
    monitor = StagedOODMonitor(config, reference_forces=CountingForces())
    stages = [result.stage for result in _run(monitor, [h4_frame(step, 2.0) for step in range(4)])]
    assert stages == [Stage.PHYSICS, Stage.LIGHT, Stage.PHYSICS, Stage.LIGHT]


def test_light_interval_skips_scoring_but_keeps_temporal_state_and_stop_rules() -> None:
    config = staged_config(stages={"light_interval": 2, "scheduled_full_interval": 1})
    config["metrics"]["rmin"]["threshold"] = 0.5
    config["actions"] = {"stop_min_distance": 0.7}
    monitor = StagedOODMonitor(config, reference_forces=CountingForces())
    first = monitor.evaluate(h4_frame(0, 0.1))
    close = h4_frame(1, 0.1)
    close.positions[1] = [0.65, 0.0, 0.0]
    skipped = monitor.evaluate(close)
    assert first.stage is Stage.LIGHT and skipped.stage is Stage.SKIPPED
    assert skipped.metadata["skipped_by_light_interval"] and not skipped.triggered
    assert np.all(skipped.atom_scores == 0.0)
    assert skipped.action is FrameAction.STOP  # hard safety rule still evaluated
    assert monitor.scorer.stats["force"].frame_count == 1  # skipped frame did not update stats
    third = monitor.evaluate(h4_frame(2, 0.1))
    np.testing.assert_allclose(third.metrics["displacement"].values[1], 0.35)  # relative to the skipped frame


def test_lj_is_fitted_once_per_escalated_frame_f16() -> None:
    """PHAL fitted the LJ residual twice per escalated frame (intermediate + final scoring pass)."""

    config = staged_config(stages={"scheduled_full_interval": 1})
    config["metrics"]["force"]["weight"] = 1.0
    config["metrics"]["lj_residual"] = {"weight": 2.0, "lazy_threshold": 1.0}
    lj = CountingLJ()
    monitor = StagedOODMonitor(config, reference_forces=CountingForces(), committee_provider=CountingCommittee(), lj_estimator=lj)
    results = _run(monitor, [h4_frame(0, 2.0), h4_frame(1, 2.0)])
    escalated = sum(result.stage in {Stage.PHYSICS, Stage.FULL} for result in results)
    assert escalated == 2 and lj.calls == 2
    assert results[0].metrics["lj_residual"].status is MetricStatus.MEASURED


# -- metric availability and feature switches ----------------------------------------------------


def test_committee_is_unavailable_not_zero_without_committee_f06() -> None:
    frames = ethane_box_frames(3, committee_members=0)
    monitor = StagedOODMonitor({"stages": {"scheduled_full_interval": 1, "physics_interval": "always", "full_interval": "always"}})
    result = _run(monitor, frames)[-1]
    assert result.stage is Stage.FULL
    committee = result.metrics["committee"]
    assert committee.status is MetricStatus.UNAVAILABLE and not np.any(committee.evaluated)
    assert np.all(np.isnan(committee.values))
    assert result.metrics["mlip_force_deviation"].status is MetricStatus.UNAVAILABLE  # no provider given
    assert monitor.scorer.stats["committee"].count == 0


def test_single_member_committee_is_unavailable() -> None:
    frames = ethane_box_frames(2, committee_members=1)
    monitor = StagedOODMonitor({"stages": {"physics_interval": "always", "full_interval": "always"}})
    result = _run(monitor, frames)[-1]
    assert result.metrics["committee"].status is MetricStatus.UNAVAILABLE
    assert "1 member" in (result.metrics["committee"].reason or "")


def test_committee_from_frame_is_measured() -> None:
    frames = ethane_box_frames(2, committee_members=3)
    monitor = StagedOODMonitor({"stages": {"physics_interval": "always", "full_interval": "always"}})
    result = _run(monitor, frames)[-1]
    assert result.metrics["committee"].status is MetricStatus.MEASURED


def test_first_frame_deltas_are_not_evaluated_by_default() -> None:
    frames = ethane_box_frames(2)
    monitor = StagedOODMonitor({})
    first, second = _run(monitor, frames)
    for name in ("delta_force", "displacement", "delta_q"):
        assert first.metrics[name].status is MetricStatus.NOT_EVALUATED
        assert second.metrics[name].status is MetricStatus.MEASURED
    legacy = StagedOODMonitor({"running_stats": {"first_frame_deltas": "zero"}})
    assert np.all(legacy.evaluate(frames[0]).metrics["delta_force"].values == 0.0)


METRIC_SWITCHES = ["force", "delta_force", "displacement", "rmin", "delta_q", "lj_residual", "mlip_force_deviation", "committee"]


@pytest.mark.parametrize("name", METRIC_SWITCHES)
def test_each_metric_can_be_disabled(name: str) -> None:
    frames = ethane_box_frames(4, burst={3: (2, [8.0, 0.0, 0.0])})
    config: dict[str, Any] = {
        "metrics": {name: {"enabled": False}},
        "stages": {"physics_interval": "always", "full_interval": "always"},
        "running_stats": {"warmup_frames": 0},
    }
    if name == "rmin":
        config["actions"] = {"stop_min_distance": None}
    lj = CountingLJ(cutoff=6.0)
    reference = CountingForces()
    monitor = StagedOODMonitor(config, reference_forces=reference, lj_estimator=lj)
    results = _run(monitor, frames)
    for result in results:
        metric = result.metrics[name]
        assert metric.status is MetricStatus.NOT_EVALUATED and metric.reason == "disabled"
        assert monitor.scorer.stats[name].count == 0
    if name == "lj_residual":
        assert lj.calls == 0
    if name == "mlip_force_deviation":
        assert reference.calls == 0
    enabled_others = [other for other in METRIC_SWITCHES if other != name and other not in {"delta_force", "displacement", "delta_q"}]
    assert all(results[-1].metrics[other].status is MetricStatus.MEASURED for other in enabled_others if other != "mlip_force_deviation")


def test_physics_and_full_stages_can_be_disabled() -> None:
    frames = ethane_box_frames(6, burst={4: (2, [30.0, 0.0, 0.0])})
    lj, reference = CountingLJ(cutoff=6.0), CountingForces()
    monitor = StagedOODMonitor(
        {"stages": {"physics_interval": "never", "full_interval": "never"}, "running_stats": {"warmup_frames": 2}},
        reference_forces=reference,
        lj_estimator=lj,
    )
    results = _run(monitor, frames)
    assert all(result.stage in {Stage.LIGHT} for result in results)
    assert results[4].triggered  # the burst still triggers at the light stage
    assert lj.calls == 0 and reference.calls == 0
    no_full = StagedOODMonitor({"stages": {"physics_interval": "always", "full_interval": "never"}})
    assert all(result.stage is Stage.PHYSICS for result in _run(no_full, frames))


def test_monitor_can_be_disabled() -> None:
    monitor = StagedOODMonitor({"enabled": False})
    result = monitor.evaluate(ethane_box_frames(1)[0])
    assert result.stage is Stage.SKIPPED and result.action is FrameAction.CONTINUE
    assert all(metric.status is MetricStatus.NOT_EVALUATED for metric in result.metrics.values())
    assert monitor.frame_index == 0


def test_lazy_lj_can_be_switched_off() -> None:
    frames = ethane_box_frames(2)
    monitor = StagedOODMonitor({"metrics": {"lj_residual": {"lazy": False}}, "stages": {"physics_interval": "always"}})
    result = _run(monitor, frames)[-1]
    assert np.all(result.metrics["lj_residual"].evaluated)
    assert result.metadata["lj_fit_count"] == frames[0].n_atoms


def test_missing_forces_raise() -> None:
    frame = ethane_box_frames(1)[0]
    frame.forces = None
    with pytest.raises(MonitorError, match="no forces"):
        StagedOODMonitor({}).evaluate(frame)


# -- close contacts and actions ------------------------------------------------------------------


def test_close_contact_is_scored_with_default_threshold_and_stops() -> None:
    frames = ethane_box_frames(2)
    close = frames[1]
    close.positions[1] = close.positions[0] + [0.5, 0.0, 0.0]  # C-C at 0.5 A
    monitor = StagedOODMonitor({})
    result = _run(monitor, frames)[-1]
    assert result.metrics["rmin"].values[1] == pytest.approx(0.5)
    assert monitor.config.metrics.rmin.threshold == 0.8
    assert result.action is FrameAction.STOP and "stop_min_distance" in result.metadata["action_reasons"]
    no_stop = StagedOODMonitor({"actions": {"stop_min_distance": None}})
    assert _run(no_stop, ethane_box_frames(2))[-1].action is FrameAction.CONTINUE


def test_rmin_score_for_a_0_6_angstrom_contact_with_threshold_1() -> None:
    frames = ethane_box_frames(1)
    frames[0].positions[1] = frames[0].positions[0] + [0.6, 0.0, 0.0]
    monitor = StagedOODMonitor({"metrics": {"rmin": {"threshold": 1.0}}, "actions": {"stop_min_distance": None}})
    evaluation = monitor.evaluate_detailed(frames[0])
    assert evaluation.metric_scores["rmin"][1] == pytest.approx(0.4)


def test_action_policy_precedence() -> None:
    frames = ethane_box_frames(6, burst={5: (3, [40.0, 0.0, 0.0])})
    base = {"running_stats": {"warmup_frames": 2}, "stages": {"physics_interval": "never"}}
    save = _run(StagedOODMonitor(base), frames)[-1]
    assert save.triggered and save.action is FrameAction.SAVE
    checkpoint = _run(StagedOODMonitor({**base, "actions": {"checkpoint_score": 1.0}}), frames)[-1]
    assert checkpoint.action is FrameAction.CHECKPOINT
    stop = _run(StagedOODMonitor({**base, "actions": {"checkpoint_score": 1.0, "stop_score": 2.0}}), frames)[-1]
    assert stop.action is FrameAction.STOP and stop.metadata["action_reasons"][-1] == "stop_score"
    force_stop = _run(StagedOODMonitor({**base, "actions": {"stop_max_force": 30.0}}), frames)[-1]
    assert force_stop.action is FrameAction.STOP and "stop_max_force" in force_stop.metadata["action_reasons"]
    quiet = _run(StagedOODMonitor({**base, "actions": {"save_on_trigger": False}}), frames)[-1]
    assert quiet.triggered and quiet.action is FrameAction.CONTINUE


# -- identity and state --------------------------------------------------------------------------


def test_temporal_metrics_follow_stable_atom_ids_not_positions_in_arrays() -> None:
    frames = ethane_box_frames(3)
    reference = _run(StagedOODMonitor({}), frames)
    permutation = np.random.default_rng(0).permutation(frames[1].n_atoms)
    shuffled = frames[1]
    permuted = Frame(
        shuffled.run_id,
        shuffled.step,
        shuffled.atom_ids[permutation],
        shuffled.numbers[permutation],
        shuffled.positions[permutation],
        shuffled.cell,
        shuffled.pbc,
        forces=shuffled.forces[permutation],
        committee_forces=shuffled.committee_forces[:, permutation],
    )
    results = _run(StagedOODMonitor({}), [frames[0], permuted, frames[2]])
    for name in ("delta_force", "displacement", "delta_q"):
        np.testing.assert_allclose(results[1].metrics[name].values, reference[1].metrics[name].values[permutation], atol=1e-12)
        np.testing.assert_allclose(results[2].metrics[name].values, reference[2].metrics[name].values, atol=1e-12)


def test_changed_atom_set_restarts_temporal_metrics() -> None:
    frames = ethane_box_frames(2)
    monitor = StagedOODMonitor({})
    monitor.evaluate(frames[0])
    renumbered = frames[1]
    renumbered.atom_ids = renumbered.atom_ids + 1000
    assert monitor.evaluate(renumbered).metrics["delta_force"].status is MetricStatus.NOT_EVALUATED


def test_state_round_trip_resumes_identically() -> None:
    frames = ethane_box_frames(10, burst={6: (4, [10.0, 0.0, 0.0])})
    config = {"running_stats": {"warmup_frames": 2}, "stages": {"scheduled_full_interval": 3}}
    continuous = _run(StagedOODMonitor(config, reference_forces=CountingForces()), frames)
    first = StagedOODMonitor(config, reference_forces=CountingForces())
    _run(first, frames[:5])
    state = json.loads(json.dumps(first.to_state()))
    second = StagedOODMonitor(config, reference_forces=CountingForces())
    second.load_state(state)
    resumed = _run(second, frames[5:])
    for expected, actual in zip(continuous[5:], resumed):
        assert actual.stage is expected.stage and actual.triggered == expected.triggered
        assert actual.hotspot_atom_ids == expected.hotspot_atom_ids
        np.testing.assert_array_equal(actual.atom_scores, expected.atom_scores)
    with pytest.raises(MonitorError):
        second.load_state({**state, "engine": "global_lj"})


# -- engine selection ----------------------------------------------------------------------------


def test_global_lj_engine_wraps_reliability_monitor() -> None:
    from ase import Atoms
    from ase.calculators.lj import LennardJones

    positions = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [1.9, 3.3, 0.0], [1.9, 1.1, 3.1]])
    atoms = Atoms("Ar4", positions=positions)
    atoms.calc = LennardJones(epsilon=0.0104, sigma=3.40, rc=50.0, smooth=False)
    frame = Frame("ar", 0, np.arange(4) + 10, atoms.numbers, positions, None, (False,) * 3, forces=atoms.get_forces())
    monitor = build_frame_monitor({"engine": "global_lj", "global_lj": {"ridge_lambda": 1e-8}})
    result = monitor.evaluate(frame)
    assert result.metadata["status"] == "SAFE" and not result.triggered and result.action is FrameAction.CONTINUE
    assert result.metrics["committee"].status is MetricStatus.UNAVAILABLE
    bad = Frame("ar", 1, np.arange(3) + 10, np.full(3, 18), [[0, 0, 0], [0.3, 0, 0], [1.5, 0, 0]], None, (False,) * 3,
                forces=[[50.0, 0, 0], [-60.0, 0, 0], [10.0, 0, 0]])
    stop = build_frame_monitor({"engine": "global_lj"}).evaluate(bad)
    assert stop.triggered and stop.action is FrameAction.STOP and len(stop.hotspot_atom_ids) >= 1
    restored = build_frame_monitor({"engine": "global_lj"})
    restored.load_state(json.loads(json.dumps(monitor.to_state())))
    assert restored.frame_index == 1
