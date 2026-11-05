"""Parity of the staged monitor and event buffer with PHAL's golden outputs (tests/fixtures/phal_reference).

Two replays per scenario:

* ``phal_math``: PHAL's scoring conventions (``monitor_config_from_phal(legacy_scoring=True)``) with
  the sign-corrected LJ estimator fed the NEGATED forces. Since corrected(-F) == PHAL(F), this must
  reproduce every recorded per-frame decision, per-atom metric score and event exactly (up to the
  8-decimal rounding of the stored input frames).
* ``corrected``: the same conventions with the physically correct LJ sign (F-01). The deviations
  are pinned below and documented in docs/monitoring.md; every non-LJ metric still matches PHAL.
"""

from __future__ import annotations

import numpy as np
import pytest

from gmd_active_learning.events import events_config_from_phal
from gmd_active_learning.monitors import LocalLJEstimator, OnlineMonitorDriver, StagedOODMonitor, monitor_config_from_phal
from gmd_active_learning.monitors.staged_monitor import StagedEvaluation

from .phal_replay import SCENARIOS, NegatedForceLJ, Scenario, load_scenario, primary_model_forces

# Input frames are stored with 8 decimals, so scores agree to ~1e-6 rather than bit-exactly.
RTOL, ATOL = 1.0e-4, 2.0e-5

# Frames whose hotspot atoms / reasons / trigger change once the LJ sign is corrected (F-01),
# with PHAL's scoring conventions otherwise unchanged.
CORRECTED_LJ_DEVIATIONS = {
    "ethane_box": {"triggered": set(), "hotspots_or_reasons": {5, 6, 7, 8, 13}},
    "polyethylene_chains": {"triggered": {7, 8}, "hotspots_or_reasons": {5, 7, 8, 9, 11}},
    "pt111_co_slab": {"triggered": set(), "hotspots_or_reasons": {6, 7, 11, 12}},
}


def _replay(scenario: Scenario, *, negate_lj: bool) -> tuple[list[StagedEvaluation], list, OnlineMonitorDriver]:
    """Replay a scenario through the driver; the monitor's LJ estimator is reachable via ``driver.monitor``."""

    config = monitor_config_from_phal(scenario.config, legacy_scoring=True)
    estimator = LocalLJEstimator(cutoff=config.neighbors.lj_cutoff)
    monitor = StagedOODMonitor(
        config,
        reference_forces=primary_model_forces,
        lj_estimator=NegatedForceLJ(estimator) if negate_lj else estimator,
    )
    evaluations: list[StagedEvaluation] = []
    original = monitor.evaluate

    def evaluate(frame):  # keep the detailed evaluation of every frame
        evaluation = monitor.evaluate_detailed(frame)
        evaluations.append(evaluation)
        return evaluation.result

    monitor.evaluate = evaluate  # type: ignore[method-assign]
    driver = OnlineMonitorDriver(
        config,
        events_config_from_phal(scenario.config),
        {"merge_radius": scenario.config["hotspot"]["merge_radius"]},
        monitor=monitor,
    )
    summary = driver.run(scenario.frames)
    monitor.evaluate = original  # type: ignore[method-assign]
    assert summary.stop_reason == "end_of_stream"
    return evaluations, summary.events, driver


@pytest.fixture(scope="module", params=SCENARIOS)
def scenario(request: pytest.FixtureRequest) -> Scenario:
    return load_scenario(request.param)


def test_phal_math_reproduces_every_frame_decision_and_metric(scenario: Scenario) -> None:
    evaluations, _events, _driver = _replay(scenario, negate_lj=True)
    assert len(evaluations) == len(scenario.results)
    for index, (evaluation, record) in enumerate(zip(evaluations, scenario.results)):
        result = evaluation.result
        context = f"{scenario.name} frame {index}"
        assert result.step == record["step"], context
        assert result.stage.value == record["stage"], context
        assert result.triggered == record["triggered"], context
        assert result.hotspot_atom_ids == record["hotspot_indices"], context
        assert result.trigger_reasons == record["trigger_reason"], context
        assert result.max_score == pytest.approx(record["max_score"], rel=RTOL, abs=ATOL), context
        assert result.metadata.get("lj_fit_count", 0) == record["metadata"]["lj_fit_count"], context
        np.testing.assert_allclose(
            result.atom_scores, scenario.arrays[f"frame{index:03d}/atom_scores"], rtol=RTOL, atol=ATOL, err_msg=context
        )
        assert sorted(evaluation.metric_scores) == record["evaluated_metrics"], context
        for metric in record["evaluated_metrics"]:
            np.testing.assert_allclose(
                evaluation.metric_scores[metric],
                scenario.arrays[f"frame{index:03d}/metric/{metric}"],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"{context} metric {metric}",
            )
        # Metrics outside the stage are recorded as not evaluated, never as zeros.
        for name, metric in result.metrics.items():
            if name not in record["evaluated_metrics"]:
                assert metric.status.value in {"not_evaluated", "unavailable"}, f"{context} {name}"


def test_phal_math_reproduces_events(scenario: Scenario) -> None:
    _evaluations, events, _driver = _replay(scenario, negate_lj=True)
    assert len(events) == len(scenario.events)
    for event, reference in zip(events, scenario.events):
        assert event.trigger_step == reference["step"]
        assert [frame.step for frame in event.pre_frames] == reference["pre_frame_steps"]
        assert [frame.step for frame in event.post_frames] == reference["post_frame_steps"]
        assert event.trigger_steps == reference["metadata"]["trigger_steps"]
        assert event.metadata["trigger_count"] == reference["metadata"]["trigger_count"]
        assert event.hotspot_atom_ids == reference["hotspot_atoms"]
        assert event.trigger_reasons == reference["trigger_reason"]
        np.testing.assert_allclose(event.metadata["atom_scores"], reference["ood_scores"], rtol=RTOL, atol=ATOL)


def test_truncated_event_is_marked_incomplete_f07() -> None:
    """PHAL's polyethylene event at step 90 has 2 of 3 post frames and no marker; ours says so."""

    scenario = load_scenario("polyethylene_chains")
    _evaluations, events, _driver = _replay(scenario, negate_lj=True)
    truncated = [event for event in events if event.trigger_step == 90]
    assert len(truncated) == 1
    event = truncated[0]
    assert [frame.step for frame in event.post_frames] == [100, 110]
    assert event.post_frames_requested == 3
    assert event.complete is False
    assert event.truncation_reason == "end_of_stream"
    assert all(other.complete for other in events if other is not event)


def test_event_ids_are_deterministic_across_reruns_f15(scenario: Scenario) -> None:
    from gmd_active_learning.contracts import stable_id

    _e1, first, _d1 = _replay(scenario, negate_lj=True)
    _e2, second, _d2 = _replay(scenario, negate_lj=True)
    assert [event.event_id for event in first] == [event.event_id for event in second]
    for event in first:
        assert event.event_id == stable_id("evt", event.run_id, event.trigger_steps[0])
    assert len({event.event_id for event in first}) == len(first)


def test_corrected_lj_deviates_only_where_documented_f01(scenario: Scenario) -> None:
    corrected, events, _driver = _replay(scenario, negate_lj=False)
    expected = CORRECTED_LJ_DEVIATIONS[scenario.name]
    triggered_diff, detail_diff = set(), set()
    for index, (evaluation, record) in enumerate(zip(corrected, scenario.results)):
        result = evaluation.result
        assert result.stage.value == record["stage"], f"{scenario.name} frame {index}: stage must not change"
        if result.triggered != record["triggered"]:
            triggered_diff.add(index)
        if result.hotspot_atom_ids != record["hotspot_indices"] or result.trigger_reasons != record["trigger_reason"]:
            detail_diff.add(index)
        # Every metric other than the LJ residual is unaffected by the sign fix.
        for metric in record["evaluated_metrics"]:
            if metric == "lj_residual":
                continue
            np.testing.assert_allclose(
                evaluation.metric_scores[metric],
                scenario.arrays[f"frame{index:03d}/metric/{metric}"],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"{scenario.name} frame {index} metric {metric}",
            )
    assert triggered_diff == expected["triggered"]
    assert detail_diff == expected["hotspots_or_reasons"]
    if scenario.name == "polyethylene_chains":
        # Frame 7 (step 70) now triggers and frame 8 (step 80) no longer does, so the first
        # event's merged trigger steps become [50, 60, 70] instead of PHAL's [50, 60, 80].
        assert [event.trigger_steps for event in events] == [[50, 60, 70], [90, 100, 110]]
    else:
        assert [event.trigger_steps for event in events] == [ref["metadata"]["trigger_steps"] for ref in scenario.events]


def test_lj_residuals_match_phal_on_negated_forces() -> None:
    """Direct check of the LJ math on the recorded frames: corrected(-F) == PHAL(F) for fitted atoms."""

    scenario = load_scenario("ethane_box")
    evaluations, _events, _driver = _replay(scenario, negate_lj=True)
    checked = 0
    for index, evaluation in enumerate(evaluations):
        lj = evaluation.result.metrics["lj_residual"]
        if lj.status.value != "measured":
            continue
        reference = scenario.arrays[f"frame{index:03d}/metric/lj_residual"]
        # Scores are max(residual, z) on fitted atoms and exactly 0 on atoms lazily skipped.
        mine = evaluation.metric_scores["lj_residual"]
        np.testing.assert_allclose(mine[lj.evaluated], reference[lj.evaluated], rtol=RTOL, atol=ATOL)
        assert np.all(reference[~lj.evaluated] == 0.0)
        checked += int(np.sum(lj.evaluated))
    assert checked > 0


def test_lj_is_fitted_once_per_escalated_frame_on_the_fixture_f16() -> None:
    """PHAL fitted LJ 21 times for ethane_box's 13 escalated frames (scratch reproduction with the
    reference tree: 5 scheduled full frames x 1 + 8 cascaded frames x 2). We fit once per frame that
    has a suspicious atom."""

    scenario = load_scenario("ethane_box")
    evaluations, _events, driver = _replay(scenario, negate_lj=True)
    escalated = [e for e in evaluations if e.result.stage.value in {"physics", "full"}]
    fitted = [e for e in escalated if e.result.metadata.get("lj_fit_count", 0) > 0]
    assert len(escalated) == 13
    assert driver.monitor.lj_estimator.calls == len(fitted) == 12  # frame 0 has no suspicious atom
