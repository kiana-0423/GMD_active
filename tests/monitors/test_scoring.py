"""OOD scorer: score transforms, staged aggregation, running stats (MON-010, MON-011, MON-012; F-06, F-16)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from gmd_active_learning.contracts.monitoring import MetricValues, Stage
from gmd_active_learning.core.exceptions import DataContractError
from gmd_active_learning.monitors.config import load_monitor_config
from gmd_active_learning.monitors.running_stats import RunningMetricStats
from gmd_active_learning.monitors.scoring import METRIC_NAMES, OODScorer

ZERO_WEIGHTS = {name: {"weight": 0.0} for name in METRIC_NAMES}


def _config(weights: dict[str, float], **overrides: Any) -> Any:
    metrics: dict[str, Any] = {name: dict(ZERO_WEIGHTS[name]) for name in METRIC_NAMES}
    for name, weight in weights.items():
        metrics[name]["weight"] = weight
    for name, extra in overrides.pop("metrics", {}).items():
        metrics[name].update(extra)
    data = {
        "metrics": metrics,
        "stages": {"screen_threshold": 4.0, "physics_threshold": 5.0, "label_threshold": 6.0, **overrides.pop("stages", {})},
        "running_stats": {"enabled": True, "warmup_frames": 5, "min_std": 1.0, **overrides.pop("running_stats", {})},
        "actions": {"stop_min_distance": None},
        **overrides,
    }
    return load_monitor_config(data)


def _metrics(n_atoms: int = 2, **values: np.ndarray) -> dict[str, MetricValues]:
    base = {name: np.zeros(n_atoms) for name in METRIC_NAMES}
    base["rmin"] = np.ones(n_atoms)
    base.update(values)
    return {name: MetricValues.measured(name, array) for name, array in base.items()}


def _score(scorer: OODScorer, metrics: dict[str, MetricValues], stage: Stage):
    n_atoms = len(next(iter(metrics.values())).values)
    return scorer.score_stage(metrics, stage, n_atoms=n_atoms)


def test_physics_stage_includes_mlip_force_deviation() -> None:
    config = _config({"mlip_force_deviation": 1.0}, stages={"physics_threshold": 2.0, "label_threshold": 2.0}, running_stats={"warmup_frames": 0})
    result = _score(OODScorer(config), _metrics(mlip_force_deviation=np.array([3.0, 0.0])), Stage.PHYSICS)
    assert result.triggered and result.hotspot_indices == [0]
    assert "mlip_force_deviation" in result.metric_scores
    assert "committee" not in result.metric_scores


def test_light_no_trigger_below_screen_threshold() -> None:
    config = _config({"force": 1.0, "mlip_force_deviation": 1.0}, running_stats={"warmup_policy": "raw"})
    result = _score(OODScorer(config), _metrics(force=np.array([1.0, 3.9])), Stage.LIGHT)
    assert not result.triggered and result.hotspot_indices == []


def test_full_triggers_on_mlip_force_deviation_with_reason() -> None:
    config = _config({"force": 1.0, "mlip_force_deviation": 1.0}, running_stats={"warmup_policy": "raw"})
    result = _score(OODScorer(config), _metrics(mlip_force_deviation=np.array([6.5, 0.0])), Stage.FULL)
    assert result.triggered and result.hotspot_indices == [0]
    assert result.trigger_reasons == ["model_drift"]


def test_full_combines_multiple_reasons() -> None:
    config = _config(
        {"force": 1.0, "rmin": 1.0, "delta_q": 1.0, "lj_residual": 1.0},
        metrics={"rmin": {"threshold": 1.0}, "delta_q": {"threshold": 1.0}, "force": {"z_threshold": None}},
        stages={"screen_threshold": 0.5, "physics_threshold": 0.5, "label_threshold": 0.5},
        running_stats={"enabled": False},
    )
    metrics = _metrics(
        3,
        force=np.array([1.0, 0.0, 0.0]),
        rmin=np.array([1.0, 0.0, 1.0]),
        delta_q=np.array([0.0, 0.0, 2.0]),
        lj_residual=np.array([0.5, 0.5, 0.5]),
    )
    result = _score(OODScorer(config), metrics, Stage.FULL)
    assert result.triggered and result.hotspot_indices == [0, 1, 2]
    assert set(result.trigger_reasons) >= {"force_large", "close_contact", "coordination_change", "lj_residual"}


def test_warmup_raw_policy_reproduces_phal_and_suppress_drops_z_parts_f16() -> None:
    """PHAL compared raw eV/A to the z threshold during warmup; the default suppresses that."""

    metrics = _metrics(force=np.array([2.5, 4.5]), mlip_force_deviation=np.array([3.0, 0.0]))
    raw = OODScorer(_config({"force": 1.0}, running_stats={"warmup_policy": "raw"}))
    np.testing.assert_allclose(raw.metric_score("force", metrics["force"]), [0.0, 4.5])
    np.testing.assert_allclose(raw.metric_score("mlip_force_deviation", metrics["mlip_force_deviation"]), [3.0, 0.0])
    suppress = OODScorer(_config({"force": 1.0}))
    assert np.all(suppress.metric_score("force", metrics["force"]) == 0.0)
    assert np.all(suppress.metric_score("mlip_force_deviation", metrics["mlip_force_deviation"]) == 0.0)
    # Threshold-based parts stay active during warmup.
    dq = MetricValues.measured("delta_q", np.array([1.0, 0.2]))
    np.testing.assert_allclose(suppress.metric_score("delta_q", dq), [1.0, 0.0])
    lj = MetricValues.measured("lj_residual", np.array([0.7, 0.1]))
    np.testing.assert_allclose(suppress.metric_score("lj_residual", lj), [0.7, 0.1])


def test_delta_q_threshold_scoring_without_stats() -> None:
    config = _config({"delta_q": 1.0}, running_stats={"enabled": False})
    score = OODScorer(config).metric_score("delta_q", MetricValues.measured("delta_q", np.array([1.0])))
    np.testing.assert_allclose(score, [1.0])  # max(|dq| = 1.0, (1.0 - 0.5) / 0.5 = 1.0)


def test_warmup_counts_frames_not_atoms() -> None:
    scorer = OODScorer(_config({"force": 1.0}, running_stats={"warmup_frames": 2, "warmup_policy": "raw"}))
    scorer.update_stats(_metrics(4, force=np.array([1.0, 2.0, 3.0, 4.0])))
    assert scorer.stats["force"].count == 4 and scorer.stats["force"].frame_count == 1
    second = MetricValues.measured("force", np.array([10.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(scorer.metric_score("force", second), [10.0, 0.0, 0.0, 0.0])


def test_gated_z_thresholds() -> None:
    config = _config({"displacement": 1.0, "force": 1.0}, running_stats={"warmup_frames": 0, "min_std": 10.0},
                     metrics={"displacement": {"z_threshold": 4.0}, "force": {"z_threshold": 4.0}})
    scorer = OODScorer(config)
    assert np.all(scorer.metric_score("displacement", MetricValues.measured("displacement", np.array([5.0, 0.0]))) == 0.0)
    config = _config({"force": 1.0}, running_stats={"warmup_frames": 0, "min_std": 1.0})
    np.testing.assert_allclose(OODScorer(config).metric_score("force", MetricValues.measured("force", np.array([3.9, 4.1]))), [0.0, 4.1])


def test_min_trigger_atoms_and_empty_frame() -> None:
    config = _config({"force": 1.0}, stages={"min_trigger_atoms": 3}, running_stats={"warmup_policy": "raw"})
    assert not _score(OODScorer(config), _metrics(3, force=np.array([7.0, 0.0, 0.0])), Stage.LIGHT).triggered
    empty = _score(OODScorer(config), _metrics(0), Stage.FULL)
    assert not empty.triggered and empty.hotspot_indices == [] and empty.max_score == 0.0


def test_triggered_without_label_hotspots_falls_back_to_argmax() -> None:
    config = _config({"force": 1.0}, running_stats={"warmup_policy": "raw"})
    result = _score(OODScorer(config), _metrics(3, force=np.array([4.5, 5.0, 5.0])), Stage.LIGHT)
    assert result.triggered and result.hotspot_indices == [1, 2]


def test_rmin_scoring_threshold_and_null() -> None:
    contact = MetricValues.measured("rmin", np.array([0.6, 2.5]))
    np.testing.assert_allclose(OODScorer(_config({"rmin": 1.0}, metrics={"rmin": {"threshold": 1.0}})).metric_score("rmin", contact), [0.4, 0.0])
    assert np.all(OODScorer(_config({"rmin": 1.0}, metrics={"rmin": {"threshold": None}})).metric_score("rmin", contact) == 0.0)


def test_unavailable_and_unevaluated_metrics_score_zero_and_skip_stats_f06() -> None:
    config = _config({"committee": 2.0, "lj_residual": 2.0}, running_stats={"warmup_frames": 0})
    scorer = OODScorer(config)
    scorer.stats["committee"].update(np.array([0.2, 0.4, 0.3]))  # a baseline exists...
    unavailable = MetricValues.unavailable("committee", 3, "no committee")
    assert np.all(scorer.metric_score("committee", unavailable) == 0.0)  # ...but z(0) is never invented
    partial = MetricValues.measured("lj_residual", np.array([0.9, 0.0, 0.3]), np.array([True, False, True]))
    assert scorer.metric_score("lj_residual", partial)[1] == 0.0
    scorer.update_stats({"committee": unavailable, "lj_residual": partial})
    assert scorer.stats["committee"].count == 3
    assert scorer.stats["lj_residual"].count == 2


def test_non_finite_evaluated_metric_is_rejected() -> None:
    with pytest.raises(DataContractError, match="finite"):
        MetricValues.measured("mlip_force_deviation", np.full(3, np.nan))


def test_running_stats_batch_chunked_and_state_round_trip() -> None:
    values = np.random.default_rng(9).normal(size=1000)
    batch, single, chunked = RunningMetricStats(), RunningMetricStats(), RunningMetricStats()
    batch.update(values)
    for value in values:
        single.update(np.array([value]))
    for start in range(0, len(values), 100):
        chunked.update(values[start : start + 100])
    for stats in (single, chunked):
        assert stats.count == batch.count
        assert stats.mean == pytest.approx(batch.mean, rel=1e-12)
        assert stats.std == pytest.approx(batch.std, rel=1e-12)
    assert batch.mean == pytest.approx(np.mean(values), rel=1e-12)
    assert batch.std == pytest.approx(np.std(values, ddof=1), rel=1e-12)
    assert chunked.frame_count == 10
    assert RunningMetricStats.from_state(batch.to_state()) == batch
    scorer = OODScorer(_config({"force": 1.0}))
    scorer.stats["force"] = batch
    restored = OODScorer(_config({"force": 1.0}))
    restored.load_state(scorer.to_state())
    assert restored.stats["force"] == batch


def test_stage_weighted_sum_order_and_reasons_rank_unweighted() -> None:
    config = _config({"force": 10.0, "delta_q": 1.0}, running_stats={"enabled": False}, metrics={"force": {"z_threshold": None}},
                     stages={"screen_threshold": 1.0, "physics_threshold": 1.0, "label_threshold": 1.0})
    metrics = _metrics(1, force=np.array([0.5]), delta_q=np.array([3.0]))
    result = _score(OODScorer(config), metrics, Stage.LIGHT)
    assert result.atom_scores[0] == pytest.approx(10.0 * 0.5 + 1.0 * 5.0)
    assert result.trigger_reasons == ["coordination_change", "force_large"]
