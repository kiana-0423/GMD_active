from __future__ import annotations


def _normalize_ratio(value: float, threshold: float) -> float:
    if threshold <= 0.0:
        return value
    return max(0.0, value / threshold)


def compute_risk_score(metrics: dict[str, float | bool | None], monitor_config: dict[str, object]) -> float:
    thresholds = monitor_config.get("thresholds", {})
    weights = monitor_config.get("weights", {})
    lj_residual = float(metrics.get("lj_residual", 0.0) or 0.0)
    param_anomaly = 1.0 if bool(metrics.get("param_anomaly", False)) else 0.0
    param_jump = float(metrics.get("param_jump_ratio", 0.0) or 0.0)
    ensemble = float(metrics.get("ensemble_max_deviation", 0.0) or 0.0)
    max_force = float(metrics.get("max_force", 0.0) or 0.0)
    min_distance = float(metrics.get("min_distance", 999.0) or 999.0)
    unsafe_probability = float(metrics.get("unsafe_probability", 0.0) or 0.0)
    inverse_min_distance = 0.0 if min_distance <= 0.0 else 1.0 / min_distance
    distance_threshold = float(thresholds.get("min_distance_stop", 1.0))
    return float(
        float(weights.get("lj_residual", 1.0)) * _normalize_ratio(lj_residual, float(thresholds.get("lj_residual_candidate", 1.0)))
        + float(weights.get("param_anomaly", 1.0)) * param_anomaly
        + float(weights.get("param_jump", 1.0)) * _normalize_ratio(param_jump, float(thresholds.get("parameter_jump_ratio", 1.0)))
        + float(weights.get("ensemble_deviation", 1.0)) * _normalize_ratio(ensemble, float(thresholds.get("ensemble_deviation_candidate", 1.0)))
        + float(weights.get("max_force", 1.0)) * _normalize_ratio(max_force, float(thresholds.get("max_force_stop", 1.0)))
        + float(weights.get("min_distance", 1.0)) * _normalize_ratio(inverse_min_distance, 1.0 / max(distance_threshold, 1.0e-12))
        + float(weights.get("reliability_model", 1.0)) * unsafe_probability
    )
