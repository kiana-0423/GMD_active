"""Configuration of the online monitor (YAML section ``monitor``; defaults in configs/sections/monitor.yaml).

Every threshold, weight, interval and feature switch used by ``gmd_active_learning.monitors`` is
defined here. Mapping from PHAL keys: see :func:`monitor_config_from_phal` and docs/monitoring.md.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from gmd_active_learning.core.exceptions import ConfigurationError

IntervalSpec = int | str
INTERVAL_KEYWORDS = {"triggered", "never", "disabled", "off", "always", "all"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _normalise_interval(value: Any) -> IntervalSpec:
    """``triggered``/``never``/``always`` keywords or a non-negative integer frame interval.

    PHAL silently treated unparsable strings as ``triggered``; here they are a configuration error.
    """

    if isinstance(value, bool):
        raise ValueError("interval must be an integer or one of " + ", ".join(sorted(INTERVAL_KEYWORDS)))
    if isinstance(value, int):
        if value < 0:
            raise ValueError("integer interval must be >= 0 (0 means never)")
        return value
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in INTERVAL_KEYWORDS:
            return normalised
        try:
            return _normalise_interval(int(normalised))
        except ValueError:
            pass
    raise ValueError(f"invalid interval {value!r}: use an integer or one of {sorted(INTERVAL_KEYWORDS)}")


class NeighborConfig(_Strict):
    lj_cutoff: float = Field(6.0, gt=0.0, description="Angstrom; local LJ fit neighbourhood")
    coordination_cutoff: float = Field(6.0, gt=0.0, description="Angstrom; smooth coordination neighbourhood")


class ZScoredMetricConfig(_Strict):
    """force / delta_force / displacement: score = z if z >= z_threshold else 0."""

    enabled: bool = True
    weight: float = Field(1.0, ge=0.0)
    z_threshold: float | None = Field(4.0, ge=0.0, description="null = ungated z-score")


class DeltaForceConfig(ZScoredMetricConfig):
    weight: float = Field(1.5, ge=0.0)


class RminConfig(_Strict):
    """Close contact: score = max(0, (threshold - rmin) / threshold); threshold null = measured, not scored."""

    enabled: bool = True
    weight: float = Field(1.0, ge=0.0)
    threshold: float | None = Field(0.8, gt=0.0, description="Angstrom")


class CoordinationConfig(_Strict):
    """delta_q of q_i = sum_j 1 / (1 + (r_ij / (scale * (R_i + R_j)))^power), r_ij <= coordination_cutoff."""

    enabled: bool = True
    weight: float = Field(1.0, ge=0.0)
    threshold: float | None = Field(0.5, gt=0.0, description="relative-exceedance threshold on |delta_q|")
    scale: float = Field(1.15, gt=0.0, description="multiplies the covalent-radius sum")
    power: float = Field(6.0, gt=0.0, description="exponent of the rational switching function")


class LJResidualConfig(_Strict):
    """Local per-atom LJ projection residual (sign-corrected, docs/migration/findings.md F-01)."""

    enabled: bool = True
    weight: float = Field(2.0, ge=0.0)
    estimator: Literal["linear", "nonlinear"] = "linear"
    lazy: bool = True
    lazy_threshold: float = Field(3.0, ge=0.0, description="fit atoms whose weighted light score >= this")
    condition_limit: float = Field(1.0e4, gt=1.0)
    sigma_min: float = Field(0.2, gt=0.0)
    sigma_max: float = Field(10.0, gt=0.0)
    relative_delta: float = Field(1.0e-8, gt=0.0, description="eV/A added to |F| in the residual denominator")
    nonlinear_max_nfev: int = Field(400, ge=1)


class WeightedMetricConfig(_Strict):
    enabled: bool = True
    weight: float = Field(2.0, ge=0.0)


class CommitteeConfig(_Strict):
    enabled: bool = True
    weight: float = Field(2.0, ge=0.0)
    min_members: int = Field(2, ge=2, description="fewer members -> metric UNAVAILABLE (never zero)")


class MetricsConfig(_Strict):
    force: ZScoredMetricConfig = Field(default_factory=ZScoredMetricConfig)
    delta_force: DeltaForceConfig = Field(default_factory=DeltaForceConfig)
    displacement: ZScoredMetricConfig = Field(default_factory=ZScoredMetricConfig)
    rmin: RminConfig = Field(default_factory=RminConfig)
    delta_q: CoordinationConfig = Field(default_factory=CoordinationConfig)
    lj_residual: LJResidualConfig = Field(default_factory=LJResidualConfig)
    mlip_force_deviation: WeightedMetricConfig = Field(default_factory=WeightedMetricConfig)
    committee: CommitteeConfig = Field(default_factory=CommitteeConfig)

    def weight(self, name: str) -> float:
        return float(getattr(self, name).weight)

    def enabled(self, name: str) -> bool:
        return bool(getattr(self, name).enabled)


class RunningStatsConfig(_Strict):
    enabled: bool = Field(True, description="false: z-scores are replaced by raw |x|")
    warmup_frames: int = Field(5, ge=0)
    min_std: float = Field(1.0e-8, gt=0.0)
    warmup_policy: Literal["suppress", "raw"] = Field(
        "suppress",
        description="during warmup: suppress = z-based score parts are 0; raw = PHAL (raw |x| compared to z thresholds)",
    )
    first_frame_deltas: Literal["not_evaluated", "zero"] = Field(
        "not_evaluated",
        description="temporal metrics with no previous frame: not_evaluated (default) or PHAL's measured zeros",
    )


class StagesConfig(_Strict):
    light_interval: IntervalSpec = 1
    physics_interval: IntervalSpec = "triggered"
    full_interval: IntervalSpec = "triggered"
    scheduled_full_interval: int = Field(1, ge=1, description="PHAL online.monitor_freq; >1 forces a full frame every k frames")
    screen_threshold: float = 4.0
    physics_threshold: float = 5.0
    label_threshold: float = Field(6.0, description="hotspot atoms: atom score >= label_threshold (the only hotspot threshold)")
    min_trigger_atoms: int = Field(1, ge=1)

    @field_validator("light_interval", "physics_interval", "full_interval", mode="before")
    @classmethod
    def _check_interval(cls, value: Any) -> IntervalSpec:
        return _normalise_interval(value)


class ActionsConfig(_Strict):
    """Frame action policy (precedence stop > checkpoint > save > continue)."""

    save_on_trigger: bool = True
    checkpoint_score: float | None = Field(None, description="max atom score >= this -> CHECKPOINT")
    stop_score: float | None = Field(None, description="max atom score >= this -> STOP")
    stop_min_distance: float | None = Field(0.6, gt=0.0, description="Angstrom; any rmin below -> STOP")
    stop_max_force: float | None = Field(None, gt=0.0, description="eV/A; any |F_i| above -> STOP")


class GlobalLJConfig(_Strict):
    """Settings of the frame-level ``global_lj`` engine (GMD ReliabilityMonitor)."""

    window_size: int = Field(20, ge=1)
    cutoff: float = Field(6.0, gt=0.0)
    ridge_lambda: float = Field(1.0e-6, ge=0.0)
    thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "lj_residual_candidate": 0.30,
            "lj_residual_stop": 0.60,
            "max_force_stop": 20.0,
            "min_distance_stop": 0.60,
            "ensemble_deviation_candidate": 0.15,
            "ensemble_deviation_stop": 0.30,
            "parameter_jump_ratio": 5.0,
        }
    )
    parameter_bounds: dict[str, float] | None = Field(
        default_factory=lambda: {"epsilon_min": 1.0e-6, "epsilon_max": 10.0, "sigma_min": 0.5, "sigma_max": 8.0}
    )
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "lj_residual": 1.0,
            "param_anomaly": 1.0,
            "param_jump": 0.5,
            "ensemble_deviation": 1.0,
            "max_force": 0.5,
            "min_distance": 0.5,
            "reliability_model": 0.5,
        }
    )

    def as_reliability_config(self) -> dict[str, Any]:
        return self.model_dump()


class DriverConfig(_Strict):
    continue_on_error: bool = False
    max_errors: int = Field(1, ge=1, description="consecutive frame errors tolerated when continue_on_error")
    max_consecutive_timeouts: int = Field(10, ge=1, description="poll timeouts in a row before the driver gives up")
    poll_timeout_s: float | None = Field(None, gt=0.0, description="timeout passed to FrameSource.next_frame")
    stop_on_stop_action: bool = True
    summary_interval: int = Field(0, ge=0, description="log a progress summary every k frames (0 = never)")
    progress_file: str | None = Field(None, description="JSON progress file rewritten atomically every frame")


class MonitorConfig(_Strict):
    enabled: bool = True
    engine: Literal["staged", "global_lj"] = "staged"
    neighbors: NeighborConfig = Field(default_factory=NeighborConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    running_stats: RunningStatsConfig = Field(default_factory=RunningStatsConfig)
    stages: StagesConfig = Field(default_factory=StagesConfig)
    actions: ActionsConfig = Field(default_factory=ActionsConfig)
    global_lj: GlobalLJConfig = Field(default_factory=GlobalLJConfig)
    driver: DriverConfig = Field(default_factory=DriverConfig)

    @model_validator(mode="after")
    def _consistency(self) -> MonitorConfig:
        rmin = self.metrics.rmin
        cutoff = max(self.neighbors.lj_cutoff, self.neighbors.coordination_cutoff)
        if rmin.enabled and rmin.threshold is not None and rmin.threshold >= cutoff:
            raise ValueError(f"metrics.rmin.threshold ({rmin.threshold}) must be below the neighbour cutoff ({cutoff})")
        if self.actions.stop_min_distance is not None and not rmin.enabled:
            raise ValueError("actions.stop_min_distance needs metrics.rmin.enabled (set it to null to disable the rule)")
        if self.actions.stop_max_force is not None and not self.metrics.force.enabled:
            raise ValueError("actions.stop_max_force needs metrics.force.enabled (set it to null to disable the rule)")
        if self.metrics.lj_residual.sigma_min >= self.metrics.lj_residual.sigma_max:
            raise ValueError("metrics.lj_residual.sigma_min must be < sigma_max")
        return self


def load_monitor_config(section: dict[str, Any] | MonitorConfig | None) -> MonitorConfig:
    """Validate the ``monitor`` section; raises ConfigurationError with the pydantic message."""

    if isinstance(section, MonitorConfig):
        return section
    try:
        return MonitorConfig.model_validate(section or {})
    except ValidationError as exc:
        raise ConfigurationError(f"invalid monitor configuration: {exc}") from exc


def monitor_config_from_phal(phal: dict[str, Any], *, legacy_scoring: bool = True) -> MonitorConfig:
    """Translate a full PHAL configuration (monitor/ood_score/online sections) into ``MonitorConfig``.

    ``legacy_scoring=True`` also selects PHAL's warmup (raw values against z thresholds) and
    first-frame (zero deltas) conventions, so that a replay reproduces PHAL scores; the LJ sign is
    always the corrected one (F-01). Used for migration and parity tests.
    """

    monitor = phal.get("monitor", {})
    score = phal.get("ood_score", {})
    weights = score.get("weights", {})
    stats = score.get("running_stats", {})
    online = phal.get("online", {})

    def zcfg(name: str, key: str) -> dict[str, Any]:
        return {"weight": float(weights.get(name, 0.0)), "z_threshold": monitor.get(key)}

    lj_cutoff = float(monitor.get("lj_cutoff", 6.0))
    data = {
        "neighbors": {"lj_cutoff": lj_cutoff, "coordination_cutoff": float(monitor.get("coordination_cutoff", lj_cutoff))},
        "metrics": {
            "force": zcfg("force", "force_z_threshold"),
            "delta_force": zcfg("delta_force", "delta_force_z_threshold"),
            "displacement": zcfg("displacement", "displacement_z_threshold"),
            "rmin": {"weight": float(weights.get("rmin", 0.0)), "threshold": monitor.get("rmin_threshold")},
            "delta_q": {
                "weight": float(weights.get("delta_q", 0.0)),
                "threshold": monitor.get("delta_q_threshold"),
                "scale": float(monitor.get("coordination_scale", 1.15)),
                "power": float(monitor.get("coordination_smoothing_power", 6)),
            },
            "lj_residual": {
                "weight": float(weights.get("lj_residual", 0.0)),
                "lazy_threshold": float(score.get("lj_lazy_threshold", 3.0)),
            },
            "mlip_force_deviation": {"weight": float(weights.get("mlip_force_deviation", 0.0))},
            "committee": {"weight": float(weights.get("committee", 0.0))},
        },
        "running_stats": {
            "enabled": bool(stats.get("enabled", True)),
            "warmup_frames": int(stats.get("warmup_frames", 5)),
            "min_std": float(stats.get("min_std", 1.0e-8)),
            "warmup_policy": "raw" if legacy_scoring else "suppress",
            "first_frame_deltas": "zero" if legacy_scoring else "not_evaluated",
        },
        "stages": {
            "light_interval": monitor.get("light_interval", 1),
            "physics_interval": monitor.get("physics_interval", "triggered"),
            "full_interval": monitor.get("committee_interval", "triggered"),
            "scheduled_full_interval": max(1, int(online.get("monitor_freq", 1))),
            "screen_threshold": float(score.get("screen_threshold", 4.0)),
            "physics_threshold": float(score.get("physics_threshold", 5.0)),
            "label_threshold": float(score.get("label_threshold", 6.0)),
            "min_trigger_atoms": int(score.get("min_trigger_atoms", 1)),
        },
        # PHAL had no frame actions; keep a pure "save on trigger" policy for replays.
        "actions": {"stop_min_distance": None},
        "driver": {
            "continue_on_error": bool(online.get("continue_on_error", False)),
            "max_errors": max(1, int(online.get("max_errors", 1))),
            "summary_interval": max(0, int(online.get("summary_interval", 0))),
        },
    }
    return load_monitor_config(data)


__all__ = [
    "ActionsConfig",
    "CommitteeConfig",
    "CoordinationConfig",
    "DeltaForceConfig",
    "DriverConfig",
    "GlobalLJConfig",
    "LJResidualConfig",
    "MetricsConfig",
    "MonitorConfig",
    "NeighborConfig",
    "RminConfig",
    "RunningStatsConfig",
    "StagesConfig",
    "ZScoredMetricConfig",
    "load_monitor_config",
    "monitor_config_from_phal",
]
