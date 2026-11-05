# Ported from PHAL (hotspot_al.training.retrain_trigger) at 96263b6; MIT License, see NOTICE.
"""Retraining trigger: sample count, elapsed time, or manual request.

Changes from PHAL (ledger TRN-005, finding F-13): the first labeled sample no longer bypasses
``min_new_samples`` unless explicitly configured, the decision never advances state by itself
(the controller records the training run), and training is refused while any labeling task of the
current selection is still non-terminal (no retraining on incomplete data).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, ConfigDict, Field


class RetrainTriggerConfig(BaseModel):
    """``retraining`` config section."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    min_new_samples: int = Field(10, ge=1)
    interval_hours: float | None = Field(24.0, gt=0.0, description="train after this long if >=1 new sample; None disables")
    train_on_first_samples: bool = False
    wait_for_pending_labels: bool = True


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    due: bool
    reason: str
    new_samples: int


def evaluate_trigger(
    config: RetrainTriggerConfig,
    *,
    new_samples: int,
    pending_tasks: int,
    last_trained_at: str | None,
    manual: bool = False,
    now: datetime | None = None,
) -> TriggerDecision:
    if not config.enabled and not manual:
        return TriggerDecision(False, "disabled", new_samples)
    if config.wait_for_pending_labels and pending_tasks > 0:
        return TriggerDecision(False, "labels_pending", new_samples)
    if new_samples <= 0:
        return TriggerDecision(False, "no_new_samples", new_samples)
    if manual:
        return TriggerDecision(True, "manual", new_samples)
    if new_samples >= config.min_new_samples:
        return TriggerDecision(True, "sample_count", new_samples)
    if last_trained_at is None:
        if config.train_on_first_samples:
            return TriggerDecision(True, "initial_samples", new_samples)
        return TriggerDecision(False, "below_min_new_samples", new_samples)
    if config.interval_hours is not None:
        elapsed = (now or datetime.now(timezone.utc)) - datetime.fromisoformat(last_trained_at)
        if elapsed >= timedelta(hours=config.interval_hours):
            return TriggerDecision(True, "time_interval", new_samples)
    return TriggerDecision(False, "below_min_new_samples", new_samples)
