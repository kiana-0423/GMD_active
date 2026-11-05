"""Event contract: frames buffered around a trigger, with explicit completeness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gmd_active_learning.contracts.frame import Frame


@dataclass(slots=True)
class EventRecord:
    """A trigger with its pre/post context.

    ``event_id`` is deterministic (run id + first trigger step), so re-running the same
    trajectory reproduces it. ``complete`` is False when the run ended (or was stopped)
    before ``post_frames_requested`` post-trigger frames were collected; ``truncation_reason``
    then says why. Overlapping triggers inside an open event extend it (``trigger_steps``).
    """

    event_id: str
    run_id: str
    trigger_step: int
    trigger_steps: list[int]
    pre_frames: list[Frame]
    trigger_frame: Frame
    post_frames: list[Frame]
    post_frames_requested: int
    complete: bool
    hotspot_atom_ids: list[int]
    trigger_reasons: list[str]
    max_score: float
    truncation_reason: str | None = None
    model_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frames(self) -> list[Frame]:
        return [*self.pre_frames, self.trigger_frame, *self.post_frames]

    def summary(self) -> dict[str, Any]:
        """JSON-friendly description without frame payloads."""

        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "trigger_step": self.trigger_step,
            "trigger_steps": list(self.trigger_steps),
            "pre_frame_steps": [frame.step for frame in self.pre_frames],
            "post_frame_steps": [frame.step for frame in self.post_frames],
            "post_frames_requested": self.post_frames_requested,
            "complete": self.complete,
            "truncation_reason": self.truncation_reason,
            "hotspot_atom_ids": list(map(int, self.hotspot_atom_ids)),
            "trigger_reasons": list(self.trigger_reasons),
            "max_score": self.max_score,
            "model_version": self.model_version,
            "metadata": self.metadata,
        }
