# Ported from PHAL (hotspot_al.buffer.rolling_buffer) at 96263b6; MIT License, see NOTICE.
"""Rolling pre/post-trigger frame buffer producing :class:`~gmd_active_learning.contracts.EventRecord`.

Semantics (PHAL): the buffer keeps the most recent frames; a trigger opens an event with up to
``pre_trigger_frames`` preceding frames, and the next ``post_trigger_frames`` frames become its
post context. A trigger while an event is open is merged into it (``trigger_steps``; hotspot atoms
unioned, per-atom scores max-ed, reasons de-duplicated) WITHOUT extending the post window.

Deliberate changes:

* Event IDs are ``stable_id("evt", run_id, first_trigger_step)``: reruns reproduce them (F-15).
* An event finalised before its post window is full is marked ``complete=False`` with a
  ``truncation_reason`` (``end_of_stream``, ``stopped``, ``error``, ...) (F-07).
* Finished events are handed to the caller and NOT retained; only ``retain_recent`` summaries are
  kept, so memory is bounded in long runs (PHAL's ``RollingBuffer.events`` grew forever, F-07).
* ``cooldown_steps``: after an event's last frame, triggers at steps < last + cooldown open no new
  event (counted in ``counters["suppressed_by_cooldown"]``). 0 reproduces PHAL.
* :meth:`to_state` / :meth:`from_state` persist history, the open event and cooldown across
  segmented MD runs (JSON-serialisable).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gmd_active_learning.contracts.events import EventRecord
from gmd_active_learning.contracts.frame import Frame
from gmd_active_learning.contracts.ids import stable_id
from gmd_active_learning.contracts.monitoring import MonitorResult
from gmd_active_learning.core.exceptions import MonitorError
from gmd_active_learning.events.config import EventsConfig, load_events_config
from gmd_active_learning.events.serialization import frame_from_state, frame_to_state

logger = logging.getLogger(__name__)

STATE_VERSION = 1


def event_id_for(run_id: str, first_trigger_step: int) -> str:
    """Deterministic event identifier (F-15)."""

    return stable_id("evt", run_id, int(first_trigger_step))


@dataclass(slots=True)
class _PendingEvent:
    event_id: str
    pre_frames: list[Frame]
    trigger_frame: Frame
    post_frames: list[Frame]
    hotspot_atom_ids: list[int]
    atom_scores: np.ndarray
    trigger_reasons: list[str]
    max_score: float
    trigger_steps: list[int]
    trigger_metadata: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_state(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "pre_frames": [frame_to_state(frame) for frame in self.pre_frames],
            "trigger_frame": frame_to_state(self.trigger_frame),
            "post_frames": [frame_to_state(frame) for frame in self.post_frames],
            "hotspot_atom_ids": list(self.hotspot_atom_ids),
            "atom_scores": self.atom_scores.tolist(),
            "trigger_reasons": list(self.trigger_reasons),
            "max_score": self.max_score,
            "trigger_steps": list(self.trigger_steps),
            "trigger_metadata": list(self.trigger_metadata),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> _PendingEvent:
        return cls(
            event_id=state["event_id"],
            pre_frames=[frame_from_state(frame) for frame in state["pre_frames"]],
            trigger_frame=frame_from_state(state["trigger_frame"]),
            post_frames=[frame_from_state(frame) for frame in state["post_frames"]],
            hotspot_atom_ids=[int(atom_id) for atom_id in state["hotspot_atom_ids"]],
            atom_scores=np.asarray(state["atom_scores"], dtype=float),
            trigger_reasons=list(state["trigger_reasons"]),
            max_score=float(state["max_score"]),
            trigger_steps=[int(step) for step in state["trigger_steps"]],
            trigger_metadata=list(state["trigger_metadata"]),
            metadata=dict(state.get("metadata") or {}),
        )


class EventBuffer:
    """Bounded rolling buffer with one open event at a time."""

    def __init__(self, config: EventsConfig | dict[str, Any] | None = None) -> None:
        self.config = load_events_config(config)
        if self.config.maxlen is not None and self.config.maxlen < self.config.pre_trigger_frames:
            logger.warning(
                "events.maxlen=%d < pre_trigger_frames=%d: pre-trigger context will be truncated",
                self.config.maxlen,
                self.config.pre_trigger_frames,
            )
        self._history: deque[Frame] = deque(maxlen=self.config.history_length())
        self._pending: _PendingEvent | None = None
        self._cooldown_until: int | None = None
        self._last_capture_step: int | None = None
        self.recent: deque[dict[str, Any]] = deque(maxlen=self.config.retain_recent)
        self.counters = {"events": 0, "incomplete_events": 0, "merged_triggers": 0, "suppressed_by_cooldown": 0}

    # -- inspection ------------------------------------------------------------------------------

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    @property
    def pending_event_id(self) -> str | None:
        return None if self._pending is None else self._pending.event_id

    @property
    def history_steps(self) -> list[int]:
        return [frame.step for frame in self._history]

    # -- frame flow ------------------------------------------------------------------------------

    def push(self, frame: Frame) -> EventRecord | None:
        """Append a frame; returns the event it completes, if any.

        Call :meth:`capture` for a triggered frame BEFORE pushing it (the trigger frame is then not
        part of its own pre-context, and is not counted as post context).
        """

        if self.config.enabled:
            self._history.append(frame)
        if self._pending is None or frame.step == self._pending.trigger_frame.step:
            return None
        self._pending.post_frames.append(frame)
        if len(self._pending.post_frames) >= self.config.post_trigger_frames:
            return self._finalize(None)
        return None

    def capture(self, frame: Frame, result: MonitorResult, metadata: dict[str, Any] | None = None) -> EventRecord | None:
        """Open or extend an event from a triggered ``MonitorResult`` (no-op for untriggered results)."""

        if not result.triggered:
            return None
        trigger_metadata = {
            "stage": result.stage.value,
            "max_score": float(result.max_score),
            "hotspot_atom_ids": list(map(int, result.hotspot_atom_ids)),
            "trigger_reasons": list(result.trigger_reasons),
            **dict(metadata or {}),
        }
        return self.capture_event(
            frame,
            hotspot_atom_ids=result.hotspot_atom_ids,
            atom_scores=result.atom_scores,
            trigger_reasons=result.trigger_reasons,
            max_score=result.max_score,
            metadata=trigger_metadata,
        )

    def capture_event(
        self,
        trigger_frame: Frame,
        *,
        hotspot_atom_ids: list[int],
        atom_scores: np.ndarray,
        trigger_reasons: list[str],
        max_score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EventRecord | None:
        """Start an event, or merge a repeated trigger into the open one (window not extended)."""

        if not self.config.enabled:
            return None
        if self._last_capture_step == trigger_frame.step:
            return None
        scores = np.asarray(atom_scores, dtype=float)
        if scores.shape != (trigger_frame.n_atoms,):
            raise MonitorError(f"atom_scores shape {scores.shape} does not match {trigger_frame.n_atoms} atoms")
        score_max = float(np.max(scores)) if max_score is None and len(scores) else float(max_score or 0.0)
        entry = {"step": trigger_frame.step, **dict(metadata or {})}
        if self._pending is not None:
            self._merge(trigger_frame, hotspot_atom_ids, scores, trigger_reasons, score_max, entry)
            self._last_capture_step = trigger_frame.step
            return None
        if self._cooldown_until is not None and trigger_frame.step < self._cooldown_until:
            self.counters["suppressed_by_cooldown"] += 1
            logger.info("trigger at step %d suppressed by cooldown (until step %d)", trigger_frame.step, self._cooldown_until)
            return None

        frames = list(self._history)
        if frames and frames[-1].step == trigger_frame.step:
            frames = frames[:-1]
        pre_frames = frames[-self.config.pre_trigger_frames :] if self.config.pre_trigger_frames else []
        self._pending = _PendingEvent(
            event_id=event_id_for(trigger_frame.run_id, trigger_frame.step),
            pre_frames=pre_frames,
            trigger_frame=trigger_frame,
            post_frames=[],
            hotspot_atom_ids=sorted({int(atom_id) for atom_id in hotspot_atom_ids}),
            atom_scores=scores.copy(),
            trigger_reasons=list(dict.fromkeys(trigger_reasons)),
            max_score=score_max,
            trigger_steps=[trigger_frame.step],
            trigger_metadata=[entry],
        )
        self._last_capture_step = trigger_frame.step
        logger.info("opened event %s at step %d", self._pending.event_id, trigger_frame.step)
        if self.config.post_trigger_frames == 0:
            return self._finalize(None)
        return None

    def _merge(
        self,
        trigger_frame: Frame,
        hotspot_atom_ids: list[int],
        scores: np.ndarray,
        trigger_reasons: list[str],
        max_score: float,
        entry: dict[str, Any],
    ) -> None:
        pending = self._pending
        assert pending is not None
        if not np.array_equal(trigger_frame.atom_ids, pending.trigger_frame.atom_ids):
            raise MonitorError(
                f"cannot merge the trigger at step {trigger_frame.step} into event {pending.event_id}: atom set or order differs"
            )
        pending.hotspot_atom_ids = sorted(set(pending.hotspot_atom_ids).union(int(atom_id) for atom_id in hotspot_atom_ids))
        pending.atom_scores = np.maximum(pending.atom_scores, scores)
        pending.trigger_reasons = list(dict.fromkeys([*pending.trigger_reasons, *trigger_reasons]))
        pending.max_score = max(pending.max_score, max_score)
        if trigger_frame.step not in pending.trigger_steps:
            pending.trigger_steps.append(trigger_frame.step)
        pending.trigger_metadata.append(entry)
        self.counters["merged_triggers"] += 1

    def flush(self, reason: str = "end_of_stream") -> EventRecord | None:
        """Finalise the open event now; it is marked incomplete (with ``reason``) if post context is missing."""

        if self._pending is None:
            return None
        return self._finalize(reason)

    def _finalize(self, reason: str | None) -> EventRecord:
        pending = self._pending
        assert pending is not None
        requested = self.config.post_trigger_frames
        complete = len(pending.post_frames) >= requested
        event = EventRecord(
            event_id=pending.event_id,
            run_id=pending.trigger_frame.run_id,
            trigger_step=pending.trigger_frame.step,
            trigger_steps=list(pending.trigger_steps),
            pre_frames=list(pending.pre_frames),
            trigger_frame=pending.trigger_frame,
            post_frames=list(pending.post_frames),
            post_frames_requested=requested,
            pre_frames_requested=self.config.pre_trigger_frames,
            complete=complete,
            hotspot_atom_ids=list(pending.hotspot_atom_ids),
            trigger_reasons=list(pending.trigger_reasons),
            max_score=float(pending.max_score),
            truncation_reason=None if complete else (reason or "unknown"),
            model_version=pending.trigger_frame.model_version,
            metadata={
                **pending.metadata,
                "pre_frames_requested": self.config.pre_trigger_frames,
                "trigger_count": len(pending.trigger_metadata),
                "latest_trigger_step": pending.trigger_steps[-1],
                "trigger_metadata": list(pending.trigger_metadata),
                "atom_scores": pending.atom_scores.tolist(),
            },
        )
        self._pending = None
        last_step = event.frames[-1].step
        if self.config.cooldown_steps > 0:
            self._cooldown_until = last_step + self.config.cooldown_steps
        self.counters["events"] += 1
        if not complete:
            self.counters["incomplete_events"] += 1
        self.recent.append({key: value for key, value in event.summary().items() if key != "metadata"})
        logger.info(
            "finalised event %s (trigger steps %s, %d pre / %d post frames, complete=%s)",
            event.event_id, event.trigger_steps, len(event.pre_frames), len(event.post_frames), complete,
        )
        return event

    # -- persistence -----------------------------------------------------------------------------

    def to_state(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "history": [frame_to_state(frame) for frame in self._history],
            "pending": None if self._pending is None else self._pending.to_state(),
            "cooldown_until": self._cooldown_until,
            "last_capture_step": self._last_capture_step,
            "recent": list(self.recent),
            "counters": dict(self.counters),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if int(state.get("version", STATE_VERSION)) != STATE_VERSION:
            raise MonitorError(f"unsupported event buffer state version {state.get('version')}")
        self._history.clear()
        for frame in state.get("history", []):
            self._history.append(frame_from_state(frame))
        pending = state.get("pending")
        self._pending = None if pending is None else _PendingEvent.from_state(pending)
        self._cooldown_until = state.get("cooldown_until")
        self._last_capture_step = state.get("last_capture_step")
        self.recent.clear()
        self.recent.extend(state.get("recent", []))
        self.counters.update(state.get("counters", {}))

    @classmethod
    def from_state(cls, state: dict[str, Any], config: EventsConfig | dict[str, Any] | None = None) -> EventBuffer:
        buffer = cls(config)
        buffer.load_state(state)
        return buffer


__all__ = ["EventBuffer", "event_id_for"]
