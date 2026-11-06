# Ported from PHAL (hotspot_al.monitor.online_monitor.OnlineMonitor run loop) at 96263b6; MIT License, see NOTICE.
"""Online monitor driver: frames in, ``MonitorResult`` + ``FrameAction`` out, events via callback.

The driver owns no MD process. The GMD adapter (or any caller) either

* calls :meth:`OnlineMonitorDriver.process` once per frame (callback mode) and honours
  ``result.action`` (CONTINUE / SAVE / CHECKPOINT / STOP) at its next control boundary, then
  :meth:`finish` (end of run) or :meth:`to_state` (end of a segment; the open event, cooldown,
  running statistics and temporal references resume with :meth:`load_state`); or
* hands :meth:`run` an iterable of frames or a :class:`FrameSource`. ``next_frame`` returning
  ``None`` means end of stream and :data:`POLL_TIMEOUT` means "no frame yet" (PHAL overloaded None
  for both, F-19); ``driver.max_consecutive_timeouts`` timeouts in a row end the run.

Per frame (PHAL order): evaluate -> capture a trigger into the event buffer -> push the frame
(which may complete an event). Completed events go to ``on_event`` exactly once (and to the
configured event store). When the run ends or is stopped with an event still open, the event is
flushed with ``complete=False`` and a ``truncation_reason`` (F-07).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from gmd_active_learning.contracts.events import EventRecord
from gmd_active_learning.contracts.frame import Frame
from gmd_active_learning.contracts.monitoring import FrameAction, MonitorResult, Stage
from gmd_active_learning.core.exceptions import MonitorError
from gmd_active_learning.events.config import EventsConfig, load_events_config
from gmd_active_learning.events.event_store import EventStore
from gmd_active_learning.events.phal_view import monitor_metadata_for_event
from gmd_active_learning.events.rolling_buffer import EventBuffer
from gmd_active_learning.hotspots.config import HotspotsConfig, load_hotspots_config
from gmd_active_learning.hotspots.detector import hotspots_from_result
from gmd_active_learning.monitors.config import MonitorConfig, load_monitor_config
from gmd_active_learning.monitors.force_disagreement import CommitteeForceProvider, ReferenceForceProvider
from gmd_active_learning.monitors.global_lj_monitor import GlobalLJMonitor
from gmd_active_learning.monitors.staged_monitor import StagedOODMonitor
from gmd_active_learning.utils.atomic_io import write_json_atomic

logger = logging.getLogger(__name__)

EventCallback = Callable[[EventRecord], None]
ProgressCallback = Callable[[dict[str, Any]], None]
STATE_VERSION = 1


class _PollTimeout:
    _instance: _PollTimeout | None = None

    def __new__(cls) -> _PollTimeout:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "POLL_TIMEOUT"


POLL_TIMEOUT = _PollTimeout()


@runtime_checkable
class FrameSource(Protocol):
    """Pull-style frame source: a Frame, ``None`` (end of stream) or ``POLL_TIMEOUT`` (nothing yet)."""

    def next_frame(self, timeout: float | None = None) -> Frame | None | _PollTimeout: ...


@runtime_checkable
class FrameMonitor(Protocol):
    engine: str

    def evaluate(self, frame: Frame) -> MonitorResult: ...

    def to_state(self) -> dict[str, Any]: ...

    def load_state(self, state: dict[str, Any]) -> None: ...


def build_frame_monitor(
    config: MonitorConfig | dict[str, Any] | None,
    *,
    reference_forces: ReferenceForceProvider | None = None,
    committee_provider: CommitteeForceProvider | None = None,
    lj_estimator: Any | None = None,
) -> FrameMonitor:
    """The monitor selected by ``monitor.engine`` (``staged`` = PHAL atom-wise, ``global_lj`` = GMD frame-level)."""

    settings = load_monitor_config(config)
    if settings.engine == "global_lj":
        if lj_estimator is not None:
            raise MonitorError("lj_estimator applies to the staged engine only")
        return GlobalLJMonitor(settings)
    return StagedOODMonitor(
        settings,
        reference_forces=reference_forces,
        committee_provider=committee_provider,
        lj_estimator=lj_estimator,
    )


@dataclass(slots=True)
class RunSummary:
    results: list[MonitorResult] = field(default_factory=list)
    events: list[EventRecord] = field(default_factory=list)
    stop_reason: str = "end_of_stream"
    errors: int = 0
    last_action: FrameAction = FrameAction.CONTINUE


class OnlineMonitorDriver:
    """Drives a :class:`FrameMonitor` over a frame stream with events, hotspots and actions."""

    def __init__(
        self,
        monitor_config: MonitorConfig | dict[str, Any] | None = None,
        events_config: EventsConfig | dict[str, Any] | None = None,
        hotspots_config: HotspotsConfig | dict[str, Any] | None = None,
        *,
        monitor: FrameMonitor | None = None,
        reference_forces: ReferenceForceProvider | None = None,
        committee_provider: CommitteeForceProvider | None = None,
        on_event: EventCallback | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.config = load_monitor_config(monitor_config)
        self.events_config = load_events_config(events_config)
        self.hotspots_config = load_hotspots_config(hotspots_config)
        self.monitor: FrameMonitor = monitor or build_frame_monitor(
            self.config, reference_forces=reference_forces, committee_provider=committee_provider
        )
        self.buffer = EventBuffer(self.events_config)
        self.on_event = on_event
        self.on_progress = on_progress
        self.store = (
            EventStore(self.events_config.store.directory, write_extxyz=self.events_config.store.write_extxyz)
            if self.events_config.store.enabled and self.events_config.store.directory
            else None
        )
        self.last_action = FrameAction.CONTINUE
        self.stats: dict[str, int] = {
            "processed_frames": 0,
            "triggered_frames": 0,
            "events": 0,
            "errors": 0,
            "skipped_frames": 0,
            "light_frames": 0,
            "physics_frames": 0,
            "full_frames": 0,
        }

    # -- per-frame API ---------------------------------------------------------------------------

    def process(self, frame: Frame) -> tuple[MonitorResult, list[EventRecord]]:
        """Monitor one frame; returns the result (with ``action``) and any events completed by it."""

        result = self.monitor.evaluate(frame)
        if result.triggered and self.hotspots_config.enabled:
            result.metadata["hotspots"] = [
                hotspot.to_dict() for hotspot in hotspots_from_result(frame, result, self.hotspots_config)
            ]
        completed: list[EventRecord] = []
        event = self.buffer.capture(
            frame,
            result,
            metadata={"frame_index": result.metadata.get("frame_index"), "monitor": monitor_metadata_for_event(result.metadata)},
        )
        if event is not None:
            completed.append(event)
        event = self.buffer.push(frame)
        if event is not None:
            completed.append(event)
        if self.buffer.has_pending:
            result.metadata["open_event_id"] = self.buffer.pending_event_id
        for item in completed:
            self._emit(item)
        self.last_action = result.action
        self._record_progress(result)
        return result, completed

    def finish(self, reason: str = "end_of_stream") -> EventRecord | None:
        """End of run: flush the open event (incomplete -> ``truncation_reason=reason``)."""

        event = self.buffer.flush(reason)
        if event is not None:
            self._emit(event)
        return event

    # -- loop API --------------------------------------------------------------------------------

    def run(self, frames: Iterable[Frame] | FrameSource, *, max_frames: int | None = None) -> RunSummary:
        """Process frames until end of stream, ``max_frames``, a STOP action, or too many errors/timeouts."""

        summary = RunSummary()
        driver_cfg = self.config.driver
        consecutive_errors = 0
        consecutive_timeouts = 0
        processed = 0
        pull = self._puller(frames, driver_cfg.poll_timeout_s)
        reason = "end_of_stream"
        try:
            while max_frames is None or processed < max_frames:
                item = pull()
                if item is None:
                    reason = "end_of_stream"
                    break
                if item is POLL_TIMEOUT:
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= driver_cfg.max_consecutive_timeouts:
                        reason = "timeout"
                        logger.warning("frame source timed out %d times in a row; stopping", consecutive_timeouts)
                        break
                    continue
                consecutive_timeouts = 0
                assert isinstance(item, Frame)
                try:
                    result, events = self.process(item)
                except Exception:
                    consecutive_errors += 1
                    summary.errors += 1
                    self._record_error(item, consecutive_errors)
                    if not driver_cfg.continue_on_error or consecutive_errors >= driver_cfg.max_errors:
                        reason = "error"
                        raise
                    continue
                consecutive_errors = 0
                processed += 1
                summary.results.append(result)
                summary.events.extend(events)
                if result.action is FrameAction.STOP and driver_cfg.stop_on_stop_action:
                    reason = "stopped"
                    logger.warning("STOP requested at step %d (%s)", result.step, result.metadata.get("action_reasons"))
                    break
            else:
                reason = "max_frames"
        except KeyboardInterrupt:
            reason = "interrupted"
            logger.info("online monitor interrupted after %d frames", processed)
        finally:
            event = self.finish(reason)
            if event is not None:
                summary.events.append(event)
            summary.stop_reason = reason
            summary.last_action = self.last_action
        return summary

    @staticmethod
    def _puller(frames: Iterable[Frame] | FrameSource, timeout: float | None) -> Callable[[], Frame | None | _PollTimeout]:
        if isinstance(frames, FrameSource):
            source = frames
            return lambda: source.next_frame(timeout)
        iterator: Iterator[Frame] = iter(frames)
        return lambda: next(iterator, None)

    # -- persistence -----------------------------------------------------------------------------

    def to_state(self) -> dict[str, Any]:
        """JSON-serialisable state to resume a segmented run (monitor, open event, cooldown, counters)."""

        return {
            "version": STATE_VERSION,
            "monitor": self.monitor.to_state(),
            "events": self.buffer.to_state(),
            "stats": dict(self.stats),
            "last_action": self.last_action.value,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if int(state.get("version", STATE_VERSION)) != STATE_VERSION:
            raise MonitorError(f"unsupported driver state version {state.get('version')}")
        self.monitor.load_state(state["monitor"])
        self.buffer.load_state(state["events"])
        self.stats.update(state.get("stats", {}))
        self.last_action = FrameAction(state.get("last_action", FrameAction.CONTINUE.value))

    def save_state(self, path: str) -> None:
        write_json_atomic(path, self.to_state())

    # -- internals -------------------------------------------------------------------------------

    def _emit(self, event: EventRecord) -> None:
        self.stats["events"] += 1
        if self.store is not None:
            self.store(event)
        if self.on_event is not None:
            self.on_event(event)

    def _record_progress(self, result: MonitorResult) -> None:
        self.stats["processed_frames"] += 1
        if result.triggered:
            self.stats["triggered_frames"] += 1
        key = {Stage.SKIPPED: "skipped_frames", Stage.LIGHT: "light_frames", Stage.PHYSICS: "physics_frames", Stage.FULL: "full_frames"}
        self.stats[key[result.stage]] += 1
        payload = {
            **self.stats,
            "last_step": result.step,
            "last_stage": result.stage.value,
            "last_max_score": result.max_score,
            "last_action": result.action.value,
        }
        self._emit_progress(payload)
        interval = self.config.driver.summary_interval
        if interval and self.stats["processed_frames"] % interval == 0:
            logger.info("monitor progress %s", payload)

    def _record_error(self, frame: Frame, consecutive: int) -> None:
        self.stats["errors"] += 1
        logger.exception("monitor error at step %d (consecutive errors %d)", frame.step, consecutive)
        self._emit_progress({**self.stats, "last_step": frame.step, "last_stage": "error", "consecutive_errors": consecutive})

    def _emit_progress(self, payload: dict[str, Any]) -> None:
        if self.on_progress is not None:
            self.on_progress(payload)
        if self.config.driver.progress_file:
            write_json_atomic(self.config.driver.progress_file, payload)


__all__ = [
    "POLL_TIMEOUT",
    "EventCallback",
    "FrameMonitor",
    "FrameSource",
    "OnlineMonitorDriver",
    "ProgressCallback",
    "RunSummary",
    "build_frame_monitor",
]
