"""Builds a ready-to-run controller from a validated configuration (composition root).

This is the only place where concrete backends are constructed; the controller and the domain
components only see the protocols in ``workflow.protocols``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gmd_active_learning.adapters.gmd.engine import GMDConfig, GMDEngine
from gmd_active_learning.adapters.sgt.backend import GMDSGTBackend, SGTConfig
from gmd_active_learning.config import LoadedConfig
from gmd_active_learning.contracts import EventRecord, ExecutionStatus, Frame, LabelResult, LabelTask, MonitorResult
from gmd_active_learning.core.exceptions import ConfigurationError
from gmd_active_learning.workflow.controller import ActiveLearningController, WorkflowConfig


class DriverFrameMonitor:
    """Adapts ``monitors.online.OnlineMonitorDriver`` to the controller's ``FrameMonitor`` protocol."""

    def __init__(self, driver: Any) -> None:
        self.driver = driver
        self._completed: list[EventRecord] = []

    def process(self, frame: Frame) -> MonitorResult:
        result, events = self.driver.process(frame)
        self._completed.extend(events)
        return result

    def drain_events(self) -> list[EventRecord]:
        events, self._completed = self._completed, []
        return events

    def close(self, reason: str) -> list[EventRecord]:
        event = self.driver.finish(reason)
        events = self.drain_events()
        if event is not None and all(item.event_id != event.event_id for item in events):
            events.append(event)
        return events

    def to_state(self) -> dict[str, Any]:
        return self.driver.to_state()

    def load_state(self, state: dict[str, Any]) -> None:
        self.driver.load_state(state)


class ServiceLabeler:
    """Adapts ``labeling.service.LabelingService`` to the controller's ``Labeler`` protocol.

    The service already owns the retry ladder (per failure class, bounded) and multi-stage
    labeling (cap relaxation then single point), so ``retry`` returns None here and the task is
    reported terminal only once the service has finished it.
    """

    def __init__(self, service: Any) -> None:
        self.service = service

    def prepare(self, region: Any) -> LabelTask:
        return self.service.prepare(region)

    def find_submission(self, task: LabelTask) -> LabelTask | None:
        current = self.service.load(task.task_id)
        return current if current.metadata.get("attempts") else None

    def submit(self, task: LabelTask) -> LabelTask:
        return self._view(self.service.submit(task.task_id))

    def poll(self, task: LabelTask) -> LabelTask:
        return self._view(self.service.poll(task.task_id))

    def reconcile(self, task: LabelTask) -> LabelTask:
        return self._view(self.service.reconcile(task.task_id))

    def collect(self, task: LabelTask) -> LabelResult:
        return self.service.collect(task.task_id)

    def retry(self, task: LabelTask, result: LabelResult | None) -> LabelTask | None:
        return None

    def cancel(self, task: LabelTask) -> LabelTask:
        return self.service.cancel(task.task_id)

    @staticmethod
    def _view(task: LabelTask) -> LabelTask:
        done = task.metadata.get("stage") == "done"
        if done:
            # FAILED and CANCELLED stay what they are (a cancelled job is not a completed calculation
            # with an invalid result); every other finished state means the service collected a result.
            if task.execution not in (ExecutionStatus.FAILED, ExecutionStatus.CANCELLED):
                task.execution = ExecutionStatus.COMPLETED
        elif task.execution.terminal:
            task.execution = ExecutionStatus.RUNNING  # an intermediate stage finished; the next one is queued
        return task


def hotspot_grouper(merge_radius: float) -> Any:
    """One region per hotspot cluster, as PHAL's ``detect_hotspots`` + extraction strategy did.

    Atoms are clustered by minimum-image single linkage within ``merge_radius``. With per-atom
    scores the clusters are ordered by max score descending (ties: first appearance), PHAL's region
    order (``hotspots.sort(key=max_score, reverse=True)``); without scores they keep
    first-appearance order.
    """

    from gmd_active_learning.hotspots.cluster import cluster_anomalous_atoms

    def group(frame: Frame, atom_ids: list[int], scores: Any = None) -> list[list[int]]:
        indices = sorted(frame.index_of(atom_ids).tolist())
        clusters = cluster_anomalous_atoms(indices, frame.positions, merge_radius=merge_radius, cell=frame.cell, pbc=frame.pbc)
        if scores is not None:
            values = np.asarray(scores, dtype=float)
            # Python's sort is stable and clusters arrive ordered by smallest member index.
            clusters = sorted(clusters, key=lambda members: -float(np.max(values[members])))
        return [[int(frame.atom_ids[index]) for index in cluster] for cluster in clusters]

    return group


def build_controller(config: LoadedConfig, *, work_dir: str | Path) -> ActiveLearningController:
    from gmd_active_learning.datasets.config import DatasetsConfig
    from gmd_active_learning.datasets.store import VersionedDatasetStore
    from gmd_active_learning.events.config import EventsConfig
    from gmd_active_learning.extraction.config import CappingConfig, EmbeddingConfig, ExtractionConfig
    from gmd_active_learning.extraction.dispatch import EventRegionExtractor
    from gmd_active_learning.hotspots.config import HotspotsConfig
    from gmd_active_learning.labeling.service import LabelingService
    from gmd_active_learning.monitors.config import MonitorConfig
    from gmd_active_learning.monitors.online import OnlineMonitorDriver

    work_dir = Path(work_dir).resolve()
    sections = config.sections
    workflow = sections["workflow"]
    if not isinstance(workflow, WorkflowConfig):
        raise ConfigurationError("workflow section has the wrong type")
    from gmd_active_learning.adapters.registry import BackendRole, create_backend

    md = create_backend(config.raw, BackendRole.MD)
    sgt = create_backend(config.raw, BackendRole.MLIP)
    if hasattr(sgt, "set_engine_smoke") and hasattr(md, "smoke_test"):
        sgt.set_engine_smoke(md.smoke_test)
    hotspots = sections.get("hotspots") or HotspotsConfig()
    driver = OnlineMonitorDriver(
        sections.get("monitor") or MonitorConfig(),
        sections.get("events") or EventsConfig(),
        hotspots,
    )
    extractor = EventRegionExtractor(
        sections.get("extraction") or ExtractionConfig(),  # type: ignore[arg-type]
        sections.get("capping") or CappingConfig(),  # type: ignore[arg-type]
        sections.get("embedding") or EmbeddingConfig(),  # type: ignore[arg-type]
        grouper=hotspot_grouper(hotspots.merge_radius) if hotspots.enabled else None,  # type: ignore[attr-defined]
        state_dir=work_dir / "extraction_state",
    )
    service = LabelingService.from_config(config.raw, base_dir=work_dir)
    datasets = VersionedDatasetStore(work_dir / "datasets", sections.get("datasets") or DatasetsConfig())  # type: ignore[arg-type]
    return ActiveLearningController(
        workflow, work_dir=work_dir, md=md, monitor=DriverFrameMonitor(driver), extractor=extractor,
        labeler=ServiceLabeler(service), datasets=datasets, trainer=sgt, enricher=sgt, config_hash=config.config_hash,
    )


__all__ = ["DriverFrameMonitor", "ServiceLabeler", "build_controller", "GMDConfig", "SGTConfig"]
