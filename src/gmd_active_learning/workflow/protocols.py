"""Interfaces the workflow controller depends on. Concrete adapters (GMD, GMD_SGT, CP2K,
schedulers) and domain components (monitoring, extraction, datasets) implement these; the
controller never constructs backends itself (see ``workflow.assembly``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from gmd_active_learning.contracts import (
    DatasetVersion,
    EventRecord,
    ExecutionStatus,
    ExtractedRegion,
    Frame,
    FrameAction,
    LabelResult,
    LabelTask,
    ModelArtifact,
    MonitorResult,
)


@dataclass(slots=True)
class SegmentRequest:
    run_id: str
    segment_index: int
    work_dir: Path
    model: ModelArtifact
    n_steps: int
    restart_from: Path | None  # checkpoint of the previous segment; None for the first segment
    previous_model: ModelArtifact | None  # set when the model changes at this boundary (recorded promotion)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SegmentResult:
    status: str  # "completed" | "stopped" | "failed"
    end_step: int | None
    checkpoint: Path | None
    frames_path: Path | None
    detail: dict[str, Any] = field(default_factory=dict)


FrameCallback = Callable[[Frame], FrameAction]


@runtime_checkable
class MDEngine(Protocol):
    def run_segment(self, request: SegmentRequest, on_frame: FrameCallback) -> SegmentResult: ...

    def completed_segment(self, request: SegmentRequest) -> SegmentResult | None:
        """Result of a segment that already finished (e.g. before a controller crash), else None."""

    def read_frames(self, result: SegmentResult, request: SegmentRequest) -> Iterable[Frame]: ...


@runtime_checkable
class FrameEnricher(Protocol):
    """Adds committee forces (or other model evaluations) to a frame before monitoring."""

    def enrich(self, frame: Frame, model: ModelArtifact) -> Frame: ...


@runtime_checkable
class FrameMonitor(Protocol):
    def process(self, frame: Frame) -> MonitorResult: ...

    def drain_events(self) -> list[EventRecord]:
        """Events completed since the last call."""

    def close(self, reason: str) -> list[EventRecord]:
        """Flush open events (marked incomplete with ``reason``)."""

    def to_state(self) -> dict[str, Any]: ...

    def load_state(self, state: dict[str, Any]) -> None: ...


@runtime_checkable
class RegionExtractor(Protocol):
    def extract(self, event: EventRecord) -> list[tuple[ExtractedRegion, float]]:
        """Labelable regions (already capped if configured) with their candidate scores."""


@runtime_checkable
class Labeler(Protocol):
    def prepare(self, region: ExtractedRegion) -> LabelTask:
        """Deterministic: the same region and settings always give the same task_id/work_dir."""

    def find_submission(self, task: LabelTask) -> LabelTask | None:
        """Return the task with its job if it was already submitted (e.g. before a crash)."""

    def submit(self, task: LabelTask) -> LabelTask: ...

    def poll(self, task: LabelTask) -> LabelTask:
        """Refresh execution status; UNKNOWN must be resolved via ``reconcile`` before any retry."""

    def reconcile(self, task: LabelTask) -> LabelTask: ...

    def collect(self, task: LabelTask) -> LabelResult: ...

    def retry(self, task: LabelTask, result: LabelResult | None) -> LabelTask | None:
        """A new attempt for a failed/invalid task, or None when the retry policy is exhausted."""

    def cancel(self, task: LabelTask) -> LabelTask: ...


@runtime_checkable
class DatasetStore(Protocol):
    def add_sample(self, region: ExtractedRegion, label: LabelResult, *, provenance: dict[str, Any]) -> tuple[str, Path]:
        """Write one labeled sample; idempotent per (region_id, task_id). Returns (sample_id, path)."""

    def publish(self, sample_paths: list[Path], *, parent: DatasetVersion | None, split_groups: dict[str, str]) -> DatasetVersion: ...

    def get(self, version: str) -> DatasetVersion: ...


@dataclass(slots=True)
class TrainOutput:
    checkpoint: Path
    species: list[int]
    cutoff: float | None
    mode: str  # "fresh" | "finetune"
    summary: dict[str, Any]


@runtime_checkable
class Trainer(Protocol):
    def train(self, dataset: DatasetVersion, *, parent: ModelArtifact | None, work_dir: Path) -> TrainOutput: ...

    def evaluate(self, checkpoint: Path, dataset: DatasetVersion) -> dict[str, float]:
        """Metrics on the dataset's validation split: force_rmse, energy_rmse_per_atom, n_validation."""

    def export(self, checkpoint: Path, work_dir: Path) -> Path: ...

    def validate_export(self, checkpoint: Path, export: Path, dataset: DatasetVersion) -> dict[str, Any]:
        """Compare exported vs checkpoint predictions; returns at least max_force_deviation."""

    def engine_smoke_test(self, export: Path, work_dir: Path) -> dict[str, Any]:
        """Run the export inside the MD engine briefly; returns at least {"passed": bool}."""
