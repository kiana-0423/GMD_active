"""Deterministic fake backends for workflow tests (never used in production code paths).

Physics: an Ar cluster under ASE Lennard-Jones. The "DFT" labeler returns exact LJ forces for a
region; the fake MD engine produces frames whose forces are LJ plus a model-dependent error, so a
better-trained fake model really yields smaller errors. Failure injection hooks simulate crashes,
unknown scheduler states, invalid labels, bad models and failing MD segments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.io import read, write

from gmd_active_learning.contracts import (
    DatasetVersion,
    EventRecord,
    ExecutionStatus,
    ExtractedRegion,
    Frame,
    FrameAction,
    LabelResult,
    LabelTask,
    MetricValues,
    ModelArtifact,
    MonitorResult,
    RegionRole,
    SchedulerJob,
    Stage,
    ValidationStatus,
    content_hash,
    stable_id,
)
from gmd_active_learning.utils.atomic_io import write_json_atomic
from gmd_active_learning.workflow.protocols import SegmentRequest, SegmentResult, TrainOutput


class SimulatedCrash(BaseException):
    """Stands in for a killed controller process (not an Exception, so nothing swallows it)."""


EPS, SIGMA = 0.0104, 3.40


def lj_forces(positions: np.ndarray) -> np.ndarray:
    atoms = Atoms("Ar" * len(positions), positions=positions)
    atoms.calc = LennardJones(epsilon=EPS, sigma=SIGMA, rc=10.0, smooth=True)
    return atoms.get_forces()


def base_cluster() -> np.ndarray:
    grid = np.array([[i, j, k] for i in range(3) for j in range(3) for k in range(3)], dtype=float) * 3.8
    return grid


def model_error_scale(model: ModelArtifact) -> float:
    data = json.loads(Path(model.checkpoint_path).read_text())
    return float(data["error_scale"])


# --------------------------------------------------------------------------- MD
@dataclass
class FakeMD:
    frames_per_segment: int = 6
    anomaly_segments: tuple[int, ...] = (0,)
    fail_segments_with_model: set[str] = field(default_factory=set)
    crash_after_segment_files: bool = False
    runs: list[dict[str, Any]] = field(default_factory=list)

    def _frames(self, request: SegmentRequest) -> list[Frame]:
        rng = np.random.default_rng(1000 + request.segment_index)
        base = base_cluster()
        scale = model_error_scale(request.model)
        start = 0 if request.restart_from is None else json.loads(Path(request.restart_from).read_text())["step"]
        frames = []
        for k in range(self.frames_per_segment):
            positions = base + rng.normal(0.0, 0.05, base.shape)
            if request.segment_index in self.anomaly_segments and k == 3:
                positions[13] += np.array([1.4, 0.0, 0.0])  # squeeze the central atom toward a neighbour
            forces = lj_forces(positions) + scale * rng.normal(0.0, 1.0, positions.shape)
            frames.append(Frame(run_id=request.run_id, step=start + (k + 1) * 10, atom_ids=np.arange(len(base)) + 100,
                                numbers=np.full(len(base), 18), positions=positions, cell=None, pbc=(False,) * 3, forces=forces,
                                model_version=request.model.model_id))
        return frames

    def run_segment(self, request: SegmentRequest, on_frame) -> SegmentResult:  # noqa: ANN001
        self.runs.append({"segment": request.segment_index, "model": request.model.model_id,
                          "previous_model": None if request.previous_model is None else request.previous_model.model_id})
        request.work_dir.mkdir(parents=True, exist_ok=True)
        if request.model.model_id in self.fail_segments_with_model:
            return SegmentResult("failed", None, None, None, {"error": "model produced exploding forces"})
        processed, status = [], "completed"
        for frame in self._frames(request):
            processed.append(frame)
            if on_frame(frame) is FrameAction.STOP:
                status = "stopped"
                break
        return self._write(request, processed, status)

    def _write(self, request: SegmentRequest, frames: list[Frame], status: str) -> SegmentResult:
        write(request.work_dir / "frames.extxyz", [frame.to_atoms() for frame in frames], format="extxyz")
        checkpoint = request.work_dir / "checkpoint.json"
        write_json_atomic(checkpoint, {"step": frames[-1].step})
        write_json_atomic(request.work_dir / "status.json", {"state": status, "step": frames[-1].step})
        if self.crash_after_segment_files:
            self.crash_after_segment_files = False
            raise SimulatedCrash("controller killed after MD finished")
        return self.completed_segment(request)  # type: ignore[return-value]

    def completed_segment(self, request: SegmentRequest) -> SegmentResult | None:
        status = request.work_dir / "status.json"
        if not status.is_file():
            return None
        data = json.loads(status.read_text())
        return SegmentResult(data["state"], data["step"], request.work_dir / "checkpoint.json", request.work_dir / "frames.extxyz", {})

    def read_frames(self, result: SegmentResult, request: SegmentRequest) -> Iterable[Frame]:
        for atoms in read(result.frames_path, index=":"):
            yield Frame.from_atoms(atoms, run_id=request.run_id, model_version=atoms.info.get("model_version"))


# --------------------------------------------------------------------------- monitor
class FakeMonitor:
    """Force-deviation-vs-LJ threshold monitor with 1 pre / 1 post frame events."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.history: list[Frame] = []
        self.open: dict[str, Any] | None = None
        self.done: list[EventRecord] = []

    def process(self, frame: Frame) -> MonitorResult:
        deviation = np.linalg.norm(frame.forces - lj_forces(frame.positions), axis=1) if frame.forces is not None else np.zeros(frame.n_atoms)
        deviation = np.maximum(deviation, np.linalg.norm(frame.forces, axis=1) / 5.0)
        hot = frame.atom_ids[deviation > self.threshold].tolist()
        triggered = bool(hot)
        if self.open is not None:
            self.open["post"].append(frame)
            if len(self.open["post"]) >= 1:
                self._finish(complete=True)
        elif triggered:
            self.open = {"trigger": frame, "pre": self.history[-1:], "post": [], "hot": hot, "score": float(deviation.max())}
        self.history = (self.history + [frame])[-2:]
        action = FrameAction.SAVE if triggered else FrameAction.CONTINUE
        return MonitorResult(frame.run_id, frame.step, Stage.FULL, deviation, {"dev": MetricValues.measured("dev", deviation)},
                             float(deviation.max()), triggered, hot, ["force_deviation"] if triggered else [], action)

    def _finish(self, *, complete: bool, reason: str | None = None) -> None:
        trig = self.open["trigger"]
        self.done.append(EventRecord(stable_id("evt", trig.run_id, trig.step), trig.run_id, trig.step, [trig.step], self.open["pre"], trig,
                                     self.open["post"], 1, complete, self.open["hot"], ["force_deviation"], self.open["score"], reason))
        self.open = None

    def drain_events(self) -> list[EventRecord]:
        events, self.done = self.done, []
        return events

    def close(self, reason: str) -> list[EventRecord]:
        if self.open is not None:
            self._finish(complete=False, reason=reason)
        return self.drain_events()

    def to_state(self) -> dict[str, Any]:
        return {"threshold": self.threshold}

    def load_state(self, state: dict[str, Any]) -> None:
        self.threshold = state["threshold"]


# --------------------------------------------------------------------------- extraction
class FakeExtractor:
    def extract(self, event: EventRecord) -> list[tuple[ExtractedRegion, float]]:
        frame = event.trigger_frame
        out = []
        for hot_id in event.hotspot_atom_ids[:2]:
            center = frame.positions[frame.index_of([hot_id])[0]]
            keep = np.linalg.norm(frame.positions - center, axis=1) <= 5.0
            ids = frame.atom_ids[keep]
            roles = [RegionRole.CORE if i == hot_id else RegionRole.INNER_BUFFER for i in ids]
            weights = np.array([1.0 if r is RegionRole.CORE else 0.3 for r in roles])
            region = ExtractedRegion(stable_id("reg", event.event_id, hot_id), "cluster", frame.run_id, frame.step, event.event_id,
                                     frame.numbers[keep], frame.positions[keep], None, (False,) * 3, ids, np.zeros((len(ids), 3), int),
                                     roles, weights, np.zeros(len(ids), bool), [hot_id], model_version=frame.model_version)
            out.append((region, event.max_score))
        return out


# --------------------------------------------------------------------------- labeling
@dataclass
class FakeLabeler:
    root: Path
    polls_to_complete: int = 1
    invalid_first_attempt: set[str] = field(default_factory=set)
    unknown_once: set[str] = field(default_factory=set)
    crash_after_submit: bool = False
    submissions: dict[str, int] = field(default_factory=dict)
    _polls: dict[str, int] = field(default_factory=dict)

    def prepare(self, region: ExtractedRegion) -> LabelTask:
        task_id = stable_id("task", region.region_id, "fake-lj-v1")
        work_dir = self.root / task_id
        work_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(work_dir / "region.json", region.to_dict())
        return LabelTask(task_id, region.region_id, "fake_lj", "fake-lj-v1", str(work_dir))

    def find_submission(self, task: LabelTask) -> LabelTask | None:
        marker = Path(task.work_dir) / f"job_attempt{task.attempt}.json"
        if not marker.is_file():
            return None
        data = json.loads(marker.read_text())
        task.job = SchedulerJob(data["job_id"], "fake", ExecutionStatus.SUBMITTED, task.work_dir)
        task.execution = ExecutionStatus.SUBMITTED
        return task

    def submit(self, task: LabelTask) -> LabelTask:
        self.submissions[task.task_id] = self.submissions.get(task.task_id, 0) + 1
        job_id = f"job-{task.task_id}-{task.attempt}"
        write_json_atomic(Path(task.work_dir) / f"job_attempt{task.attempt}.json", {"job_id": job_id})
        if self.crash_after_submit:
            self.crash_after_submit = False
            raise SimulatedCrash("controller killed right after sbatch returned")
        task.job = SchedulerJob(job_id, "fake", ExecutionStatus.SUBMITTED, task.work_dir)
        task.execution = ExecutionStatus.SUBMITTED
        return task

    def poll(self, task: LabelTask) -> LabelTask:
        if task.task_id in self.unknown_once:
            self.unknown_once.discard(task.task_id)
            task.execution = ExecutionStatus.UNKNOWN
            return task
        key = f"{task.task_id}:{task.attempt}"
        self._polls[key] = self._polls.get(key, 0) + 1
        task.execution = ExecutionStatus.COMPLETED if self._polls[key] >= self.polls_to_complete else ExecutionStatus.RUNNING
        return task

    def reconcile(self, task: LabelTask) -> LabelTask:
        task.execution = ExecutionStatus.RUNNING  # scheduler answers on the second ask
        return task

    def collect(self, task: LabelTask) -> LabelResult:
        region = ExtractedRegion.from_dict(json.loads((Path(task.work_dir) / "region.json").read_text()))
        if task.region_id in self.invalid_first_attempt and task.attempt == 0:
            return LabelResult(task.task_id, task.region_id, np.zeros((region.n_atoms, 3)), None, False, ValidationStatus.INVALID, ["SCF not converged"])
        atoms = Atoms(numbers=region.numbers, positions=region.positions)
        atoms.calc = LennardJones(epsilon=EPS, sigma=SIGMA, rc=10.0, smooth=True)
        return LabelResult(task.task_id, task.region_id, atoms.get_forces(), float(atoms.get_potential_energy()), True, ValidationStatus.VALID, engine="fake_lj")

    def retry(self, task: LabelTask, result: LabelResult | None) -> LabelTask | None:
        if task.attempt >= 2:
            return None
        return LabelTask(task.task_id, task.region_id, task.engine, task.settings_hash, task.work_dir, attempt=task.attempt + 1,
                         failure_reasons=list(task.failure_reasons))

    def cancel(self, task: LabelTask) -> LabelTask:
        task.execution = ExecutionStatus.CANCELLED
        return task


# --------------------------------------------------------------------------- datasets
@dataclass
class FakeDatasetStore:
    root: Path
    crash_after_write: bool = False
    versions: dict[str, DatasetVersion] = field(default_factory=dict)

    def add_sample(self, region: ExtractedRegion, label: LabelResult, *, provenance: dict[str, Any]) -> tuple[str, Path]:
        sample_id = stable_id("sample", region.region_id, label.task_id)
        path = self.root / "samples" / f"{sample_id}.extxyz"
        path.parent.mkdir(parents=True, exist_ok=True)
        atoms = region.to_atoms()
        atoms.arrays["forces"] = label.forces
        write(path, atoms, format="extxyz")
        if self.crash_after_write:
            self.crash_after_write = False
            raise SimulatedCrash("controller killed after writing the sample file")
        return sample_id, path

    def publish(self, sample_paths: list[Path], *, parent: DatasetVersion | None, split_groups: dict[str, str]) -> DatasetVersion:
        ids = tuple(sorted({*(parent.sample_ids if parent else ()), *(p.stem for p in sample_paths)}))
        version = stable_id("ds", ids)
        groups = sorted({split_groups[i] for i in ids})
        val_groups = set(groups[::4]) if len(groups) > 1 else set()
        n_val = sum(1 for i in ids if split_groups[i] in val_groups)
        record = DatasetVersion(version, str(self.root / version), content_hash(ids), len(ids), len(ids) - n_val, n_val,
                                None if parent is None else parent.version, ids, "now", {"groups": groups, "val_groups": sorted(val_groups)})
        self.versions[version] = record
        return record

    def get(self, version: str) -> DatasetVersion:
        return self.versions[version]


# --------------------------------------------------------------------------- training
@dataclass
class FakeTrainer:
    bad_model: bool = False
    trained: list[str] = field(default_factory=list)

    def train(self, dataset: DatasetVersion, *, parent: ModelArtifact | None, work_dir: Path) -> TrainOutput:
        work_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = work_dir / "ckpt_best.json"
        scale = 5.0 if self.bad_model else 0.05 / (1.0 + dataset.n_structures)
        checkpoint.write_text(json.dumps({"error_scale": scale, "dataset": dataset.version, "parent": None if parent is None else parent.model_id}))
        self.trained.append(dataset.version)
        return TrainOutput(checkpoint, [18], 6.0, "finetune" if parent else "fresh", {"best_epoch": 3})

    def evaluate(self, checkpoint: Path, dataset: DatasetVersion) -> dict[str, float]:
        scale = json.loads(Path(checkpoint).read_text())["error_scale"]
        return {"force_rmse": float(scale), "n_validation": float(max(dataset.n_val, 1))}

    def export(self, checkpoint: Path, work_dir: Path) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        export = work_dir / "model_export.json"
        export.write_text(Path(checkpoint).read_text())
        return export

    def validate_export(self, checkpoint: Path, export: Path, dataset: DatasetVersion) -> dict[str, Any]:
        return {"max_force_deviation": 0.0 if Path(export).read_text() == Path(checkpoint).read_text() else 1.0}

    def engine_smoke_test(self, export: Path, work_dir: Path) -> dict[str, Any]:
        return {"passed": True}
