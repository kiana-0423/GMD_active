"""GMD molecular-dynamics adapter (segmented runs, flushed-frame monitoring, cooperative control).

Replaces PHAL's LAMMPS controller/runner/input writer (ledger MD-001, MD-003, MD-004, BE-009).
Requires a GMD build with the controller-integration directives (checkpoint v3, ``frame_file``,
``stop_file``, ``checkpoint_request_file``, ``status_file``, ``model_change``) and, for ML runs,
``GMD_ENABLE_TORCH=ON``.

Monitoring cadence and stop latency: frames are read as soon as GMD flushes them (every
``frame_interval`` steps). A STOP decision is delivered by creating the stop file, and GMD acts on
it at its next ``control_poll_interval`` boundary. The stop therefore takes effect up to
``control_poll_interval`` steps plus the reader's polling delay after the triggering frame. This is
cooperative control, not instantaneous per-step control.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from ase.io import read
from pydantic import BaseModel, ConfigDict, Field

from gmd_active_learning.contracts import Frame, FrameAction
from gmd_active_learning.core.exceptions import (
    AdapterError,
    CapabilityUnavailableError,
    ConfigurationError,
    UnsupportedCombinationError,
)
from gmd_active_learning.utils.logging import get_logger
from gmd_active_learning.workflow.protocols import FrameCallback, SegmentRequest, SegmentResult

LOGGER = get_logger(__name__)
FRAME_FORMAT = "gmd-frame-v1"
STATUS_FORMAT = "gmd-status-v1"


class GMDConfig(BaseModel):
    """``md`` config section for the GMD backend."""

    model_config = ConfigDict(extra="forbid")

    executable: str = Field(..., description="path to a GMD binary built with GMD_ENABLE_TORCH=ON for ML runs")
    initial_structure: str = Field(..., description="GMD xyz input (element symbols) for the first segment")
    directives: dict[str, Any] = Field(default_factory=lambda: {"time_step": 0.5, "velocity": 300.0, "velocity_init": "random", "velocity_seed": 1})
    frame_interval: int = Field(10, ge=1)
    control_poll_interval: int = Field(1, ge=1)
    checkpoint_every: int = Field(0, ge=0, description="0 = only at segment end")
    reader_poll_s: float = Field(0.05, gt=0.0)
    segment_timeout_s: float = Field(86400.0, gt=0.0)
    env: dict[str, str] = Field(default_factory=lambda: {"OMP_NUM_THREADS": "1"})
    smoke_steps: int = Field(5, ge=1)


RESERVED = {"run", "force_field", "model_path", "frame_file", "frame_interval", "stop_file", "checkpoint_request_file",
            "status_file", "checkpoint_file", "restart_from", "model_change", "run_id", "control_poll_interval"}


def parse_frames(text: str, *, run_id: str) -> list[Frame]:
    if not text.strip():
        return []
    frames = []
    for atoms in read(io.StringIO(text), index=":", format="extxyz"):
        if atoms.info.get("format") not in (None, FRAME_FORMAT):
            raise AdapterError(f"unexpected GMD frame format {atoms.info.get('format')}")
        sha = atoms.info.get("model_sha256")
        frames.append(Frame.from_atoms(atoms, run_id=run_id, model_sha256=None if sha in (None, "none", "") else str(sha),
                                       metadata={key: atoms.info[key] for key in ("model_changed_from", "force_field") if key in atoms.info}))
    return frames


class FrameTail:
    """Incremental reader of a GMD frame file that only returns complete frames."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path, self.run_id = path, run_id
        self._offset = 0
        self._buffer = ""

    def read_new(self) -> list[Frame]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
            self._offset = handle.tell()
        self._buffer += chunk
        lines = self._buffer.split("\n")
        complete_end, cursor = 0, 0
        while cursor < len(lines) - 1:  # last element is the (possibly partial) tail
            header = lines[cursor].strip()
            if not header:
                cursor += 1
                complete_end = cursor
                continue
            n_atoms = int(header)
            frame_end = cursor + 2 + n_atoms
            if frame_end > len(lines) - 1:
                break
            cursor = frame_end
            complete_end = cursor
        text = "\n".join(lines[:complete_end]) + ("\n" if complete_end else "")
        self._buffer = "\n".join(lines[complete_end:])
        return parse_frames(text, run_id=self.run_id)


class GMDEngine:
    """Implements ``workflow.protocols.MDEngine``."""

    name = "gmd"
    role = "md"
    capabilities = frozenset({"segmented_runs", "flushed_frames", "cooperative_stop", "checkpoint_request",
                              "checkpoint_v3", "model_change_at_restart", "torchscript_ml", "stable_atom_ids"})

    @classmethod
    def from_config(cls, config: Any) -> GMDEngine:
        """Registry factory: ``config`` is the full configuration mapping (section ``md``)."""

        section = config.get("md") if hasattr(config, "get") else None
        return cls(GMDConfig.model_validate(section or {}))

    def check_runtime(self) -> Any:
        from gmd_active_learning.adapters.registry import BackendRole, RuntimeStatus

        return RuntimeStatus(self.name, BackendRole.MD, True, str(self.executable))

    def __init__(self, config: GMDConfig) -> None:
        self.config = config
        self.executable = Path(config.executable).expanduser()
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise CapabilityUnavailableError(f"GMD executable not found or not executable: {self.executable}")
        overlap = RESERVED & set(config.directives)
        if overlap:
            raise ConfigurationError(f"md.directives may not set controller-owned directives {sorted(overlap)}")

    # ------------------------------------------------------------------ inputs
    def _write_inputs(self, request: SegmentRequest) -> tuple[Path, Path]:
        work = request.work_dir
        work.mkdir(parents=True, exist_ok=True)
        if request.model.export_path is None:
            raise UnsupportedCombinationError(f"model {request.model.model_id} has no exported TorchScript artifact for GMD")
        xyz = work / "in.xyz"
        xyz.write_text(Path(self.config.initial_structure).expanduser().read_text())
        lines = [f"run {request.n_steps}", "force_field ml", f"model_path {request.model.export_path}",
                 f"run_id {request.run_id}", "frame_file frames.extxyz", f"frame_interval {self.config.frame_interval}",
                 f"control_poll_interval {self.config.control_poll_interval}", "stop_file STOP",
                 "checkpoint_request_file CHECKPOINT", "status_file status.json", "checkpoint_file segment.gmdchk"]
        if self.config.checkpoint_every:
            lines.append(f"write_checkpoint_every {self.config.checkpoint_every}")
        directives = dict(self.config.directives)
        if request.restart_from is not None:
            lines.append(f"restart_from {Path(request.restart_from).resolve()}")
            directives["velocity_init"] = "input"
            directives.pop("velocity_seed", None)
            if request.previous_model is not None:
                lines.append("model_change allow")
        lines += [f"{key} {value}" for key, value in directives.items()]
        run_in = work / "run.in"
        run_in.write_text("\n".join(lines) + "\n")
        return xyz, run_in

    # ------------------------------------------------------------------ execution
    def run_segment(self, request: SegmentRequest, on_frame: FrameCallback) -> SegmentResult:
        xyz, run_in = self._write_inputs(request)
        for stale in ("STOP", "CHECKPOINT", "status.json", "frames.extxyz"):
            (request.work_dir / stale).unlink(missing_ok=True)
        env = {**os.environ, **self.config.env}
        stdout = (request.work_dir / "gmd.stdout").open("w")
        stderr = (request.work_dir / "gmd.stderr").open("w")
        process = subprocess.Popen([str(self.executable), str(xyz), str(run_in)], cwd=request.work_dir, stdout=stdout, stderr=stderr, env=env)
        tail = FrameTail(request.work_dir / "frames.extxyz", request.run_id)
        deadline = time.monotonic() + self.config.segment_timeout_s
        decisions: list[dict[str, Any]] = []
        stop_sent = False
        try:
            while True:
                finished = process.poll() is not None
                for frame in tail.read_new():
                    self._check_model(frame, request)
                    frame.model_version = request.model.model_id  # registry identity of the model that ran
                    action = on_frame(frame)
                    if action is FrameAction.STOP and not stop_sent:
                        (request.work_dir / "STOP").write_text(f"stop requested after step {frame.step}\n")
                        stop_sent = True
                        decisions.append({"action": "stop", "step": frame.step})
                    elif action is FrameAction.CHECKPOINT:
                        (request.work_dir / "CHECKPOINT").write_text(f"checkpoint requested after step {frame.step}\n")
                        decisions.append({"action": "checkpoint", "step": frame.step})
                if finished:
                    break
                if time.monotonic() > deadline:
                    process.terminate()  # GMD handles SIGTERM cooperatively (final frame + checkpoint)
                    process.wait(timeout=60)
                    raise AdapterError(f"GMD segment exceeded {self.config.segment_timeout_s} s")
                time.sleep(self.config.reader_poll_s)
        finally:
            stdout.close()
            stderr.close()
        result = self.completed_segment(request)
        if result is None or process.returncode != 0:
            message = (request.work_dir / "gmd.stderr").read_text()[-2000:]
            return SegmentResult("failed", None, None, None, {"returncode": process.returncode, "stderr": message, "decisions": decisions})
        result.detail["decisions"] = decisions
        if stop_sent:
            stop_step = next(item["step"] for item in decisions if item["action"] == "stop")
            result.detail["stop_latency_steps"] = None if result.end_step is None else int(result.end_step) - int(stop_step)
        return result

    @staticmethod
    def _check_model(frame: Frame, request: SegmentRequest) -> None:
        if frame.model_sha256 is not None and frame.model_sha256 != request.model.export_sha256:
            raise AdapterError(f"GMD ran model {frame.model_sha256}, expected {request.model.export_sha256}")

    def completed_segment(self, request: SegmentRequest) -> SegmentResult | None:
        status_path = request.work_dir / "status.json"
        if not status_path.is_file():
            return None
        status = json.loads(status_path.read_text())
        if status.get("format") not in (None, STATUS_FORMAT):
            raise AdapterError(f"unexpected GMD status format {status.get('format')}")
        state = status.get("state")
        if state == "running":
            return None
        if state not in {"completed", "stopped"}:
            return SegmentResult("failed", status.get("step"), None, None, {"status": status})
        checkpoint = Path(status["checkpoint"]) if status.get("checkpoint") else None
        if checkpoint is not None and not checkpoint.is_absolute():
            checkpoint = request.work_dir / checkpoint
        if checkpoint is None or not checkpoint.is_file():
            return SegmentResult("failed", status.get("step"), None, None, {"status": status, "error": "no checkpoint"})
        frames = request.work_dir / "frames.extxyz"
        return SegmentResult(state, int(status["step"]), checkpoint, frames, {"status": status})

    def read_frames(self, result: SegmentResult, request: SegmentRequest) -> Iterable[Frame]:
        if result.frames_path is None or not result.frames_path.is_file():
            return []
        frames = parse_frames(result.frames_path.read_text(), run_id=request.run_id)
        for frame in frames:
            self._check_model(frame, request)
            frame.model_version = request.model.model_id
        return frames

    # ------------------------------------------------------------------ export smoke test
    def smoke_test(self, export: Path, work_dir: Path) -> dict[str, Any]:
        """Run a few ML steps with ``export`` on the initial structure; checks a finite, completed run."""

        from gmd_active_learning.contracts import ModelArtifact, UNITS
        from gmd_active_learning.contracts.ids import file_sha256

        model = ModelArtifact("smoke", str(export), "", str(export), file_sha256(export), None, None, (), None, dict(UNITS), {}, "smoke", "gmd_sgt", "")
        request = SegmentRequest("smoke", 0, Path(work_dir), model, self.config.smoke_steps, None, None)
        frames: list[Frame] = []
        try:
            result = self.run_segment(request, lambda frame: frames.append(frame) or FrameAction.CONTINUE)
        except AdapterError as exc:
            return {"passed": False, "reason": str(exc)}
        finite = all(np.isfinite(frame.forces).all() for frame in frames if frame.forces is not None)
        energies = [frame.energy for frame in frames if frame.energy is not None]
        passed = result.status == "completed" and bool(frames) and finite and all(np.isfinite(energies))
        return {"passed": passed, "status": result.status, "frames": len(frames), "detail": {k: v for k, v in result.detail.items() if k != "status"}}


def iter_frames(path: Path, run_id: str) -> Iterator[Frame]:
    yield from parse_frames(Path(path).read_text(), run_id=run_id)
