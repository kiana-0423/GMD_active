"""GMD molecular-dynamics adapter (segmented runs, flushed-frame monitoring, cooperative control).

Replaces PHAL's LAMMPS controller/runner/input writer (ledger MD-001, MD-003, MD-004, MD-005, BE-009).
Requires a GMD build with the controller-integration directives (checkpoint v3, ``frame_file``,
``stop_file``, ``checkpoint_request_file``, ``status_file``, ``model_change``) and, for ML runs,
``GMD_ENABLE_TORCH=ON``.

Monitoring cadence and stop latency: frames are read as soon as GMD flushes them (every
``frame_interval`` steps). A STOP decision is delivered by creating the stop file, and GMD acts on
it at its next ``control_poll_interval`` boundary. The stop therefore takes effect up to
``control_poll_interval`` steps plus the reader's polling delay after the triggering frame. This is
cooperative control, not instantaneous per-step control.

GMD's TorchScript (ML) force path is serial: GMD itself is never launched with ``--np``, and a
configured ``launcher`` (e.g. ``["taskset", "-c", "0"]`` or ``["srun", "-n", "1"]``) must not ask for
more than one MPI rank. The model's species are checked against the initial structure before GMD
is started (GMD silently maps unknown elements to Z=0, which a GMD_SGT model treats as padding).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from ase.io import read
from ase.data import atomic_numbers
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    launcher: list[str] = Field(default_factory=list, description="argv prefix, e.g. [taskset, -c, '0'] or [srun, -n, '1']; ML runs are serial")
    check_species: bool = Field(True, description="refuse to launch when the initial structure has species the model does not know")

    @field_validator("directives")
    @classmethod
    def _positive_time_step(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "time_step" in value and not (isinstance(value["time_step"], (int, float)) and not isinstance(value["time_step"], bool) and value["time_step"] > 0):
            raise ValueError(f"md.directives.time_step must be a positive number of fs, got {value['time_step']!r}")
        return value

    @field_validator("launcher")
    @classmethod
    def _serial_launcher(cls, value: list[str]) -> list[str]:
        rank_flags = {"-n", "-np", "--np", "--ntasks", "-N", "--nodes"}
        for flag, count in zip(value, value[1:]):
            if flag in rank_flags and count.isdigit() and int(count) > 1:
                raise ValueError(f"md.launcher requests {count} ranks ({flag} {count}); GMD's TorchScript force path is serial-only")
        for item in value:
            if item.startswith(("--ntasks=", "--np=", "-np=")) and item.split("=", 1)[1].isdigit() and int(item.split("=", 1)[1]) > 1:
                raise ValueError(f"md.launcher requests {item}; GMD's TorchScript force path is serial-only")
        return value


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
    """Incremental reader of a GMD frame file that returns every complete frame exactly once.

    Only bytes appended since the previous call are read; a partially written frame stays buffered
    until it is complete. If the file shrinks (truncated or rewritten) the reader resynchronises
    from the start and skips frames whose step was already delivered (PHAL's LAMMPS dump tail reset
    the offset but could deliver a frame twice).
    """

    def __init__(self, path: Path, run_id: str) -> None:
        self.path, self.run_id = Path(path), run_id
        self._offset = 0
        self._buffer = ""
        self.last_step: int | None = None
        self.resyncs = 0

    def read_new(self) -> list[Frame]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self._offset:
            LOGGER.warning("GMD frame file %s shrank (%d < %d bytes); resynchronising", self.path, size, self._offset)
            self._offset, self._buffer = 0, ""
            self.resyncs += 1
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
            try:
                n_atoms = int(header)
            except ValueError as exc:
                raise AdapterError(f"{self.path}: expected an atom count, got {header[:40]!r}") from exc
            frame_end = cursor + 2 + n_atoms
            if frame_end > len(lines) - 1:
                break
            cursor = frame_end
            complete_end = cursor
        text = "\n".join(lines[:complete_end]) + ("\n" if complete_end else "")
        self._buffer = "\n".join(lines[complete_end:])
        frames = []
        for frame in parse_frames(text, run_id=self.run_id):
            if self.last_step is not None and frame.step <= self.last_step:
                continue  # already delivered (resync after truncation)
            self.last_step = frame.step
            frames.append(frame)
        return frames


def structure_species(path: str | Path) -> list[int]:
    """Atomic numbers of a GMD xyz input (``N``, comment/box line, ``Symbol x y z`` lines)."""

    lines = Path(path).expanduser().read_text().splitlines()
    try:
        n_atoms = int(lines[0].split()[0])
    except (IndexError, ValueError) as exc:
        raise ConfigurationError(f"{path}: not a GMD xyz file (first line must be the atom count)") from exc
    numbers = []
    for line in lines[2 : 2 + n_atoms]:
        symbol = line.split()[0] if line.split() else ""
        if symbol not in atomic_numbers or atomic_numbers[symbol] == 0:
            raise ConfigurationError(f"{path}: unknown element {symbol!r} (GMD would silently use Z=0)")
        numbers.append(atomic_numbers[symbol])
    if len(numbers) != n_atoms:
        raise ConfigurationError(f"{path}: declares {n_atoms} atoms but lists {len(numbers)}")
    return numbers


def check_model_species(structure: str | Path, model_species: Iterable[int], *, model: str = "model") -> None:
    """Reject a run whose structure contains elements the model was not trained on."""

    known = {int(z) for z in model_species}
    if not known:
        raise UnsupportedCombinationError(f"{model} declares no species; cannot check it against {structure}")
    missing = sorted(set(structure_species(structure)) - known)
    if missing:
        raise UnsupportedCombinationError(f"{structure} contains species {missing} that {model} (species {sorted(known)}) does not know")


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
        executable = config.executable
        if os.sep not in executable and not Path(executable).is_file():
            executable = shutil.which(executable) or executable  # bare name: PATH lookup
        self.executable = Path(executable).expanduser()
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise CapabilityUnavailableError(f"GMD executable not found or not executable: {self.executable}")
        overlap = RESERVED & set(config.directives)
        if overlap:
            raise ConfigurationError(f"md.directives may not set controller-owned directives {sorted(overlap)}")

    # ------------------------------------------------------------------ inputs
    def command(self, xyz: str | Path, run_in: str | Path) -> list[str]:
        """argv for one GMD run: ``[*launcher, gmd, input.xyz, run.in]`` (never ``--np``: ML is serial)."""

        return [*self.config.launcher, str(self.executable), str(xyz), str(run_in)]

    def prepare(self, request: SegmentRequest) -> list[str]:
        """Write the segment inputs and return the argv without executing it (PHAL's dry run)."""

        xyz, run_in = self._write_inputs(request)
        return self.command(xyz, run_in)

    def _write_inputs(self, request: SegmentRequest) -> tuple[Path, Path]:
        work = request.work_dir
        if request.model.export_path is None:
            raise UnsupportedCombinationError(f"model {request.model.model_id} has no exported TorchScript artifact for GMD")
        if self.config.check_species and not request.metadata.get("skip_species_check"):
            check_model_species(self.config.initial_structure, request.model.species, model=f"model {request.model.model_id}")
        work.mkdir(parents=True, exist_ok=True)
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
        try:
            process = subprocess.Popen(self.command(xyz, run_in), cwd=request.work_dir, stdout=stdout, stderr=stderr, env=env)
        except OSError as exc:
            stdout.close()
            stderr.close()
            raise AdapterError(f"cannot launch GMD ({self.command(xyz, run_in)}): {exc}") from exc
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
        # The export's species are unknown here (no GMD_SGT import in the MD adapter); the registry
        # checks them when the model is registered, and running it is the point of the smoke test.
        request = SegmentRequest("smoke", 0, Path(work_dir), model, self.config.smoke_steps, None, None, {"skip_species_check": True})
        frames: list[Frame] = []
        try:
            result = self.run_segment(request, lambda frame: frames.append(frame) or FrameAction.CONTINUE)
        except AdapterError as exc:
            return {"passed": False, "reason": str(exc)}
        finite = all(np.isfinite(frame.forces).all() for frame in frames if frame.forces is not None)
        energies = [frame.energy for frame in frames if frame.energy is not None]
        passed = result.status == "completed" and bool(frames) and finite and all(np.isfinite(energies))
        return {"passed": passed, "status": result.status, "frames": len(frames), "detail": {k: v for k, v in result.detail.items() if k != "status"}}


class GMDFrameSource:
    """Pull-style ``FrameSource`` over one running GMD segment (``next_frame(timeout)``).

    Returns each complete frame exactly once, ``POLL_TIMEOUT`` when no new frame arrived within
    ``timeout`` (distinct from end of stream), and ``None`` once GMD has exited and every frame has
    been delivered; a non-zero exit raises :class:`AdapterError` instead. ``request_stop`` asks GMD
    to stop at its next control boundary; ``close`` terminates a still-running process and closes
    the log handles. ``OnlineMonitorDriver.run`` consumes it unchanged.
    """

    def __init__(self, engine: GMDEngine, request: SegmentRequest) -> None:
        self.engine, self.request = engine, request
        self.process: subprocess.Popen[bytes] | None = None
        self.tail = FrameTail(request.work_dir / "frames.extxyz", request.run_id)
        self._pending: list[Frame] = []
        self._logs: list[Any] = []

    def start(self) -> GMDFrameSource:
        xyz, run_in = self.engine._write_inputs(self.request)  # noqa: SLF001 - same adapter
        for stale in ("STOP", "CHECKPOINT", "status.json", "frames.extxyz"):
            (self.request.work_dir / stale).unlink(missing_ok=True)
        self._logs = [(self.request.work_dir / "gmd.stdout").open("w"), (self.request.work_dir / "gmd.stderr").open("w")]
        env = {**os.environ, **self.engine.config.env}
        self.process = subprocess.Popen(self.engine.command(xyz, run_in), cwd=self.request.work_dir, stdout=self._logs[0], stderr=self._logs[1], env=env)
        return self

    def __enter__(self) -> GMDFrameSource:
        return self if self.process is not None else self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _take(self) -> Frame:
        frame = self._pending.pop(0)
        GMDEngine._check_model(frame, self.request)  # noqa: SLF001
        frame.model_version = self.request.model.model_id
        return frame

    def next_frame(self, timeout: float | None = None) -> Any:
        from gmd_active_learning.monitors.online import POLL_TIMEOUT

        if self.process is None:
            self.start()
        assert self.process is not None
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._pending:
                return self._take()
            exited = self.process.poll() is not None
            self._pending.extend(self.tail.read_new())
            if self._pending:
                return self._take()
            if exited:
                self._close_logs()
                if self.process.returncode != 0:
                    stderr = (self.request.work_dir / "gmd.stderr").read_text()[-2000:]
                    raise AdapterError(f"GMD exited with code {self.process.returncode}: {stderr}")
                return None
            if deadline is not None and time.monotonic() >= deadline:
                return POLL_TIMEOUT
            time.sleep(self.engine.config.reader_poll_s)

    def request_stop(self, reason: str = "stop requested") -> None:
        (self.request.work_dir / "STOP").write_text(reason + "\n")

    def _close_logs(self) -> None:
        for handle in self._logs:
            handle.close()
        self._logs = []

    def close(self, timeout: float = 30.0) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()  # GMD handles SIGTERM cooperatively
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self._close_logs()

    @property
    def logs_closed(self) -> bool:
        return not self._logs


def iter_frames(path: Path, run_id: str) -> Iterator[Frame]:
    yield from parse_frames(Path(path).read_text(), run_id=run_id)
