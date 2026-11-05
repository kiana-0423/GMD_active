# Ported from PHAL (hotspot_al.monitor.online_monitor.OnlineMonitor.write_event) at 96263b6; MIT License, see NOTICE.
"""Persist events: ``<root>/<event_id>/event.json`` (atomic), ``frames.npz`` (bit-exact) and ``frames.extxyz``.

``event.json`` holds ``EventRecord.summary()`` plus frame steps; ``frames.npz`` holds every frame's
arrays exactly (positions, forces, atom ids, numbers, cell, pbc, step, optional committee forces);
``frames.extxyz`` is a human/ASE-readable copy (8-decimal text, not bit-exact). Writing the same
event twice yields the same files (deterministic IDs), so ingestion is idempotent.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from gmd_active_learning.contracts.events import EventRecord
from gmd_active_learning.contracts.frame import Frame
from gmd_active_learning.core.exceptions import MonitorError
from gmd_active_learning.utils.atomic_io import write_json_atomic

logger = logging.getLogger(__name__)

EVENT_FILE = "event.json"
FRAMES_NPZ = "frames.npz"
FRAMES_EXTXYZ = "frames.extxyz"


def _replace_atomically(target: Path, write: Any, suffix: str) -> None:
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=suffix)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        write(temporary_path)
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _frames_payload(frames: list[Frame]) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {"steps": np.asarray([frame.step for frame in frames], dtype=np.int64)}
    for index, frame in enumerate(frames):
        prefix = f"frame{index:04d}"
        payload[f"{prefix}/atom_ids"] = frame.atom_ids
        payload[f"{prefix}/numbers"] = frame.numbers
        payload[f"{prefix}/positions"] = frame.positions
        payload[f"{prefix}/pbc"] = np.asarray(frame.pbc, dtype=bool)
        if frame.cell is not None:
            payload[f"{prefix}/cell"] = frame.cell
        if frame.forces is not None:
            payload[f"{prefix}/forces"] = frame.forces
        if frame.committee_forces is not None:
            payload[f"{prefix}/committee_forces"] = frame.committee_forces
    return payload


def write_event(event: EventRecord, root: str | Path, *, write_extxyz: bool = True) -> Path:
    """Write one event below ``root``; returns the event directory."""

    event_dir = Path(root) / event.event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    frames = event.frames
    payload = {
        **event.summary(),
        "trigger_frame_step": event.trigger_frame.step,
        "n_frames": len(frames),
        "frame_steps": [frame.step for frame in frames],
    }
    arrays = _frames_payload(frames)

    def write_npz(path: Path) -> None:
        with path.open("wb") as handle:
            np.savez(handle, **arrays)

    _replace_atomically(event_dir / FRAMES_NPZ, write_npz, ".npz")
    if write_extxyz:
        from ase.io import write

        def write_xyz(path: Path) -> None:
            write(path, [frame.to_atoms() for frame in frames], format="extxyz")

        _replace_atomically(event_dir / FRAMES_EXTXYZ, write_xyz, ".extxyz")
    # event.json last: its presence marks a fully written event.
    write_json_atomic(event_dir / EVENT_FILE, payload)
    logger.info("wrote event %s (%d frames) to %s", event.event_id, len(frames), event_dir)
    return event_dir


def read_event(event_dir: str | Path) -> EventRecord:
    """Rebuild an ``EventRecord`` from a directory written by :func:`write_event` (frames bit-exact)."""

    import json

    event_dir = Path(event_dir)
    try:
        payload = json.loads((event_dir / EVENT_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MonitorError(f"cannot read event summary in {event_dir}: {exc}") from exc
    frames: list[Frame] = []
    with np.load(event_dir / FRAMES_NPZ) as data:
        for index, step in enumerate(data["steps"].tolist()):
            prefix = f"frame{index:04d}"
            frames.append(
                Frame(
                    run_id=payload["run_id"],
                    step=int(step),
                    atom_ids=data[f"{prefix}/atom_ids"],
                    numbers=data[f"{prefix}/numbers"],
                    positions=data[f"{prefix}/positions"],
                    cell=data[f"{prefix}/cell"] if f"{prefix}/cell" in data.files else None,
                    pbc=tuple(bool(flag) for flag in data[f"{prefix}/pbc"]),  # type: ignore[arg-type]
                    forces=data[f"{prefix}/forces"] if f"{prefix}/forces" in data.files else None,
                    committee_forces=data[f"{prefix}/committee_forces"] if f"{prefix}/committee_forces" in data.files else None,
                    model_version=payload.get("model_version"),
                )
            )
    n_pre = len(payload["pre_frame_steps"])
    return EventRecord(
        event_id=payload["event_id"],
        run_id=payload["run_id"],
        trigger_step=int(payload["trigger_step"]),
        trigger_steps=[int(step) for step in payload["trigger_steps"]],
        pre_frames=frames[:n_pre],
        trigger_frame=frames[n_pre],
        post_frames=frames[n_pre + 1 :],
        post_frames_requested=int(payload["post_frames_requested"]),
        complete=bool(payload["complete"]),
        hotspot_atom_ids=[int(atom_id) for atom_id in payload["hotspot_atom_ids"]],
        trigger_reasons=list(payload["trigger_reasons"]),
        max_score=float(payload["max_score"]),
        truncation_reason=payload.get("truncation_reason"),
        model_version=payload.get("model_version"),
        metadata=dict(payload.get("metadata") or {}),
    )


class EventStore:
    """Callable sink: ``EventStore(root)(event)`` writes the event (usable as the driver's ``on_event``)."""

    def __init__(self, root: str | Path, *, write_extxyz: bool = True) -> None:
        self.root = Path(root)
        self.write_extxyz = write_extxyz

    def __call__(self, event: EventRecord) -> None:
        write_event(event, self.root, write_extxyz=self.write_extxyz)

    def event_ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(path.name for path in self.root.iterdir() if (path / EVENT_FILE).is_file())

    def read(self, event_id: str) -> EventRecord:
        return read_event(self.root / event_id)


__all__ = ["EventStore", "read_event", "write_event"]
