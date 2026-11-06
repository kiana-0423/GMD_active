"""JSON-friendly (de)serialisation of frames and events for persisted workflow state.

Floats go through ``repr`` in ``json``, so positions/forces round-trip bit-exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gmd_active_learning.contracts.events import EventRecord
from gmd_active_learning.contracts.frame import Frame


def _list_or_none(value: np.ndarray | None) -> list[Any] | None:
    return None if value is None else np.asarray(value).tolist()


def frame_to_state(frame: Frame) -> dict[str, Any]:
    return {
        "run_id": frame.run_id,
        "step": frame.step,
        "atom_ids": frame.atom_ids.tolist(),
        "numbers": frame.numbers.tolist(),
        "positions": frame.positions.tolist(),
        "cell": _list_or_none(frame.cell),
        "pbc": list(frame.pbc),
        "forces": _list_or_none(frame.forces),
        "time_fs": frame.time_fs,
        "energy": frame.energy,
        "velocities": _list_or_none(frame.velocities),
        "committee_forces": _list_or_none(frame.committee_forces),
        "model_version": frame.model_version,
        "model_sha256": frame.model_sha256,
        "metadata": frame.metadata,
    }


def frame_from_state(state: dict[str, Any]) -> Frame:
    def array(key: str, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
        value = state.get(key)
        if value is None:
            return None
        result = np.asarray(value, dtype=float)
        return result.reshape(shape) if shape is not None else result

    n_atoms = len(state["numbers"])
    committee = state.get("committee_forces")
    return Frame(
        run_id=state["run_id"],
        step=int(state["step"]),
        atom_ids=np.asarray(state["atom_ids"], dtype=np.int64),
        numbers=np.asarray(state["numbers"], dtype=np.int64),
        positions=array("positions", (n_atoms, 3)),
        cell=array("cell", (3, 3)),
        pbc=tuple(bool(flag) for flag in state["pbc"]),  # type: ignore[arg-type]
        forces=array("forces", (n_atoms, 3)),
        time_fs=state.get("time_fs"),
        energy=state.get("energy"),
        velocities=array("velocities", (n_atoms, 3)),
        committee_forces=None if committee is None else np.asarray(committee, dtype=float).reshape(-1, n_atoms, 3),
        model_version=state.get("model_version"),
        model_sha256=state.get("model_sha256"),
        metadata=dict(state.get("metadata") or {}),
    )


def event_to_state(event: EventRecord) -> dict[str, Any]:
    return {
        **event.summary(),
        "pre_frames": [frame_to_state(frame) for frame in event.pre_frames],
        "trigger_frame": frame_to_state(event.trigger_frame),
        "post_frames": [frame_to_state(frame) for frame in event.post_frames],
    }


def event_from_state(state: dict[str, Any]) -> EventRecord:
    return EventRecord(
        event_id=state["event_id"],
        run_id=state["run_id"],
        trigger_step=int(state["trigger_step"]),
        trigger_steps=[int(step) for step in state["trigger_steps"]],
        pre_frames=[frame_from_state(frame) for frame in state["pre_frames"]],
        trigger_frame=frame_from_state(state["trigger_frame"]),
        post_frames=[frame_from_state(frame) for frame in state["post_frames"]],
        post_frames_requested=int(state["post_frames_requested"]),
        complete=bool(state["complete"]),
        hotspot_atom_ids=[int(atom_id) for atom_id in state["hotspot_atom_ids"]],
        trigger_reasons=list(state["trigger_reasons"]),
        max_score=float(state["max_score"]),
        truncation_reason=state.get("truncation_reason"),
        model_version=state.get("model_version"),
        metadata=dict(state.get("metadata") or {}),
        pre_frames_requested=None if state.get("pre_frames_requested") is None else int(state["pre_frames_requested"]),
    )


__all__ = ["event_from_state", "event_to_state", "frame_from_state", "frame_to_state"]
