# Ported from PHAL (hotspot_al.monitor.online_monitor.OnlineMonitor.write_event) at 96263b6; MIT License, see NOTICE.
"""PHAL-compatible ``event.json`` view of a GMD :class:`~gmd_active_learning.contracts.EventRecord`.

GMD's own ``event.json`` is ``EventRecord.summary()`` (contract keys, see ``event_store``). Tools that
read PHAL event directories can use :func:`phal_event_payload`, which rebuilds PHAL's payload
(``event_id, step, time, hotspot_atoms, trigger_reason, ood_scores, backend, model_version,
metadata``) and its metadata (``event_id, backend, model_version, ood, latest_trigger_step,
trigger_steps, trigger_count, trigger_metadata``) from the GMD event alone. Two inputs are not part
of the event: the MLIP ``backend`` name and ``lj_lazy_threshold``, a config constant that PHAL
echoed in every result's metadata.

Mapping rules:

* ``ood`` (per trigger) is the trigger result's monitor metadata, which the online driver records
  under ``trigger_metadata[i]["monitor"]``. ``lj_fit_count`` is 0 when no LJ fit ran, as in PHAL.
  ``screen_max_score`` is left out for scheduled full frames, because PHAL's ``score_full`` did not
  record it.
* ``event_id`` is GMD's stable id (F-15). PHAL's ids were ``evt-<step>-<uuid8>``.
* ``ood_scores`` are the per-atom scores max-merged over the event's triggers, as in PHAL.
"""

from __future__ import annotations

from typing import Any

from gmd_active_learning.contracts.events import EventRecord

#: monitor-result metadata keys the online driver keeps for every trigger (PHAL's ``ood`` dict)
MONITOR_METADATA_KEYS = (
    "label_threshold",
    "physics_threshold",
    "screen_threshold",
    "lj_fit_count",
    "screen_max_score",
    "physics_max_score",
    "scheduled_full",
)


def monitor_metadata_for_event(result_metadata: dict[str, Any]) -> dict[str, Any]:
    """The subset of a ``MonitorResult.metadata`` an event keeps per trigger."""

    return {key: result_metadata[key] for key in MONITOR_METADATA_KEYS if key in result_metadata}


def phal_ood_metadata(monitor_metadata: dict[str, Any], *, lj_lazy_threshold: float) -> dict[str, Any]:
    """PHAL's ``OODFrameResult.metadata`` rebuilt from GMD monitor metadata."""

    ood: dict[str, Any] = {key: monitor_metadata[key] for key in ("label_threshold", "physics_threshold", "screen_threshold")}
    ood["lj_fit_count"] = int(monitor_metadata.get("lj_fit_count", 0))
    ood["lj_lazy_threshold"] = float(lj_lazy_threshold)
    for key in ("screen_max_score", "physics_max_score"):
        if key in monitor_metadata and not (key == "screen_max_score" and monitor_metadata.get("scheduled_full")):
            ood[key] = monitor_metadata[key]
    return dict(sorted(ood.items()))


def phal_event_payload(event: EventRecord, *, backend: str | None, lj_lazy_threshold: float) -> dict[str, Any]:
    """PHAL ``write_event``'s ``event.json`` payload for ``event``.

    Raises ``KeyError`` when the event's trigger metadata has no ``monitor`` entry (events that were
    not captured through the online driver), because PHAL's ``ood`` cannot be rebuilt then.
    """

    triggers = list(event.metadata.get("trigger_metadata") or [])
    if not triggers:
        raise KeyError(f"event {event.event_id} has no trigger_metadata")
    trigger_metadata = [
        {"step": int(entry["step"]), "ood": phal_ood_metadata(entry["monitor"], lj_lazy_threshold=lj_lazy_threshold)}
        for entry in triggers
    ]
    metadata = {
        "event_id": event.event_id,
        "backend": backend,
        "model_version": event.model_version,
        "ood": trigger_metadata[0]["ood"],
        "latest_trigger_step": int(event.trigger_steps[-1]),
        "trigger_steps": [int(step) for step in event.trigger_steps],
        "trigger_count": len(triggers),
        "trigger_metadata": trigger_metadata,
    }
    return {
        "event_id": event.event_id,
        "step": int(event.trigger_step),
        "time": event.trigger_frame.time_fs,
        "hotspot_atoms": sorted(event.trigger_frame.index_of(event.hotspot_atom_ids).tolist()),  # PHAL: frame indices
        "trigger_reason": list(event.trigger_reasons),
        "ood_scores": [float(value) for value in event.metadata["atom_scores"]],
        "backend": backend,
        "model_version": event.model_version,
        "metadata": metadata,
    }


__all__ = ["MONITOR_METADATA_KEYS", "monitor_metadata_for_event", "phal_event_payload", "phal_ood_metadata"]
