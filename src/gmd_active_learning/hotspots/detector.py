# Ported from PHAL (hotspot_al.hotspot.hotspot_detector) at 96263b6; MIT License, see NOTICE.
"""Hotspot detection on stable atom IDs: flagged atoms -> PBC clusters -> centre, score, reasons.

Ordering is canonical: flagged atoms are sorted by atom ID before clustering, cluster members are
sorted by atom ID, and hotspots are sorted by (max_score descending, smallest atom ID), so the
result does not depend on the engine's atom order. The centre is the minimum-image centroid
anchored at the cluster's smallest atom ID (clusters wider than half a cell period fold).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gmd_active_learning.contracts.frame import Frame
from gmd_active_learning.contracts.ids import stable_id
from gmd_active_learning.contracts.monitoring import MonitorResult
from gmd_active_learning.hotspots.cluster import cluster_anomalous_atoms
from gmd_active_learning.hotspots.config import HotspotsConfig, load_hotspots_config
from gmd_active_learning.utils.geometry import hotspot_center

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Hotspot:
    """A spatial cluster of hotspot atoms in one frame (stable IDs, Angstrom)."""

    hotspot_id: str
    run_id: str
    step: int
    atom_ids: list[int]
    center: np.ndarray
    max_score: float
    trigger_reasons: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hotspot_id": self.hotspot_id,
            "run_id": self.run_id,
            "step": self.step,
            "atom_ids": list(map(int, self.atom_ids)),
            "center": np.asarray(self.center, dtype=float).tolist(),
            "max_score": float(self.max_score),
            "trigger_reasons": list(self.trigger_reasons),
            "metadata": dict(self.metadata),
        }


def cluster_atom_ids(
    frame: Frame,
    atom_ids: list[int],
    atom_scores: np.ndarray,
    *,
    merge_radius: float,
    trigger_reasons: list[str] | None = None,
) -> list[Hotspot]:
    """Cluster the given atoms of ``frame`` (by stable ID) into hotspots."""

    if not atom_ids:
        return []
    scores = np.asarray(atom_scores, dtype=float)
    ordered_ids = sorted({int(atom_id) for atom_id in atom_ids})
    indices = frame.index_of(ordered_ids).tolist()
    clusters = cluster_anomalous_atoms(indices, frame.positions, merge_radius=merge_radius, cell=frame.cell, pbc=frame.pbc)
    hotspots: list[Hotspot] = []
    for members in clusters:
        member_ids = sorted(int(frame.atom_ids[index]) for index in members)
        member_indices = frame.index_of(member_ids).tolist()  # anchor = smallest atom ID
        center = hotspot_center(frame.positions, member_indices, cell=frame.cell, pbc=frame.pbc)
        hotspots.append(
            Hotspot(
                hotspot_id=stable_id("hot", frame.run_id, frame.step, member_ids),
                run_id=frame.run_id,
                step=frame.step,
                atom_ids=member_ids,
                center=center,
                max_score=float(np.max(scores[member_indices])),
                trigger_reasons=list(trigger_reasons or []),
                metadata={"n_atoms": len(member_ids)},
            )
        )
    hotspots.sort(key=lambda item: (-item.max_score, item.atom_ids[0]))
    return hotspots


def detect_hotspots(
    frame: Frame,
    atom_scores: np.ndarray,
    *,
    threshold: float,
    merge_radius: float = 8.0,
    trigger_reasons: list[str] | None = None,
) -> list[Hotspot]:
    """PHAL ``detect_hotspots``: atoms with score >= threshold, clustered."""

    scores = np.asarray(atom_scores, dtype=float)
    flagged = np.flatnonzero(scores >= threshold)
    return cluster_atom_ids(
        frame,
        [int(frame.atom_ids[index]) for index in flagged],
        scores,
        merge_radius=merge_radius,
        trigger_reasons=trigger_reasons,
    )


def hotspots_from_result(
    frame: Frame,
    result: MonitorResult,
    config: HotspotsConfig | dict[str, Any] | None = None,
) -> list[Hotspot]:
    """Cluster ``result.hotspot_atom_ids`` (the monitor's single hotspot threshold) into hotspots."""

    settings = load_hotspots_config(config)
    if not settings.enabled or not result.hotspot_atom_ids:
        return []
    if result.step != frame.step or result.run_id != frame.run_id:
        logger.warning("hotspots: result %s:%d applied to frame %s:%d", result.run_id, result.step, frame.run_id, frame.step)
    return cluster_atom_ids(
        frame,
        result.hotspot_atom_ids,
        result.atom_scores,
        merge_radius=settings.merge_radius,
        trigger_reasons=result.trigger_reasons,
    )


__all__ = ["Hotspot", "cluster_atom_ids", "detect_hotspots", "hotspots_from_result"]
