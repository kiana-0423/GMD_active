# Ported from PHAL (hotspot_al.extraction.cluster_extractor) at 96263b6; MIT License, see NOTICE.
"""Spherical, non-periodic cluster extraction around a hotspot group."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gmd_active_learning.contracts import ExtractedRegion, Frame
from gmd_active_learning.core.exceptions import ExtractionError
from gmd_active_learning.datasets.masks import force_weights_for_roles
from gmd_active_learning.extraction._placement import (
    apply_periodic_guard,
    assemble_region,
    hotspot_frame_indices,
    mic_placement,
    min_periodic_half_width,
)
from gmd_active_learning.extraction.config import ExtractionConfig
from gmd_active_learning.extraction.regions import radial_roles
from gmd_active_learning.utils.bonds import bonded_neighbors
from gmd_active_learning.utils.geometry import distances_to_group, hotspot_center, padded_cluster_cell

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RadialSelection:
    """Frame indices selected around a seed group, with the explicit size-policy record."""

    indices: list[int]
    distances: np.ndarray  # distance of every frame atom to the seed group
    center: np.ndarray
    extract_radius: float
    record: dict[str, Any] = field(default_factory=dict)


def select_radial(
    frame: Frame,
    seed_indices: list[int],
    config: ExtractionConfig,
    *,
    extract_radius: float,
    min_atoms: int | None,
    protected: list[int] | None = None,
) -> RadialSelection:
    """PHAL's selection rule with explicit min/max-atom policies.

    PHAL padded up to ``min_atoms`` nearest-first and trimmed to ``max_atoms`` nearest-first, both
    silently, and could drop hotspot atoms. Here padding and trimming follow
    ``min_atoms_policy`` / ``max_atoms_policy``, every added or removed atom is recorded, and
    ``protected`` (hotspot) atoms are never trimmed.
    """

    pbc = frame.pbc if frame.cell is not None else False
    center = hotspot_center(frame.positions, list(seed_indices), cell=frame.cell, pbc=pbc)
    distances = distances_to_group(frame.positions, list(seed_indices), cell=frame.cell, pbc=pbc)
    n_atoms = frame.n_atoms
    max_atoms = n_atoms if config.max_atoms is None else int(config.max_atoms)
    min_atoms = min(n_atoms, len(seed_indices)) if min_atoms is None else int(min_atoms)
    record: dict[str, Any] = {"min_atoms": min_atoms, "max_atoms": max_atoms, "padded_atom_ids": [], "trimmed_atom_ids": []}

    selected = np.where(distances <= extract_radius)[0].tolist()
    if len(selected) < min_atoms:
        if config.min_atoms_policy == "error":
            raise ExtractionError(
                f"min_atoms_not_met: {len(selected)} atoms within {extract_radius} A, extraction.min_atoms={min_atoms}"
            )
        if config.min_atoms_policy == "pad":
            padded = sorted(np.argsort(distances).tolist()[: min(min_atoms, n_atoms)])
            added = sorted(set(padded) - set(selected))
            record["padded_atom_ids"] = frame.atom_ids[added].tolist()
            selected = padded
        else:
            record["below_min_atoms"] = True
    if len(selected) > max_atoms:
        keep = set(protected or [])
        if len(keep) > max_atoms:
            raise ExtractionError(
                f"max_atoms_exceeded: {len(keep)} hotspot atoms but extraction.max_atoms={max_atoms}"
            )
        if config.max_atoms_policy == "error":
            raise ExtractionError(f"max_atoms_exceeded: {len(selected)} atoms selected, extraction.max_atoms={max_atoms}")
        order = np.argsort(distances[selected])
        kept = sorted(selected[i] for i in order[:max_atoms])
        missing = keep - set(kept)
        if missing:  # only possible with ties at distance 0; keep hotspots, drop the farthest others
            others = [index for index in sorted(kept, key=lambda item: -distances[item]) if index not in keep]
            for index in others[: len(missing)]:
                kept.remove(index)
            kept = sorted([*kept, *missing])
        removed = sorted(set(selected) - set(kept))
        record["trimmed_atom_ids"] = frame.atom_ids[removed].tolist()
        LOGGER.warning("trimmed %d atoms to extraction.max_atoms=%d (recorded in metadata)", len(removed), max_atoms)
        selected = kept
    if config.terminal_atoms == "follow_partner":
        selected = _terminal_atoms_follow_partner(frame, selected, config, protected or [], record)
    return RadialSelection(indices=selected, distances=distances, center=center, extract_radius=extract_radius, record=record)


def _terminal_atoms_follow_partner(
    frame: Frame, selected: list[int], config: ExtractionConfig, protected: list[int], record: dict[str, Any]
) -> list[int]:
    """``extraction.terminal_atoms = follow_partner``: a terminal atom is selected iff its partner is.

    Runs after padding/trimming, so the result can exceed ``max_atoms`` by the added terminal atoms
    (recorded). A protected (hotspot) terminal atom whose partner is not selected raises.
    """

    adjacency = bonded_neighbors(
        frame.positions,
        frame.numbers,
        cell=frame.cell,
        pbc=frame.pbc if frame.cell is not None else False,
        scale=float(config.graph.covalent_scale),
    )
    chosen = set(int(index) for index in selected)
    terminal = {index: partners[0] for index, partners in adjacency.items() if len(partners) == 1}
    removed = sorted(index for index in chosen if index in terminal and terminal[index] not in chosen)
    lost = sorted(set(removed) & set(int(index) for index in protected))
    if lost:
        raise ExtractionError(
            f"terminal_atoms=follow_partner: hotspot atoms {frame.atom_ids[lost].tolist()} are terminal atoms whose "
            "bonded partner is outside the selection; enlarge the region"
        )
    added = sorted(index for index, partner in terminal.items() if partner in chosen and index not in chosen)
    result = sorted((chosen - set(removed)) | set(added))
    record["terminal_atoms"] = {
        "policy": "follow_partner",
        "added_atom_ids": frame.atom_ids[added].tolist(),
        "removed_atom_ids": frame.atom_ids[removed].tolist(),
    }
    return result


def cluster_from_selection(
    frame: Frame,
    selection: RadialSelection,
    config: ExtractionConfig,
    *,
    mode: str,
    hotspot_atom_ids: list[int],
    event_id: str | None,
    metadata: dict[str, Any],
    config_payload: Any,
) -> ExtractedRegion:
    """Place a radial selection in a padded non-periodic box and assign radial roles."""

    indices = np.asarray(selection.indices, dtype=np.int64)
    displacements, shifts = mic_placement(frame, indices, selection.center)
    vacuum = float(config.vacuum_padding)
    positions = displacements - displacements.min(axis=0) + 0.5 * vacuum
    cell = padded_cluster_cell(positions, padding=vacuum)
    roles = radial_roles(
        selection.distances[indices],
        core_radius=float(config.core_radius),
        extract_radius=selection.extract_radius,
        boundary_thickness=float(config.boundary_thickness),
    )
    half_width = min_periodic_half_width(frame)
    guard = apply_periodic_guard(
        frame,
        indices,
        shifts,
        config.periodic_guard,
        mode=mode,
        weights=force_weights_for_roles(roles, config.force_weights),
        extra={
            "radius_exceeds_half_cell": bool(half_width is not None and selection.extract_radius > half_width),
            "min_periodic_half_width": half_width,
        },
    )
    full_metadata = {
        "center": np.asarray(selection.center, dtype=float).tolist(),
        "extract_radius": selection.extract_radius,
        "core_radius": float(config.core_radius),
        "boundary_thickness": float(config.boundary_thickness),
        "vacuum_padding": vacuum,
        "size_policy": selection.record,
        "periodic_guard": guard,
        **metadata,
    }
    return assemble_region(
        frame,
        mode=mode,
        indices=indices,
        shifts=shifts,
        positions=positions,
        cell=cell,
        pbc=(False, False, False),
        roles=roles,
        hotspot_atom_ids=hotspot_atom_ids,
        config=config,
        event_id=event_id,
        metadata=full_metadata,
        config_payload=config_payload,
    )


def extract_cluster(
    frame: Frame,
    hotspot_atom_ids: Any,
    config: ExtractionConfig | None = None,
    *,
    event_id: str | None = None,
) -> ExtractedRegion:
    """Non-periodic spherical cluster: every atom within ``extract_radius`` of any hotspot atom.

    Positions are minimum-image displacements from the hotspot centroid, shifted so the lowest
    coordinate sits at ``vacuum_padding / 2`` inside an orthorhombic box of ``ptp + 2 *
    vacuum_padding`` (PHAL geometry, kept for parity).
    """

    config = config or ExtractionConfig()
    hotspots = hotspot_frame_indices(frame, hotspot_atom_ids)
    selection = select_radial(
        frame,
        hotspots,
        config,
        extract_radius=config.effective_extract_radius,
        min_atoms=config.min_atoms,
        protected=hotspots,
    )
    hotspot_ids = frame.atom_ids[hotspots].tolist()
    return cluster_from_selection(
        frame,
        selection,
        config,
        mode="cluster",
        hotspot_atom_ids=hotspot_ids,
        event_id=event_id,
        metadata={},
        config_payload=config.identity_payload({"slab", "graph", "block", "mode"}),
    )


__all__ = ["RadialSelection", "cluster_from_selection", "extract_cluster", "select_radial"]
