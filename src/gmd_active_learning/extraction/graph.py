# Ported from PHAL (hotspot_al.extraction.graph_extractor) at 96263b6; MIT License, see NOTICE.
"""Bonded-subgraph extraction for molecular and polymeric systems."""

from __future__ import annotations

from collections import deque
from typing import Any

from gmd_active_learning.contracts import ExtractedRegion, Frame
from gmd_active_learning.extraction._placement import hotspot_frame_indices
from gmd_active_learning.extraction.cluster import cluster_from_selection, select_radial
from gmd_active_learning.extraction.config import ExtractionConfig
from gmd_active_learning.utils.bonds import bonded_neighbors


def bonded_subgraph(frame: Frame, seeds: list[int], *, hops: int, covalent_scale: float) -> list[int]:
    """Frame indices reachable from ``seeds`` within ``hops`` covalent bonds (MIC-aware), sorted."""

    adjacency = bonded_neighbors(
        frame.positions,
        frame.numbers,
        cell=frame.cell,
        pbc=frame.pbc if frame.cell is not None else False,
        scale=covalent_scale,
    )
    visited = set(seeds)
    queue: deque[tuple[int, int]] = deque((index, 0) for index in seeds)
    while queue:
        node, depth = queue.popleft()
        if depth >= hops:
            continue
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return sorted(visited)


def extract_graph(
    frame: Frame,
    hotspot_atom_ids: Any,
    config: ExtractionConfig | None = None,
    *,
    event_id: str | None = None,
) -> ExtractedRegion:
    """BFS over the covalent graph from the hotspot atoms, then a radial pass seeded by the subgraph.

    With ``graph.radial_shell`` (default, documented as deliberate in PHAL's known issues) the
    region adds an ``extract_radius`` shell around every graph atom and the core role means
    "within ``core_radius`` of any graph atom". Without it the region is exactly the subgraph and
    every atom is core. ``hotspot_atom_ids`` records the triggering atoms, not the subgraph.
    """

    config = config or ExtractionConfig()
    hotspots = hotspot_frame_indices(frame, hotspot_atom_ids)
    graph_atoms = bonded_subgraph(
        frame, hotspots, hops=int(config.graph.hops), covalent_scale=float(config.graph.covalent_scale)
    )
    if config.graph.radial_shell:
        extract_radius, min_atoms = config.effective_extract_radius, config.min_atoms
    else:
        extract_radius, min_atoms = 0.0, len(graph_atoms)
    selection = select_radial(
        frame, graph_atoms, config, extract_radius=extract_radius, min_atoms=min_atoms, protected=hotspots
    )
    present = set(selection.indices)
    hotspot_ids = frame.atom_ids[sorted(index for index in hotspots if index in present)].tolist()
    return cluster_from_selection(
        frame,
        selection,
        config,
        mode="graph",
        hotspot_atom_ids=hotspot_ids,
        event_id=event_id,
        metadata={
            "graph_hops": int(config.graph.hops),
            "graph_radial_shell": bool(config.graph.radial_shell),
            "graph_atom_ids": frame.atom_ids[graph_atoms].tolist(),
            "n_graph_atoms": len(graph_atoms),
        },
        config_payload=config.identity_payload({"slab", "block", "mode"}),
    )


__all__ = ["bonded_subgraph", "extract_graph"]
