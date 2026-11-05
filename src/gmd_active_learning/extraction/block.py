# Ported from PHAL (hotspot_al.extraction.block) at 96263b6; MIT License, see NOTICE.
"""Block-localized extraction: a fixed spatial grid gives stable region identities (PHAL ADR 002).

Each anomalous atom maps to a grid block; adjacent blocks merge (26-neighbourhood, PBC-aware) into
groups of at most ``max_merged_blocks``. A group's atoms are the *label core*; non-core atoms within
``buffer.inner`` of the core form the inner buffer; the shell out to ``max(buffer.outer, halo,
inner)`` is split into an outer buffer and a frozen boundary of thickness ``frozen.thickness``.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gmd_active_learning.contracts import ExtractedRegion, Frame, RegionRole
from gmd_active_learning.core.exceptions import ExtractionError
from gmd_active_learning.datasets.masks import force_weights_for_roles
from gmd_active_learning.extraction._placement import (
    apply_periodic_guard,
    assemble_region,
    frame_cell,
    mic_placement,
)
from gmd_active_learning.extraction.config import BlockConfig, ExtractionConfig
from gmd_active_learning.extraction.cooldown import BlockCooldownTracker, BlockId
from gmd_active_learning.utils.geometry import distances_to_group, hotspot_center, padded_cluster_cell

LOGGER = logging.getLogger(__name__)


def _require_cell(frame: Frame) -> np.ndarray:
    if frame.cell is None or abs(float(np.linalg.det(frame.cell))) < 1.0e-12:
        raise ExtractionError("block extraction needs a non-degenerate cell (the spatial grid is defined on it)")
    return frame_cell(frame)


def block_grid_shape(cell: np.ndarray, block_size: Iterable[float]) -> tuple[int, int, int]:
    """Blocks per cell axis: ``max(1, ceil(|a_i| / size_i))`` (zero-length axes use ``size``)."""

    size = np.asarray(list(block_size), dtype=float)
    if size.shape != (3,) or np.any(size <= 0.0):
        raise ExtractionError("extraction.block.size must contain three positive numbers")
    lengths = np.linalg.norm(np.asarray(cell, dtype=float), axis=1)
    lengths = np.where(lengths > 1.0e-12, lengths, size)
    shape = np.maximum(1, np.ceil(lengths / size)).astype(int).tolist()
    return (int(shape[0]), int(shape[1]), int(shape[2]))


def assign_atoms_to_spatial_blocks(frame: Frame, block_size: Iterable[float]) -> dict[int, BlockId]:
    """Frame index -> block id. Fractional coordinates wrap on periodic axes, clip on the others."""

    cell = _require_cell(frame)
    grid = block_grid_shape(cell, block_size)
    scaled = np.linalg.solve(cell.T, frame.positions.T).T
    for axis, periodic in enumerate(frame.pbc):
        if periodic:
            scaled[:, axis] = np.mod(scaled[:, axis], 1.0)
    coords = np.floor(scaled * np.asarray(grid, dtype=float)).astype(int)
    for axis, count in enumerate(grid):
        coords[:, axis] = np.clip(coords[:, axis], 0, count - 1)
    return {index: (int(row[0]), int(row[1]), int(row[2])) for index, row in enumerate(coords)}


def invert_block_mapping(atom_to_block: dict[int, BlockId]) -> dict[BlockId, list[int]]:
    block_to_atoms: dict[BlockId, list[int]] = defaultdict(list)
    for atom_index, block_id in atom_to_block.items():
        block_to_atoms[block_id].append(atom_index)
    return {block_id: sorted(indices) for block_id, indices in block_to_atoms.items()}


def anomalous_blocks(atom_to_block: dict[int, BlockId], atom_indices: Iterable[int]) -> list[BlockId]:
    return sorted({atom_to_block[index] for index in atom_indices if index in atom_to_block})


def _blocks_are_adjacent(
    left: BlockId,
    right: BlockId,
    *,
    grid_shape: tuple[int, int, int] | None,
    pbc: tuple[bool, bool, bool],
) -> bool:
    for axis in range(3):
        diff = abs(left[axis] - right[axis])
        if grid_shape is not None and pbc[axis] and grid_shape[axis] > 1:
            diff = min(diff, grid_shape[axis] - diff)
        if diff > 1:
            return False
    return True


def _split_connected_block_group(
    group: list[BlockId],
    *,
    max_blocks: int,
    grid_shape: tuple[int, int, int] | None,
    pbc: tuple[bool, bool, bool],
) -> list[list[BlockId]]:
    remaining = set(group)
    chunks: list[list[BlockId]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        chunk = [start]
        queue: deque[BlockId] = deque([start])
        while queue and len(chunk) < max_blocks:
            current = queue.popleft()
            neighbors = sorted(
                other for other in remaining if _blocks_are_adjacent(current, other, grid_shape=grid_shape, pbc=pbc)
            )
            for other in neighbors:
                if len(chunk) >= max_blocks:
                    break
                remaining.remove(other)
                chunk.append(other)
                queue.append(other)
        chunks.append(sorted(chunk))
    return chunks


def merge_adjacent_blocks(
    block_ids: Iterable[BlockId],
    *,
    max_merged_blocks: int | None = None,
    grid_shape: tuple[int, int, int] | None = None,
    pbc: tuple[bool, bool, bool] = (False, False, False),
) -> list[list[BlockId]]:
    """26-neighbour BFS merge; oversized groups split into connected chunks (PHAL algorithm)."""

    remaining = set(block_ids)
    groups: list[list[BlockId]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group = [start]
        queue: deque[BlockId] = deque([start])
        while queue:
            current = queue.popleft()
            connected = [
                other for other in remaining if _blocks_are_adjacent(current, other, grid_shape=grid_shape, pbc=pbc)
            ]
            for other in connected:
                remaining.remove(other)
                group.append(other)
                queue.append(other)
        group = sorted(group)
        if max_merged_blocks is not None and max_merged_blocks > 0 and len(group) > max_merged_blocks:
            groups.extend(
                _split_connected_block_group(group, max_blocks=max_merged_blocks, grid_shape=grid_shape, pbc=pbc)
            )
        else:
            groups.append(group)
    return groups


@dataclass(slots=True)
class BlockSelection:
    core: list[int]
    inner: list[int]
    outer: list[int]
    frozen: list[int]
    distances: np.ndarray
    inner_radius: float
    outer_radius: float
    frozen_thickness: float
    trimmed: dict[str, list[int]] = field(default_factory=dict)


def select_block_atoms(frame: Frame, selected_blocks: set[BlockId], atom_to_block: dict[int, BlockId], block: BlockConfig) -> BlockSelection:
    core = sorted(index for index, block_id in atom_to_block.items() if block_id in selected_blocks)
    if not core:
        raise ExtractionError(f"block extraction found no atoms in block group {sorted(selected_blocks)!r}")
    inner_radius = float(block.buffer.inner)
    outer_radius = max(float(block.buffer.outer), float(block.halo), inner_radius)
    frozen_enabled = bool(block.frozen.enabled)
    thickness = max(0.0, float(block.frozen.thickness)) if frozen_enabled else 0.0
    distances = distances_to_group(frame.positions, core, cell=frame.cell, pbc=frame.pbc)
    core_set = set(core)
    inner = sorted(index for index, distance in enumerate(distances) if index not in core_set and distance <= inner_radius)
    inner_set = set(inner)
    shell = [
        index
        for index, distance in enumerate(distances)
        if index not in core_set and index not in inner_set and distance <= outer_radius
    ]
    frozen_start = max(inner_radius, outer_radius - thickness)
    frozen = sorted(index for index in shell if frozen_enabled and distances[index] >= frozen_start)
    frozen_set = set(frozen)
    outer = sorted(index for index in shell if index not in frozen_set)
    return BlockSelection(core, inner, outer, frozen, distances, inner_radius, outer_radius, thickness)


def _trim_group(indices: list[int], *, target: int, distances: np.ndarray) -> list[int]:
    if target >= len(indices):
        return indices
    if target <= 0:
        return []
    return sorted(sorted(indices, key=lambda index: float(distances[index]))[:target])


def enforce_max_atoms(selection: BlockSelection, max_atoms: int) -> BlockSelection:
    """Raise when the core alone exceeds ``max_atoms``; otherwise trim frozen, outer, inner (farthest first).

    PHAL trimmed silently; the removed atoms are recorded in ``selection.trimmed``.
    """

    if len(selection.core) > max_atoms:
        raise ExtractionError(f"max_atoms_exceeded: label_core has {len(selection.core)} atoms but max_atoms={max_atoms}")
    before = {"inner": list(selection.inner), "outer": list(selection.outer), "frozen": list(selection.frozen)}
    core, inner, outer, frozen = selection.core, list(selection.inner), list(selection.outer), list(selection.frozen)
    frozen = _trim_group(frozen, target=max_atoms - len(core) - len(inner) - len(outer), distances=selection.distances)
    outer = _trim_group(outer, target=max_atoms - len(core) - len(inner) - len(frozen), distances=selection.distances)
    inner = _trim_group(inner, target=max_atoms - len(core) - len(outer) - len(frozen), distances=selection.distances)
    selection.inner, selection.outer, selection.frozen = sorted(inner), sorted(outer), sorted(frozen)
    after = {"inner": selection.inner, "outer": selection.outer, "frozen": selection.frozen}
    selection.trimmed = {key: sorted(set(before[key]) - set(after[key])) for key in before}
    return selection


def _block_roles(n_atoms: int, local: dict[int, int], selection: BlockSelection) -> list[RegionRole]:
    roles = [RegionRole.UNASSIGNED] * n_atoms
    for group, role in (
        (selection.outer, RegionRole.OUTER_BUFFER),
        (selection.frozen, RegionRole.BOUNDARY),
        (selection.inner, RegionRole.INNER_BUFFER),
        (selection.core, RegionRole.CORE),
    ):
        for index in group:
            roles[local[index]] = role
    return roles


def extract_block_region(
    frame: Frame,
    block_ids: list[BlockId],
    *,
    ood_atom_ids: Any,
    config: ExtractionConfig,
    atom_to_block: dict[int, BlockId] | None = None,
    event_id: str | None = None,
) -> ExtractedRegion:
    """One block group -> one non-periodic region with label core / buffers / frozen boundary."""

    block = config.block
    if block.size is None:
        raise ExtractionError("extraction.block.size is required for block extraction")
    atom_to_block = atom_to_block or assign_atoms_to_spatial_blocks(frame, block.size)
    ood_indices = frame.index_of(ood_atom_ids).tolist() if len(np.atleast_1d(ood_atom_ids)) else []
    max_atoms = int(block.max_atoms or config.max_atoms or frame.n_atoms)
    min_atoms = int(block.min_atoms if block.min_atoms is not None else (config.min_atoms if config.min_atoms is not None else 1))
    selection = enforce_max_atoms(select_block_atoms(frame, set(block_ids), atom_to_block, block), max_atoms)
    trimmed_total = sum(len(values) for values in selection.trimmed.values())
    if trimmed_total:
        LOGGER.warning("block region trimmed %d shell atoms to max_atoms=%d (recorded)", trimmed_total, max_atoms)
    selected = sorted([*selection.core, *selection.inner, *selection.outer, *selection.frozen])
    below_min = len(selected) < min_atoms
    if below_min:
        message = (
            f"block region has {len(selected)} atoms, below extraction.block.min_atoms={min_atoms}; "
            "the labeled core may be too small to be informative"
        )
        if block.min_atoms_policy == "error":
            raise ExtractionError(f"min_atoms_not_met: {message}")
        if block.min_atoms_policy == "warn":
            LOGGER.warning("%s", message)

    indices = np.asarray(selected, dtype=np.int64)
    center = hotspot_center(frame.positions, selected, cell=frame.cell, pbc=frame.pbc)
    displacements, shifts = mic_placement(frame, indices, center)
    positions = displacements - displacements.min(axis=0) + float(block.vacuum_offset)
    cell = padded_cluster_cell(positions, padding=float(block.cell_padding))
    local = {original: position for position, original in enumerate(selected)}
    roles = _block_roles(len(selected), local, selection)
    guard = apply_periodic_guard(
        frame, indices, shifts, config.periodic_guard, mode="block", weights=force_weights_for_roles(roles, config.force_weights)
    )
    hotspot_ids = [int(frame.atom_ids[index]) for index in ood_indices if index in local]
    metadata = {
        "extraction_mode": "block",
        "block_scheme": block.scheme,
        "block_ids": [list(block_id) for block_id in sorted(block_ids)],
        "source_frame_id": int(frame.step),
        "ood_atom_ids": [int(frame.atom_ids[index]) for index in ood_indices],
        "training_weights": config.force_weights.as_dict(),
        "periodic": bool(any(frame.pbc)),
        "cell": frame_cell(frame).tolist(),
        "block_size": [float(value) for value in block.size],
        "cooldown_steps": int(block.cooldown_steps),
        "inner_buffer": selection.inner_radius,
        "outer_buffer": selection.outer_radius,
        "frozen_thickness": selection.frozen_thickness,
        "inner_buffer_definition": "distance_to_core_atoms",
        "outer_buffer_definition": "distance_to_core_atoms",
        "core_max_distance_to_group": float(max(selection.distances[index] for index in selection.core)),
        "approximate_spatial_grid": True,
        "below_min_atoms": below_min,
        "min_atoms": min_atoms,
        "max_atoms": max_atoms,
        "trimmed_atom_ids": {key: frame.atom_ids[values].tolist() for key, values in selection.trimmed.items()},
        "center": np.asarray(center, dtype=float).tolist(),
        "vacuum_offset": float(block.vacuum_offset),
        "cell_padding": float(block.cell_padding),
        "periodic_guard": guard,
    }
    return assemble_region(
        frame,
        mode="block",
        indices=indices,
        shifts=shifts,
        positions=positions,
        cell=cell,
        pbc=(False, False, False),
        roles=roles,
        hotspot_atom_ids=hotspot_ids,
        config=config,
        event_id=event_id,
        metadata=metadata,
        config_payload={
            "block": block.model_dump(mode="json", exclude={"cooldown_steps", "cooldown_state_path"}),
            "force_weights": config.force_weights.model_dump(mode="json"),
            "guard": config.periodic_guard.model_dump(mode="json"),
            "freeze_boundary": config.freeze_boundary,
            "max_atoms": config.max_atoms,
            "min_atoms": config.min_atoms,
            "group": [list(block_id) for block_id in sorted(block_ids)],
        },
    )


def plan_block_groups(frame: Frame, ood_atom_ids: Any, config: ExtractionConfig) -> tuple[dict[int, BlockId], list[list[BlockId]]]:
    """Atom->block map and the merged block groups for the anomalous atoms."""

    block = config.block
    if block.size is None:
        raise ExtractionError("extraction.block.size is required for block extraction")
    atom_to_block = assign_atoms_to_spatial_blocks(frame, block.size)
    ood_indices = frame.index_of(ood_atom_ids).tolist()
    block_ids = anomalous_blocks(atom_to_block, ood_indices)
    if not block_ids:
        return atom_to_block, []
    if block.merge_adjacent:
        groups = merge_adjacent_blocks(
            block_ids,
            max_merged_blocks=int(block.max_merged_blocks),
            grid_shape=block_grid_shape(frame_cell(frame), block.size),
            pbc=tuple(bool(flag) for flag in frame.pbc),  # type: ignore[arg-type]
        )
    else:
        groups = [[block_id] for block_id in block_ids]
    return atom_to_block, groups


def extract_block_regions(
    frame: Frame,
    ood_atom_ids: Any,
    config: ExtractionConfig | None = None,
    *,
    cooldown: BlockCooldownTracker | None = None,
    event_id: str | None = None,
) -> list[ExtractedRegion]:
    """All block regions for one frame; groups still in cooldown are skipped (and logged)."""

    config = config or ExtractionConfig()
    ids = np.atleast_1d(np.asarray(ood_atom_ids, dtype=np.int64))
    if ids.size == 0:
        return []
    atom_to_block, groups = plan_block_groups(frame, ids, config)
    regions: list[ExtractedRegion] = []
    step = int(frame.step)
    for index, group in enumerate(groups):
        if cooldown is not None and cooldown.should_skip(group, step):
            LOGGER.info("block group %s skipped: cooldown", group)
            continue
        region = extract_block_region(
            frame, group, ood_atom_ids=ids, config=config, atom_to_block=atom_to_block, event_id=event_id
        )
        region.metadata["block_group_index"] = index
        regions.append(region)
        if cooldown is not None:
            cooldown.update(group, step)
    return regions


__all__ = [
    "BlockSelection",
    "anomalous_blocks",
    "assign_atoms_to_spatial_blocks",
    "block_grid_shape",
    "enforce_max_atoms",
    "extract_block_region",
    "extract_block_regions",
    "invert_block_mapping",
    "merge_adjacent_blocks",
    "plan_block_groups",
    "select_block_atoms",
]
