# Ported from PHAL (hotspot_al.extraction.slab_extractor) at 96263b6; MIT License, see NOTICE.
"""Slab-patch extraction for surfaces and interfaces (surface normal = Cartesian z)."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from gmd_active_learning.contracts import ExtractedRegion, Frame
from gmd_active_learning.core.exceptions import ExtractionError, UnsupportedCombinationError
from gmd_active_learning.datasets.masks import force_weights_for_roles
from gmd_active_learning.extraction._placement import (
    apply_periodic_guard,
    assemble_region,
    frame_cell,
    hotspot_frame_indices,
    mic_placement,
)
from gmd_active_learning.extraction.config import ExtractionConfig
from gmd_active_learning.extraction.regions import radial_roles
from gmd_active_learning.utils.geometry import distances_to_group, hotspot_center

LOGGER = logging.getLogger(__name__)

_ALIGN_TOLERANCE = 1.0e-8


@dataclass(slots=True)
class LateralPlan:
    windows: list[float]
    periodic: list[bool]
    cell_lengths: list[float]
    repeats: list[int]
    record: dict[str, Any]


def _in_plane_aligned(cell: np.ndarray) -> bool:
    """a along x and b along y (the PHAL Cartesian window is exact only for such cells)."""

    return bool(
        abs(cell[0, 1]) <= _ALIGN_TOLERANCE
        and abs(cell[0, 2]) <= _ALIGN_TOLERANCE
        and abs(cell[1, 0]) <= _ALIGN_TOLERANCE
        and abs(cell[1, 2]) <= _ALIGN_TOLERANCE
    )


def plan_lateral_axes(frame: Frame, config: ExtractionConfig) -> LateralPlan:
    """Decide, per lateral axis, the window, its periodicity and the number of source periods.

    A lateral axis stays periodic only when the source axis is periodic and the requested window
    spans at least one period (PHAL rule: an arbitrary periodic window would wrap the cut edges onto
    each other). PHAL then used ``round(window / period)`` periods for the cell but kept a single
    image of the atoms (F-08); here ``slab.multi_period`` replicates, reduces or refuses.
    """

    slab = config.slab
    requested = [float(slab.lx), float(slab.ly)]
    cell = frame_cell(frame)
    lengths = np.linalg.norm(cell, axis=1)
    pbc = np.asarray(frame.pbc, dtype=bool) if frame.cell is not None else np.zeros(3, dtype=bool)
    record: dict[str, Any] = {"requested_window": requested, "window_reduced": [False, False], "skewed_fallback": False}
    windows, periodic, cell_lengths, repeats = [], [], [], []
    for axis in (0, 1):
        period = float(lengths[axis])
        can_tile = bool(pbc[axis]) and period > 1.0e-8 and requested[axis] >= period - slab.period_tolerance
        if not can_tile:
            windows.append(requested[axis])
            periodic.append(False)
            cell_lengths.append(requested[axis] + float(config.vacuum_padding))
            repeats.append(0)
            continue
        count = max(1, int(round(requested[axis] / period)))
        if count >= 2 and slab.multi_period == "error":
            raise ExtractionError(
                f"slab window {requested[axis]} A spans {count} periods of axis {axis} ({period:.4f} A); "
                "extraction.slab.multi_period=error (finding F-08)"
            )
        if count >= 2 and slab.multi_period == "single_period":
            record["window_reduced"][axis] = True
            count = 1
        windows.append(count * period)
        periodic.append(True)
        cell_lengths.append(count * period)
        repeats.append(count)
    if any(periodic) and not _in_plane_aligned(cell):
        horizontal = abs(cell[0, 2]) <= _ALIGN_TOLERANCE and abs(cell[1, 2]) <= _ALIGN_TOLERANCE
        if not (all(periodic) and horizontal):
            if slab.skewed_policy == "error":
                raise UnsupportedCombinationError(
                    "slab extraction cannot keep one lateral axis periodic in a skewed in-plane cell with a Cartesian "
                    "window; set extraction.slab.skewed_policy=nonperiodic or use a full-period window on both axes"
                )
            LOGGER.warning("skewed source cell: slab patch made laterally non-periodic (recorded)")
            record["skewed_fallback"] = True
            windows = requested
            periodic = [False, False]
            cell_lengths = [value + float(config.vacuum_padding) for value in requested]
            repeats = [0, 0]
    return LateralPlan(windows, periodic, cell_lengths, repeats, record)


def extract_slab(
    frame: Frame,
    hotspot_atom_ids: Any,
    config: ExtractionConfig | None = None,
    *,
    event_id: str | None = None,
) -> ExtractedRegion:
    """Lateral window ``(lx, ly)`` x vertical window ``2 * z_margin`` around the hotspot centroid.

    Periodic lateral axes are filled with the source's periodic images (``image_shifts`` differ per
    copy, parent atom IDs repeat), so the output cell holds exactly what it claims. For a skewed
    source cell with both axes periodic, the output keeps the source lateral cell vectors. z is
    always non-periodic with ``slab.z_vacuum`` above the top atom (PHAL: 6 A, hard-coded).
    """

    config = config or ExtractionConfig()
    hotspots = hotspot_frame_indices(frame, hotspot_atom_ids)
    pbc = frame.pbc if frame.cell is not None else False
    center = hotspot_center(frame.positions, hotspots, cell=frame.cell, pbc=pbc)
    all_indices = np.arange(frame.n_atoms)
    displacements, _ = mic_placement(frame, all_indices, center)
    plan = plan_lateral_axes(frame, config)
    z_half = float(config.slab.z_margin)

    mask = np.abs(displacements[:, 2]) <= z_half
    for axis in (0, 1):
        if not plan.periodic[axis]:
            mask &= np.abs(displacements[:, axis]) <= plan.windows[axis] * 0.5
    selected = np.where(mask)[0]
    record = dict(plan.record)
    if len(selected) == 0:
        LOGGER.warning("empty slab window; falling back to the hotspot atoms (recorded)")
        record["empty_window_fallback"] = True
        selected = np.asarray(sorted(hotspots), dtype=np.int64)
    local, shifts = mic_placement(frame, selected, center)

    cell = frame_cell(frame)
    ranges = [range(max(1, count)) if plan.periodic[axis] else range(1) for axis, count in enumerate(plan.repeats)]
    copies_local, copies_shift, copies_index = [], [], []
    for k0, k1 in itertools.product(*ranges):
        offset = np.array([k0, k1, 0], dtype=np.int64)
        copies_local.append(local + offset @ cell)
        copies_shift.append(shifts + offset)
        copies_index.append(selected)
    local = np.vstack(copies_local)
    shifts = np.vstack(copies_shift)
    indices = np.concatenate(copies_index)

    positions = local - local.min(axis=0)
    for axis in (0, 1):
        if not plan.periodic[axis]:
            positions[:, axis] += 0.5 * float(config.vacuum_padding)
    z_length = max(float(np.ptp(positions[:, 2]) + config.slab.z_vacuum), float(config.slab.z_vacuum), 1.0)
    if all(plan.periodic) and not _in_plane_aligned(cell):
        out_cell = np.vstack([plan.repeats[0] * cell[0], plan.repeats[1] * cell[1], [0.0, 0.0, z_length]])
    else:
        out_cell = np.diag([max(plan.cell_lengths[0], 1.0), max(plan.cell_lengths[1], 1.0), z_length])

    distances = distances_to_group(frame.positions, hotspots, cell=frame.cell, pbc=pbc)[indices]
    roles = radial_roles(
        distances,
        core_radius=float(config.core_radius),
        extract_radius=config.effective_extract_radius,
        boundary_thickness=float(config.boundary_thickness),
    )
    output_periods = np.array([plan.repeats[0] if plan.periodic[0] else 0, plan.repeats[1] if plan.periodic[1] else 0, 0])
    guard = apply_periodic_guard(
        frame,
        indices,
        shifts,
        config.periodic_guard,
        mode="slab",
        weights=force_weights_for_roles(roles, config.force_weights),
        output_periods=output_periods,
    )
    metadata = {
        "center": np.asarray(center, dtype=float).tolist(),
        "xy_lengths": [float(config.slab.lx), float(config.slab.ly)],
        "z_thickness": 2.0 * z_half,
        "windows": [float(value) for value in plan.windows],
        "lateral_pbc": list(plan.periodic),
        "lateral_repeats": [int(value) for value in plan.repeats],
        "z_vacuum": float(config.slab.z_vacuum),
        "extract_radius": config.effective_extract_radius,
        "core_radius": float(config.core_radius),
        "boundary_thickness": float(config.boundary_thickness),
        "slab_policy": record,
        "hotspots_outside_window": [int(frame.atom_ids[index]) for index in hotspots if index not in set(selected.tolist())],
        "periodic_guard": guard,
    }
    return assemble_region(
        frame,
        mode="slab",
        indices=indices,
        shifts=shifts,
        positions=positions,
        cell=out_cell,
        pbc=(bool(plan.periodic[0]), bool(plan.periodic[1]), False),
        roles=roles,
        hotspot_atom_ids=[int(frame.atom_ids[index]) for index in hotspots if index in set(selected.tolist())],
        config=config,
        event_id=event_id,
        metadata=metadata,
        config_payload=config.model_dump(mode="json", exclude={"graph", "block", "mode"}),
    )


__all__ = ["LateralPlan", "extract_slab", "plan_lateral_axes"]
