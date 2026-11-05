"""Shared machinery for every extraction mode: image shifts, the periodic guard, region assembly.

Every extracted atom is recorded by its stable parent atom ID and the integer lattice vector
(``image_shift``) that was added to its parent position, so that

    region_position = parent_position + image_shift @ parent_cell + origin_translation

where ``origin_translation`` is one rigid vector per region (stored in
``metadata['origin_translation']``) that places the region inside its own box. PHAL recorded
neither (finding F-09).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from gmd_active_learning.contracts import ExtractedRegion, Frame, RegionRole, content_hash, stable_id
from gmd_active_learning.core.exceptions import ExtractionError
from gmd_active_learning.datasets.masks import force_weights_for_roles
from gmd_active_learning.extraction.config import ExtractionConfig, PeriodicGuardConfig
from gmd_active_learning.utils.periodic import mic_displacements_from_reference

LOGGER = logging.getLogger(__name__)

_SHIFT_TOLERANCE = 1.0e-6
_TRANSLATION_TOLERANCE = 1.0e-8


def frame_is_periodic(frame: Frame) -> bool:
    return frame.cell is not None and any(frame.pbc)


def frame_cell(frame: Frame) -> np.ndarray:
    return np.zeros((3, 3)) if frame.cell is None else np.asarray(frame.cell, dtype=float)


def frame_atoms(frame: Frame) -> Atoms:
    """ASE view of the frame (cell zeroed when absent) for neighbour lists."""

    atoms = Atoms(numbers=frame.numbers, positions=frame.positions, pbc=frame.pbc if frame.cell is not None else False)
    if frame.cell is not None:
        atoms.set_cell(frame.cell)
    return atoms


def image_shifts(frame: Frame, indices: np.ndarray, placed: np.ndarray) -> np.ndarray:
    """Integer lattice vectors ``n`` with ``placed = parent + n @ cell`` (exactly integer, or raise)."""

    indices = np.asarray(indices, dtype=np.int64)
    placed = np.asarray(placed, dtype=float).reshape(-1, 3)
    if not frame_is_periodic(frame):
        if not np.allclose(placed, frame.positions[indices], atol=_SHIFT_TOLERANCE, rtol=0.0):
            raise ExtractionError("non-periodic frame: placed positions must equal parent positions")
        return np.zeros((len(indices), 3), dtype=np.int64)
    cell = frame_cell(frame)
    fractional = (placed - frame.positions[indices]) @ np.linalg.inv(cell)
    shifts = np.rint(fractional)
    if np.any(np.abs(fractional - shifts) > _SHIFT_TOLERANCE):
        raise ExtractionError("placed positions are not lattice translations of the parent positions")
    pbc = np.asarray(frame.pbc, dtype=bool)
    if np.any(shifts[:, ~pbc] != 0):
        raise ExtractionError("an image shift along a non-periodic axis was requested")
    return shifts.astype(np.int64)


def mic_placement(frame: Frame, indices: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PHAL placement: each atom at its minimum image relative to ``center``.

    Returns ``(displacements_from_center, image_shifts)``.
    """

    indices = np.asarray(indices, dtype=np.int64)
    displacements = mic_displacements_from_reference(
        center, frame.positions[indices], cell=frame.cell, pbc=frame.pbc if frame.cell is not None else False
    )
    shifts = image_shifts(frame, indices, np.asarray(center, dtype=float) + displacements)
    return displacements, shifts


def perpendicular_widths(cell: np.ndarray) -> np.ndarray:
    """Distance between opposite faces of the cell along each axis."""

    volume = abs(float(np.linalg.det(cell)))
    widths = np.empty(3)
    for axis in range(3):
        cross = np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3])
        widths[axis] = volume / float(np.linalg.norm(cross))
    return widths


def min_periodic_half_width(frame: Frame) -> float | None:
    """Half the smallest perpendicular width over periodic axes (None for non-periodic frames)."""

    if not frame_is_periodic(frame):
        return None
    widths = perpendicular_widths(frame_cell(frame))
    return float(0.5 * np.min(widths[np.asarray(frame.pbc, dtype=bool)]))


def find_torn_pairs(
    frame: Frame,
    indices: np.ndarray,
    shifts: np.ndarray,
    cutoff: float,
    *,
    output_periods: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Frame-index pairs within ``cutoff`` in the parent whose relative image differs in the region.

    For each parent neighbour entry ``(i, j, S)`` with both atoms in the region, some copy of
    ``j`` must sit at ``shift_j - shift_i == S`` relative to each copy of ``i``. Along axes where
    the region itself is periodic with ``output_periods[k]`` source periods, the comparison is
    modulo that period. The test is exact (integer shifts), not a distance tolerance.
    """

    if not frame_is_periodic(frame):
        return []
    indices = np.asarray(indices, dtype=np.int64)
    shifts = np.asarray(shifts, dtype=np.int64).reshape(-1, 3)
    periods = np.zeros(3, dtype=np.int64) if output_periods is None else np.asarray(output_periods, dtype=np.int64)
    copies: dict[int, list[np.ndarray]] = defaultdict(list)
    for index, shift in zip(indices.tolist(), shifts):
        copies[int(index)].append(shift)
    left, right, lattice = neighbor_list("ijS", frame_atoms(frame), float(cutoff))
    in_region = np.isin(left, indices) & np.isin(right, indices) & (left < right)
    torn: list[tuple[int, int]] = []
    for i, j, shift in zip(left[in_region].tolist(), right[in_region].tolist(), lattice[in_region]):
        for shift_i in copies[i]:
            ok = False
            for shift_j in copies[j]:
                difference = shift_j - shift_i - shift
                wrapped = np.where(periods > 0, np.mod(difference, np.maximum(periods, 1)), difference)
                if not np.any(wrapped):
                    ok = True
                    break
            if not ok:
                torn.append((int(i), int(j)))
    return torn


def apply_periodic_guard(
    frame: Frame,
    indices: np.ndarray,
    shifts: np.ndarray,
    guard: PeriodicGuardConfig,
    *,
    mode: str,
    weights: np.ndarray,
    extra: dict[str, Any] | None = None,
    output_periods: np.ndarray | None = None,
) -> dict[str, Any]:
    """Torn-pair check (finding F-09); applies ``guard.policy`` to pairs touching weighted atoms.

    A torn pair between two zero-weight atoms behaves like any other cut at the region surface (and
    a torn covalent bond is capped by the capping step), so it is recorded only. A torn pair that
    involves an atom with a non-zero force weight means a trained atom lost a neighbour that exists
    in the parent: that is the violation ``guard.policy`` acts on.
    """

    report: dict[str, Any] = {"checked": frame_is_periodic(frame), "policy": guard.policy, "pair_cutoff": guard.pair_cutoff}
    report.update(extra or {})
    weight_of: dict[int, float] = defaultdict(float)
    for index, weight in zip(np.asarray(indices).tolist(), np.asarray(weights, dtype=float).tolist()):
        weight_of[int(index)] = max(weight_of[int(index)], weight)
    torn = find_torn_pairs(frame, indices, shifts, guard.pair_cutoff, output_periods=output_periods)
    weighted = [pair for pair in torn if weight_of[pair[0]] > 0.0 or weight_of[pair[1]] > 0.0]
    report["torn_pairs"] = len(torn)
    report["torn_weighted_pairs"] = len(weighted)
    report["torn_pair_examples"] = [[int(frame.atom_ids[i]), int(frame.atom_ids[j])] for i, j in (weighted or torn)[:20]]
    report["violation"] = bool(weighted)
    if report.get("radius_exceeds_half_cell"):
        LOGGER.warning("%s region: extraction radius exceeds half the periodic cell width (recorded)", mode)
    if weighted:
        message = (
            f"{mode} region is torn by the periodic boundary: {len(weighted)} parent neighbour pairs within "
            f"{guard.pair_cutoff} A that involve weighted atoms are placed at different images (finding F-09)"
        )
        if guard.policy == "error":
            raise ExtractionError(message + "; reduce the region size or set extraction.periodic_guard.policy=flag")
        if guard.policy == "flag":
            LOGGER.warning("%s; kept because periodic_guard.policy=flag", message)
    return report


def region_identity(
    frame: Frame,
    *,
    mode: str,
    event_id: str | None,
    hotspot_atom_ids: list[int],
    parent_atom_ids: np.ndarray,
    shifts: np.ndarray,
    config_payload: Any,
) -> str:
    return stable_id(
        "region",
        frame.run_id,
        int(frame.step),
        mode,
        event_id,
        sorted(int(atom_id) for atom_id in hotspot_atom_ids),
        np.asarray(parent_atom_ids, dtype=np.int64).tolist(),
        np.asarray(shifts, dtype=np.int64).tolist(),
        content_hash(config_payload),
    )


def assemble_region(
    frame: Frame,
    *,
    mode: str,
    indices: np.ndarray,
    shifts: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray | None,
    pbc: tuple[bool, bool, bool],
    roles: list[RegionRole],
    hotspot_atom_ids: list[int],
    config: ExtractionConfig,
    event_id: str | None,
    metadata: dict[str, Any],
    config_payload: Any,
) -> ExtractedRegion:
    """Build the contract object, checking the position/shift/translation invariant."""

    indices = np.asarray(indices, dtype=np.int64)
    shifts = np.asarray(shifts, dtype=np.int64).reshape(-1, 3)
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    unwrapped = frame.positions[indices] + shifts @ frame_cell(frame)
    offsets = positions - unwrapped
    translation = offsets[0] if len(offsets) else np.zeros(3)
    if len(offsets) and float(np.abs(offsets - translation).max()) > _TRANSLATION_TOLERANCE * max(1.0, float(np.abs(unwrapped).max())):
        raise ExtractionError(f"{mode} region positions are not a rigid translation of the parent images")
    parent_ids = frame.atom_ids[indices]
    roles = [RegionRole(role) for role in roles]
    frozen = np.asarray([config.freeze_boundary and role is RegionRole.BOUNDARY for role in roles], dtype=bool)
    region_id = region_identity(
        frame,
        mode=mode,
        event_id=event_id,
        hotspot_atom_ids=hotspot_atom_ids,
        parent_atom_ids=parent_ids,
        shifts=shifts,
        config_payload=config_payload,
    )
    full_metadata = {
        "mode": mode,
        "origin_translation": [float(value) for value in translation],
        "parent_cell": None if frame.cell is None else frame_cell(frame).tolist(),
        "parent_pbc": list(frame.pbc),
        "parent_frame_indices": indices.tolist(),
        "force_weight_policy": config.force_weights.as_dict(),
        **metadata,
    }
    return ExtractedRegion(
        region_id=region_id,
        mode=mode,
        run_id=frame.run_id,
        parent_step=int(frame.step),
        event_id=event_id,
        numbers=frame.numbers[indices],
        positions=positions,
        cell=cell,
        pbc=pbc,
        parent_atom_ids=parent_ids,
        image_shifts=shifts,
        roles=roles,
        force_weights=force_weights_for_roles(roles, config.force_weights),
        frozen=frozen,
        hotspot_atom_ids=[int(atom_id) for atom_id in hotspot_atom_ids],
        model_version=frame.model_version,
        metadata=full_metadata,
    )


def hotspot_frame_indices(frame: Frame, hotspot_atom_ids: Any) -> list[int]:
    ids = [int(atom_id) for atom_id in np.atleast_1d(np.asarray(hotspot_atom_ids, dtype=np.int64))]
    if not ids:
        raise ExtractionError("at least one hotspot atom is required for extraction")
    if len(set(ids)) != len(ids):
        raise ExtractionError("hotspot atom ids must be unique")
    return frame.index_of(ids).tolist()


__all__ = [
    "apply_periodic_guard",
    "assemble_region",
    "find_torn_pairs",
    "frame_atoms",
    "frame_cell",
    "frame_is_periodic",
    "hotspot_frame_indices",
    "image_shifts",
    "mic_placement",
    "min_periodic_half_width",
    "perpendicular_widths",
]
