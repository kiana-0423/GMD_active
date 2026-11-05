# Ported from PHAL (hotspot_al.extraction.h_capping) at 96263b6; MIT License, see NOTICE.
"""Hydrogen capping of cut covalent bonds, as a separate step for regions of any mode.

Differences from PHAL (documented in docs/extraction_and_datasets.md):

* Caps are appended with :meth:`ExtractedRegion.with_caps`, so roles, force weights, frozen flags,
  parent IDs and image shifts all grow together. PHAL extended some per-atom lists but not block
  mode's ``atom_role``, so capped block regions could not be written (finding F-04).
* A bond counts as cut when its partner image is not in the region, decided with integer image
  shifts. For regions that are not torn this is PHAL's "partner not selected" rule; for torn
  regions (periodic guard ``flag``) it also caps bonds whose partner exists only at another image.
* Every cut bond that stays dangling is recorded (``uncapped_cut_bonds``, ``partial``); PHAL left
  partial capping untagged.
* ``allow_core_capping`` keeps PHAL's real meaning (also cap inner-buffer atoms); ``cap_core_atoms``
  is the switch that caps core atoms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list

from gmd_active_learning.contracts import ExtractedRegion, Frame, RegionRole
from gmd_active_learning.core.exceptions import ExtractionError
from gmd_active_learning.extraction._placement import frame_atoms, frame_cell
from gmd_active_learning.extraction.config import CappingConfig

LOGGER = logging.getLogger(__name__)

METAL_LIKE: frozenset[str] = frozenset(
    {"Li", "Na", "K", "Mg", "Ca", "Al", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Zr", "Mo", "Ag", "Sn", "Pt", "Au"}
)
OXIDE_FORMERS: frozenset[str] = METAL_LIKE | {"Si"}


@dataclass(frozen=True, slots=True)
class CutBond:
    local_index: int
    parent_index: int
    partner_index: int
    lattice_shift: tuple[int, int, int]
    vector: np.ndarray  # parent -> partner image, Angstrom


def covalent_bonds_with_shifts(frame: Frame, scale: float) -> list[tuple[int, int, tuple[int, int, int], np.ndarray]]:
    """Directed covalent bonds ``(i, j, S, D)`` of the parent frame, ``D = r_j + S @ cell - r_i``.

    Same criterion as :func:`gmd_active_learning.utils.bonds.infer_bonds` (``d <= scale * (r_i +
    r_j)``) but keeping the lattice shift of every image, which capping needs to decide whether a
    partner image is present in the region.
    """

    if frame.n_atoms < 2:
        return []
    radii = covalent_radii[frame.numbers]
    cutoff = float(scale * 2.0 * np.max(radii))
    left, right, distance, lattice, vector = neighbor_list("ijdSD", frame_atoms(frame), cutoff)
    bonded = distance <= scale * (radii[left] + radii[right])
    bonds = [
        (int(i), int(j), (int(s[0]), int(s[1]), int(s[2])), np.asarray(d, dtype=float))
        for i, j, s, d in zip(left[bonded], right[bonded], lattice[bonded], vector[bonded])
    ]
    bonds.sort(key=lambda item: (item[0], item[1], item[2]))
    return bonds


def _output_periods(region: ExtractedRegion, frame: Frame) -> np.ndarray:
    """Source periods per axis along which the region itself is periodic (0 = non-periodic)."""

    periods = np.zeros(3, dtype=np.int64)
    if region.cell is None or not any(region.pbc) or frame.cell is None:
        return periods
    parent_lengths = np.linalg.norm(frame_cell(frame), axis=1)
    region_lengths = np.linalg.norm(region.cell, axis=1)
    for axis in range(3):
        if region.pbc[axis]:
            periods[axis] = max(1, int(round(region_lengths[axis] / parent_lengths[axis])))
    return periods


def find_cut_bonds(region: ExtractedRegion, frame: Frame, scale: float) -> list[CutBond]:
    """Bonds from region atoms to partner images that are not in the region (caps excluded)."""

    real = [index for index, role in enumerate(region.roles) if role is not RegionRole.H_CAP]
    if not real:
        return []
    parent_indices = frame.index_of(region.parent_atom_ids[real])
    periods = _output_periods(region, frame)
    present: dict[int, list[np.ndarray]] = {}
    for local, parent in zip(real, parent_indices.tolist()):
        present.setdefault(parent, []).append(region.image_shifts[local])
    by_atom: dict[int, list[tuple[int, int, tuple[int, int, int], np.ndarray]]] = {}
    for bond in covalent_bonds_with_shifts(frame, scale):
        by_atom.setdefault(bond[0], []).append(bond)
    cut: list[CutBond] = []
    for local, parent in zip(real, parent_indices.tolist()):
        shift_i = region.image_shifts[local]
        for _i, partner, lattice, vector in by_atom.get(parent, []):
            wanted = shift_i + np.asarray(lattice, dtype=np.int64)
            intact = False
            for shift_j in present.get(partner, []):
                difference = shift_j - wanted
                wrapped = np.where(periods > 0, np.mod(difference, np.maximum(periods, 1)), difference)
                if not np.any(wrapped):
                    intact = True
                    break
            if not intact:
                cut.append(CutBond(local, parent, partner, lattice, vector))
    return cut


def _skip(region: ExtractedRegion, reason: str, cut: list[CutBond], frame: Frame) -> ExtractedRegion:
    data = region.to_dict()
    record = _record(region, frame, cut, capped=[], reason=reason)
    data["metadata"] = {**region.metadata, "h_capping": record, "h_capping_skipped": True, "h_capping_skipped_reason": reason}
    return ExtractedRegion.from_dict(data)


def _bond_entry(region: ExtractedRegion, frame: Frame, bond: CutBond, why: str) -> dict[str, Any]:
    return {
        "local_index": bond.local_index,
        "parent_atom_id": int(frame.atom_ids[bond.parent_index]),
        "partner_atom_id": int(frame.atom_ids[bond.partner_index]),
        "lattice_shift": list(bond.lattice_shift),
        "element": frame.symbols[bond.parent_index],
        "role": region.roles[bond.local_index].value,
        "reason": why,
    }


def _record(
    region: ExtractedRegion, frame: Frame, cut: list[CutBond], *, capped: list[CutBond], reason: str | None,
    uncapped: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if uncapped is None:
        uncapped = [_bond_entry(region, frame, bond, reason or "skipped") for bond in cut]
    return {
        "applied": reason is None,
        "skipped_reason": reason,
        "n_cut_bonds": len(cut),
        "n_caps": len(capped),
        "uncapped_cut_bonds": uncapped,
        "partial": bool(reason is None and uncapped),
    }


def add_h_caps(region: ExtractedRegion, frame: Frame, config: CappingConfig | None = None) -> ExtractedRegion:
    """Cap cut bonds of boundary / outer-buffer atoms (configurable) with H at a fixed X-H length.

    Skipped, with ``h_capping_skipped`` / ``h_capping_skipped_reason`` recorded, when disabled
    (``disabled_by_config``), when the region contains a metal-like element (``metal_system``), or
    when it is an oxide or any cut bond is O-(metal|Si) (``oxide_system``). The cap sits at
    ``parent + L_XH * unit(bond vector)``; one H per cut bond, no clash check, cell unchanged.
    """

    config = config or CappingConfig()
    if any(role is RegionRole.H_CAP for role in region.roles):
        raise ExtractionError(f"region {region.region_id} is already capped")
    if not config.enabled:
        return _skip(region, "disabled_by_config", find_cut_bonds(region, frame, config.covalent_scale), frame)
    cut = find_cut_bonds(region, frame, config.covalent_scale)
    symbols = {frame.symbols[index] for index in frame.index_of(region.parent_atom_ids)}
    if config.disabled_for_metals and symbols & METAL_LIKE:
        return _skip(region, "metal_system", cut, frame)
    source_symbols = frame.symbols
    region_is_oxide = "O" in symbols and bool(symbols & OXIDE_FORMERS)
    cut_oxide = any(
        (source_symbols[bond.parent_index] == "O" and source_symbols[bond.partner_index] in OXIDE_FORMERS)
        or (source_symbols[bond.partner_index] == "O" and source_symbols[bond.parent_index] in OXIDE_FORMERS)
        for bond in cut
    )
    if config.disabled_for_oxides_by_default and (region_is_oxide or cut_oxide):
        return _skip(region, "oxide_system", cut, frame)

    allowed = {RegionRole.BOUNDARY, RegionRole.OUTER_BUFFER}
    if config.allow_core_capping:
        allowed.add(RegionRole.INNER_BUFFER)
    if config.cap_core_atoms:
        allowed.add(RegionRole.CORE)
    positions: list[np.ndarray] = []
    caps_meta: list[dict[str, Any]] = []
    capped: list[CutBond] = []
    uncapped: list[dict[str, Any]] = []
    for bond in cut:
        symbol = source_symbols[bond.parent_index]
        role = region.roles[bond.local_index]
        norm = float(np.linalg.norm(bond.vector))
        if role not in allowed:
            uncapped.append(_bond_entry(region, frame, bond, f"role_{role.value}_not_capped"))
            continue
        if symbol not in config.bond_lengths:
            uncapped.append(_bond_entry(region, frame, bond, f"no_bond_length_for_{symbol}"))
            continue
        if norm < 1.0e-8:
            uncapped.append(_bond_entry(region, frame, bond, "zero_length_bond"))
            continue
        length = float(config.bond_lengths[symbol])
        positions.append(region.positions[bond.local_index] + length * bond.vector / norm)
        capped.append(bond)
        caps_meta.append(
            {
                "parent_local_index": bond.local_index,
                "parent_atom_id": int(frame.atom_ids[bond.parent_index]),
                "broken_partner_atom_id": int(frame.atom_ids[bond.partner_index]),
                "lattice_shift": list(bond.lattice_shift),
                "bond_length": length,
                "element": symbol,
            }
        )
    record = _record(region, frame, cut, capped=capped, reason=None, uncapped=uncapped)
    if record["partial"]:
        LOGGER.warning("region %s: %d cut bonds left uncapped (recorded)", region.region_id, len(uncapped))
    capped_region = region.with_caps(np.asarray(positions).reshape(-1, 3), caps_meta) if positions else ExtractedRegion.from_dict(region.to_dict())
    capped_region.metadata.update(
        {
            "h_capping": record,
            "h_capping_skipped": False,
            "h_cap_enabled": True,
            "h_cap_parent_local_indices": [entry["parent_local_index"] for entry in caps_meta],
        }
    )
    return capped_region


@dataclass(frozen=True, slots=True)
class CappingAssessment:
    """What a labeling backend needs to know about dangling bonds (data, not a decision)."""

    capped: bool
    skipped_reason: str | None
    n_dangling: int
    partial: bool

    @property
    def has_dangling_bonds(self) -> bool:
        return self.n_dangling > 0


def assess_capping(region: ExtractedRegion) -> CappingAssessment:
    record = region.metadata.get("h_capping")
    if record is None:
        raise ExtractionError(f"region {region.region_id} has not been through the capping step")
    return CappingAssessment(
        capped=bool(record["applied"]),
        skipped_reason=record.get("skipped_reason"),
        n_dangling=len(record.get("uncapped_cut_bonds", [])),
        partial=bool(record.get("partial", False)),
    )


def check_labelable(region: ExtractedRegion, config: CappingConfig | None = None, *, periodic_labeling: bool) -> CappingAssessment:
    """Fail-closed gate for uncapped fragments (PHAL known issue 1, ``cp2k.uncapped_fragment_policy``).

    An isolated (non-periodic) calculation of a fragment whose cut bonds were left dangling because
    capping was skipped (metal/oxide/disabled) raises under ``uncapped_policy=error`` (default).
    Partially capped regions follow ``partial_policy`` (default ``warn``).
    """

    config = config or CappingConfig()
    assessment = assess_capping(region)
    if assessment.has_dangling_bonds and not periodic_labeling:
        if not assessment.capped:
            message = (
                f"region {region.region_id}: {assessment.n_dangling} cut bonds left uncapped "
                f"({assessment.skipped_reason}) in an isolated calculation"
            )
            if config.uncapped_policy == "error":
                raise ExtractionError(message + "; set capping.uncapped_policy to override (research decision pending)")
            if config.uncapped_policy == "warn":
                LOGGER.warning("%s", message)
        elif assessment.partial:
            message = f"region {region.region_id}: partially capped, {assessment.n_dangling} dangling bonds"
            if config.partial_policy == "error":
                raise ExtractionError(message)
            if config.partial_policy == "warn":
                LOGGER.warning("%s", message)
    return assessment


__all__ = [
    "METAL_LIKE",
    "OXIDE_FORMERS",
    "CappingAssessment",
    "CutBond",
    "add_h_caps",
    "assess_capping",
    "check_labelable",
    "covalent_bonds_with_shifts",
    "find_cut_bonds",
]
