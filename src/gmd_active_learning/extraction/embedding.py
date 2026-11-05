# Ported from PHAL (hotspot_al.extraction.embedding) at 96263b6; MIT License, see NOTICE.
"""Point-charge embedding *data* for extracted regions. Executing QM/MM is unsupported.

The charges represent the environment atoms that were left out of the region, placed in the
region's coordinate frame. PHAL recovered the parent->region transform from the positions; here it
follows from the recorded image shifts and ``origin_translation``.

Known issue (PHAL docs/known-issues.md #2, kept fail-closed): no labeling backend can run an
electrostatically embedded calculation. :data:`EMBEDDING_CAPABILITIES` states that as data and
:func:`require_embedding_backend` raises for any enabled embedding, so the labeling area can check
it before writing an input.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from gmd_active_learning.contracts import ExtractedRegion, Frame, RegionRole
from gmd_active_learning.core.exceptions import CapabilityUnavailableError
from gmd_active_learning.extraction._placement import frame_cell
from gmd_active_learning.extraction.config import EmbeddingConfig
from gmd_active_learning.utils.geometry import distances_to_group
from gmd_active_learning.utils.periodic import mic_displacements_from_reference

LOGGER = logging.getLogger(__name__)

SUPPORTED_METHODS = frozenset({"point_charge", "point_charges"})

#: What exists today. ``qmmm_execution`` is False for every backend: CP2K QM/MM is not wired.
EMBEDDING_CAPABILITIES: dict[str, Any] = {
    "point_charge_data": True,
    "periodic_images": False,  # one minimum image per environment atom, no Ewald sum
    "charge_neutralisation": False,  # no boundary charge redistribution
    "qmmm_execution": {"cp2k": False},
}


def _result(region: ExtractedRegion, payload: dict[str, Any]) -> ExtractedRegion:
    data = region.to_dict()
    data["metadata"] = {**region.metadata, "embedding": {**payload, "capabilities": EMBEDDING_CAPABILITIES}}
    return ExtractedRegion.from_dict(data)


def build_embedding(region: ExtractedRegion, frame: Frame, config: EmbeddingConfig | None = None) -> ExtractedRegion:
    """Return a copy of ``region`` with ``metadata['embedding']`` (status disabled/unsupported/unavailable/ok)."""

    config = config or EmbeddingConfig()
    if not config.enabled:
        return _result(region, {"enabled": False, "status": "disabled"})
    if config.method not in SUPPORTED_METHODS:
        return _result(region, {"enabled": True, "status": "unsupported", "method": config.method})
    if any(region.pbc):
        return _result(region, {"enabled": True, "status": "unsupported", "method": "point_charge", "reason": "periodic region"})

    real = [index for index, role in enumerate(region.roles) if role is not RegionRole.H_CAP]
    if not real or frame.run_id != region.run_id or int(frame.step) != int(region.parent_step):
        return _result(region, {"enabled": True, "status": "unavailable", "method": "point_charge", "reason": "frame is not the region's parent"})
    parent_indices = frame.index_of(region.parent_atom_ids[real])
    translation = np.asarray(region.metadata.get("origin_translation", [0.0, 0.0, 0.0]), dtype=float)
    expected = frame.positions[parent_indices] + region.image_shifts[real] @ frame_cell(frame) + translation
    if float(np.abs(region.positions[real] - expected).max()) > config.rigid_tolerance:
        return _result(
            region,
            {"enabled": True, "status": "unavailable", "method": "point_charge", "reason": "region coordinates are not a rigid transform of the source frame"},
        )

    in_region = set(parent_indices.tolist())
    environment = [index for index in range(frame.n_atoms) if index not in in_region]
    pbc = frame.pbc if frame.cell is not None else False
    distances = distances_to_group(frame.positions, sorted(in_region), cell=frame.cell, pbc=pbc)
    selected = [index for index in environment if distances[index] <= config.cutoff]
    anchor_local, anchor_parent = real[0], int(parent_indices[0])
    if selected:
        local_positions = region.positions[anchor_local] + mic_displacements_from_reference(
            frame.positions[anchor_parent], frame.positions[selected], cell=frame.cell, pbc=pbc
        )
    else:
        local_positions = np.zeros((0, 3))
    symbols = frame.symbols
    charges: list[dict[str, Any]] = []
    skipped: list[int] = []
    for position, index in zip(local_positions, selected):
        symbol = symbols[index]
        if symbol in config.charges:
            charge = float(config.charges[symbol])
        elif config.default_charge is not None:
            charge = float(config.default_charge)
        else:
            skipped.append(int(frame.atom_ids[index]))
            continue
        charges.append(
            {"source_atom_id": int(frame.atom_ids[index]), "element": symbol, "charge": charge, "position": [float(v) for v in position]}
        )
    if skipped:
        LOGGER.warning("embedding: %d environment atoms have no charge and were skipped (recorded)", len(skipped))
    return _result(
        region,
        {
            "enabled": True,
            "status": "ok",
            "method": "point_charge",
            "n_point_charges": len(charges),
            "point_charges": charges,
            "skipped_atom_ids": skipped,
            "cutoff": float(config.cutoff),
        },
    )


def require_embedding_backend(region: ExtractedRegion, backend: str = "cp2k") -> None:
    """Fail-closed gate: an enabled embedding cannot be executed by any labeling backend."""

    embedding = region.metadata.get("embedding") or {}
    if embedding.get("enabled") and not EMBEDDING_CAPABILITIES["qmmm_execution"].get(backend, False):
        raise CapabilityUnavailableError(
            f"region {region.region_id}: embedding is enabled but {backend} QM/MM embedding is unsupported "
            "(known issue); disable embedding.enabled"
        )


__all__ = ["EMBEDDING_CAPABILITIES", "SUPPORTED_METHODS", "build_embedding", "require_embedding_backend"]
