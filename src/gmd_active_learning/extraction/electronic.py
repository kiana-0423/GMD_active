"""Charge and spin multiplicity of an extracted (possibly capped) region.

Thin wrapper over :func:`gmd_active_learning.utils.electronic_state.resolve_electronic_state`, run
on the final atoms (caps included), as PHAL's ``cp2k_input.region_electronic_state`` did. Precedence:
``region.metadata['charge'|'multiplicity']`` > settings (``charge``, ``multiplicity``,
``spin_policy``, typically the labeling engine section) > parity-derived multiplicity.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from ase import Atoms

from gmd_active_learning.contracts import ExtractedRegion
from gmd_active_learning.utils.electronic_state import ElectronicState, resolve_electronic_state

LOGGER = logging.getLogger(__name__)


def resolve_region_electronic_state(region: ExtractedRegion, settings: Mapping[str, Any] | None = None) -> tuple[ExtractedRegion, ElectronicState]:
    """Return a copy of ``region`` with ``charge``/``multiplicity`` set, and the resolved state."""

    atoms = Atoms(numbers=region.numbers, positions=region.positions)
    state = resolve_electronic_state(atoms, dict(settings or {}), region_metadata=region.metadata, logger=LOGGER)
    data = region.to_dict()
    data["charge"] = state.charge
    data["multiplicity"] = state.multiplicity
    data["metadata"] = {**region.metadata, "electronic_state": state.as_metadata()}
    return ExtractedRegion.from_dict(data), state


__all__ = ["resolve_region_electronic_state"]
