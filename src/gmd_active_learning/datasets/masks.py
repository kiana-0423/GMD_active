# Ported from PHAL (hotspot_al.training.mask_generator, hotspot_al.training.region_codes) at 96263b6; MIT License, see NOTICE.
"""Per-atom force weights and the region-code on-disk contract.

PHAL kept role information in five parallel index lists plus (for block regions) a separate
``metadata["atom_role"]`` list, and derived masks and codes from whichever it found first. Adding H
caps extended some of them but not ``atom_role``, so block regions with caps could not be written
(finding F-04). Here the role of every atom lives in exactly one place, ``ExtractedRegion.roles``,
and both the force weights and the integer codes are pure functions of it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

import numpy as np

from gmd_active_learning.contracts import REGION_CODES, ExtractedRegion, RegionRole
from gmd_active_learning.core.exceptions import DataContractError
from gmd_active_learning.datasets.config import RoleWeights

#: PHAL label strings (training/region_codes.py) -> integer codes. Byte-identical to PHAL.
REGION_CODE_MAP: dict[str, int] = {
    "core": 0,
    "label_core": 0,
    "inner_buffer": 1,
    "outer_buffer": 2,
    "boundary": 3,
    "frozen_boundary": 3,
    "h_cap": 4,
}

#: Integer code -> canonical label, serialized into extxyz ``region_label_map`` (PHAL format).
REGION_LABEL_MAP: dict[int, str] = {
    0: "core",
    1: "inner_buffer",
    2: "outer_buffer",
    3: "boundary",
    4: "h_cap",
}

#: PHAL label strings -> contract roles (block ``label_core``/``frozen_boundary`` included).
PHAL_LABEL_TO_ROLE: dict[str, RegionRole] = {
    "core": RegionRole.CORE,
    "label_core": RegionRole.CORE,
    "inner_buffer": RegionRole.INNER_BUFFER,
    "outer_buffer": RegionRole.OUTER_BUFFER,
    "boundary": RegionRole.BOUNDARY,
    "frozen_boundary": RegionRole.BOUNDARY,
    "h_cap": RegionRole.H_CAP,
    "unassigned": RegionRole.UNASSIGNED,
}

_CODE_TO_ROLE: dict[int, RegionRole] = {code: role for role, code in REGION_CODES.items()}

# The contract's codes must stay identical to PHAL's on-disk codes.
assert all(REGION_CODES[PHAL_LABEL_TO_ROLE[label]] == code for label, code in REGION_CODE_MAP.items())


def region_codes_for_labels(labels: Iterable[str]) -> np.ndarray:
    """PHAL-compatible: unknown labels (e.g. ``unassigned``) map to -1."""

    return np.asarray([REGION_CODE_MAP.get(label, -1) for label in labels], dtype=int)


def region_label_map_json() -> str:
    """The extxyz ``region_label_map`` info value, byte-identical to PHAL."""

    return json.dumps(REGION_LABEL_MAP)


def roles_from_labels(labels: Sequence[str]) -> list[RegionRole]:
    try:
        return [PHAL_LABEL_TO_ROLE[str(label)] for label in labels]
    except KeyError as exc:
        raise DataContractError(f"unknown region label {exc.args[0]!r}") from exc


def roles_from_codes(codes: Sequence[int]) -> list[RegionRole]:
    try:
        return [_CODE_TO_ROLE[int(code)] for code in codes]
    except KeyError as exc:
        raise DataContractError(f"unknown region code {exc.args[0]!r}") from exc


def force_weights_for_roles(roles: Sequence[RegionRole], weights: RoleWeights | None = None) -> np.ndarray:
    """Per-atom force-loss weights as a step function of role."""

    weights = weights or RoleWeights()
    return np.asarray([weights.weight(role) for role in roles], dtype=float)


def reweight(region: ExtractedRegion, weights: RoleWeights) -> ExtractedRegion:
    """Copy of ``region`` whose force weights follow ``weights`` (roles unchanged)."""

    data = region.to_dict()
    data["force_weights"] = force_weights_for_roles(region.roles, weights).tolist()
    return ExtractedRegion.from_dict(data)


__all__ = [
    "PHAL_LABEL_TO_ROLE",
    "REGION_CODE_MAP",
    "REGION_LABEL_MAP",
    "force_weights_for_roles",
    "region_codes_for_labels",
    "region_label_map_json",
    "reweight",
    "roles_from_codes",
    "roles_from_labels",
]
