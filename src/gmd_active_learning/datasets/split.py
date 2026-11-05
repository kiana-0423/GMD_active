"""Deterministic, group-aware train/validation split.

Every sample of one split group (by default one MD event) lands on the same side, so correlated
frames of an event never straddle train and validation. A group's side is a pure function of
``(seed, group)`` (hash below ``val_fraction`` -> validation), and groups already assigned in the
parent dataset version keep their side, so validation data never migrates into training in later
versions. When no group would be in validation although ``val_fraction > 0`` and there are at least
two groups, the *new* group with the smallest hash is moved to validation (deterministic).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from gmd_active_learning.datasets.config import SplitConfig

TRAIN = "train"
VAL = "val"


def group_hash(seed: int, group: str) -> float:
    digest = hashlib.sha256(f"{int(seed)}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign_groups(
    groups: Iterable[str],
    config: SplitConfig,
    *,
    inherited: dict[str, str] | None = None,
) -> dict[str, str]:
    """``group -> 'train' | 'val'``."""

    unique = sorted(set(groups))
    inherited = dict(inherited or {})
    if not config.enabled or config.val_fraction <= 0.0:
        return {group: inherited.get(group, TRAIN) for group in unique}
    assignment: dict[str, str] = {}
    new_groups = []
    for group in unique:
        if group in inherited:
            assignment[group] = inherited[group]
        else:
            new_groups.append(group)
            assignment[group] = VAL if group_hash(config.seed, group) < config.val_fraction else TRAIN
    if len(unique) >= 2 and VAL not in assignment.values() and new_groups:
        candidate = min(new_groups, key=lambda group: group_hash(config.seed, group))
        assignment[candidate] = VAL
    if TRAIN not in assignment.values() and len(unique) >= 2 and new_groups:
        candidate = max(new_groups, key=lambda group: group_hash(config.seed, group))
        assignment[candidate] = TRAIN
    return assignment


__all__ = ["TRAIN", "VAL", "assign_groups", "group_hash"]
