"""Deterministic group-level splits: every parent of one base conformation lands in the same split.

Allocation: ``round(fraction x n_groups)`` by largest remainder (ties broken by
``splits.assignment_order``), with at least one group for every split whose fraction is positive
when there are enough groups. Order of groups: within each stratum (chain length) by a seeded hash,
then interleaved across strata, and dealt to the splits in ``assignment_order``; so with two chain
lengths the first split in the order (calibration by default) gets one group of each length.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence

from gmd_active_learning.contracts.ids import content_hash
from tools.experiments.alkane_hotspots.config import SPLITS, SplitsConfig

LOGGER = logging.getLogger(__name__)


def split_counts(n_groups: int, config: SplitsConfig) -> dict[str, int]:
    order = list(config.assignment_order)
    raw = {name: config.fractions[name] * n_groups for name in order}
    counts = {name: int(raw[name]) for name in order}
    remaining = n_groups - sum(counts.values())
    by_remainder = sorted(order, key=lambda name: (-(raw[name] - counts[name]), order.index(name)))
    for name in by_remainder[:remaining]:
        counts[name] += 1
    positive = [name for name in order if config.fractions[name] > 0]
    if n_groups >= len(positive):
        for name in positive:
            if counts[name] == 0:
                donor = max(order, key=lambda key: (counts[key], -order.index(key)))
                counts[donor] -= 1
                counts[name] += 1
    return counts


def assign_splits(groups: Sequence[tuple[str, str]], config: SplitsConfig, seed: int) -> dict[str, str]:
    """``groups``: (group_id, stratum) pairs (duplicates ignored). Returns group_id -> split."""

    strata: dict[str, list[str]] = defaultdict(list)
    for group, stratum in dict.fromkeys(groups):
        strata[stratum if config.stratify_by != "none" else "all"].append(group)
    ordered_strata = [sorted(set(members), key=lambda g: content_hash([int(seed), g])) for _, members in sorted(strata.items())]
    interleaved: list[str] = []
    for position in range(max((len(s) for s in ordered_strata), default=0)):
        interleaved.extend(stratum[position] for stratum in ordered_strata if position < len(stratum))
    counts = split_counts(len(interleaved), config)
    sequence = [name for name in config.assignment_order for _ in range(counts[name])]
    assignment = dict(zip(interleaved, sequence))
    LOGGER.info("split %d groups: %s", len(interleaved), counts)
    return assignment


def check_no_leakage(records: Sequence[dict]) -> None:
    """Raise if a split group appears in two splits or a split name is unknown."""

    seen: dict[str, str] = {}
    for record in records:
        split, group = record["split"], record["split_group"]
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r} for {record.get('parent_id')}")
        if seen.setdefault(group, split) != split:
            raise ValueError(f"split group {group} straddles {seen[group]} and {split}")


__all__ = ["assign_splits", "check_no_leakage", "split_counts"]
