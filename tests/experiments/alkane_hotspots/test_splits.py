"""Group splits: deterministic, no group straddles two splits, stratified by chain length."""

from __future__ import annotations

import pytest

from ._helpers import tiny_config
from tools.experiments.alkane_hotspots.config import SplitsConfig
from tools.experiments.alkane_hotspots.families import generate
from tools.experiments.alkane_hotspots.splits import assign_splits, check_no_leakage, split_counts


def _groups(n_per_length: int) -> list[tuple[str, str]]:
    return [(f"C{n}-b{b:02d}", f"C{n}") for n in (16, 32) for b in range(n_per_length)]


def test_assignment_is_deterministic_and_covers_every_group() -> None:
    config = SplitsConfig()
    first = assign_splits(_groups(5), config, seed=7)
    assert first == assign_splits(list(reversed(_groups(5))), config, seed=7)
    assert set(first) == {g for g, _ in _groups(5)}
    assert first != assign_splits(_groups(5), config, seed=8) or len(set(first.values())) == 1


def test_counts_follow_fractions_and_every_positive_split_gets_a_group() -> None:
    config = SplitsConfig(fractions={"initial_train": 0.2, "calibration": 0.3, "validation": 0.2, "final_eval": 0.3})
    assert split_counts(6, config) == {"calibration": 2, "final_eval": 2, "initial_train": 1, "validation": 1}
    assert sum(split_counts(13, config).values()) == 13
    sparse = SplitsConfig(fractions={"initial_train": 0.94, "calibration": 0.02, "validation": 0.02, "final_eval": 0.02})
    assert all(v >= 1 for v in split_counts(4, sparse).values())


def test_first_split_in_the_order_gets_both_chain_lengths() -> None:
    config = SplitsConfig(fractions={"initial_train": 0.2, "calibration": 0.3, "validation": 0.2, "final_eval": 0.3})
    assignment = assign_splits(_groups(3), config, seed=7)
    calibration = [g for g, s in assignment.items() if s == "calibration"]
    assert sorted(g.split("-")[0] for g in calibration) == ["C16", "C32"]


def test_generated_parents_never_straddle(tmp_path) -> None:
    result = generate(tiny_config(tmp_path).config)
    check_no_leakage(result.parents)
    groups: dict[str, set[str]] = {}
    for parent in result.parents:
        groups.setdefault(parent["split_group"], set()).add(parent["split"])
    assert all(len(s) == 1 for s in groups.values()) and len(groups) == 4


def test_leakage_is_detected() -> None:
    with pytest.raises(ValueError, match="straddles"):
        check_no_leakage([{"parent_id": "a", "split": "calibration", "split_group": "g"}, {"parent_id": "b", "split": "final_eval", "split_group": "g"}])
