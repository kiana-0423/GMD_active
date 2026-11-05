"""Charge/multiplicity resolution (UTIL-007): port of PHAL tests/unit/test_electronic_state.py (core rules)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from gmd_active_learning.core.exceptions import ConfigurationError
from gmd_active_learning.utils.electronic_state import (
    minimum_multiplicity,
    multiplicity_is_consistent,
    resolve_electronic_state,
    total_electron_count,
)


def _atoms(formula: str) -> Atoms:
    template = Atoms(formula)
    positions = np.random.default_rng(0).uniform(0.0, 3.0, (len(template), 3))
    return Atoms(formula, positions=positions, cell=[12.0, 12.0, 12.0])


def test_total_electron_count_accounts_for_charge() -> None:
    assert total_electron_count(_atoms("H2O"), 0) == 10
    assert total_electron_count(_atoms("H2O"), -1) == 11
    assert total_electron_count(_atoms("H2O"), 1) == 9


@pytest.mark.parametrize(("electrons", "expected"), [(10, 1), (9, 2), (0, 1), (1, 2)])
def test_minimum_multiplicity_follows_electron_parity(electrons: int, expected: int) -> None:
    assert minimum_multiplicity(electrons) == expected


def test_multiplicity_consistency_rules() -> None:
    assert multiplicity_is_consistent(10, 1) and multiplicity_is_consistent(10, 3)
    assert not multiplicity_is_consistent(10, 2)
    assert multiplicity_is_consistent(9, 2)
    assert not multiplicity_is_consistent(9, 1) and not multiplicity_is_consistent(9, 3)
    assert not multiplicity_is_consistent(1, 4) and not multiplicity_is_consistent(10, 0)


def test_radical_fragment_becomes_a_doublet() -> None:
    state = resolve_electronic_state(_atoms("OH"), {"charge": 0, "multiplicity": 1, "spin_policy": "auto"})
    assert (state.n_electrons, state.multiplicity) == (9, 2)
    assert state.corrected and state.is_open_shell


def test_strict_policy_raises() -> None:
    with pytest.raises(ConfigurationError, match="impossible"):
        resolve_electronic_state(_atoms("OH"), {"charge": 0, "multiplicity": 1, "spin_policy": "strict"})


def test_multiplicity_is_derived_when_unspecified() -> None:
    closed = resolve_electronic_state(_atoms("H2O"), {"charge": 0, "multiplicity": None})
    radical = resolve_electronic_state(_atoms("CH3"), {"charge": 0, "multiplicity": None})
    assert (closed.multiplicity, closed.source, closed.is_open_shell) == (1, "derived", False)
    assert (radical.multiplicity, radical.source, radical.is_open_shell) == (2, "derived", True)


def test_region_metadata_overrides_configuration() -> None:
    state = resolve_electronic_state(_atoms("O2"), {"charge": 0, "multiplicity": 1}, region_metadata={"multiplicity": 3})
    assert (state.multiplicity, state.source, state.corrected) == (3, "region", False)
    charged = resolve_electronic_state(_atoms("OH"), {"charge": 0, "multiplicity": None}, region_metadata={"charge": -1})
    assert (charged.charge, charged.n_electrons, charged.multiplicity) == (-1, 10, 1)


def test_unknown_spin_policy_and_negative_electron_count_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="Unsupported spin_policy"):
        resolve_electronic_state(_atoms("OH"), {"multiplicity": 1, "spin_policy": "whatever"})
    with pytest.raises(ConfigurationError, match="negative electron count"):
        resolve_electronic_state(_atoms("H"), {"charge": 3})
