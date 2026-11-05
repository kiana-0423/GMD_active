# Ported from PHAL (tests.scf.test_scf_valence) at 96263b6; MIT License, see NOTICE.
"""Tests for the valence electron count a density must integrate to.

The bug these exist for: the exported model normalised to ``sum(Z)``. Under GTH
pseudopotentials water carries 8 electrons, not 10, so every prediction was 25%
heavy and CP2K rejected it -- silently, by running the atomic guess instead.

The counts are *fitted* from recorded densities rather than transcribed, so the
tests check both the arithmetic and the cases where fitting must refuse.
"""

from __future__ import annotations

import numpy as np
import pytest

from gmd_active_learning.adapters.dft.scf.density.gnn_model import DensityModelError, ValenceTable

#: GTH-PBE valences for the elements used below, from CP2K's POTENTIAL file.
GTH_PBE = {1: 1, 6: 4, 7: 5, 8: 6, 14: 4, 22: 12, 29: 11}

WATER = (8, 1, 1)
AMMONIA = (7, 1, 1, 1)
METHANE = (6, 1, 1, 1, 1)
CO2 = (6, 8, 8)


def table(**overrides) -> ValenceTable:
    counts = {key: value for key, value in GTH_PBE.items()}
    counts.update(overrides)
    return ValenceTable(valence_per_element=counts, potential_family="GTH-PBE")


# -- the number itself --------------------------------------------------


def test_water_under_gth_has_eight_electrons_not_ten():
    """The case the whole fix is about."""

    counts = table()

    assert sum(WATER) == 10, "sum(Z) for H2O"
    assert counts.electrons(WATER) == 8.0
    assert counts.electrons(WATER) != float(sum(WATER))


def test_charge_is_subtracted():
    counts = table()

    assert counts.electrons(WATER, charge=1.0) == 7.0
    assert counts.electrons(WATER, charge=-1.0) == 9.0


def test_a_transition_metal_oxide_is_further_off():
    """Cu2O: 66 nuclear against 28 valence -- a factor of 2.4."""

    counts = table()

    assert counts.electrons((29, 29, 8)) == 28.0
    assert sum((29, 29, 8)) == 66


def test_an_unknown_element_is_refused():
    with pytest.raises(DensityModelError, match="no valence electron count known"):
        table().electrons((92, 8, 8))


def test_a_valence_above_the_nuclear_charge_is_refused():
    """More electrons than protons is not a pseudopotential, it is a typo."""

    with pytest.raises(DensityModelError, match="never more than Z"):
        ValenceTable(valence_per_element={8: 9})


def test_a_non_positive_count_is_refused():
    with pytest.raises(DensityModelError, match="non-positive"):
        ValenceTable(valence_per_element={8: 0})


# -- fitting ------------------------------------------------------------


def test_fitting_recovers_the_gth_valences():
    """Solved from recorded electron counts, not transcribed from a table."""

    compositions = [
        (WATER, 8.0),
        (AMMONIA, 8.0),
        (METHANE, 8.0),
        (CO2, 16.0),
    ]

    counts = ValenceTable.fit(compositions, potential_family="GTH-PBE")

    assert counts.valence_per_element == {1: 1, 6: 4, 7: 5, 8: 6}
    assert counts.potential_family == "GTH-PBE"


def test_fitting_accounts_for_charge():
    compositions = [(WATER, 7.0), (AMMONIA, 8.0), (METHANE, 8.0), (CO2, 16.0)]
    charges = [1.0, 0.0, 0.0, 0.0]

    counts = ValenceTable.fit(compositions, charges=charges)

    assert counts.valence_per_element[8] == 6


def test_an_underdetermined_dataset_is_refused():
    """H2O alone admits (H=1, O=6) and (H=3, O=2); both give 8.

    Guessing between them puts the wrong number of electrons on every atom of
    every other system, so the fit refuses rather than picking one.
    """

    with pytest.raises(DensityModelError, match="does not determine"):
        ValenceTable.fit([(WATER, 8.0), (WATER, 8.0)])


def test_a_fractional_fit_is_refused():
    """Electron counts are integers; a fractional solution fits something else."""

    with pytest.raises(DensityModelError, match="not integral"):
        ValenceTable.fit([(WATER, 8.5), (AMMONIA, 8.0), (METHANE, 8.0), (CO2, 16.0)])


def test_an_inconsistent_dataset_is_refused():
    """One system computed with a different pseudopotential poisons the fit."""

    compositions = [
        (WATER, 8.0),
        (AMMONIA, 8.0),
        (METHANE, 8.0),
        (CO2, 16.0),
        ((8, 8), 20.0),  # O2 with semicore oxygen: inconsistent with O=6
    ]

    with pytest.raises(DensityModelError, match="not integral|no consistent"):
        ValenceTable.fit(compositions)


def test_fitting_from_zero_systems_is_refused():
    with pytest.raises(DensityModelError, match="zero systems"):
        ValenceTable.fit([])


# -- telling a pseudopotential table from a nuclear-charge one ----------


def test_a_nuclear_charge_table_is_identifiable():
    """Z for every element: right for all-electron, the bug for GTH."""

    assert ValenceTable(valence_per_element={1: 1, 6: 6, 8: 8}).is_nuclear_charge() is True


def test_a_pseudopotential_table_is_not():
    assert table().is_nuclear_charge() is False


def test_hydrogen_alone_cannot_distinguish_them():
    """H has one valence electron and Z = 1, so a probe of H proves nothing.

    This is why the export probe picks the element with the largest Z-valence
    gap instead of whatever comes first.
    """

    assert ValenceTable(valence_per_element={1: 1}).is_nuclear_charge() is True


# -- round trip ---------------------------------------------------------


def test_the_table_round_trips():
    counts = table()

    restored = ValenceTable.from_dict(counts.to_dict())

    assert restored.valence_per_element == counts.valence_per_element
    assert restored.potential_family == "GTH-PBE"


def test_the_serialised_form_uses_string_keys():
    """JSON has no integer keys; a silent int/str mismatch loses the table."""

    payload = table().to_dict()

    assert all(isinstance(key, str) for key in payload["valence_per_element"])
    assert payload["valence_per_element"]["8"] == 6


# -- against the real dataset -------------------------------------------


def test_the_fitted_table_matches_cp2ks_own_potential_file():
    """The fit is checked against the q-values CP2K ships, not against itself."""

    from pathlib import Path

    import os

    dataset = Path("runtime/cp2k/pulay_dataset")
    # The fork sits beside the repo locally and at /workspace in the container;
    # CP2K_DATA_DIR names it in both. Checking all three keeps this from
    # silently skipping in the environment where it matters most.
    candidates = [
        Path(os.environ.get("CP2K_DATA_DIR", "")),
        Path("../cp2k-ml-scf/cp2k/data"),
        Path("/workspace/cp2k-ml-scf/cp2k/data"),
    ]
    data_dir = next((item for item in candidates if (item / "POTENTIAL").is_file()), None)
    if not dataset.is_dir() or data_dir is None:
        pytest.skip("needs the generated dataset and the CP2K data directory")

    from tools.scf.training.dataset_adapter import samples_from_dataset
    from tools.scf.structures import resolve_kinds
    from tools.scf.training.dataset_loader import build_examples
    from tools.scf.training.density_gnn import fit_valence

    samples, _ = samples_from_dataset(dataset)
    examples = build_examples(samples, target="density_matrix")
    fitted = fit_valence(examples, potential_family="GTH-PBE")

    from ase.data import chemical_symbols

    elements = [chemical_symbols[number] for number in fitted.valence_per_element]
    kinds = resolve_kinds(elements, data_dir=data_dir)

    for number, count in fitted.valence_per_element.items():
        symbol = chemical_symbols[number]
        expected = int(kinds[symbol].potential.rsplit("-q", 1)[1])
        assert count == expected, f"{symbol}: fitted {count}, CP2K ships q{expected}"
