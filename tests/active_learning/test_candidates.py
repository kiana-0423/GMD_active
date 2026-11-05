from __future__ import annotations

import numpy as np
import pytest
from ase.build import molecule

from gmd_active_learning.active_learning.candidates import CandidateConfig, CandidateSelector, fingerprint_region
from gmd_active_learning.contracts import ExtractedRegion, RegionRole


def _region(region_id: str, positions: np.ndarray, numbers: np.ndarray, *, event: str = "evt-a", cell=None, pbc=(False,) * 3) -> ExtractedRegion:
    n = len(numbers)
    return ExtractedRegion(
        region_id=region_id, mode="cluster", run_id="run", parent_step=0, event_id=event,
        numbers=numbers, positions=positions, cell=cell, pbc=pbc,
        parent_atom_ids=np.arange(n), image_shifts=np.zeros((n, 3), int), roles=[RegionRole.CORE] * n,
        force_weights=np.ones(n), frozen=np.zeros(n, bool), hotspot_atom_ids=[0],
    )


def _mol(name: str, region_id: str, shift: float = 0.0, **kw) -> ExtractedRegion:
    mol = molecule(name)
    return _region(region_id, mol.positions * (1.0 + shift), mol.numbers, **kw)


def test_fingerprint_is_finite_when_no_pair_is_inside_the_cutoff() -> None:
    """PHAL produced NaN here (density histogram of zero counts), F-30."""

    far = _region("far", np.array([[0.0, 0, 0], [20.0, 0, 0]]), np.array([18, 18]))
    single = _region("one", np.zeros((1, 3)), np.array([18]))
    for mode in ("pair_distance_histogram", "type_weighted_pair_distance_histogram"):
        config = CandidateConfig(fingerprint=mode)
        assert np.all(np.isfinite(fingerprint_region(far, config)))
        assert np.all(np.isfinite(fingerprint_region(single, config)))


def test_isolated_atoms_candidate_does_not_block_later_candidates() -> None:
    selector = CandidateSelector(CandidateConfig(max_per_cycle=5, max_per_event=5))
    far = selector.make_candidate(_region("far", np.array([[0.0, 0, 0], [20.0, 0, 0]]), np.array([18, 18])), score=99.0)
    water = selector.make_candidate(_mol("H2O", "water"), score=1.0)
    decision = selector.select([far, water])
    assert [item.candidate_id for item in decision.selected] == ["far", "water"]


def test_periodic_regions_use_minimum_image_distances() -> None:
    cell = np.eye(3) * 6.0
    across = _region("pbc", np.array([[0.2, 3, 3], [5.8, 3, 3]]), np.array([6, 6]), cell=cell, pbc=(True,) * 3)
    config = CandidateConfig(fingerprint="pair_distance_histogram", fingerprint_bins=16, fingerprint_cutoff=8.0,
                             histogram_normalisation="probability")
    histogram = fingerprint_region(across, config)
    assert histogram[0] == 1.0  # 0.4 A apart through the boundary, not 5.6 A
    density = fingerprint_region(across, config.model_copy(update={"histogram_normalisation": "density"}))
    assert density[0] == 2.0  # PHAL normalisation: probability / bin width (0.5 A)
    plain = fingerprint_region(across, config.model_copy(update={"pair_distances": "plain"}))
    assert plain[11] == 1.0  # PHAL's plain pdist: 5.6 A


def test_selection_ranks_by_score_deduplicates_and_records_reasons() -> None:
    selector = CandidateSelector(CandidateConfig(max_per_cycle=10, max_per_event=10, diversity_threshold=0.05))
    a = selector.make_candidate(_mol("CH4", "a"), score=5.0)
    a_copy = selector.make_candidate(_mol("CH4", "a_copy", shift=1e-4), score=4.0)
    b = selector.make_candidate(_mol("C6H6", "b"), score=3.0)
    low = selector.make_candidate(_mol("NH3", "low"), score=-1.0)
    decision = selector.select([b, low, a_copy, a])
    assert [item.candidate_id for item in decision.selected] == ["a", "b"]
    assert decision.rejected == {"a_copy": "duplicate", "low": "below_min_score"}


def test_labeled_history_suppresses_repeats() -> None:
    selector = CandidateSelector(CandidateConfig())
    a = selector.make_candidate(_mol("CH4", "a"), score=5.0)
    decision = selector.select([a], labeled_fingerprints=[a.fingerprint.copy()])
    assert decision.selected == [] and decision.rejected == {"a": "duplicate"}
    no_history = CandidateSelector(CandidateConfig(compare_with_labeled=False)).select([a], labeled_fingerprints=[a.fingerprint])
    assert len(no_history.selected) == 1


@pytest.mark.parametrize(
    ("config", "expected", "reason"),
    [
        (dict(max_per_cycle=2, max_per_event=10), ["m0", "m1"], "cycle_budget"),
        (dict(max_per_cycle=10, max_per_event=1), ["m0"], "event_budget"),
        (dict(max_per_cycle=10, max_per_event=10, max_total=3), ["m0"], "total_budget"),
    ],
)
def test_budgets(config: dict, expected: list[str], reason: str) -> None:
    selector = CandidateSelector(CandidateConfig(deduplicate=False, **config))
    names = ["CH4", "H2O", "NH3", "C2H6"]
    candidates = [selector.make_candidate(_mol(name, f"m{i}"), score=10.0 - i) for i, name in enumerate(names)]
    decision = selector.select(candidates, already_labeled_total=2)
    assert [item.candidate_id for item in decision.selected] == expected
    assert reason in decision.rejected.values()


def test_selection_can_be_disabled() -> None:
    selector = CandidateSelector(CandidateConfig(enabled=False))
    decision = selector.select([selector.make_candidate(_mol("CH4", "a"), score=1.0)])
    assert decision.selected == [] and decision.rejected == {"a": "selection_disabled"}
