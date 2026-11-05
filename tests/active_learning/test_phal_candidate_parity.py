"""PHAL parity of fingerprints, deduplication and pool selection (ledger AL-002, AL-003, AL-004).

Ports of PHAL tests/unit/test_deduplication.py and test_candidate_pool.py against GMD's
``CandidateSelector``, plus fingerprints of every golden PHAL region compared with PHAL's formula
(re-implemented below as an oracle: ``scipy.spatial.distance.pdist`` + ``np.histogram(density=True)``),
and pool selection over the golden regions compared with PHAL's ``CandidatePool`` algorithm.

Deliberate deviation asserted: where PHAL's density histogram is NaN (no pair inside the cutoff,
F-30) GMD returns zeros, so such a candidate neither disappears nor blocks later ones.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from ase.data import atomic_numbers
from scipy.spatial.distance import pdist

from gmd_active_learning.active_learning.candidates import (
    Candidate,
    CandidateConfig,
    CandidateSelector,
    deduplicate_candidates,
    fingerprint_distance,
    fingerprint_region,
)
from gmd_active_learning.contracts import ExtractedRegion, RegionRole

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "phal_reference"
SCENARIOS = ("ethane_box", "polyethylene_chains", "pt111_co_slab")
MODES = ("cluster", "slab", "graph", "block")

#: PHAL's candidate_pool semantics expressed in GMD config (no per-event budget, no score filter,
#: no labeled history, plain pdist, density histogram, PHAL default fingerprint).
PHAL = dict(fingerprint="pair_distance_histogram", pair_distances="plain", histogram_normalisation="density",
            max_per_event=None, min_score=None, compare_with_labeled=False, tie_break="insertion")


# ----------------------------------------------------------------------------- PHAL oracle
def phal_pair_distance_histogram(positions: np.ndarray, bins: int = 16, cutoff: float = 8.0) -> np.ndarray:
    if len(positions) < 2:
        return np.zeros(bins)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # PHAL's NaN case
        histogram, _ = np.histogram(pdist(positions), bins=bins, range=(0.0, cutoff), density=True)
    return histogram.astype(float)


def phal_type_weighted(symbols: list[str], positions: np.ndarray, bins: int = 16, cutoff: float = 8.0) -> np.ndarray:
    if len(positions) < 2:
        return np.zeros(3 * bins)
    distances = pdist(positions)
    numbers = np.fromiter((atomic_numbers.get(symbol, 0) for symbol in symbols), dtype=float, count=len(symbols))
    left, right = np.triu_indices(len(symbols), k=1)
    weights = 0.5 * (numbers[left] + numbers[right]) / 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        plain, _ = np.histogram(distances, bins=bins, range=(0.0, cutoff), density=True)
    weighted_sum, _ = np.histogram(distances, bins=bins, range=(0.0, cutoff), weights=weights)
    counts, _ = np.histogram(distances, bins=bins, range=(0.0, cutoff))
    weighted_mean = np.divide(weighted_sum, counts, out=np.zeros_like(weighted_sum, dtype=float), where=counts > 0)
    return np.concatenate([plain.astype(float), weighted_sum / float(len(distances)), weighted_mean])


def phal_pool_select(entries: list[tuple[str, float, np.ndarray]], *, threshold: float, cap: int, deduplicate: bool) -> list[str]:
    """PHAL CandidatePool.select: stable sort by score, greedy >= threshold dedup, then cap."""

    ordered = sorted(entries, key=lambda item: item[1], reverse=True)
    if not deduplicate:
        return [name for name, _, _ in ordered[:cap]]
    unique: list[tuple[str, float, np.ndarray]] = []
    for entry in ordered:
        if all(float(np.linalg.norm(entry[2] - kept[2])) >= threshold for kept in unique):
            unique.append(entry)
    return [name for name, _, _ in unique[:cap]]


# ----------------------------------------------------------------------------- helpers
def _region(region_id: str, positions, numbers, *, event: str | None = "evt", cell=None, pbc=(False,) * 3) -> ExtractedRegion:
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    n = len(positions)
    return ExtractedRegion(
        region_id=region_id, mode="cluster", run_id="run", parent_step=0, event_id=event,
        numbers=np.asarray(numbers, dtype=int), positions=positions, cell=cell, pbc=pbc,
        parent_atom_ids=np.arange(n), image_shifts=np.zeros((n, 3), int), roles=[RegionRole.CORE] * n,
        force_weights=np.ones(n), frozen=np.zeros(n, bool), hotspot_atom_ids=[0] if n else [],
    )


def _h2(region_id: str, bond: float = 0.74) -> ExtractedRegion:
    return _region(region_id, [[0.0, 0.0, 0.0], [bond, 0.0, 0.0]], [1, 1])


def _fixed(name: str, score: float, fingerprint, event: str | None = "evt") -> Candidate:
    return Candidate(_h2(name), float(score), np.asarray(fingerprint, dtype=float), event)


def _selector(**overrides) -> CandidateSelector:
    return CandidateSelector(CandidateConfig(**{**PHAL, "max_per_cycle": 50, "diversity_threshold": 0.1, **overrides}))


def _names(decision) -> list[str]:  # noqa: ANN001
    return [item.candidate_id for item in decision.selected]


# ----------------------------------------------------------------------------- test_deduplication.py ports
def test_empty_and_single_candidate() -> None:
    assert _selector().select([]).selected == []
    assert deduplicate_candidates([], diversity_threshold=0.1) == []
    only = _fixed("only", 1.0, [0.0, 1.0])
    assert _names(_selector().select([only])) == ["only"]


def test_identical_fingerprints_keep_high_score() -> None:
    decision = _selector().select([_fixed("low", 1.0, [1.0, 0.0]), _fixed("high", 2.0, [1.0, 0.0])])
    assert _names(decision) == ["high"] and decision.rejected == {"low": "duplicate"}


def test_threshold_boundary_is_kept() -> None:
    decision = _selector().select([_fixed("left", 2.0, [0.0]), _fixed("right", 1.0, [0.1])])
    assert _names(decision) == ["left", "right"]


def test_fingerprint_distance_is_l2() -> None:
    assert fingerprint_distance(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == 5.0


def test_pair_distance_histogram_empty_single_and_small_regions() -> None:
    config = CandidateConfig(**PHAL)
    # (an empty region is refused by the ExtractedRegion contract, so PHAL's empty case cannot arise)
    assert np.allclose(fingerprint_region(_region("one", [[0.0, 0, 0]], [1]), config), np.zeros(16))
    small = fingerprint_region(_h2("h2"), config.model_copy(update={"fingerprint_bins": 8, "fingerprint_cutoff": 4.0}))
    assert small.shape == (8,) and np.count_nonzero(small) == 1


def test_type_weighted_histogram_distinguishes_atom_types() -> None:
    config = CandidateConfig(**{**PHAL, "fingerprint": "type_weighted_pair_distance_histogram", "fingerprint_bins": 8, "fingerprint_cutoff": 4.0})
    positions = [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]
    h2 = fingerprint_region(_region("h2", positions, [1, 1]), config)
    co = fingerprint_region(_region("co", positions, [6, 8]), config)
    assert h2.shape == (24,) and not np.allclose(h2, co) and fingerprint_distance(h2, co) > 0.0


@pytest.mark.parametrize("alias", ["pair_distance", "distance", "PAIR_DISTANCE_HISTOGRAM"])
def test_phal_mode_aliases_and_unknown_mode(alias: str) -> None:
    assert CandidateConfig(fingerprint=alias).fingerprint == "pair_distance_histogram"
    assert CandidateConfig(fingerprint="weighted_pair").fingerprint == "type_weighted_pair_distance_histogram"
    with pytest.raises(ValueError, match="Unsupported candidate fingerprint mode"):
        CandidateConfig(fingerprint="unknown")


def _sized(n_atoms: int, seed: int = 0) -> ExtractedRegion:
    positions = np.random.default_rng(seed).uniform(0.0, 8.0, (n_atoms, 3))
    return _region(f"c{n_atoms}", positions, [6] * n_atoms)


def test_type_weighted_fingerprint_is_intensive_and_blocks_are_comparable() -> None:
    config = CandidateConfig(**{**PHAL, "fingerprint": "type_weighted_pair_distance_histogram"})
    small, larger = fingerprint_region(_sized(40), config), fingerprint_region(_sized(41), config)
    assert fingerprint_distance(small[:16], larger[:16]) > 0.5 * fingerprint_distance(small, larger)
    magnitudes = [float(np.abs(small[k : k + 16]).max()) for k in (0, 16, 32)]
    assert max(magnitudes) < 100.0 * max(min(magnitudes), 1.0e-12)


# ----------------------------------------------------------------------------- test_candidate_pool.py ports
def test_pool_can_disable_deduplication() -> None:
    same = [_h2("a"), _h2("b")]
    deduplicated = _selector(max_per_cycle=5)
    assert len(deduplicated.select([deduplicated.make_candidate(r, score=s) for r, s in zip(same, (1.0, 2.0))]).selected) == 1
    unfiltered = _selector(max_per_cycle=5, deduplicate=False)
    decision = unfiltered.select([unfiltered.make_candidate(r, score=s) for r, s in zip(same, (1.0, 2.0))])
    assert [item.score for item in decision.selected] == [2.0, 1.0]


def test_pool_selects_top_by_score_with_cap() -> None:
    selector = _selector(diversity_threshold=0.0, max_per_cycle=2, deduplicate=False)
    candidates = [selector.make_candidate(_h2(f"h{b}", b), score=s) for b, s in ((0.72, 0.2), (0.74, 2.0), (0.76, 1.1))]
    assert [item.score for item in selector.select(candidates).selected] == [2.0, 1.1]


def test_pool_deduplicate_removes_similar() -> None:
    selector = _selector(diversity_threshold=0.01, max_per_cycle=5)
    decision = selector.select([selector.make_candidate(_h2("a", 0.740), score=1.0), selector.make_candidate(_h2("b", 0.741), score=2.0)])
    assert [item.score for item in decision.selected] == [2.0]


def test_pool_cap_larger_than_pool_keeps_score_order() -> None:
    selector = _selector(diversity_threshold=0.0, max_per_cycle=10, deduplicate=False)
    decision = selector.select([selector.make_candidate(_h2(f"h{s}", 0.7 + s), score=s) for s in (0.5, 1.5, 1.0)])
    assert [item.score for item in decision.selected] == [1.5, 1.0, 0.5]


def test_pair_distance_histogram_matches_reference_loop() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.8]])
    loop = [float(np.linalg.norm(positions[i] - positions[j])) for i in range(4) for j in range(i + 1, 4)]
    expected, _ = np.histogram(loop, bins=16, range=(0.0, 8.0), density=True)
    assert np.allclose(fingerprint_region(_region("h4", positions, [1] * 4), CandidateConfig(**PHAL)), expected)


def test_eviction_scenario_readmits_the_third_candidate() -> None:
    """candidate_pool.py:19-25: a higher-scoring entry that suppresses a suppressor re-admits a third."""

    selector = _selector(diversity_threshold=1.0)
    b = _fixed("B", 2.0, [1.0])
    c = _fixed("C", 1.0, [1.6])  # 0.6 from B: suppressed by B
    pool = [b, c]
    assert _names(selector.select(pool)) == ["B"]
    a = _fixed("A", 3.0, [0.2])  # 0.8 from B (suppresses it), 1.4 from C
    pool.append(a)
    assert _names(selector.select(pool)) == ["A", "C"]  # recomputed from scratch, C is back
    assert phal_pool_select([(x.candidate_id, x.score, x.fingerprint) for x in pool], threshold=1.0, cap=50, deduplicate=True) == ["A", "C"]


# ----------------------------------------------------------------------------- deliberate deviation F-30
def test_nan_case_of_phal_is_finite_and_does_not_block_later_candidates() -> None:
    far = _region("far", [[0.0, 0, 0], [20.0, 0, 0]], [6, 6])
    assert np.isnan(phal_pair_distance_histogram(far.positions)).all()  # PHAL's defect
    selector = _selector()
    fingerprint = selector.make_candidate(far, score=9.0).fingerprint
    assert np.all(fingerprint == 0.0)
    decision = selector.select([selector.make_candidate(far, score=9.0), selector.make_candidate(_h2("h2"), score=1.0)])
    assert _names(decision) == ["far", "h2"]  # PHAL kept "far" and rejected every later candidate


# ----------------------------------------------------------------------------- golden regions
def _golden_regions() -> list[tuple[str, str, dict]]:
    out = []
    for scenario in SCENARIOS:
        for event_dir in sorted((ROOT / scenario / "regions").iterdir()):
            for mode in MODES:
                for index, record in enumerate(json.loads((event_dir / f"{mode}.json").read_text())):
                    for kind in ("raw_region", "capped_region"):
                        if record.get(kind):
                            out.append((f"{scenario}/{event_dir.name}/{mode}/{index}/{kind}", event_dir.name, record[kind]))
    return out


GOLDEN = _golden_regions()


def _from_golden(name: str, event: str, record: dict) -> ExtractedRegion:
    atoms = record["atoms"]
    numbers = [atomic_numbers[symbol] for symbol in atoms["symbols"]]
    pbc = tuple(bool(flag) for flag in atoms["pbc"])
    cell = np.asarray(atoms["cell"], dtype=float) if any(pbc) else None
    return _region(name, atoms["positions"], numbers, event=event, cell=cell, pbc=pbc)


def test_golden_fixture_regions_are_present() -> None:
    assert len(GOLDEN) >= 80
    # PHAL wrote every golden region (cluster, slab, graph, block) with pbc=False, so plain and
    # minimum-image distances coincide on all of them.
    assert not any(any(record["atoms"]["pbc"]) for _, _, record in GOLDEN)


@pytest.mark.parametrize("fingerprint", ["pair_distance_histogram", "type_weighted_pair_distance_histogram"])
def test_fingerprints_of_golden_regions_match_phal_formula(fingerprint: str) -> None:
    config = CandidateConfig(**{**PHAL, "fingerprint": fingerprint})
    mic = config.model_copy(update={"pair_distances": "minimum_image"})
    for name, event, record in GOLDEN:
        region = _from_golden(name, event, record)
        positions = np.asarray(record["atoms"]["positions"], dtype=float)
        if fingerprint == "pair_distance_histogram":
            expected = phal_pair_distance_histogram(positions)
        else:
            expected = phal_type_weighted(record["atoms"]["symbols"], positions)
        ours = fingerprint_region(region, config)
        if np.isnan(expected).any():  # F-30: GMD is finite where PHAL is NaN
            assert np.all(np.isfinite(ours)), name
            continue
        np.testing.assert_allclose(ours, expected, rtol=1e-12, atol=0.0, err_msg=name)
        if not any(region.pbc):  # minimum image is identical for non-periodic regions
            np.testing.assert_allclose(fingerprint_region(region, mic), expected, rtol=1e-12, atol=0.0, err_msg=name)


def _event_scores(scenario: str) -> dict[str, float]:
    events = json.loads((ROOT / scenario / "events.json").read_text())
    scores = {}
    for event in events:
        values = event["ood_scores"]
        values = json.loads(values) if isinstance(values, str) else values
        scores[event["event_id"]] = float(np.max(values))
    return scores


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("threshold", [0.1, 0.5, 2.0])
def test_pool_over_golden_regions_matches_phal_for_every_cap(scenario: str, mode: str, threshold: float) -> None:
    """AL-002: every region of a frame gets the frame-level max score (ties keep insertion order)."""

    scores = _event_scores(scenario)
    regions = [(name, event, record) for name, event, record in GOLDEN
               if name.startswith(f"{scenario}/") and f"/{mode}/" in name and name.endswith("capped_region")]
    assert regions
    for deduplicate in (True, False):
        for cap in range(1, len(regions) + 2):
            selector = CandidateSelector(CandidateConfig(**{**PHAL, "max_per_cycle": cap, "diversity_threshold": threshold, "deduplicate": deduplicate}))
            candidates = [selector.make_candidate(_from_golden(name, event, record), score=scores[event]) for name, event, record in regions]
            expected = phal_pool_select([(c.candidate_id, c.score, c.fingerprint) for c in candidates], threshold=threshold, cap=cap, deduplicate=deduplicate)
            assert _names(selector.select(candidates)) == expected, (cap, deduplicate)
