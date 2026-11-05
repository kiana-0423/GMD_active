"""Hotspot detection and clustering on stable IDs (HOT-001, HOT-002): ports of PHAL test_hotspot_detection.py."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from gmd_active_learning.contracts import stable_id
from gmd_active_learning.contracts.frame import Frame
from gmd_active_learning.contracts.monitoring import MetricValues, MonitorResult, Stage
from gmd_active_learning.hotspots import cluster_anomalous_atoms, detect_hotspots, hotspots_from_result
from gmd_active_learning.utils.periodic import mic_displacement


def _frame(positions: Any, *, cell: Any = None, pbc: Any = (False,) * 3, atom_ids: Any = None) -> Frame:
    positions = np.asarray(positions, dtype=float)
    n_atoms = len(positions)
    ids = np.arange(n_atoms) if atom_ids is None else np.asarray(atom_ids)
    return Frame("run", 12, ids, np.ones(n_atoms, dtype=int), positions, cell, pbc, forces=np.zeros((n_atoms, 3)))


def _reference_clusters(flagged: list[int], positions: np.ndarray, merge_radius: float, cell: Any, pbc: Any) -> list[list[int]]:
    """PHAL's union-find loop over explicit pairwise minimum-image distances."""

    parent = {index: index for index in flagged}

    def find(index: int) -> int:
        while parent[index] != index:
            index = parent[index]
        return index

    for i, left in enumerate(flagged):
        for right in flagged[i + 1 :]:
            if np.linalg.norm(mic_displacement(positions[left], positions[right], cell=cell, pbc=pbc)) <= merge_radius:
                root_left, root_right = find(left), find(right)
                if root_left != root_right:
                    parent[root_right] = root_left
    clusters: dict[int, list[int]] = {}
    for index in flagged:
        clusters.setdefault(find(index), []).append(index)
    return [sorted(indices) for indices in clusters.values()]


def test_nearby_atoms_merge_and_far_ones_split() -> None:
    frame = _frame([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [7.5, 0.0, 0.0], [20.0, 0.0, 0.0]])
    hotspots = detect_hotspots(frame, np.array([7.2, 6.4, 6.1, 1.0]), threshold=6.0, merge_radius=2.0, trigger_reasons=["force_burst"])
    assert [hotspot.atom_ids for hotspot in hotspots] == [[0, 1], [2]]
    assert hotspots[0].step == 12 and hotspots[0].trigger_reasons == ["force_burst"]
    assert hotspots[0].max_score == 7.2 and hotspots[0].metadata["n_atoms"] == 2
    np.testing.assert_allclose(hotspots[0].center, [0.5, 0.0, 0.0])
    assert hotspots[0].hotspot_id == stable_id("hot", "run", 12, [0, 1])


def test_sorted_by_max_score() -> None:
    frame = _frame([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [6.0, 0.0, 0.0], [6.5, 0.0, 0.0]])
    hotspots = detect_hotspots(frame, np.array([7.0, 7.1, 8.0, 7.9]), threshold=6.0, merge_radius=1.0)
    assert [hotspot.atom_ids for hotspot in hotspots] == [[2, 3], [0, 1]]


def test_none_below_threshold_single_atom_and_exact_threshold() -> None:
    line = _frame([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert detect_hotspots(line, np.array([1.0, 5.99, 2.0]), threshold=6.0, merge_radius=1.0) == []
    spread = _frame([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    assert [hotspot.atom_ids for hotspot in detect_hotspots(spread, np.array([1.0, 6.2, 2.0]), threshold=6.0, merge_radius=1.0)] == [[1]]
    periodic = _frame([[0.0, 0.0, 0.0], [9.8, 0.0, 0.0]], cell=np.eye(3) * 10.0, pbc=(True,) * 3)
    hotspots = detect_hotspots(periodic, np.array([6.0, 6.1]), threshold=6.0, merge_radius=0.5)
    assert [hotspot.atom_ids for hotspot in hotspots] == [[0, 1]]
    np.testing.assert_allclose(hotspots[0].center, [-0.1, 0.0, 0.0], atol=1e-12)  # contiguous across the boundary


def test_distance_equal_to_merge_radius_merges() -> None:
    clusters = cluster_anomalous_atoms([0, 1], np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), merge_radius=2.0)
    assert clusters == [[0, 1]]


@pytest.mark.parametrize(
    ("cell", "pbc"),
    [(np.diag([10.0, 10.0, 10.0]), True), (np.array([[10.0, 0.0, 0.0], [4.0, 9.0, 0.0], [2.0, 3.0, 11.0]]), True),
     (np.array([[6.0, 0.0, 0.0], [5.0, 3.0, 0.0], [4.0, 2.5, 2.0]]), (True, True, False))],
    ids=["orthorhombic", "triclinic", "skewed-partial"],
)
def test_clusters_match_reference_loop_under_pbc(cell: np.ndarray, pbc: Any) -> None:
    rng = np.random.default_rng(11)
    positions = rng.uniform(0.0, 10.0, size=(60, 3))
    positions[1] = positions[0] + np.array([0.2, 0.0, 0.0])
    positions[3] = np.array([9.9, 0.0, 0.0])
    positions[4] = np.array([0.1, 0.0, 0.0])
    flagged = list(range(60))
    expected = _reference_clusters(flagged, positions, 1.2, cell, pbc)
    actual = cluster_anomalous_atoms(flagged, positions, merge_radius=1.2, cell=cell, pbc=pbc)
    assert actual == expected  # same sets AND same (first-appearance) order


def test_hotspots_are_invariant_to_engine_atom_order() -> None:
    rng = np.random.default_rng(5)
    positions = rng.uniform(0.0, 12.0, size=(40, 3))
    ids = np.arange(100, 140)
    scores = rng.uniform(0.0, 10.0, size=40)
    cell = np.eye(3) * 12.0
    reference = detect_hotspots(_frame(positions, cell=cell, pbc=(True,) * 3, atom_ids=ids), scores, threshold=6.0, merge_radius=3.0)
    permutation = rng.permutation(40)
    shuffled = detect_hotspots(
        _frame(positions[permutation], cell=cell, pbc=(True,) * 3, atom_ids=ids[permutation]), scores[permutation], threshold=6.0, merge_radius=3.0
    )
    assert [hotspot.to_dict() for hotspot in shuffled] == [hotspot.to_dict() for hotspot in reference]
    assert all(min(hotspot.atom_ids) >= 100 for hotspot in reference)


def _result(frame: Frame, scores: np.ndarray, hotspot_ids: list[int]) -> MonitorResult:
    return MonitorResult(
        run_id=frame.run_id,
        step=frame.step,
        stage=Stage.LIGHT,
        atom_scores=scores,
        metrics={"force": MetricValues.measured("force", scores)},
        max_score=float(scores.max()),
        triggered=bool(hotspot_ids),
        hotspot_atom_ids=hotspot_ids,
        trigger_reasons=["force_large"],
    )


def test_hotspots_from_result_use_the_monitor_hotspot_atoms_only() -> None:
    """A light-stage trigger with argmax hotspot atoms below label_threshold still yields a cluster (F-16)."""

    frame = _frame([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [9.0, 0.0, 0.0]], atom_ids=[7, 8, 9])
    scores = np.array([4.5, 1.0, 4.2])
    hotspots = hotspots_from_result(frame, _result(frame, scores, [7]), {"merge_radius": 2.0})
    assert [hotspot.atom_ids for hotspot in hotspots] == [[7]] and hotspots[0].trigger_reasons == ["force_large"]
    assert hotspots_from_result(frame, _result(frame, scores, [7]), {"enabled": False}) == []
    assert hotspots_from_result(frame, _result(frame, scores, []), {}) == []
