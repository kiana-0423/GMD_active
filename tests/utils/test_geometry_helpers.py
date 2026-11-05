"""Geometry helpers (UTIL-001 / UTIL-002): ports of PHAL test_periodic_utils.py and padded_cluster_cell."""

from __future__ import annotations

import numpy as np
import pytest

from gmd_active_learning.utils.geometry import distances_to_group, hotspot_center, padded_cluster_cell, row_norms
from gmd_active_learning.utils.periodic import as_cell_matrix, mic_displacement, mic_displacements_from_reference


def test_vectorised_mic_matches_loop_for_orthogonal_and_skewed_cells() -> None:
    rng = np.random.default_rng(456)
    reference = np.array([11.0, -3.0, 5.0])
    positions = rng.uniform(-8.0, 18.0, size=(100, 3))
    for cell, pbc in ((np.diag([10.0, 10.0, 10.0]), True), (np.array([[10.0, 2.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 12.0]]), (True, True, False))):
        vectorised = mic_displacements_from_reference(reference, positions, cell=cell, pbc=pbc)
        looped = np.vstack([mic_displacement(reference, position, cell=cell, pbc=pbc) for position in positions])
        np.testing.assert_allclose(vectorised, looped, atol=1e-12)
    np.testing.assert_allclose(mic_displacements_from_reference(reference, positions, pbc=False), positions - reference)


def test_non_periodic_axis_is_untouched() -> None:
    displacement = mic_displacements_from_reference(
        np.array([9.5, 9.5, 0.0]), np.array([[0.5, 0.5, 20.0]]), cell=np.diag([10.0, 10.0, 10.0]), pbc=[True, True, False]
    )
    np.testing.assert_allclose(displacement[0], [1.0, 1.0, 20.0])


def test_as_cell_matrix() -> None:
    np.testing.assert_allclose(as_cell_matrix(np.array([10.0, 8.0, 6.0])), np.diag([10.0, 8.0, 6.0]))
    with pytest.raises(ValueError, match="Unsupported cell shape"):
        as_cell_matrix(np.ones((2, 2)))


def test_distances_to_group_matches_brute_force_on_a_non_orthogonal_cell() -> None:
    rng = np.random.default_rng(321)
    positions = rng.uniform(-2.0, 12.0, size=(40, 3))
    group = [0, 7, 13]
    cell = np.array([[10.0, 1.5, 0.2], [0.0, 8.0, 0.7], [0.0, 0.0, 12.0]])
    pbc = [True, True, False]
    expected = np.min(
        [[np.linalg.norm(mic_displacement(positions[index], position, cell=cell, pbc=pbc)) for position in positions] for index in group],
        axis=0,
    )
    np.testing.assert_allclose(distances_to_group(positions, group, cell=cell, pbc=pbc), expected, atol=1e-12)
    assert np.all(np.isinf(distances_to_group(positions, [])))


def test_padded_cluster_cell_and_row_norms() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]])
    np.testing.assert_allclose(padded_cluster_cell(positions, padding=4.0), np.diag([10.0, 9.0, 9.0]))
    np.testing.assert_allclose(padded_cluster_cell(positions[:1], padding=3.0), np.diag([6.0, 6.0, 6.0]))
    np.testing.assert_allclose(row_norms(np.array([[3.0, 4.0, 0.0]])), [5.0])


def test_hotspot_center_is_anchor_relative_and_rejects_empty() -> None:
    positions = np.array([[0.5, 1.0, 1.0], [9.7, 1.0, 1.0], [9.9, 1.0, 1.0]])
    center = hotspot_center(positions, [0, 1, 2], cell=np.eye(3) * 10.0, pbc=True)
    np.testing.assert_allclose(center, [0.5 + (0.0 - 0.8 - 0.6) / 3.0, 1.0, 1.0], atol=1e-12)
    with pytest.raises(ValueError):
        hotspot_center(positions, [])


def test_padded_cluster_cell_reproduces_phal_cluster_region_cells() -> None:
    import json
    from pathlib import Path

    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "phal_reference"
    checked = 0
    for scenario in ("ethane_box", "polyethylene_chains", "pt111_co_slab"):
        config = json.loads((fixtures / scenario / "config.json").read_text(encoding="utf-8"))
        padding = float(config["extraction"]["vacuum_padding"])
        for event in json.loads((fixtures / scenario / "events.json").read_text(encoding="utf-8")):
            payload = json.loads((fixtures / scenario / "regions" / event["event_id"] / "cluster.json").read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                continue
            for record in payload:
                atoms = record["raw_region"]["atoms"]
                np.testing.assert_allclose(padded_cluster_cell(np.asarray(atoms["positions"]), padding), np.asarray(atoms["cell"]), atol=1e-12)
                checked += 1
    assert checked >= 8
