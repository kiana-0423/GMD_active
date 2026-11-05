"""Parity with PHAL golden regions (all four modes, all three scenarios), plus corrected defects.

Deliberate deviations asserted here:
* F-04: capped block regions have one role / weight / code per atom and write a dataset (PHAL raised).
* F-09: every region records parent atom IDs and integer image shifts; regions torn by the periodic
  boundary are refused under the default guard policy (PHAL emitted them silently), and with
  ``policy=flag`` H capping also caps torn covalent bonds (PHAL left them dangling).
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.io import read

from gmd_active_learning.contracts import LabelResult, RegionRole, ValidationStatus
from gmd_active_learning.core.exceptions import ExtractionError
from gmd_active_learning.datasets import DatasetsConfig, build_sample, read_samples_extxyz, write_samples_extxyz
from gmd_active_learning.datasets.masks import region_codes_for_labels, roles_from_labels
from gmd_active_learning.extraction import add_h_caps, extract_block_regions, extract_cluster, extract_graph, extract_slab
from gmd_active_learning.extraction.config import ExtractionConfig

from extraction.phal_fixtures import (
    FIXTURE_POSITION_ATOL,
    ROOT,
    all_cases,
    capping_config,
    extraction_config,
    frame_at,
    hotspots_for,
    region_records,
)

CASES = all_cases()
EXTRACTORS = {"cluster": extract_cluster, "slab": extract_slab, "graph": extract_graph}


def _extract(scenario: str, event: str, mode: str, index: int, config: ExtractionConfig):
    record = region_records(scenario, event, mode)[index]
    raw = record["raw_region"]
    frame = frame_at(scenario, int(raw["metadata"]["original_frame_id"]))
    hotspots = hotspots_for(scenario, event, mode, index)
    if mode == "block":
        regions = extract_block_regions(frame, hotspots, config, event_id=event)
        matches = [region for region in regions if region.metadata["block_ids"] == raw["metadata"]["block_ids"]]
        assert len(matches) == 1
        return frame, matches[0], record
    return frame, EXTRACTORS[mode](frame, hotspots, config, event_id=event), record


def _flag(scenario: str, mode: str) -> ExtractionConfig:
    return extraction_config(scenario, mode, periodic_guard={"policy": "flag"})


def test_every_scenario_and_mode_is_covered() -> None:
    covered = {(scenario, mode) for scenario, _event, mode, _index in CASES}
    assert covered == {(s, m) for s in ("ethane_box", "polyethylene_chains", "pt111_co_slab") for m in ("cluster", "slab", "graph", "block")}
    assert len(CASES) == 42


@pytest.mark.parametrize(("scenario", "event", "mode", "index"), CASES)
def test_raw_region_matches_phal(scenario: str, event: str, mode: str, index: int) -> None:
    frame, region, record = _extract(scenario, event, mode, index, _flag(scenario, mode))
    raw = record["raw_region"]
    # PHAL original_indices are 0-based frame indices == atom IDs of the fixture frames.
    assert region.parent_atom_ids.tolist() == raw["original_indices"]
    assert region.roles == roles_from_labels(raw["region_labels"])
    np.testing.assert_allclose(region.positions, np.asarray(raw["atoms"]["positions"]), atol=FIXTURE_POSITION_ATOL, rtol=0)
    np.testing.assert_allclose(region.cell, np.asarray(raw["atoms"]["cell"]), atol=FIXTURE_POSITION_ATOL, rtol=0)
    assert list(region.pbc) == raw["atoms"]["pbc"]
    expected_hotspots = sorted(raw["original_indices"][local] for local in raw["hotspot_indices"])
    assert sorted(region.hotspot_atom_ids) == expected_hotspots
    np.testing.assert_allclose(region.metadata["center"], raw["metadata"].get("center", region.metadata["center"]), atol=FIXTURE_POSITION_ATOL)
    if mode == "graph":
        assert region.metadata["graph_atom_ids"] == raw["metadata"]["graph_atom_indices"]
    # F-09: image shifts reproduce every region position from its parent position.
    translation = np.asarray(region.metadata["origin_translation"])
    rebuilt = frame.positions[frame.index_of(region.parent_atom_ids)] + region.image_shifts @ frame.cell + translation
    np.testing.assert_allclose(region.positions, rebuilt, atol=1e-9, rtol=0)


@pytest.mark.parametrize(("scenario", "event", "mode", "index"), CASES)
def test_capped_region_matches_phal_or_documents_the_torn_bond_fix(scenario: str, event: str, mode: str, index: int) -> None:
    frame, region, record = _extract(scenario, event, mode, index, _flag(scenario, mode))
    capped = add_h_caps(region, frame, capping_config(scenario))
    phal = record["capped_region"]
    phal_positions = np.asarray(phal["atoms"]["positions"])
    n_raw = region.n_atoms
    assert capped.metadata["h_capping"]["skipped_reason"] == phal["metadata"].get("h_capping_skipped_reason")
    if phal["metadata"].get("h_capping_skipped"):
        assert capped.n_atoms == n_raw == len(phal_positions)
        assert capped.metadata["h_capping_skipped"] is True
        return
    phal_caps = phal_positions[n_raw:]
    my_caps = capped.positions[n_raw:]
    if len(my_caps) == len(phal_caps):
        np.testing.assert_allclose(my_caps, phal_caps, atol=FIXTURE_POSITION_ATOL, rtol=0)
        np.testing.assert_array_equal(capped.force_weights, np.asarray(record["mask"]))
        labels = phal["region_labels"] if mode != "block" else [*phal["metadata"]["atom_role"], *(["h_cap"] * len(phal_caps))]
        np.testing.assert_array_equal(capped.region_codes, region_codes_for_labels(labels))
    else:
        # Deviation: torn bonds (partner present only at another image) are capped as well.
        assert region.metadata["periodic_guard"]["torn_pairs"] > 0
        assert len(my_caps) > len(phal_caps)
        for position in phal_caps:
            assert np.min(np.linalg.norm(my_caps - position, axis=1)) < FIXTURE_POSITION_ATOL
    # Every per-atom array has n_atoms entries, caps included (F-04).
    assert len(capped.roles) == len(capped.force_weights) == len(capped.frozen) == capped.n_atoms
    assert capped.region_codes.shape == (capped.n_atoms,)


@pytest.mark.parametrize(
    ("scenario", "event", "index"),
    [(s, e, i) for s, e, m, i in CASES if m == "block"],
)
def test_capped_block_regions_write_a_dataset(tmp_path, scenario: str, event: str, index: int) -> None:
    """F-04: PHAL raised 'could not broadcast' for every capped block region; now it writes."""

    frame, region, record = _extract(scenario, event, "block", index, _flag(scenario, "block"))
    phal_error = record.get("dataset_error")
    capped = add_h_caps(region, frame, capping_config(scenario))
    rng = np.random.default_rng(7)
    label = LabelResult(
        task_id="task-1", region_id=capped.region_id, forces=rng.normal(size=(capped.n_atoms, 3)),
        energy=-1.0, converged=True, validation=ValidationStatus.VALID,
    )
    sample = build_sample(capped, label, DatasetsConfig())
    path = write_samples_extxyz([sample], tmp_path / "block.extxyz")
    back = read_samples_extxyz(path)[0]
    assert back.region.roles == capped.roles
    np.testing.assert_array_equal(back.region.force_weights, capped.force_weights)
    if any(role is RegionRole.H_CAP for role in capped.roles) and scenario != "pt111_co_slab":
        assert phal_error is not None and "could not broadcast" in phal_error["message"]


def test_default_guard_refuses_regions_torn_through_weighted_atoms() -> None:
    """F-09: the ethane block regions span the whole 13.8 A cell; PHAL emitted them silently."""

    with pytest.raises(ExtractionError, match="torn"):
        _extract("ethane_box", "evt-50-00000000", "block", 0, extraction_config("ethane_box", "block"))
    # A region that is not torn through weighted atoms passes the default guard unchanged.
    _frame, region, _record = _extract("ethane_box", "evt-80-00000001", "cluster", 0, extraction_config("ethane_box", "cluster"))
    assert region.metadata["periodic_guard"]["torn_weighted_pairs"] == 0


def test_pt_cluster_radius_beyond_half_cell_is_flagged() -> None:
    _frame, region, _record = _extract("pt111_co_slab", "evt-30-00000000", "cluster", 0, _flag("pt111_co_slab", "cluster"))
    assert region.metadata["periodic_guard"]["radius_exceeds_half_cell"] is True


@pytest.mark.parametrize(("scenario", "event", "mode", "index"), [c for c in CASES if c[2] != "block"])
def test_phal_capped_datasets_have_the_same_masks_and_codes(scenario: str, event: str, mode: str, index: int) -> None:
    """PHAL's written (capped) datasets are evidence of its on-disk format: compare mask_weights / region_code."""

    path = ROOT / scenario / "datasets" / event / mode / f"region{index:03d}.extxyz"
    phal = read(path, format="extxyz")
    frame, region, _record = _extract(scenario, event, mode, index, _flag(scenario, mode))
    capped = add_h_caps(region, frame, capping_config(scenario))
    if capped.n_atoms != len(phal):
        assert region.metadata["periodic_guard"]["torn_pairs"] > 0  # torn-bond capping deviation
        n = region.n_atoms
        np.testing.assert_array_equal(capped.force_weights[:n], phal.arrays["mask_weights"][:n])
        np.testing.assert_array_equal(capped.region_codes[:n], phal.arrays["region_code"][:n])
        return
    np.testing.assert_array_equal(capped.force_weights, phal.arrays["mask_weights"])
    np.testing.assert_array_equal(capped.region_codes, phal.arrays["region_code"])
    assert phal.info["region_label_map"] == '{"0": "core", "1": "inner_buffer", "2": "outer_buffer", "3": "boundary", "4": "h_cap"}'
