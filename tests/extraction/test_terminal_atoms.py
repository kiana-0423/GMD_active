"""``extraction.terminal_atoms = follow_partner``: terminal atoms (H) follow their bonded partner."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import molecule

from gmd_active_learning.contracts import Frame
from gmd_active_learning.core.exceptions import ExtractionError
from gmd_active_learning.extraction.cluster import extract_cluster
from gmd_active_learning.extraction.config import ExtractionConfig
from gmd_active_learning.extraction.graph import extract_graph
from gmd_active_learning.utils.bonds import bonded_neighbors


def _butane() -> Frame:
    atoms = molecule("trans-butane")
    return Frame("butane", 0, np.arange(len(atoms)), atoms.numbers, atoms.positions, None, (False, False, False))


def _terminal_consistent(frame: Frame, ids: set[int]) -> bool:
    adjacency = bonded_neighbors(frame.positions, frame.numbers)
    return all(
        (index in ids) == (partners[0] in ids) for index, partners in adjacency.items() if len(partners) == 1
    )


def test_graph_selection_gets_the_hydrogens_of_boundary_carbons() -> None:
    frame = _butane()
    carbons = [i for i, z in enumerate(frame.numbers) if z == 6]
    base = {"mode": "graph", "graph": {"hops": 1, "radial_shell": False}}
    plain = extract_graph(frame, [carbons[0]], ExtractionConfig.model_validate(base))
    follow = extract_graph(frame, [carbons[0]], ExtractionConfig.model_validate({**base, "terminal_atoms": "follow_partner"}))
    assert not _terminal_consistent(frame, set(plain.parent_atom_ids.tolist()))
    assert _terminal_consistent(frame, set(follow.parent_atom_ids.tolist()))
    added = follow.metadata["size_policy"]["terminal_atoms"]["added_atom_ids"]
    assert added and set(added).isdisjoint(plain.parent_atom_ids.tolist())
    # The default keeps region IDs of existing configurations unchanged.
    assert ExtractionConfig.model_validate(base).identity_payload(set()) == {
        k: v for k, v in ExtractionConfig.model_validate(base).model_dump(mode="json").items() if k != "terminal_atoms"
    }
    assert plain.region_id != follow.region_id


def test_cluster_drops_lone_hydrogens_and_adds_missing_ones() -> None:
    frame = _butane()
    carbons = [i for i, z in enumerate(frame.numbers) if z == 6]
    config = {"mode": "cluster", "extract_radius": 2.6, "core_radius": 1.0, "boundary_thickness": 0.5}
    plain = extract_cluster(frame, [carbons[0]], ExtractionConfig.model_validate(config))
    follow = extract_cluster(frame, [carbons[0]], ExtractionConfig.model_validate({**config, "terminal_atoms": "follow_partner"}))
    assert not _terminal_consistent(frame, set(plain.parent_atom_ids.tolist()))
    assert _terminal_consistent(frame, set(follow.parent_atom_ids.tolist()))
    record = follow.metadata["size_policy"]["terminal_atoms"]
    assert record["removed_atom_ids"] or record["added_atom_ids"]


def test_hotspot_terminal_atom_without_partner_raises() -> None:
    frame = _butane()
    hydrogen = int(np.flatnonzero(frame.numbers == 1)[0])
    config = ExtractionConfig.model_validate({"mode": "cluster", "extract_radius": 0.5, "terminal_atoms": "follow_partner"})
    with pytest.raises(ExtractionError, match="follow_partner"):
        extract_cluster(frame, [hydrogen], config)
