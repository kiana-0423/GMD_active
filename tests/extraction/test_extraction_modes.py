"""Unit tests for the extraction modes on small synthetic frames."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule

from gmd_active_learning.contracts import Frame, RegionRole
from gmd_active_learning.core.exceptions import ExtractionError, UnsupportedCombinationError
from gmd_active_learning.extraction import (
    BlockCooldownTracker,
    ExtractionConfig,
    assign_atoms_to_spatial_blocks,
    assign_radial_regions,
    block_grid_shape,
    extract_block_regions,
    extract_cluster,
    extract_graph,
    extract_slab,
    merge_adjacent_blocks,
    region_label_vector,
)
from gmd_active_learning.extraction._placement import find_torn_pairs


def frame_of(atoms: Atoms, *, ids=None, step: int = 0) -> Frame:
    atoms = atoms.copy()
    if ids is not None:
        atoms.arrays["atom_id"] = np.asarray(ids)
    return Frame.from_atoms(atoms, run_id="run", step=step)


def rebuilt_positions(frame: Frame, region) -> np.ndarray:
    cell = np.zeros((3, 3)) if frame.cell is None else frame.cell
    return frame.positions[frame.index_of(region.parent_atom_ids)] + region.image_shifts @ cell + np.asarray(region.metadata["origin_translation"])


def cu_frame(repeat: int = 4) -> Frame:
    return frame_of(bulk("Cu", "fcc", a=3.61, cubic=True).repeat(repeat))


# --------------------------------------------------------------------------- radial roles
def test_radial_partition_defaults_and_edges() -> None:
    distances = np.array([0.0, 4.0, 4.5, 6.0, 6.5, 8.0, 10.0, 10.5])
    regions = assign_radial_regions(distances, core_radius=4.0, extract_radius=10.0, boundary_thickness=2.0)
    assert regions == {"core": [0, 1], "inner_buffer": [2, 3], "outer_buffer": [4], "boundary": [5, 6]}
    assert region_label_vector(8, regions)[-1] == "unassigned"
    # boundary start equal to the core radius: no buffers at all
    regions = assign_radial_regions(distances, core_radius=4.0, extract_radius=5.0, boundary_thickness=2.0)
    assert regions["inner_buffer"] == [] and regions["outer_buffer"] == [] and regions["boundary"] == [1, 2]  # PHAL lists overlap; the label vector gives core priority


# --------------------------------------------------------------------------- cluster
def test_cluster_records_stable_ids_and_image_shifts_across_the_boundary() -> None:
    frame = cu_frame()
    ids = np.arange(1000, 1000 + frame.n_atoms)[::-1]
    frame = frame_of(frame.to_atoms(), ids=ids)
    corner = int(np.argmin(np.linalg.norm(frame.positions, axis=1)))
    config = ExtractionConfig(extract_radius=5.0, core_radius=2.0)
    region = extract_cluster(frame, [int(frame.atom_ids[corner])], config)
    assert set(region.parent_atom_ids.tolist()) <= set(ids.tolist())
    assert np.any(region.image_shifts != 0)  # the corner region wraps through the cell faces
    np.testing.assert_allclose(region.positions, rebuilt_positions(frame, region), atol=1e-10)
    assert region.pbc == (False, False, False)
    assert region.roles[region.parent_atom_ids.tolist().index(int(frame.atom_ids[corner]))] is RegionRole.CORE
    assert region.metadata["periodic_guard"]["torn_pairs"] == 0


def test_cluster_radius_beyond_half_cell_is_flagged_and_torn_regions_are_refused() -> None:
    frame = cu_frame(2)  # 7.22 A cell
    with pytest.raises(ExtractionError, match="torn"):
        extract_cluster(frame, [0], ExtractionConfig(extract_radius=5.0, core_radius=4.0))
    region = extract_cluster(frame, [0], ExtractionConfig(extract_radius=5.0, core_radius=4.0, periodic_guard={"policy": "flag"}))
    assert region.metadata["periodic_guard"]["radius_exceeds_half_cell"] is True
    assert region.metadata["periodic_guard"]["violation"] is True


def test_cluster_size_policies_are_explicit() -> None:
    frame = cu_frame()
    trimmed = extract_cluster(frame, [0], ExtractionConfig(extract_radius=5.0, core_radius=2.0, max_atoms=20))
    assert trimmed.n_atoms == 20 and len(trimmed.metadata["size_policy"]["trimmed_atom_ids"]) > 0
    with pytest.raises(ExtractionError, match="max_atoms_exceeded"):
        extract_cluster(frame, [0], ExtractionConfig(extract_radius=5.0, max_atoms=20, max_atoms_policy="error"))
    with pytest.raises(ExtractionError, match="hotspot atoms"):
        extract_cluster(frame, [0, 1, 2], ExtractionConfig(extract_radius=5.0, max_atoms=2))
    padded = extract_cluster(frame, [0], ExtractionConfig(extract_radius=1.0, core_radius=0.5, min_atoms=5))
    assert padded.n_atoms == 5 and len(padded.metadata["size_policy"]["padded_atom_ids"]) == 4
    assert padded.roles.count(RegionRole.UNASSIGNED) == 4 and float(padded.force_weights[padded.roles.index(RegionRole.UNASSIGNED)]) == 0.0
    with pytest.raises(ExtractionError, match="min_atoms_not_met"):
        extract_cluster(frame, [0], ExtractionConfig(extract_radius=1.0, min_atoms=5, min_atoms_policy="error"))
    allowed = extract_cluster(frame, [0], ExtractionConfig(extract_radius=1.0, min_atoms=5, min_atoms_policy="allow"))
    assert allowed.n_atoms == 1 and allowed.metadata["size_policy"]["below_min_atoms"] is True


def test_cluster_on_molecule_single_atom_and_all_hotspots() -> None:
    mol = molecule("CH3CH2OH")
    mol.center(vacuum=5.0)
    frame = frame_of(mol)
    whole = extract_cluster(frame, list(range(len(mol))), ExtractionConfig(extract_radius=3.0))
    assert whole.n_atoms == len(mol) and set(whole.roles) == {RegionRole.CORE}
    single = frame_of(Atoms("Ar", positions=[[0.0, 0.0, 0.0]]))
    region = extract_cluster(single, [0])
    assert region.n_atoms == 1 and region.positions[0].tolist() == [2.0, 2.0, 2.0]
    with pytest.raises(ExtractionError):
        extract_cluster(frame, [])


def test_triclinic_cluster_shift_invariant() -> None:
    atoms = bulk("Si", "diamond", a=5.43).repeat(3)
    frame = frame_of(atoms)
    region = extract_cluster(frame, [3], ExtractionConfig(extract_radius=4.0, core_radius=2.0))
    np.testing.assert_allclose(region.positions, rebuilt_positions(frame, region), atol=1e-9)


# --------------------------------------------------------------------------- slab
def h_lattice(spacing: float = 5.0, n: int = 1) -> Frame:
    """H atoms on a square lattice with period ``spacing``; z non-periodic."""

    positions = [[spacing * (i + 0.5) / n, spacing * (j + 0.5) / n, 10.0] for i in range(n) for j in range(n)]
    atoms = Atoms(["H"] * len(positions), positions=positions, cell=[spacing, spacing, 20.0], pbc=(True, True, False))
    return frame_of(atoms)


def test_slab_multi_period_window_replicates_images_instead_of_an_empty_cell() -> None:
    """F-08 reproduction: PHAL made a 15x15 A periodic cell holding one 5 A image."""

    frame = h_lattice(5.0, n=2)
    config = ExtractionConfig(extract_radius=8.0, slab={"lx": 16.0, "ly": 16.0, "z_margin": 4.0})
    region = extract_slab(frame, [0], config)
    assert region.pbc == (True, True, False)
    np.testing.assert_allclose(np.diag(region.cell)[:2], [15.0, 15.0])
    assert region.n_atoms == 9 * frame.n_atoms  # 3 x 3 images of the 5 A cell
    assert len(set(map(tuple, region.image_shifts.tolist()))) == 9
    # The density of the periodic patch equals that of the source (no empty strip).
    wrapped = np.mod(region.positions[:, :2], 15.0)
    spacing = np.sort(np.unique(np.round(wrapped[:, 0], 6)))
    np.testing.assert_allclose(np.diff(spacing), 2.5)
    np.testing.assert_allclose(region.positions, rebuilt_positions(frame, region), atol=1e-10)
    assert region.metadata["periodic_guard"]["torn_pairs"] == 0

    single = extract_slab(frame, [0], config.model_copy(update={"slab": config.slab.model_copy(update={"multi_period": "single_period"})}))
    assert single.n_atoms == frame.n_atoms and np.allclose(np.diag(single.cell)[:2], 5.0)
    assert single.metadata["slab_policy"]["window_reduced"] == [True, True]
    with pytest.raises(ExtractionError, match="F-08"):
        extract_slab(frame, [0], config.model_copy(update={"slab": config.slab.model_copy(update={"multi_period": "error"})}))


def test_slab_full_period_window_on_skewed_cell_keeps_source_lateral_vectors() -> None:
    atoms = Atoms("H", positions=[[1.0, 1.0, 5.0]], cell=[[4.0, 0.0, 0.0], [2.0, 3.0, 0.0], [0.0, 0.0, 15.0]], pbc=(True, True, False))
    frame = frame_of(atoms)
    region = extract_slab(frame, [0], ExtractionConfig(slab={"lx": 4.0, "ly": 4.0, "z_margin": 3.0}))
    np.testing.assert_allclose(region.cell[0], [4.0, 0.0, 0.0])
    np.testing.assert_allclose(region.cell[1], [2.0, 3.0, 0.0])
    assert region.pbc == (True, True, False)
    with pytest.raises(UnsupportedCombinationError):
        extract_slab(frame, [0], ExtractionConfig(slab={"lx": 4.0, "ly": 2.0, "z_margin": 3.0}))
    fallback = extract_slab(frame, [0], ExtractionConfig(slab={"lx": 4.0, "ly": 2.0, "z_margin": 3.0, "skewed_policy": "nonperiodic"}))
    assert fallback.pbc == (False, False, False) and fallback.metadata["slab_policy"]["skewed_fallback"] is True


def test_slab_narrow_window_is_non_periodic_without_sub_bond_contacts() -> None:
    frame = cu_frame()
    region = extract_slab(frame, [0], ExtractionConfig(extract_radius=6.0, slab={"lx": 6.0, "ly": 6.0, "z_margin": 3.0}))
    assert region.pbc == (False, False, False)
    distances = np.linalg.norm(region.positions[:, None] - region.positions[None], axis=-1)
    assert np.min(distances[np.triu_indices(region.n_atoms, 1)]) > 2.0


# --------------------------------------------------------------------------- graph
def chain_frame(n_carbon: int = 8) -> Frame:
    """A zig-zag C chain with H atoms in a large box (non-periodic)."""

    symbols, positions = [], []
    for i in range(n_carbon):
        x, y = 1.27 * i, 0.0 if i % 2 == 0 else 0.87
        symbols.append("C")
        positions.append([x + 5.0, y + 5.0, 5.0])
    for i in range(n_carbon):
        x, y, z = positions[i]
        symbols += ["H", "H"]
        positions += [[x, y + (-0.63 if i % 2 == 0 else 0.63), z + 0.89], [x, y + (-0.63 if i % 2 == 0 else 0.63), z - 0.89]]
    atoms = Atoms(symbols, positions=positions, cell=[30.0, 15.0, 15.0], pbc=False)
    return frame_of(atoms)


def test_graph_without_radial_shell_is_exactly_the_subgraph_and_all_core() -> None:
    frame = chain_frame()
    region = extract_graph(frame, [3], ExtractionConfig(graph={"hops": 1, "radial_shell": False}))
    assert sorted(region.parent_atom_ids.tolist()) == sorted([2, 3, 4, 8 + 2 * 3, 8 + 2 * 3 + 1])
    assert set(region.roles) == {RegionRole.CORE}
    assert region.hotspot_atom_ids == [3]
    shell = extract_graph(frame, [3], ExtractionConfig(extract_radius=3.0, core_radius=1.0, graph={"hops": 1}))
    assert shell.n_atoms > region.n_atoms


# --------------------------------------------------------------------------- block
def test_block_grid_assignment_wrap_clip_and_merging() -> None:
    frame = cu_frame()
    assert block_grid_shape(frame.cell, (5.0, 5.0, 5.0)) == (3, 3, 3)
    mapping = assign_atoms_to_spatial_blocks(frame, (5.0, 5.0, 5.0))
    assert set(mapping.values()) <= {(i, j, k) for i in range(3) for j in range(3) for k in range(3)}
    groups = merge_adjacent_blocks([(0, 0, 0), (2, 0, 0)], grid_shape=(3, 3, 3), pbc=(True, True, True))
    assert groups == [[(0, 0, 0), (2, 0, 0)]]  # adjacent through the periodic wrap
    assert merge_adjacent_blocks([(0, 0, 0), (2, 0, 0)], grid_shape=(3, 3, 3), pbc=(False, False, False)) == [[(0, 0, 0)], [(2, 0, 0)]]
    split = merge_adjacent_blocks([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], max_merged_blocks=2, grid_shape=(5, 1, 1))
    assert split == [[(0, 0, 0), (1, 0, 0)], [(2, 0, 0), (3, 0, 0)]]


def test_block_roles_trimming_and_policies() -> None:
    frame = frame_of(bulk("Cu", "fcc", a=3.61, cubic=True).repeat(6))  # 21.7 A
    config = ExtractionConfig(block={"size": [5.4, 5.4, 5.4], "max_atoms": 100})
    region = extract_block_regions(frame, [0], config)[0]
    core = region.indices_with(RegionRole.CORE)
    assert len(core) > 0 and region.n_atoms == 100
    assert sum(len(v) for v in region.metadata["trimmed_atom_ids"].values()) > 0
    assert region.frozen.tolist() == [role is RegionRole.BOUNDARY for role in region.roles]
    assert region.indices_with(RegionRole.OUTER_BUFFER).size == 0  # default radii: the 3-5 A shell is frozen
    np.testing.assert_allclose(region.force_weights[region.indices_with(RegionRole.INNER_BUFFER)], 0.3)  # F-18
    with pytest.raises(ExtractionError, match="max_atoms_exceeded"):
        extract_block_regions(frame, [0], ExtractionConfig(block={"size": [5.4, 5.4, 5.4], "max_atoms": 5}))
    with pytest.raises(ExtractionError, match="min_atoms_not_met"):
        extract_block_regions(frame, [0], ExtractionConfig(block={"size": [5.4, 5.4, 5.4], "min_atoms": 10000, "min_atoms_policy": "error"}))
    with pytest.raises(ValueError):
        ExtractionConfig(mode="block")


def test_block_cooldown_skips_and_state_survives_restart(tmp_path) -> None:
    frame = frame_of(bulk("Cu", "fcc", a=3.61, cubic=True).repeat(6), step=100)
    config = ExtractionConfig(block={"size": [5.4, 5.4, 5.4], "cooldown_steps": 50})
    tracker = BlockCooldownTracker(cooldown_steps=50)
    assert len(extract_block_regions(frame, [0], config, cooldown=tracker)) == 1
    assert extract_block_regions(frame, [0], config, cooldown=tracker) == []
    tracker.save(tmp_path / "cooldown.json")
    restored = BlockCooldownTracker.load(tmp_path / "cooldown.json", cooldown_steps=50)
    assert restored.to_dict() == tracker.to_dict()
    later = frame_of(frame.to_atoms(), step=151)
    assert extract_block_regions(later, [0], config, cooldown=restored) != []


def test_cooldown_replay_and_prune() -> None:
    tracker = BlockCooldownTracker(cooldown_steps=10)
    script = [((0, 0, 0), 0, False), ((0, 0, 0), 5, True), ((1, 0, 0), 5, False), ((0, 0, 0), 10, False)]
    for block, step, skip in script:
        assert tracker.should_skip([block], step) is skip
        if not skip:
            tracker.update([block], step)
    tracker.update([(2, 0, 0)], 40)  # 40 % 10 == 0 -> prune entries older than 20 steps
    assert set(tracker.last_labeled_step) == {(2, 0, 0)}
    assert BlockCooldownTracker(cooldown_steps=0).should_skip([(0, 0, 0)], 0) is False


def test_torn_pairs_are_detected_exactly() -> None:
    frame = cu_frame(2)
    pairs = find_torn_pairs(frame, np.array([0, 1]), np.array([[0, 0, 0], [5, 0, 0]]), 3.0)
    assert pairs  # a neighbour placed 5 cells away is torn
    assert find_torn_pairs(frame, np.array([0, 1]), np.zeros((2, 3), dtype=int), 0.5) == []
