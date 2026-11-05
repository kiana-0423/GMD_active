"""Optional randomized comparison against PHAL itself; skipped once ref/lammps_monitor is removed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from gmd_active_learning.extraction import BlockCooldownTracker, assign_radial_regions, merge_adjacent_blocks, region_label_vector
from gmd_active_learning.extraction.block import assign_atoms_to_spatial_blocks

from extraction.test_extraction_modes import frame_of

PHAL_SRC = Path(__file__).resolve().parents[2] / "ref" / "lammps_monitor" / "src"
pytestmark = pytest.mark.skipif(not (PHAL_SRC / "hotspot_al").is_dir(), reason="PHAL reference tree removed")


@pytest.fixture(scope="module")
def phal():
    sys.path.insert(0, str(PHAL_SRC))
    try:
        from hotspot_al.extraction import block, boundary_regions

        yield block, boundary_regions
    finally:
        sys.path.remove(str(PHAL_SRC))


def test_radial_partition_matches_phal_on_random_distances(phal) -> None:
    _block, boundary = phal
    rng = np.random.default_rng(1)
    for _ in range(200):
        distances = rng.uniform(0, 12, size=50)
        core, extract, thick = rng.uniform(0, 6), rng.uniform(0, 12), rng.uniform(0, 4)
        mine = assign_radial_regions(distances, core_radius=core, extract_radius=extract, boundary_thickness=thick)
        ref = boundary.assign_radial_regions(distances, core_radius=core, extract_radius=extract, boundary_thickness=thick)
        assert mine == ref
        assert region_label_vector(50, mine) == boundary.region_label_vector(50, ref)


def test_block_assignment_and_merging_match_phal(phal) -> None:
    block, _ = phal
    rng = np.random.default_rng(2)
    for pbc in [(True, True, True), (True, True, False), (False, False, False)]:
        atoms = bulk("Cu", "fcc", a=3.61).repeat(5)  # triclinic primitive cell
        atoms.pbc = pbc
        atoms.positions += rng.normal(scale=0.3, size=atoms.positions.shape)
        frame = frame_of(atoms)
        size = (3.0, 4.0, 5.0)
        assert assign_atoms_to_spatial_blocks(frame, size) == block.assign_atoms_to_spatial_blocks(atoms, size)
        grid = block.block_grid_shape(atoms, size)
        for _ in range(20):
            ids = [tuple(int(v) for v in rng.integers(0, grid)) for _ in range(rng.integers(1, 8))]
            for limit in (None, 2):
                assert merge_adjacent_blocks(ids, max_merged_blocks=limit, grid_shape=grid, pbc=pbc) == block.merge_adjacent_blocks(
                    ids, max_merged_blocks=limit, grid_shape=grid, pbc=pbc
                )


def test_cooldown_sequence_matches_phal(phal) -> None:
    block, _ = phal
    rng = np.random.default_rng(3)
    mine, ref = BlockCooldownTracker(25), block.BlockCooldownTracker(25)
    for step in range(0, 400, 5):
        group = [tuple(int(v) for v in rng.integers(0, 3, size=3)) for _ in range(2)]
        assert mine.should_skip(group, step) == ref.should_skip(group, step)
        if not ref.should_skip(group, step):
            mine.update(group, step)
            ref.update(group, step)
        assert mine.last_labeled_step == ref.last_labeled_step
