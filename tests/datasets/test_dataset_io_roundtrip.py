"""Serialisation round trips: labels, masks, roles and provenance survive exactly."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gmd_active_learning.contracts import Frame
from gmd_active_learning.core.exceptions import DataContractError, UnsupportedCombinationError
from gmd_active_learning.datasets import (
    build_sample,
    read_frames,
    read_samples_extxyz,
    read_samples_npz,
    write_dataset_entry,
    write_frames,
    write_samples_extxyz,
    write_samples_npz,
)

from datasets.helpers import ethanol_frame, label_for, periodic_slab_region, sample_for


def assert_same_sample(a, b, *, metadata: bool) -> None:
    assert a.sample_id == b.sample_id and a.split_group == b.split_group and a.task_id == b.task_id
    for name in ("numbers", "positions", "parent_atom_ids", "image_shifts", "force_weights", "frozen", "region_codes"):
        np.testing.assert_array_equal(getattr(a.region, name), getattr(b.region, name), err_msg=name)
    assert a.region.roles == b.region.roles and a.region.pbc == b.region.pbc
    if a.region.cell is None:
        assert b.region.cell is None
    else:
        np.testing.assert_array_equal(a.region.cell, b.region.cell)
    np.testing.assert_array_equal(a.forces, b.forces)
    assert a.energy == b.energy and a.energy_weight == b.energy_weight and a.stress_weight == b.stress_weight
    if a.stress is None:
        assert b.stress is None
    else:
        np.testing.assert_array_equal(a.stress, b.stress)
    assert a.provenance() == b.provenance()
    if metadata:
        assert a.region.metadata == json.loads(json.dumps(b.region.metadata)) and a.metadata == b.metadata


def mixed_samples():
    _frame, slab = periodic_slab_region()
    return [sample_for(0), sample_for(1, event_id="evt-2"), build_sample(slab, label_for(slab, stress=True))]


def test_extxyz_round_trip_is_exact(tmp_path) -> None:
    samples = mixed_samples()
    back = read_samples_extxyz(write_samples_extxyz(samples, tmp_path / "s.extxyz"))
    assert len(back) == len(samples)
    for a, b in zip(samples, back):
        assert_same_sample(a, b, metadata=False)


def test_canonical_npz_round_trip_is_exact_including_metadata(tmp_path) -> None:
    samples = mixed_samples()
    npz, sidecar = write_samples_npz(samples, tmp_path / "s.npz")
    assert sidecar.name == "s.json"
    for a, b in zip(samples, read_samples_npz(npz)):
        assert_same_sample(a, b, metadata=True)
    with np.load(npz, allow_pickle=False) as data:  # pickle-free
        assert "positions" in data.files


def test_per_sample_triple_keeps_phal_keys(tmp_path) -> None:
    sample = sample_for(0)
    paths = write_dataset_entry(sample, output_dir=tmp_path, prefix="region000")
    with np.load(paths["labels"]) as data:
        assert {"forces", "mask", "original_indices", "core_indices", "inner_buffer_indices", "outer_buffer_indices", "boundary_indices", "h_cap_indices"} <= set(data.files)
        np.testing.assert_array_equal(data["mask"], sample.region.force_weights)
        np.testing.assert_array_equal(data["original_indices"], sample.region.parent_atom_ids)
        assert data["h_cap_indices"].size == sample.region.roles.count(sample.region.roles[-1])
    meta = json.loads(paths["metadata"].read_text())
    assert meta["energy_weight"] == 0.0 and meta["energy"] == sample.energy
    assert meta["masked_atom_indices"] == np.where(sample.region.force_weights <= 0)[0].tolist()
    assert meta["provenance"]["energy_is_parent_energy"] is False
    assert_same_sample(sample, read_samples_extxyz(paths["structure"])[0], metadata=False)


def test_frame_round_trip_steps_and_external_forces(tmp_path) -> None:
    frames = []
    for step in (10, 20, 30):
        frame = ethanol_frame(step)
        rng = np.random.default_rng(step)
        frames.append(Frame(run_id="run-a", step=step, atom_ids=np.arange(100, 100 + frame.n_atoms), numbers=frame.numbers,
                            positions=frame.positions + rng.normal(scale=0.01, size=frame.positions.shape), cell=frame.cell,
                            pbc=frame.pbc, forces=rng.normal(size=frame.positions.shape), velocities=rng.normal(size=frame.positions.shape),
                            time_fs=step * 0.5, energy=-1.0 * step))
    path = write_frames(frames, tmp_path / "traj.extxyz")
    back = read_frames(path, run_id="ignored")
    for a, b in zip(frames, back):
        assert b.step == a.step and b.run_id == "run-a" and b.time_fs == a.time_fs and b.energy == a.energy
        for name in ("atom_ids", "positions", "forces", "velocities", "cell"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    external = np.stack([np.full((frames[0].n_atoms, 3), float(k)) for k in range(3)])
    last = read_frames(path, run_id="x", index="-1", external_forces=external)[0]
    assert last.step == 30 and np.all(last.forces == 2.0)  # aligned with the source frame, not post-slice
    np.save(tmp_path / "f.npy", external[1])
    assert np.all(read_frames(path, run_id="x", index=1, external_forces=tmp_path / "f.npy")[0].forces == 1.0)
    with pytest.raises(DataContractError):
        read_frames(path, run_id="x", external_forces=external[1])
    with pytest.raises(UnsupportedCombinationError):
        read_frames(tmp_path / "dump.lammpstrj", run_id="x")
