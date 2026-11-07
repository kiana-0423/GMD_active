"""Vacuum-padded periodic MD box: receptive field, image check (pass and fail), unwrapping, frame guard."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from tests.experiments.alkane_hotspots import _helpers  # noqa: F401  (puts the repository root on sys.path)
from gmd_active_learning.contracts import Frame
from gmd_active_learning.core.exceptions import ConfigurationError, UnsupportedCombinationError
from tools.experiments.alkane_hotspots.geometry import AlkaneTopology, build_all_trans, rotation_matrix
from tools.experiments.alkane_hotspots.md_box import (
    ImageGuard,
    ImageInteractionError,
    MDBoxConfig,
    build_box,
    diameter,
    force_receptive_field,
    min_image_distance,
    unwrap_molecule,
    write_gmd_xyz,
)

MODEL = {"n_blocks": 2, "local_cutoff": 4.0, "long_range_type": "none"}


def _chain(n: int = 16) -> tuple[AlkaneTopology, np.ndarray]:
    topology = AlkaneTopology(n)
    return topology, build_all_trans(n)


def test_receptive_field_is_twice_the_energy_field() -> None:
    field = force_receptive_field(MODEL)
    assert field.radius == 16.0 and field.energy_radius == 8.0
    assert force_receptive_field(MODEL, override=20.0).radius == 20.0
    with pytest.raises(ConfigurationError, match="below the nominal"):
        force_receptive_field(MODEL, override=10.0)
    with pytest.raises(UnsupportedCombinationError, match="long_range_type"):
        force_receptive_field({**MODEL, "long_range_type": "invariant_attention"})
    with pytest.raises(ConfigurationError, match="n_blocks"):
        force_receptive_field({"local_cutoff": 4.0})


def _brute_force(positions: np.ndarray, lengths: np.ndarray, reach: int = 2) -> float:
    best = np.inf
    for shift in itertools.product(range(-reach, reach + 1), repeat=3):
        if shift == (0, 0, 0):
            continue
        vector = np.asarray(shift) * lengths
        for a in positions:
            best = min(best, float(np.min(np.linalg.norm(positions + vector - a, axis=1))))
    return best


def test_min_image_distance_matches_brute_force() -> None:
    rng = np.random.default_rng(3)
    positions = rng.uniform(0, 6, size=(12, 3))
    lengths = np.array([9.0, 11.0, 7.5])
    assert min_image_distance(positions, lengths) == pytest.approx(_brute_force(positions, lengths))


def test_derived_box_passes_for_any_rotation() -> None:
    topology, positions = _chain(16)
    box = build_box(positions, force_receptive_field(MODEL), MDBoxConfig(margin=4.0, stretch_allowance=1.0))
    assert box.lengths[0] >= diameter(positions) + 16.0 + 4.0
    assert box.min_image_distance > 20.0
    payload = box.to_dict()
    assert payload["receptive_field"]["force_radius"] == 16.0 and "isolated" in payload["boundary_conditions"]["reference"]
    rng = np.random.default_rng(0)
    for _ in range(5):
        rotation = rotation_matrix(rng.normal(size=3), float(rng.uniform(0, 2 * np.pi)))
        rotated = (box.positions - box.positions.mean(0)) @ rotation.T + rng.uniform(0, box.lengths[0], size=3)
        assert min_image_distance(rotated, np.asarray(box.lengths)) >= box.lengths[0] - box.diameter > 20.0


def test_too_small_box_is_refused() -> None:
    _, positions = _chain(16)
    with pytest.raises(ImageInteractionError, match="receptive field"):
        build_box(positions, force_receptive_field(MODEL), MDBoxConfig(box_length=30.0))


def test_unwrap_restores_a_molecule_split_by_the_boundary() -> None:
    topology, positions = _chain(8)
    lengths = np.array([12.0, 12.0, 12.0])
    shifted = positions - positions.mean(0) + np.array([0.5, 6.0, 6.0])  # crosses x = 0
    wrapped = np.mod(shifted, lengths)
    assert np.ptp(wrapped[:, 0]) > 10.0
    whole = unwrap_molecule(wrapped, lengths, topology.bonds)
    np.testing.assert_allclose(whole - whole[0], shifted - shifted[0], atol=1e-10)


def _frame(positions: np.ndarray, length: float, topology: AlkaneTopology, step: int = 0) -> Frame:
    return Frame(run_id="r", step=step, atom_ids=np.arange(topology.n_atoms), numbers=topology.numbers, positions=np.mod(positions, length),
                 cell=np.eye(3) * length, pbc=(True, True, True))


class _Inner:
    def __init__(self) -> None:
        self.calls = 0

    def enrich(self, frame: Frame, model: object) -> Frame:
        self.calls += 1
        return frame


def test_image_guard_passes_records_and_delegates(tmp_path: Path) -> None:
    topology, positions = _chain(16)
    box = build_box(positions, force_receptive_field(MODEL), MDBoxConfig())
    inner = _Inner()
    guard = ImageGuard(inner, bonds=topology.bonds, receptive_field=force_receptive_field(MODEL), margin=4.0, stats_path=tmp_path / "s.json")
    guard.enrich(_frame(box.positions, box.lengths[0], topology, step=10), model=None)
    stats = guard.flush()
    assert inner.calls == 1 and stats["n_frames_checked"] == 1 and stats["min_image_distance"] > 20.0
    assert stats["min_at"]["step"] == 10


def test_image_guard_refuses_interacting_images(tmp_path: Path) -> None:
    topology, positions = _chain(16)
    inner = _Inner()
    guard = ImageGuard(inner, bonds=topology.bonds, receptive_field=force_receptive_field(MODEL), margin=4.0, stats_path=tmp_path / "s.json")
    with pytest.raises(ImageInteractionError, match="periodic images interact"):
        guard.enrich(_frame(positions - positions.mean(0) + 15.0, 30.0, topology), model=None)
    assert inner.calls == 0  # nothing is evaluated on an invalid frame
    disabled = ImageGuard(inner, bonds=topology.bonds, receptive_field=force_receptive_field(MODEL), margin=4.0,
                          stats_path=tmp_path / "d.json", enabled=False)
    disabled.enrich(_frame(positions, 30.0, topology), model=None)
    assert inner.calls == 1


def test_image_guard_records_margin_warnings_with_run_id(tmp_path: Path) -> None:
    topology, positions = _chain(16)
    rf = force_receptive_field(MODEL)
    centred = positions - positions.mean(0)
    length = next(L for L in np.arange(20.0, 60.0, 0.25)
                  if rf.radius < min_image_distance(centred + L / 2, np.array([L, L, L])) <= rf.radius + 4.0)
    guard = ImageGuard(None, bonds=topology.bonds, receptive_field=rf, margin=4.0, stats_path=tmp_path / "s.json")
    distance = guard.check(_frame(centred + length / 2, float(length), topology, step=40))
    stats = guard.flush()
    assert stats["below_margin"] == [{"run_id": "r", "step": 40, "distance": distance}]


def test_gmd_xyz_format(tmp_path: Path) -> None:
    topology, positions = _chain(4)
    path = write_gmd_xyz(tmp_path / "in.xyz", topology.numbers, positions, (30.0, 30.0, 30.0))
    lines = path.read_text().splitlines()
    assert lines[0] == str(topology.n_atoms) and lines[1].split() == ["30.000000"] * 3
    assert lines[2].startswith("C ") and lines[-1].startswith("H ")
    from gmd_active_learning.adapters.gmd.engine import structure_species

    assert structure_species(path) == topology.numbers.tolist()
