from __future__ import annotations

import numpy as np
import pytest
from ase.build import molecule

from gmd_active_learning.contracts import (
    ExecutionStatus,
    ExtractedRegion,
    Frame,
    LabelResult,
    MetricStatus,
    MetricValues,
    RegionRole,
    ValidationStatus,
    content_hash,
    stable_id,
)
from gmd_active_learning.core.exceptions import DataContractError


def _frame(**overrides: object) -> Frame:
    kwargs: dict[str, object] = dict(
        run_id="run",
        step=10,
        atom_ids=[7, 3, 9],
        numbers=[8, 1, 1],
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=np.eye(3) * 10.0,
        pbc=(True, True, True),
        forces=np.zeros((3, 3)),
    )
    kwargs.update(overrides)
    return Frame(**kwargs)  # type: ignore[arg-type]


def test_frame_validates_identity_species_and_shapes() -> None:
    frame = _frame()
    assert frame.symbols == ["O", "H", "H"]
    assert frame.index_of([9, 7]).tolist() == [2, 0]
    with pytest.raises(DataContractError):
        _frame(atom_ids=[1, 1, 2])
    with pytest.raises(DataContractError):
        _frame(numbers=[8, 0, 1])  # Z = 0 is an unknown species, not padding
    with pytest.raises(DataContractError):
        _frame(forces=np.zeros((2, 3)))
    with pytest.raises(DataContractError):
        _frame(cell=None)  # periodic frame without a cell
    with pytest.raises(DataContractError):
        _frame(positions=[[np.nan, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_frame_round_trips_through_ase() -> None:
    frame = _frame(committee_forces=np.zeros((4, 3, 3)))
    back = Frame.from_atoms(frame.to_atoms(), run_id="run")
    assert back.step == 10
    assert back.atom_ids.tolist() == [7, 3, 9]
    np.testing.assert_array_equal(back.positions, frame.positions)


def test_unavailable_metric_is_not_zero() -> None:
    unavailable = MetricValues.unavailable("committee", 3, "no committee configured")
    measured_zero = MetricValues.measured("committee", np.zeros(3))
    assert unavailable.status is MetricStatus.UNAVAILABLE and unavailable.max() is None
    assert measured_zero.status is MetricStatus.MEASURED and measured_zero.max() == 0.0
    lazy = MetricValues.measured("lj", np.array([0.5, 9.0, 0.1]), evaluated=np.array([True, False, True]))
    assert lazy.max() == 0.5
    assert lazy.to_dict()["values"] == [0.5, None, 0.1]
    with pytest.raises(DataContractError):
        MetricValues("x", np.zeros(2), np.ones(2, bool), MetricStatus.UNAVAILABLE)


def _region() -> ExtractedRegion:
    mol = molecule("CH4")
    return ExtractedRegion(
        region_id="reg-1",
        mode="cluster",
        run_id="run",
        parent_step=10,
        event_id="evt-1",
        numbers=mol.numbers,
        positions=mol.positions,
        cell=None,
        pbc=(False, False, False),
        parent_atom_ids=[10, 11, 12, 13, 14],
        image_shifts=np.zeros((5, 3), dtype=int),
        roles=[RegionRole.CORE, RegionRole.INNER_BUFFER, RegionRole.OUTER_BUFFER, RegionRole.BOUNDARY, RegionRole.BOUNDARY],
        force_weights=[1.0, 0.3, 0.0, 0.0, 0.0],
        frozen=[False] * 5,
        hotspot_atom_ids=[10],
    )


def test_region_caps_extend_every_per_atom_array_together() -> None:
    region = _region()
    capped = region.with_caps(np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), [{"parent": 13}, {"parent": 14}])
    assert capped.n_atoms == 7
    for array in (capped.parent_atom_ids, capped.image_shifts, capped.roles, capped.force_weights, capped.frozen):
        assert len(array) == 7
    assert capped.region_codes.tolist() == [0, 1, 2, 3, 3, 4, 4]
    assert capped.parent_atom_ids[-2:].tolist() == [-1, -1]
    assert capped.force_weights[-2:].tolist() == [0.0, 0.0]
    atoms = capped.to_atoms()
    assert atoms.arrays["region_code"].shape == (7,)
    assert ExtractedRegion.from_dict(capped.to_dict()).to_dict() == capped.to_dict()


def test_region_rejects_inconsistent_arrays() -> None:
    data = _region().to_dict()
    data["force_weights"] = data["force_weights"][:-1]
    with pytest.raises(DataContractError):
        ExtractedRegion.from_dict(data)
    data = _region().to_dict()
    data["force_weights"][0] = -1.0
    with pytest.raises(DataContractError):
        ExtractedRegion.from_dict(data)


def test_label_result_validity_requires_convergence_and_finiteness() -> None:
    with pytest.raises(DataContractError):
        LabelResult("t", "r", np.zeros((2, 3)), -1.0, converged=False, validation=ValidationStatus.VALID)
    with pytest.raises(DataContractError):
        LabelResult("t", "r", np.full((2, 3), np.nan), -1.0, converged=True, validation=ValidationStatus.VALID)
    invalid = LabelResult("t", "r", np.zeros((2, 3)), None, converged=False, validation=ValidationStatus.INVALID)
    assert invalid.validation is ValidationStatus.INVALID
    assert ExecutionStatus.COMPLETED.terminal and not ExecutionStatus.UNKNOWN.terminal


def test_identifiers_are_deterministic() -> None:
    assert stable_id("evt", "run", 50) == stable_id("evt", "run", 50)
    assert stable_id("evt", "run", 50) != stable_id("evt", "run", 60)
    array = np.arange(6.0).reshape(2, 3)
    assert content_hash({"a": array}) == content_hash({"a": array.copy()})
    assert content_hash({"a": array}) != content_hash({"a": array + 1e-12})
