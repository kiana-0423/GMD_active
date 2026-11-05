"""Region codes, force-weight masks, and the energy / stress policies of labeled samples."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gmd_active_learning.contracts import REGION_CODES, LabelResult, RegionRole, ValidationStatus
from gmd_active_learning.core.exceptions import DataContractError
from gmd_active_learning.datasets import (
    REGION_CODE_MAP,
    DatasetsConfig,
    RoleWeights,
    build_sample,
    force_weights_for_roles,
    region_codes_for_labels,
    region_label_map_json,
    reweight,
)
from gmd_active_learning.datasets.masks import roles_from_codes, roles_from_labels

from datasets.helpers import capped_cluster, label_for, periodic_slab_region


def test_region_codes_and_label_map_are_phal_byte_identical() -> None:
    assert REGION_CODE_MAP == {"core": 0, "label_core": 0, "inner_buffer": 1, "outer_buffer": 2, "boundary": 3, "frozen_boundary": 3, "h_cap": 4}
    assert region_label_map_json() == '{"0": "core", "1": "inner_buffer", "2": "outer_buffer", "3": "boundary", "4": "h_cap"}'
    assert region_codes_for_labels(["core", "unassigned", "frozen_boundary", "h_cap"]).tolist() == [0, -1, 3, 4]
    assert all(REGION_CODES[role] == code for role, code in zip(roles_from_codes([0, 1, 2, 3, 4, -1]), [0, 1, 2, 3, 4, -1]))
    assert roles_from_labels(["label_core", "frozen_boundary"]) == [RegionRole.CORE, RegionRole.BOUNDARY]
    with pytest.raises(DataContractError):
        roles_from_codes([7])


def test_force_weights_follow_roles_with_a_single_inner_buffer_default() -> None:
    roles = [RegionRole.CORE, RegionRole.INNER_BUFFER, RegionRole.OUTER_BUFFER, RegionRole.BOUNDARY, RegionRole.H_CAP, RegionRole.UNASSIGNED]
    assert force_weights_for_roles(roles).tolist() == [1.0, 0.3, 0.0, 0.0, 0.0, 0.0]  # F-18: 0.3, not block's 0.2
    _frame, region = capped_cluster()
    heavy = reweight(region, RoleWeights(inner_buffer=0.5, boundary=0.1))
    assert heavy.roles == region.roles
    np.testing.assert_array_equal(heavy.force_weights, force_weights_for_roles(region.roles, RoleWeights(inner_buffer=0.5, boundary=0.1)))


def test_cluster_energy_is_written_but_not_trained_by_default() -> None:
    _frame, region = capped_cluster()
    sample = build_sample(region, label_for(region), DatasetsConfig())
    assert sample.energy == -42.5 and sample.energy_weight == 0.0
    assert sample.metadata["energy_policy"] == "capped_region"
    assert sample.provenance()["energy_scope"] == "region" and sample.provenance()["energy_is_parent_energy"] is False
    trained = build_sample(region, label_for(region), DatasetsConfig(energy={"nonperiodic_weight": 1.0, "allow_capped": True}))
    assert trained.energy_weight == 1.0
    with pytest.raises(DataContractError, match="no energy"):
        build_sample(region, label_for(region, energy=None), DatasetsConfig(energy={"nonperiodic_weight": 1.0, "allow_capped": True}))
    no_energy = build_sample(region, label_for(region), DatasetsConfig(energy={"write_energy": False}))
    assert no_energy.energy is None and no_energy.energy_weight == 0.0


def test_stress_only_for_labeled_periodic_regions() -> None:
    _frame, cluster = capped_cluster()
    cluster_sample = build_sample(cluster, label_for(cluster, stress=True))
    assert cluster_sample.stress is None and cluster_sample.stress_weight == 0.0
    assert cluster_sample.metadata["stress_policy"] == "nonperiodic_region"
    _frame, slab = periodic_slab_region()
    assert any(slab.pbc)
    slab_sample = build_sample(slab, label_for(slab, stress=True), DatasetsConfig(energy={"periodic_weight": 0.5}))
    assert slab_sample.stress is not None and slab_sample.stress_weight == 1.0 and slab_sample.energy_weight == 0.5
    off = build_sample(slab, label_for(slab, stress=True), DatasetsConfig(stress={"enabled": False}))
    assert off.stress is None and off.stress_weight == 0.0
    unlabeled = build_sample(slab, label_for(slab))
    assert unlabeled.stress is None and unlabeled.stress_weight == 0.0


def test_only_valid_matching_labels_become_samples() -> None:
    _frame, region = capped_cluster()
    invalid = LabelResult(task_id="t", region_id=region.region_id, forces=np.zeros((region.n_atoms, 3)), energy=None, converged=False, validation=ValidationStatus.INVALID)
    with pytest.raises(DataContractError, match="VALID"):
        build_sample(region, invalid)
    wrong = label_for(region)
    wrong.region_id = "region-other"
    with pytest.raises(DataContractError, match="belongs to region"):
        build_sample(region, wrong)
    short = LabelResult(task_id="t", region_id=region.region_id, forces=np.zeros((region.n_atoms - 1, 3)), energy=1.0, converged=True, validation=ValidationStatus.VALID)
    with pytest.raises(DataContractError, match="forces"):
        build_sample(region, short)


def test_sample_identity_and_split_groups() -> None:
    _frame, region = capped_cluster(event_id="evt-9")
    a = build_sample(region, label_for(region, "task-1"))
    b = build_sample(region, label_for(region, "task-1"))
    c = build_sample(region, label_for(region, "task-2"))
    assert a.sample_id == b.sample_id != c.sample_id
    assert a.split_group == "event:evt-9"
    _frame, no_event = capped_cluster(event_id=None)
    assert build_sample(no_event, label_for(no_event)).split_group == "run:run-a"
    assert build_sample(region, label_for(region), DatasetsConfig(split={"group_by": "region"})).split_group == f"region:{region.region_id}"
    json.dumps(a.provenance())
