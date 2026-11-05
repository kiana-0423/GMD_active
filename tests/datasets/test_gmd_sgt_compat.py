"""Published datasets are read by GMD_SGT's own reader, and its weighted loss matches the NumPy oracle.

Runs when GMD_SGT is importable (installed, ``GMD_SGT_PATH``, or the sibling checkout). The weight
round trip needs GMD_SGT >= 0.4 (``force_weight`` support); older versions are skipped with a reason.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from gmd_active_learning.datasets import DatasetsConfig, VersionedDatasetStore, build_sample, export_for_gmd_sgt
from reference.masked_force_mse import weighted_force_loss

from datasets.helpers import label_for, periodic_slab_region, sample_for

SGT_PATH = os.environ.get("GMD_SGT_PATH", str(Path(__file__).resolve().parents[3] / "GMD_SGT"))


@pytest.fixture(scope="module")
def sgt():
    if SGT_PATH not in sys.path and Path(SGT_PATH).is_dir():
        sys.path.append(SGT_PATH)
    try:
        reader = importlib.import_module("gmd_sgt.data.reader")
        validation = importlib.import_module("gmd_sgt.data.validation")
    except ImportError as exc:
        pytest.skip(f"GMD_SGT not importable: {exc}")
    if not hasattr(validation, "FORCE_WEIGHT_KEY"):
        pytest.skip("this GMD_SGT has no force_weight support yet (needs the al-weighted-training branch / v0.4)")
    return reader


def mixed():
    _frame, slab = periodic_slab_region()
    samples = [sample_for(k, event_id=f"evt-{k % 3}", task=f"t{k}") for k in range(6)]
    samples.append(build_sample(slab, label_for(slab, "slab-task", stress=True), DatasetsConfig(energy={"periodic_weight": 1.0})))
    return samples


def test_gmd_sgt_reads_published_versions_with_all_weights(tmp_path, sgt) -> None:
    store = VersionedDatasetStore(tmp_path / "store")
    samples = mixed()
    version = store.publish([store.write_sample(s) for s in samples], parent=None, split_groups={})
    by_id = {s.sample_id: s for s in samples}
    read_back = []
    for key in ("train_file", "val_file"):
        items = sgt.read_extxyz(version.metadata[key])
        published = store.read_samples(version)
        side = store.split_of(version)["samples"]
        expected = [s for s in published if side[s.sample_id] == key.split("_")[0]]
        assert len(items) == len(expected)
        for item, sample in zip(items, expected):
            np.testing.assert_allclose(item["forces"].numpy(), sample.forces, rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(item["force_weight"].numpy(), sample.region.force_weights.astype(np.float32))
            assert float(item["energy_weight"]) == sample.energy_weight
            assert float(item["stress_weight"]) == sample.stress_weight
            assert item["split_group"] == sample.split_group
            assert ("stress" in item) == (sample.stress is not None)
            assert ("pbc" in item) == any(sample.region.pbc)
            read_back.append(sample.sample_id)
    assert sorted(read_back) == sorted(by_id)
    npz_items = sgt.read_npz(version.metadata["sgt_npz"])
    assert len(npz_items) == len(samples)
    for item, sample in zip(npz_items, store.read_samples(version)):
        np.testing.assert_array_equal(item["force_weight"].numpy(), sample.region.force_weights.astype(np.float32))
        assert float(item["energy_weight"]) == sample.energy_weight and item["split_group"] == sample.split_group


def test_gmd_sgt_weighted_loss_matches_numpy_oracle(tmp_path, sgt) -> None:
    torch = pytest.importorskip("torch")
    data = importlib.import_module("gmd_sgt.data")
    loss_module = importlib.import_module("gmd_sgt.training.loss")
    paths = export_for_gmd_sgt(mixed(), tmp_path / "export")
    items = sgt.read_extxyz(str(paths["all"]))
    batch = data.collate_fn(items)
    target_forces = batch["forces"].double()
    weights = batch["force_weight"].double()
    predicted = (target_forces + 0.1 * torch.randn_like(target_forces)).requires_grad_(True)
    loss = loss_module.EnergyForceLoss(force_loss="rmse")
    for kind in ("rmse", "mae", "mse"):
        value = loss._weighted(kind, predicted, target_forces, weights)
        oracle = weighted_force_loss(predicted.detach().numpy(), target_forces.numpy(), weights.numpy(), kind)
        assert np.isclose(float(value.detach()), oracle, rtol=1e-6)
    value = loss._weighted("rmse", predicted, target_forces, weights)
    (gradient,) = torch.autograd.grad(value, predicted)
    assert torch.all(gradient[weights == 0] == 0)  # masked atoms (caps, boundary) carry no gradient
    assert torch.any(gradient[weights > 0] != 0)
