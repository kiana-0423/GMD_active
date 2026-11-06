"""Training data policy: role-based force weights, energy weight 0 for capped fragments (reason recorded),
full-system energy supervision; zero-weight atoms and structures change neither GMD_SGT's loss nor any
parameter gradient (checked through GMD_SGT's own reader, collate, model and loss)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from .beta_synthetic import import_sgt, make_labels, make_parents

from tools.experiments.alkane_hotspots.arms import PoolConfig, build_pool  # noqa: E402
from tools.experiments.alkane_hotspots.training import (  # noqa: E402
    CAPPED_ENERGY_REASON,
    TrainingConfig,
    sample_for_label,
    write_dataset,
)
from gmd_active_learning.contracts import RegionRole  # noqa: E402
from gmd_active_learning.core.exceptions import DataContractError  # noqa: E402

POOL = {"source": "explicit", "splits": ["validation"], "extraction": {"mode": "cluster", "terminal_atoms": "follow_partner", "core_radius": 1.2, "buffer_radius": 2.0, "boundary_thickness": 1.0, "extract_radius": 3.2}}
WEIGHTS = {"core": 1.0, "inner_buffer": 0.3, "outer_buffer": 0.1, "boundary": 0.0, "h_cap": 0.0}
TINY = {"n_species": 10, "n_blocks": 1, "scalar_dim": 8, "irreps": "8x0e", "n_basis": 4, "l_max": 0, "local_cutoff": 3.0,
        "lr_cutoff": 5.0, "long_range_type": "none", "n_heads": 1, "avg_neighbors": 6.0}


@pytest.fixture()
def data(tmp_path: Path):
    root = tmp_path / "art"
    parents = make_parents(root)
    pool = build_pool(parents, PoolConfig.model_validate(POOL), [root])
    labels = make_labels(root, parents, pool)
    config = TrainingConfig.model_validate({"model": TINY, "force_weights": WEIGHTS, "full_system_force_weight": 1.0,
                                            "energy": {"full_system_weight": 1.0}})
    return root, {p["parent_id"]: p for p in parents}, labels, config


def _pick(labels, kind):
    return next(r for r in labels if r["kind"] == kind)


def test_fragment_and_full_samples_follow_the_policy(data) -> None:
    root, parents, labels, config = data
    fragment = sample_for_label(_pick(labels, "fragment"), parents, config, root)
    roles = fragment.region.roles
    assert RegionRole.H_CAP in roles  # graph fragments of a chain cut C-C bonds -> capped
    expected = np.asarray([WEIGHTS[role.value] if role.value in WEIGHTS else 0.0 for role in roles])
    np.testing.assert_array_equal(fragment.region.force_weights, expected)
    assert fragment.energy_weight == 0.0 and fragment.metadata["energy_weight_reason"] == CAPPED_ENERGY_REASON
    assert fragment.energy == -50.0  # kept as provenance of the fragment itself, never the parent's energy
    full = sample_for_label(_pick(labels, "full"), parents, config, root)
    assert np.all(full.region.force_weights == 1.0) and full.energy_weight == 1.0
    assert full.split_group == parents[full.metadata["parent_id"]]["split_group"]
    with pytest.raises(DataContractError, match="unknown parent"):
        sample_for_label({**_pick(labels, "full"), "parent_id": "nope"}, parents, config, root)
    with pytest.raises(DataContractError, match="status"):
        sample_for_label({**_pick(labels, "full"), "status": "failed"}, parents, config, root)


def test_config_refuses_final_eval_and_fixed_seed() -> None:
    with pytest.raises(ValueError, match="final_eval"):
        TrainingConfig.model_validate({"validation_splits": ["final_eval"]})
    with pytest.raises(ValueError, match="seed"):
        TrainingConfig.model_validate({"train": {"seed": 1}})


def test_dataset_manifest_records_energy_reasons(data, tmp_path: Path) -> None:
    root, parents, labels, config = data
    samples = [sample_for_label(r, parents, config, root) for r in labels[:2] + [_pick(labels, "fragment")]]
    dataset = write_dataset(samples[:2] + samples[2:], samples[:1], tmp_path / "ds", version="v1")
    manifest = json.loads(Path(dataset.metadata["manifest"]).read_text())
    kinds = {item["kind"]: item for item in manifest["train"]}
    assert kinds["fragment"]["energy_weight"] == 0.0 and "capped fragment" in kinds["fragment"]["energy_weight_reason"]
    assert kinds["full"]["energy_weight"] == 1.0
    with pytest.raises(DataContractError, match="validation"):
        write_dataset(samples, [], tmp_path / "ds2", version="v2")


def _loss_and_grads(model, batch, loss_fn):
    import torch

    model.zero_grad()
    pred = model(species=batch["species"], positions=batch["positions"], batch=batch["batch"], cell=batch.get("cell"), compute_forces=True)
    loss, _ = loss_fn(pred, batch, batch.get("n_atoms"))
    loss.backward()
    grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]
    return float(loss.detach()), grads


def test_zero_weights_change_neither_loss_nor_gradients_in_gmd_sgt(data, tmp_path: Path) -> None:
    import_sgt()
    import torch
    from gmd_sgt.data import collate_fn
    from gmd_sgt.data.reader import read_extxyz
    from gmd_sgt.models.factory import instantiate_model
    from gmd_sgt.training.common import build_loss

    root, parents, labels, config = data
    fragment = sample_for_label(_pick(labels, "fragment"), parents, config, root)
    full = sample_for_label(_pick(labels, "full"), parents, config, root)
    dataset = write_dataset([fragment, full], [full], tmp_path / "ds", version="zw")
    items = read_extxyz(dataset.metadata["train_file"])
    batch = collate_fn(items)
    weights = batch["force_weight"]
    assert torch.any(weights == 0) and torch.any((weights > 0) & (weights < 1))
    assert batch["energy_weight"].tolist() == [0.0, 1.0]
    torch.manual_seed(0)
    model = instantiate_model("UnifiedEquivariantMLIP", TINY)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim >= 2 and float(parameter.abs().max()) == 0.0:
                parameter.normal_(0.0, 0.5)
    model = model.double().train()
    batch = {k: (v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in batch.items()}
    loss_fn = build_loss({"w_energy": 1.0, "w_force": 1.0, "force_loss": "rmse", "energy_loss": "mse"})
    reference_loss, reference_grads = _loss_and_grads(model, batch, loss_fn)

    perturbed = dict(batch)
    perturbed["forces"] = batch["forces"].clone()
    perturbed["forces"][weights == 0] += 1.0e3  # caps / boundary: masked
    perturbed["energy"] = batch["energy"].clone()
    perturbed["energy"][0] += 1.0e4  # capped fragment: energy weight 0
    loss, grads = _loss_and_grads(model, perturbed, loss_fn)
    assert loss == reference_loss
    assert all(torch.equal(a, b) for a, b in zip(grads, reference_grads))

    control = dict(batch)
    control["forces"] = batch["forces"].clone()
    control["forces"][torch.nonzero(weights == 1.0)[0]] += 1.0
    loss, grads = _loss_and_grads(model, control, loss_fn)
    assert loss != reference_loss and any(not torch.equal(a, b) for a, b in zip(grads, reference_grads))


def test_backend_training_round_trip(data, tmp_path: Path) -> None:
    """One tiny real GMD_SGT training through GMDSGTBackend (primary + 1 committee member), then prediction."""

    import_sgt()
    from tools.experiments.alkane_hotspots.training import make_backend, predictor_for, train_model

    root, parents, labels, _ = data
    config = TrainingConfig.model_validate({"model": TINY, "train": {"n_epochs": 2, "batch_size": 2, "warmup_steps": 1, "patience": 0},
                                            "sgt": {"source_path": str(Path(__file__).resolve().parents[4] / "GMD_SGT"), "committee_size": 1}})
    train = [sample_for_label(r, parents, config, root) for r in labels if r["kind"] == "full"][:4]
    dataset = write_dataset(train[:3], train[3:], tmp_path / "ds", version="bk")
    backend = make_backend(config)
    model = train_model(backend, dataset, tmp_path / "train", seed=11)
    assert len(model.checkpoints) == 2 and all(Path(p).is_file() for p in model.checkpoints)
    numbers = np.asarray(train[0].region.numbers)
    forces, committee = predictor_for(backend, model)(numbers, train[0].region.positions)
    assert forces.shape == (len(numbers), 3) and committee.shape == (2, len(numbers), 3)
