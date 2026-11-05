"""Real GMD_SGT training/export through the adapter (no fakes). Runs when GMD_SGT >= 0.4 is
importable (installed, or via GMD_SGT_PATH / the sibling checkout)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from gmd_active_learning.adapters.sgt.backend import GMDSGTBackend, SGTConfig
from gmd_active_learning.contracts import DatasetVersion, Frame
from gmd_active_learning.core.exceptions import CapabilityUnavailableError
from gmd_active_learning.retraining.registry import ModelRegistry
from gmd_active_learning.workflow.store import WorkflowStore

SGT_PATH = os.environ.get("GMD_SGT_PATH", str(Path(__file__).resolve().parents[4] / "GMD_SGT"))
TINY = {
    "model": {"n_species": 20, "n_blocks": 1, "scalar_dim": 8, "irreps": "8x0e", "n_basis": 4, "l_max": 0,
              "local_cutoff": 5.0, "lr_cutoff": 5.0, "long_range_type": "none", "n_heads": 1, "avg_neighbors": 8.0},
    "train": {"n_epochs": 4, "batch_size": 4, "warmup_steps": 1, "lr": 3.0e-3, "patience": 0},
}


def _backend(**overrides) -> GMDSGTBackend:
    try:
        return GMDSGTBackend(SGTConfig(source_path=SGT_PATH, train=TINY, **overrides))
    except CapabilityUnavailableError as exc:
        pytest.skip(f"GMD_SGT unavailable: {exc}")


def _labeled(positions: np.ndarray, *, cell=None, pbc=False, core: int | None = None, group: str = "g") -> Atoms:
    atoms = Atoms("Ar" * len(positions), positions=positions, cell=cell, pbc=pbc)
    calc = LennardJones(epsilon=0.0104, sigma=3.4, rc=8.0, smooth=True)
    atoms.calc = calc
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    weights = np.ones(len(atoms)) if core is None else np.where(np.arange(len(atoms)) == core, 1.0, 0.0)
    atoms.arrays["force_weight"] = weights
    atoms.info["energy_weight"] = 0.0 if not pbc else 1.0  # cluster energies excluded, as the dataset policy does
    atoms.info["split_group"] = group
    return atoms


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> DatasetVersion:
    root = tmp_path_factory.mktemp("ds")
    rng = np.random.default_rng(0)
    grid = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)], float) * 3.8
    structures = []
    for n in range(16):
        clusters = _labeled(grid + rng.normal(0, 0.12, grid.shape), core=n % 8, group=f"evt-{n // 4}")
        structures.append(clusters)
    box = np.array([[i, j, k] for i in range(3) for j in range(3) for k in range(3)], float) * 3.8
    for n in range(4):
        structures.append(_labeled(box + rng.normal(0, 0.1, box.shape), cell=np.eye(3) * 11.4, pbc=True, group=f"bulk-{n}"))
    train = [s for s in structures if s.info["split_group"] not in {"evt-3", "bulk-3"}]
    val = [s for s in structures if s.info["split_group"] in {"evt-3", "bulk-3"}]
    write(root / "train.extxyz", train, format="extxyz")
    write(root / "val.extxyz", val, format="extxyz")
    return DatasetVersion("ds-test", str(root), "sha", len(structures), len(train), len(val), None, (), "now",
                          {"train_file": str(root / "train.extxyz"), "val_file": str(root / "val.extxyz")})


def test_train_evaluate_export_validate_and_finetune(dataset: DatasetVersion, tmp_path: Path) -> None:
    backend = _backend(committee_size=1)
    output = backend.train(dataset, parent=None, work_dir=tmp_path / "train0")
    assert output.checkpoint.name == "ckpt_best.pt" and output.checkpoint.is_file()
    assert output.species == [18] and output.cutoff == pytest.approx(5.0) and output.mode == "fresh"
    assert len(output.summary["committee_checkpoints"]) == 1
    metrics = backend.evaluate(output.checkpoint, dataset)
    assert metrics["n_validation"] == 5 and np.isfinite(metrics["force_rmse"])
    export = backend.export(output.checkpoint, tmp_path / "export")
    check = backend.validate_export(output.checkpoint, export, dataset)
    assert check["n_structures"] == 5 and check["max_force_deviation"] < 1e-4
    assert check["species"] == [18]

    # register (with the committee member) and fine-tune from it: the new run must start fresh
    store = WorkflowStore(tmp_path / "s.sqlite")
    store.ensure_run("r", config_hash="c", initial_model_id=None, stage="sampling")
    registry = ModelRegistry(tmp_path / "models", store)
    parent = registry.register(checkpoint=output.checkpoint, export=export, dataset_version=dataset.version, parent_model_id=None,
                               species=output.species, cutoff=output.cutoff, metrics=metrics, train_mode=output.mode, backend="gmd_sgt",
                               extra_files={"committee_1": output.summary["committee_checkpoints"][0]})
    finetuned = backend.train(dataset, parent=parent, work_dir=tmp_path / "train1")
    summary = json.loads((tmp_path / "train1" / "member_0" / "training_summary.json").read_text())
    assert finetuned.mode == "finetune" and summary["start_checkpoint"]["sha256"] == parent.checkpoint_sha256
    assert summary["best_checkpoint"] != str(parent.checkpoint_path)
    # idempotent: re-running the same training directory reuses the finished run
    assert backend.train(dataset, parent=parent, work_dir=tmp_path / "train1").checkpoint == finetuned.checkpoint

    # committee enrichment of an MD frame uses the registered members
    frame = Frame(run_id="r", step=0, atom_ids=np.arange(8), numbers=np.full(8, 18),
                  positions=np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)], float) * 3.8,
                  cell=None, pbc=(False, False, False), forces=np.zeros((8, 3)))
    enriched = backend.enrich(frame, parent)
    assert enriched.committee_forces is not None and enriched.committee_forces.shape == (2, 8, 3)
    assert np.isfinite(enriched.committee_forces).all()


def test_dry_run_or_missing_dataset_files_are_refused(dataset: DatasetVersion, tmp_path: Path) -> None:
    backend = _backend()
    broken = DatasetVersion("ds-x", "x", "sha", 1, 1, 0, None, (), "now", {"train_file": str(tmp_path / "nope.extxyz"), "val_file": dataset.metadata["val_file"]})
    with pytest.raises(Exception, match="no readable train_file"):
        backend.train(broken, parent=None, work_dir=tmp_path / "t")
