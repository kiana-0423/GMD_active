"""GMD_SGT inference through the adapter on the tiny seeded fixture models (tests/fixtures/sgt).

Real GMD_SGT, no fakes. Skipped (with the reason) only when GMD_SGT >= 0.4 is not importable
(installed, via GMD_SGT_PATH, or as the sibling checkout). Ledger BE-005, BE-006, BE-007, FIX-002.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from gmd_active_learning.adapters.sgt.backend import (
    GMDSGTBackend,
    SGTConfig,
    check_species,
    validate_species_map,
)
from gmd_active_learning.contracts import UNITS, DatasetVersion, Frame, ModelArtifact
from gmd_active_learning.contracts.ids import file_sha256
from gmd_active_learning.core.exceptions import AdapterError, CapabilityUnavailableError, ConfigurationError

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "tests" / "fixtures" / "sgt"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
SGT_PATH = os.environ.get("GMD_SGT_PATH", str(REPO.parent / "GMD_SGT"))
WATER = read(REPO / "tests" / "fixtures" / "datasets" / "tiny_h2o.extxyz", index=0)


@pytest.fixture(scope="module")
def backend() -> GMDSGTBackend:
    try:
        return GMDSGTBackend(SGTConfig(source_path=SGT_PATH))
    except CapabilityUnavailableError as exc:
        pytest.skip(f"GMD_SGT unavailable: {exc}")


def _generator():  # noqa: ANN202
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_tiny_sgt_models", REPO / "tools" / "fixtures" / "make_tiny_sgt_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _member(index: int, key: str) -> Path:
    return FIXTURES / MANIFEST["members"][index][key]


def _artifact(with_committee: bool = True) -> ModelArtifact:
    extra = {"committee_1": {"path": str(_member(1, "checkpoint")), "sha256": MANIFEST["members"][1]["checkpoint_sha256"]}}
    return ModelArtifact("m-fixture", str(_member(0, "checkpoint")), MANIFEST["members"][0]["checkpoint_sha256"],
                         str(_member(0, "export")), MANIFEST["members"][0]["export_sha256"], None, None, (1, 8), 4.0,
                         dict(UNITS), {}, "fresh", "gmd_sgt", "now", {"extra_files": extra if with_committee else {}})


def test_fixture_files_match_the_manifest() -> None:
    assert len({member["seed"] for member in MANIFEST["members"]}) == len(MANIFEST["members"]) >= 2
    for member in MANIFEST["members"]:
        assert file_sha256(FIXTURES / member["checkpoint"]) == member["checkpoint_sha256"]
        assert file_sha256(FIXTURES / member["export"]) == member["export_sha256"]
    assert sum(path.stat().st_size for path in FIXTURES.rglob("*") if path.is_file()) < 1_000_000


@pytest.mark.parametrize("key", ["checkpoint", "export"])
def test_fixture_models_report_species_cutoff_and_give_finite_h2o_predictions(backend: GMDSGTBackend, key: str) -> None:
    for index in range(len(MANIFEST["members"])):
        info = backend.model_info(_member(index, key))
        assert info.kind == key and info.species == (1, 8) and info.cutoff == pytest.approx(4.0)
        prediction = backend.predict(_member(index, key), WATER.positions, WATER.numbers)
        assert prediction.forces.shape == (3, 3)
        assert np.isfinite(prediction.energy) and np.isfinite(prediction.forces).all()


def test_adapter_equals_gmd_sgt_own_evaluators(backend: GMDSGTBackend) -> None:
    """Checkpoint path vs GMD_SGT's MLIPCalculator, export path vs the TorchScript module itself."""

    import torch
    from gmd_sgt.inference.calculator import MLIPCalculator

    checkpoint, export = _member(0, "checkpoint"), _member(0, "export")
    reference = MLIPCalculator.from_checkpoint(str(checkpoint)).compute(WATER.positions, WATER.numbers, None, False)
    ours = backend.predict(checkpoint, WATER.positions, WATER.numbers)
    np.testing.assert_allclose(ours.forces, reference["forces"], atol=1e-6, rtol=0)
    assert ours.energy == pytest.approx(reference["energy"], abs=1e-6)

    scripted = torch.jit.load(str(export), map_location="cpu")
    n = len(WATER)
    pairs = np.array([(i, j) for i in range(n) for j in range(n) if i != j and np.linalg.norm(WATER.positions[i] - WATER.positions[j]) < 4.0]).T
    out = scripted(torch.as_tensor(WATER.numbers, dtype=torch.int64), torch.as_tensor(WATER.positions, dtype=torch.float32),
                   torch.as_tensor(pairs, dtype=torch.int64), torch.zeros((pairs.shape[1], 3), dtype=torch.float32))
    exported = backend.predict(export, WATER.positions, WATER.numbers)
    np.testing.assert_allclose(exported.forces, out["forces"].detach().numpy(), atol=1e-6, rtol=0)
    np.testing.assert_allclose(exported.forces, reference["forces"], atol=1e-4, rtol=0)  # float32 export vs eager


def test_committee_shape_and_distinct_members(backend: GMDSGTBackend) -> None:
    members = [_member(0, "checkpoint"), _member(1, "checkpoint")]
    forces = backend.predict_committee(members, WATER.positions, WATER.numbers, species=(1, 8))
    assert forces.shape == (2, 3, 3) and np.isfinite(forces).all()
    assert np.max(np.abs(forces[0] - forces[1])) > 1e-6  # distinct seeds give distinct members
    with pytest.raises(AdapterError, match="at least two members"):
        backend.predict_committee(members[:1], WATER.positions, WATER.numbers)

    frame = Frame(run_id="r", step=0, atom_ids=np.arange(3), numbers=WATER.numbers, positions=WATER.positions,
                  cell=None, pbc=(False, False, False))
    enriched = backend.enrich(frame, _artifact())
    assert enriched.committee_forces.shape == (2, 3, 3)
    np.testing.assert_allclose(enriched.committee_forces, forces)
    unchanged = backend.enrich(Frame(run_id="r", step=0, atom_ids=np.arange(3), numbers=WATER.numbers, positions=WATER.positions,
                                     cell=None, pbc=(False, False, False)), _artifact(with_committee=False))
    assert unchanged.committee_forces is None  # no committee -> metric unavailable, never zeros


def test_non_finite_output_and_model_failures_raise(backend: GMDSGTBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = _member(0, "checkpoint")
    calc = backend.calculator(checkpoint)
    original = calc.compute

    def nan_forces(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        out = original(*args, **kwargs)
        out["forces"][1, 2] = np.nan
        return out

    monkeypatch.setattr(calc, "compute", nan_forces)
    with pytest.raises(AdapterError, match="non-finite"):
        backend.predict(checkpoint, WATER.positions, WATER.numbers)

    def broken(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("CUDA error: device-side assert")

    monkeypatch.setattr(calc, "compute", broken)
    with pytest.raises(AdapterError, match="evaluation with checkpoint.pt failed"):
        backend.predict(checkpoint, WATER.positions, WATER.numbers)

    def wrong_shape(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return {"energy": 0.0, "forces": np.zeros((2, 3))}

    monkeypatch.setattr(calc, "compute", wrong_shape)
    with pytest.raises(AdapterError, match="shape"):
        backend.predict(checkpoint, WATER.positions, WATER.numbers)


def test_unknown_species_is_rejected_before_evaluation(backend: GMDSGTBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = _member(0, "checkpoint")
    calls: list[int] = []
    calc = backend.calculator(checkpoint)
    monkeypatch.setattr(calc, "compute", lambda *a, **k: calls.append(1))
    numbers = np.array([8, 1, 6])  # carbon is not in the model
    with pytest.raises(ConfigurationError, match=r"\['C'\]"):
        backend.predict(checkpoint, WATER.positions, numbers)
    with pytest.raises(ConfigurationError, match="unknown to"):
        backend.predict(checkpoint, WATER.positions, WATER.numbers, species=(8,))  # explicit ModelArtifact.species wins
    frame = Frame(run_id="r", step=0, atom_ids=np.arange(3), numbers=numbers, positions=WATER.positions, cell=None, pbc=(False,) * 3)
    with pytest.raises(ConfigurationError, match="unknown to model m-fixture"):
        backend.enrich(frame, _artifact())
    assert calls == []


@pytest.mark.parametrize(
    ("species_map", "match"),
    [
        ({1: "H", 2: "C"}, r"absent from .*\[1, 8\]"),  # name absent from the model's type list
        ({1: "H", 3: "O"}, "contiguous"),  # non-contiguous type ids
        ({0: "H", 1: "O"}, "contiguous"),  # LAMMPS-style ids start at 1
        ({1: "H", 2: "H"}, "repeats"),
        ({1: "H", 2: "Xx"}, "unknown element"),
        ({}, "empty"),
    ],
)
def test_species_map_errors(species_map: dict, match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        validate_species_map(species_map, (1, 8))


def test_species_map_accepts_subsets_and_sequences() -> None:
    assert validate_species_map({1: "O", 2: "H"}, (1, 8)) == {1: 8, 2: 1}
    assert validate_species_map(["H"], (1, 8)) == {1: 1}
    with pytest.raises(AdapterError, match="declares no species"):
        check_species([1], ())


def test_callers_config_is_never_mutated(backend: GMDSGTBackend, tmp_path: Path) -> None:
    """Port of PHAL's test_allegro_backend_does_not_mutate_the_callers_config (train with a seed)."""

    tool = _generator()
    TINY_TRAIN_CONFIG, _split = tool.TINY_TRAIN_CONFIG, tool._split  # noqa: N806, SLF001

    config = SGTConfig(source_path=SGT_PATH, train=json.loads(json.dumps(TINY_TRAIN_CONFIG)), seed=7)
    before = config.model_dump()
    local = GMDSGTBackend(config)
    train_file, val_file = _split(tmp_path)
    dataset = DatasetVersion("d", str(tmp_path), "x", 8, 6, 2, None, (), "now", {"train_file": str(train_file), "val_file": str(val_file)})
    local.train(dataset, parent=None, work_dir=tmp_path / "t", seed=99)
    local.config.train["train"] = {"n_epochs": 1}
    assert config.model_dump() == before
    summary = json.loads((tmp_path / "t" / "member_0" / "training_summary.json").read_text())
    assert summary["seed"] == 99


def test_generator_is_reproducible(backend: GMDSGTBackend, tmp_path: Path) -> None:
    """Re-running the generator gives byte-identical checkpoints and identical predictions."""

    seed = MANIFEST["members"][0]["seed"]
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run([sys.executable, "-W", "ignore", str(REPO / "tools/fixtures/make_tiny_sgt_models.py"), "--out", str(tmp_path),
                                "--gmd-sgt", SGT_PATH, "--seeds", str(seed)], capture_output=True, text=True, env=env, timeout=600)
    assert completed.returncode == 0, completed.stderr[-2000:]
    regenerated = json.loads((tmp_path / "manifest.json").read_text())["members"][0]
    assert regenerated["species"] == [1, 8] and regenerated["cutoff"] == 4.0
    fresh = backend.predict(tmp_path / regenerated["export"], WATER.positions, WATER.numbers)
    committed = backend.predict(_member(0, "export"), WATER.positions, WATER.numbers)
    np.testing.assert_allclose(fresh.forces, committed.forces, atol=1e-6, rtol=0)
    assert fresh.energy == pytest.approx(committed.energy, abs=1e-6)
