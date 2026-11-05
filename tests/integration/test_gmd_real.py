"""Real GMD (TorchScript build) + GMD_SGT integration through the adapters. Skipped only when the
GMD binary or GMD_SGT is unavailable; the reason is reported."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from gmd_active_learning.adapters.gmd.engine import GMDConfig, GMDEngine
from gmd_active_learning.adapters.sgt.backend import GMDSGTBackend, SGTConfig
from gmd_active_learning.contracts import DatasetVersion, FrameAction
from gmd_active_learning.core.exceptions import CapabilityUnavailableError
from gmd_active_learning.retraining.registry import ModelRegistry
from gmd_active_learning.workflow.protocols import SegmentRequest
from gmd_active_learning.workflow.store import WorkflowStore

pytestmark = pytest.mark.integration
WORKSPACE = Path(__file__).resolve().parents[3]
GMD_BIN = Path(os.environ.get("GMD_EXECUTABLE", WORKSPACE / "GMD" / "build-torch" / "gmd"))
SGT_PATH = os.environ.get("GMD_SGT_PATH", str(WORKSPACE / "GMD_SGT"))
BOX = 7.6
TINY = {
    "model": {"n_species": 20, "n_blocks": 1, "scalar_dim": 8, "irreps": "8x0e", "n_basis": 4, "l_max": 0,
              "local_cutoff": 4.5, "lr_cutoff": 4.5, "long_range_type": "none", "n_heads": 1, "avg_neighbors": 8.0},
    "train": {"n_epochs": 3, "batch_size": 4, "warmup_steps": 1, "lr": 3.0e-3, "patience": 0},
}


def _lattice() -> np.ndarray:
    return np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)], float) * (BOX / 2) + 0.4


@pytest.fixture(scope="module")
def models(tmp_path_factory: pytest.TempPathFactory):  # noqa: ANN201
    if not GMD_BIN.is_file():
        pytest.skip(f"GMD binary not found at {GMD_BIN} (build GMD with -DGMD_ENABLE_TORCH=ON)")
    try:
        backend = GMDSGTBackend(SGTConfig(source_path=SGT_PATH, train=TINY))
    except CapabilityUnavailableError as exc:
        pytest.skip(str(exc))
    root = tmp_path_factory.mktemp("gmd")
    rng = np.random.default_rng(3)
    structures = []
    for n in range(12):
        atoms = Atoms("Ar8", positions=_lattice() + rng.normal(0, 0.1, (8, 3)), cell=np.eye(3) * BOX, pbc=True)
        atoms.calc = LennardJones(epsilon=0.0104, sigma=3.4, rc=3.7, smooth=True)
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        atoms.info["split_group"] = f"g{n % 4}"
        structures.append(atoms)
    write(root / "train.extxyz", structures[:9], format="extxyz")
    write(root / "val.extxyz", structures[9:], format="extxyz")
    dataset = DatasetVersion("ds", str(root), "x", 12, 9, 3, None, (), "now",
                             {"train_file": str(root / "train.extxyz"), "val_file": str(root / "val.extxyz")})
    store = WorkflowStore(root / "s.sqlite")
    store.ensure_run("r", config_hash="c", initial_model_id=None, stage="sampling")
    registry = ModelRegistry(root / "models", store)
    artifacts = []
    for seed in (1, 2):
        backend.config.seed = seed
        out = backend.train(dataset, parent=None, work_dir=root / f"train{seed}")
        export = backend.export(out.checkpoint, root / f"export{seed}")
        artifacts.append(registry.register(checkpoint=out.checkpoint, export=export, dataset_version="ds", parent_model_id=None,
                                           species=out.species, cutoff=out.cutoff, metrics={}, train_mode="fresh", backend="gmd_sgt"))
    xyz = root / "ar8.xyz"
    xyz.write_text("\n".join([ "8", f"{BOX} {BOX} {BOX}", *(f"Ar {x:.6f} {y:.6f} {z:.6f}" for x, y, z in _lattice())]) + "\n")
    return backend, artifacts, xyz, root


def _engine(xyz: Path, **overrides) -> GMDEngine:
    config = dict(executable=str(GMD_BIN), initial_structure=str(xyz), frame_interval=2,
                  directives={"time_step": 1.0, "velocity": 60.0, "velocity_init": "random", "velocity_seed": 5})
    config.update(overrides)
    return GMDEngine(GMDConfig(**config))


def _request(root: Path, name: str, model, steps: int, restart=None, previous=None, index: int = 0) -> SegmentRequest:  # noqa: ANN001
    return SegmentRequest("run-gmd", index, root / name, model, steps, restart, previous)


def test_continuous_and_segmented_runs_agree_and_frames_are_complete(models) -> None:  # noqa: ANN001
    backend, (m1, _m2), xyz, root = models
    engine = _engine(xyz)
    cont_frames: list = []
    cont = engine.run_segment(_request(root, "cont", m1, 20), lambda f: cont_frames.append(f) or FrameAction.CONTINUE)
    assert cont.status == "completed" and cont.end_step == 20
    seg_frames: list = []
    s1 = engine.run_segment(_request(root, "s1", m1, 10), lambda f: seg_frames.append(f) or FrameAction.CONTINUE)
    s2 = engine.run_segment(_request(root, "s2", m1, 10, restart=s1.checkpoint, index=1), lambda f: seg_frames.append(f) or FrameAction.CONTINUE)
    assert s2.status == "completed" and s2.end_step == 20
    last_cont = [f for f in cont_frames if f.step == 20][0]
    last_seg = [f for f in seg_frames if f.step == 20][0]
    # GMD's own test shows bitwise agreement with ml_jit_optimize off and <=1e-10 A with it on.
    assert np.max(np.abs(last_cont.positions - last_seg.positions)) < 1e-6
    assert np.max(np.abs(last_cont.forces - last_seg.forces)) < 1e-5
    # Frame contract: stable IDs, species, cell, pbc, provenance, and forces equal to the model's forces.
    frame = cont_frames[3]
    assert frame.atom_ids.tolist() == list(range(8)) and set(frame.numbers.tolist()) == {18}
    assert frame.pbc == (True, True, True) and np.allclose(np.diag(frame.cell), BOX)
    assert frame.model_sha256 == m1.export_sha256 and frame.model_version == m1.model_id
    reference = backend.calculator(m1.checkpoint_path).compute(frame.positions, frame.numbers, frame.cell, frame.pbc)
    assert np.max(np.abs(np.asarray(reference["forces"]) - frame.forces)) < 1e-4  # float32 TorchScript vs eager


def test_cooperative_stop_and_model_change_at_segment_boundary(models) -> None:  # noqa: ANN001
    _backend, (m1, m2), xyz, root = models
    engine = _engine(xyz, frame_interval=1, control_poll_interval=1)
    seen: list = []

    def stop_on_third(frame):  # noqa: ANN001, ANN202
        seen.append(frame.step)
        return FrameAction.STOP if len(seen) == 3 else FrameAction.CONTINUE

    stopped = engine.run_segment(_request(root, "stop", m1, 100000), stop_on_third)
    assert stopped.status == "stopped" and stopped.checkpoint is not None
    assert stopped.end_step < 100000 and stopped.detail["stop_latency_steps"] >= 0
    # Without an explicit model change the engine refuses a different model on restart ...
    refused = engine.run_segment(_request(root, "refused", m2, 5, restart=stopped.checkpoint, index=1), lambda f: FrameAction.CONTINUE)
    assert refused.status == "failed"
    # ... and accepts and records it when the controller passes the promotion.
    frames: list = []
    changed = engine.run_segment(_request(root, "changed", m2, 5, restart=stopped.checkpoint, previous=m1, index=1),
                                 lambda f: frames.append(f) or FrameAction.CONTINUE)
    assert changed.status == "completed" and frames
    assert frames[-1].model_sha256 == m2.export_sha256
    assert changed.detail["status"]["model_changed_from"]


def test_export_smoke_test_runs_inside_gmd(models, tmp_path: Path) -> None:  # noqa: ANN001
    _backend, (m1, _m2), xyz, _root = models
    report = _engine(xyz).smoke_test(Path(m1.export_path), tmp_path / "smoke")
    assert report["passed"], report
