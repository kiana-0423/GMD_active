"""Real model rejection and rollback with GMD_SGT models and GMD segments (no fakes).

Two independently trained models are registered. A strict gate rejects a candidate on its real
validation metrics; a promoted model then runs a real GMD segment, is rolled back, and the next
GMD segment restarts from the checkpoint with the restored model, recording the change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gmd_active_learning.contracts import DatasetVersion, FrameAction
from gmd_active_learning.core.exceptions import WorkflowError
from gmd_active_learning.retraining.registry import ModelRegistry, PromotionGateConfig, evaluate_gates
from gmd_active_learning.workflow.protocols import SegmentRequest
from gmd_active_learning.workflow.store import WorkflowStore

from .test_gmd_real import _engine, models  # noqa: F401 - module-scoped fixture reused

pytestmark = pytest.mark.integration


def test_rejection_on_real_metrics_and_rollback_with_resumed_md(models, tmp_path: Path) -> None:  # noqa: ANN001, F811
    backend, (m1, m2), xyz, root = models
    dataset = DatasetVersion("ds", str(root), "x", 12, 9, 3, None, (), "now",
                             {"train_file": str(root / "train.extxyz"), "val_file": str(root / "val.extxyz")})
    store = WorkflowStore(tmp_path / "state.sqlite")
    store.ensure_run("rb", config_hash="c", initial_model_id=None, stage="sampling")
    registry = ModelRegistry(tmp_path / "models", store)
    a = registry.register(checkpoint=m1.checkpoint_path, export=m1.export_path, dataset_version="ds", parent_model_id=None,
                          species=list(m1.species), cutoff=m1.cutoff, metrics={}, train_mode="fresh", backend="gmd_sgt")
    b = registry.register(checkpoint=m2.checkpoint_path, export=m2.export_path, dataset_version="ds", parent_model_id=a.model_id,
                          species=list(m2.species), cutoff=m2.cutoff, metrics={}, train_mode="fresh", backend="gmd_sgt")
    registry.promote("rb", a.model_id, gate_report=None, reason="initial", force_initial=True)

    metrics = backend.evaluate(Path(b.checkpoint_path), dataset)
    parent = backend.evaluate(Path(a.checkpoint_path), dataset)
    export_check = backend.validate_export(Path(b.checkpoint_path), Path(b.export_path), dataset)
    strict = evaluate_gates(PromotionGateConfig(max_force_rmse=metrics["force_rmse"] / 2.0, require_engine_smoke_test=False),
                            metrics=metrics, parent_metrics=parent, export_check=export_check, engine_smoke=None)
    assert not strict.passed and not strict.checks["force_rmse"]["passed"]
    with pytest.raises(WorkflowError, match="failed promotion gates"):
        registry.promote("rb", b.model_id, gate_report=strict, reason="strict gate")
    assert store.run("rb").active_model_id == a.model_id  # rejection keeps the active model

    # A different candidate trained on the same data passes a reasonable gate and is promoted.
    c_out = backend.train(dataset, parent=registry.get(a.model_id), work_dir=root / "train_c")
    c_export = backend.export(c_out.checkpoint, root / "export_c")
    c = registry.register(checkpoint=c_out.checkpoint, export=c_export, dataset_version="ds", parent_model_id=a.model_id,
                          species=c_out.species, cutoff=c_out.cutoff, metrics={}, train_mode=c_out.mode, backend="gmd_sgt")
    engine = _engine(xyz)
    smoke = engine.smoke_test(Path(c.export_path), tmp_path / "smoke")
    gate = evaluate_gates(PromotionGateConfig(max_force_rmse=100.0, max_relative_regression=100.0),
                          metrics=backend.evaluate(Path(c.checkpoint_path), dataset), parent_metrics=parent,
                          export_check=backend.validate_export(Path(c.checkpoint_path), Path(c.export_path), dataset), engine_smoke=smoke)
    assert gate.passed, gate.checks
    registry.promote("rb", c.model_id, gate_report=gate, reason="passed")

    seg0 = engine.run_segment(SegmentRequest("rb", 0, tmp_path / "seg0", registry.get(c.model_id), 10, None, None), lambda f: FrameAction.CONTINUE)
    assert seg0.status == "completed"
    restored = registry.rollback("rb", reason="operator rollback after unstable MD")
    assert restored.model_id == a.model_id and store.run("rb").active_model_id == a.model_id
    frames: list = []
    seg1 = engine.run_segment(SegmentRequest("rb", 1, tmp_path / "seg1", restored, 10, seg0.checkpoint, registry.get(c.model_id)),
                              lambda f: frames.append(f) or FrameAction.CONTINUE)
    assert seg1.status == "completed" and seg1.end_step == 20
    assert all(frame.model_sha256 == restored.export_sha256 and frame.model_version == restored.model_id for frame in frames)
    assert seg1.detail["status"]["model_changed_from"] == c.export_sha256
    assert np.isfinite(frames[-1].forces).all()
    assert [p["action"] for p in store.promotions("rb")] == ["promote", "reject", "promote", "rollback"]
