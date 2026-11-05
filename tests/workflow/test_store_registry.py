from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gmd_active_learning.core.exceptions import WorkflowError
from gmd_active_learning.retraining.registry import GateReport, ModelRegistry, PromotionGateConfig, evaluate_gates
from gmd_active_learning.retraining.triggers import RetrainTriggerConfig, evaluate_trigger
from gmd_active_learning.workflow.store import WorkflowStore


@pytest.fixture()
def store(tmp_path: Path) -> WorkflowStore:
    store = WorkflowStore(tmp_path / "state.sqlite")
    store.ensure_run("run", config_hash="cfg", initial_model_id=None, stage="sampling")
    return store


def _artifact(tmp_path: Path, name: str, payload: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def test_store_survives_reopen_and_rejects_config_change(tmp_path: Path) -> None:
    store = WorkflowStore(tmp_path / "state.sqlite")
    store.ensure_run("run", config_hash="cfg", initial_model_id="m0", stage="sampling")
    store.transition("run", "monitoring", cycle=1, detail={"segment": 0})
    store.close()
    reopened = WorkflowStore(tmp_path / "state.sqlite")
    run = reopened.run("run")
    assert (run.stage, run.cycle, run.active_model_id) == ("monitoring", 1, "m0")
    assert [item["stage_to"] for item in reopened.transitions("run")] == ["sampling", "monitoring"]
    with pytest.raises(WorkflowError):
        reopened.ensure_run("run", config_hash="other", initial_model_id=None, stage="sampling")


def test_idempotent_records(store: WorkflowStore) -> None:
    summary = {"event_id": "evt-1", "trigger_step": 10, "complete": True}
    assert store.record_event("run", 0, summary, path=None) is True
    assert store.record_event("run", 0, summary, path=None) is False
    assert store.record_candidate("run", 0, candidate_id="c1", event_id="evt-1", score=2.0, region_path="r.json", fingerprint=[0.0]) is True
    assert store.record_candidate("run", 0, candidate_id="c1", event_id="evt-1", score=2.0, region_path="r.json", fingerprint=[0.0]) is False
    assert store.record_sample(sample_id="s1", task_id="t1", candidate_id="c1", run_id="run", cycle=0, path="s.extxyz", split_group="evt-1")
    # A second sample for the same task is a duplicate label and is refused.
    assert not store.record_sample(sample_id="s2", task_id="t1", candidate_id="c1", run_id="run", cycle=0, path="x", split_group="evt-1")
    assert [row["sample_id"] for row in store.samples("run")] == ["s1"]


def test_transaction_rolls_back_on_error(store: WorkflowStore) -> None:
    with pytest.raises(RuntimeError):
        with store.transaction() as cursor:
            cursor.execute("INSERT INTO blobs(key, value, updated_at) VALUES('k', '1', 'now')")
            raise RuntimeError("crash mid-transaction")
    assert store.get_blob("k") is None


def _passing_report() -> GateReport:
    return GateReport(True, {"all": {"passed": True}})


def test_registry_copies_hashes_and_detects_tampering(tmp_path: Path, store: WorkflowStore) -> None:
    registry = ModelRegistry(tmp_path / "models", store)
    ckpt = _artifact(tmp_path, "ckpt_best.pt", b"weights-v1")
    export = _artifact(tmp_path, "model.pt", b"torchscript-v1")
    artifact = registry.register(checkpoint=ckpt, export=export, dataset_version="ds-1", parent_model_id=None, species=[8, 1], cutoff=5.0, metrics={"force_rmse": 0.1}, train_mode="fresh", backend="gmd_sgt")
    assert Path(artifact.checkpoint_path).read_bytes() == b"weights-v1"
    assert artifact.species == (1, 8)
    # Registering the same files again returns the same record (idempotent).
    assert registry.register(checkpoint=ckpt, export=export, dataset_version="ds-1", parent_model_id=None, species=[1, 8], cutoff=5.0, metrics={}, train_mode="fresh", backend="gmd_sgt").model_id == artifact.model_id
    # Source changes after registration do not affect the stored copy.
    ckpt.write_bytes(b"changed")
    registry.verify(artifact.model_id)
    stored = Path(artifact.checkpoint_path)
    os.chmod(stored, 0o600)
    stored.write_bytes(b"tampered")
    with pytest.raises(WorkflowError, match="checksum"):
        registry.verify(artifact.model_id)


def test_promotion_requires_passed_gates_and_rollback_restores_previous(tmp_path: Path, store: WorkflowStore) -> None:
    registry = ModelRegistry(tmp_path / "models", store)
    m0 = registry.register_external(checkpoint=_artifact(tmp_path, "m0.pt", b"m0"), export=_artifact(tmp_path, "m0x.pt", b"m0x"), species=[1], cutoff=4.0, backend="gmd_sgt")
    with pytest.raises(WorkflowError):
        registry.promote("run", m0.model_id, gate_report=None, reason="no gates")  # not flagged initial
    registry.promote("run", m0.model_id, gate_report=None, reason="initial model", force_initial=True)
    m1 = registry.register(checkpoint=_artifact(tmp_path, "m1.pt", b"m1"), export=None, dataset_version="ds-1", parent_model_id=m0.model_id, species=[1], cutoff=4.0, metrics={"force_rmse": 9.0}, train_mode="finetune", backend="gmd_sgt")
    with pytest.raises(WorkflowError, match="failed promotion gates"):
        registry.promote("run", m1.model_id, gate_report=GateReport(False, {"force_rmse": {"passed": False}}), reason="cycle 1")
    assert registry.state(m1.model_id) == "rejected"
    assert store.run("run").active_model_id == m0.model_id
    m2 = registry.register(checkpoint=_artifact(tmp_path, "m2.pt", b"m2"), export=None, dataset_version="ds-2", parent_model_id=m0.model_id, species=[1], cutoff=4.0, metrics={"force_rmse": 0.1}, train_mode="finetune", backend="gmd_sgt")
    registry.promote("run", m2.model_id, gate_report=_passing_report(), reason="cycle 2")
    assert store.run("run").active_model_id == m2.model_id and registry.state(m0.model_id) == "retired"
    restored = registry.rollback("run", reason="unstable MD with m2")
    assert restored.model_id == m0.model_id
    assert store.run("run").active_model_id == m0.model_id
    assert registry.state(m2.model_id) == "rejected"
    assert [item["action"] for item in store.promotions("run")] == ["promote", "reject", "promote", "rollback"]
    with pytest.raises(WorkflowError):
        registry.rollback("run", reason="again", to_model_id=m1.model_id)  # rejected model is never a target


def test_gates_fail_closed_on_missing_evidence() -> None:
    config = PromotionGateConfig(max_force_rmse=0.2)
    good = {"force_rmse": 0.1, "n_validation": 5}
    assert evaluate_gates(config, metrics=good, parent_metrics={"force_rmse": 0.12}, export_check={"max_force_deviation": 1e-6}, engine_smoke={"passed": True}).passed
    assert not evaluate_gates(config, metrics=good, parent_metrics=None, export_check=None, engine_smoke={"passed": True}).passed
    assert not evaluate_gates(config, metrics=good, parent_metrics=None, export_check={"max_force_deviation": 1e-6}, engine_smoke=None).passed
    regression = evaluate_gates(config, metrics=good, parent_metrics={"force_rmse": 0.05}, export_check={"max_force_deviation": 0.0}, engine_smoke={"passed": True})
    assert not regression.passed and not regression.checks["no_regression_vs_active"]["passed"]
    assert not evaluate_gates(config, metrics={"force_rmse": 0.1, "n_validation": 0}, parent_metrics=None, export_check={"max_force_deviation": 0.0}, engine_smoke={"passed": True}).passed
    relaxed = PromotionGateConfig(require_export_validation=False, require_engine_smoke_test=False)
    assert evaluate_gates(relaxed, metrics=good, parent_metrics=None, export_check=None, engine_smoke=None).passed


def test_retrain_trigger_rules() -> None:
    config = RetrainTriggerConfig(min_new_samples=5, interval_hours=2.0)
    assert evaluate_trigger(config, new_samples=1, pending_tasks=0, last_trained_at=None).reason == "below_min_new_samples"  # F-13 fixed
    assert evaluate_trigger(RetrainTriggerConfig(train_on_first_samples=True), new_samples=1, pending_tasks=0, last_trained_at=None).reason == "initial_samples"
    assert evaluate_trigger(config, new_samples=5, pending_tasks=1, last_trained_at=None).reason == "labels_pending"
    assert evaluate_trigger(config, new_samples=5, pending_tasks=0, last_trained_at=None).reason == "sample_count"
    past = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    assert evaluate_trigger(config, new_samples=1, pending_tasks=0, last_trained_at=past).reason == "time_interval"
    assert evaluate_trigger(config, new_samples=0, pending_tasks=0, last_trained_at=past, manual=True).reason == "no_new_samples"
    assert evaluate_trigger(RetrainTriggerConfig(enabled=False), new_samples=1, pending_tasks=0, last_trained_at=None, manual=True).due
    assert not evaluate_trigger(RetrainTriggerConfig(enabled=False), new_samples=50, pending_tasks=0, last_trained_at=None).due


def test_registry_stores_and_verifies_extra_files(tmp_path: Path, store: WorkflowStore) -> None:
    registry = ModelRegistry(tmp_path / "models", store)
    member = _artifact(tmp_path, "member1.pt", b"committee-1")
    artifact = registry.register(checkpoint=_artifact(tmp_path, "c.pt", b"c"), export=None, dataset_version="ds", parent_model_id=None,
                                 species=[1], cutoff=4.0, metrics={}, train_mode="fresh", backend="gmd_sgt", extra_files={"committee_1": member})
    stored = Path(artifact.metadata["extra_files"]["committee_1"]["path"])
    assert stored.read_bytes() == b"committee-1"
    os.chmod(stored, 0o600)
    stored.write_bytes(b"x")
    with pytest.raises(WorkflowError, match="extra file"):
        registry.verify(artifact.model_id)
